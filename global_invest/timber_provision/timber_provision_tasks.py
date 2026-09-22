"""Timber-provision GEP tasks: the value raster summed to countries in the library.

The reported number is OUR run: the pipeline's value raster (verified equal to the
net-return layer masked and floored, see timber_provision_functions) summed per country on
the 10-arcsecond country-id raster. The committed Forestry CSV stays as the test anchor
the run is compared against, never as the output."""

import os
import numpy as np
import pandas as pd
import rasterio
import hazelbean as hb
from global_invest import utilities
from global_invest.timber_provision import timber_provision_functions as tp


def publish_inputs(p):
    """Every GEP task's first line: the timber_provision es_config row and the data reference
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'timber_provision', log=hb.log)
    utilities.hydrate_es_parameters(p, 'timber_provision', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def gep_calculation(p):
    """GEP valuation for timber provision: the value raster summed per country."""
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'timber_provision')
    if already_done:
        return

    value_src = rasterio.open(p.timber_provision_value_raster_path)
    zone_src = rasterio.open(p.timber_provision_zone_raster_path)
    n_zones = 1000
    zone_sums = np.zeros(n_zones + 1, dtype='float64')
    rows_per_block = 2048
    for row0 in range(0, value_src.shape[0], rows_per_block):
        h = min(rows_per_block, value_src.shape[0] - row0)
        win = rasterio.windows.Window(0, row0, value_src.shape[1], h)
        zone_sums += utilities.sum_by_zone(value_src.read(1, window=win), zone_src.read(1, window=win), n_zones)

    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = utilities.collapse_countries_to_r250(p.df_countries)[attr_cols]
    df_gep = tp.timber_gep_from_zone_sums(zone_sums, countries)
    df_gep['year'] = int(p.gep_base_year)

    # Both valuations, side by side, because the choice between them is live and the issues
    # document recommends the second. Publishing one silently would hide a $43bn decision.
    df_gep = tp.cwon_forest_rent_by_country(
        # pd.read_stata, like extractive_energy and fisheries: hb.df_read is csv-only and a
        # .dta reaches it as a malformed csv rather than as the wrong reader.
        pd.read_stata(str(p.get_path(p.timber_provision_cwon_forest_rent_path))),
        df_gep, int(p.gep_base_year))
    # And the bound: FAOSTAT roundwood priced at each country's own export unit value. Not a
    # third estimate -- a land share is a fraction of the gross value of the wood it comes from,
    # so a country whose GEP exceeds its own gross is saying something impossible, and that is
    # visible per country here rather than argued globally.
    df_gep = tp.roundwood_gross_value_by_country(
        hb.df_read(str(p.get_path(p.timber_provision_faostat_roundwood_path))),
        df_gep, int(p.gep_base_year))
    # The account's timber value is CWoN's rent, so `timber_provision_gep` -- the shared key
    # every other service writes and the account reads -- IS the CWoN rent. The spatial estimate
    # is kept beside it as `timber_provision_gep_spatial`: it is the only forestry layer this
    # library has.
    df_gep = df_gep.rename(columns={'timber_provision_gep': 'timber_provision_gep_spatial',
                                    'timber_provision_gep_cwon_rent': 'timber_provision_gep'})
    # The fuelwood decomposition. timber_provision_gep is CWoN's forest rent, which is built
    # from FAOSTAT items 1864 (Wood Fuel), 1866 and 1867 (industrial roundwood) summed BEFORE the
    # rental ratio, so fuelwood is already inside it. The account keeps the timber figure whole --
    # that is the decision -- and publishes the split beside it so a separate fuelwood row is
    # visibly a SUBSET rather than an addition. Adding one on top would count the same rent twice.
    faostat = hb.df_read(str(p.get_path(p.timber_provision_faostat_roundwood_path)))
    df_gep = tp.fuelwood_share_of_forest_rent(faostat, df_gep, int(p.gep_base_year))
    df_gep['timber_provision_gep_fuelwood_part'] = (
        df_gep['timber_provision_gep'] * df_gep['fuelwood_share_of_rent'])
    df_gep['timber_provision_gep_industrial_part'] = (
        df_gep['timber_provision_gep'] * (1.0 - df_gep['fuelwood_share_of_rent']))
    hb.df_write(df_gep[attr_cols + ['year', 'timber_provision_gep',
                                    'timber_provision_gep_fuelwood_part',
                                    'timber_provision_gep_industrial_part',
                                    'timber_provision_gep_spatial',
                                    'timber_roundwood_gross_value']],
                service_results['gep_by_country_base_year'])
    fuel = df_gep['timber_provision_gep_fuelwood_part'].sum()
    industrial = df_gep['timber_provision_gep_industrial_part'].sum()
    hb.log('  of which fuelwood: %s (%.1f%%); industrial roundwood: %s. A separate fuelwood '
           'service would be a SUBSET of the timber figure, not an addition.'
           % (f'{fuel:,.2f}', 100 * fuel / (fuel + industrial), f'{industrial:,.2f}'))
    for column, label in (('timber_provision_gep_spatial', 'spatial'),
                          ('timber_provision_gep', 'CWoN rent')):
        priced = df_gep[df_gep['timber_roundwood_gross_value'].gt(0) & df_gep[column].gt(0)]
        over = priced[priced[column] > priced['timber_roundwood_gross_value']]
        if len(over):
            hb.log('  %s exceeds its own gross roundwood value in %d countries: %s'
                   % (label, len(over), ', '.join(over['iso3_r250_label'].head(6))))

    spatial = df_gep['timber_provision_gep_spatial'].sum()
    rental = df_gep['timber_provision_gep'].sum()
    hb.log(f'Total timber_provision GEP for base year {p.gep_base_year}: {rental:,.2f} '
           f'(CWoN rent, the account\'s figure)')
    hb.log(f'  the superseded spatial estimate    : {spatial:,.2f} '
           f'({int(df_gep["timber_provision_gep_spatial"].gt(0).sum())} countries), '
           f'ratio {rental / spatial:.2f}')
    committed = hb.df_read(p.timber_provision_gep_path)
    hb.log(f'Committed Forestry table total: {committed.select_dtypes("number").iloc[:, -1].sum():,.2f} (the test anchor).')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)


def fuelwood_gep(p):
    """Fuelwood as its own table: our lambda-applied estimate, the reference, and the gap.

    Fuelwood is NOT an additional service on top of timber. CWoN's forest rent is built from
    FAOSTAT items 1864 (Wood Fuel), 1866 and 1867 (industrial roundwood) summed before the rental
    ratio, so the fuelwood rent is already inside `timber_provision_gep`. This task makes that part
    addressable rather than adding to it, because the account may want a fuelwood row and needs to
    see it is a subset.

    Three columns, so the choice and the discrepancy are both visible:

    - `fuelwood_gep_from_forest_rent`  the fuelwood part of CWoN's rent, published in the timber
      table as `timber_provision_gep_fuelwood_part`. This is the figure consistent with every other
      rent-based service in the account.
    - `fuelwood_gep_gross_at_export_price`  FAOSTAT wood fuel production times the country's own
      export unit value, with NO lambda. The upper bound, and the shape the reference has.
    - `fuelwood_gep_reference`  the author's committed output, staged from the drive.

    The reference applies no ecosystem share at all: across the 178 countries FAOSTAT also
    covers, it is $181,140,611,163 over 1,929,080,408 m3, an implied $93.90/m3 against FAO's export
    unit value of $67.47. That is a gross value, not a rent, and it is the lambda question the
    issues document raises for this service.
    """
    publish_inputs(p)
    p.fuelwood_gep_path = os.path.join(p.cur_dir, 'fuelwood_gep_by_country.csv')
    if not p.run_this:
        return
    if not hb.path_exists(p.fuelwood_gep_path):
        faostat = hb.df_read(str(p.get_path(p.timber_provision_faostat_roundwood_path)))
        countries = utilities.collapse_countries_to_r250(p.df_countries)[
            utilities.GEP_COUNTRY_ATTR_COLS]

        # the lambda-applied figure: the fuelwood share of CWoN's rent
        rent = tp.cwon_forest_rent_by_country(
            pd.read_stata(str(p.get_path(p.timber_provision_cwon_forest_rent_path))),
            countries, int(p.gep_base_year))
        rent = tp.fuelwood_share_of_forest_rent(faostat, rent, int(p.gep_base_year))
        rent['fuelwood_gep_from_forest_rent'] = (
            rent['timber_provision_gep_cwon_rent'] * rent['fuelwood_share_of_rent'])

        # the no-lambda upper bound, which is the shape the reference has. Wood fuel priced with
        # ITS OWN export unit value, not the fuelwood share of roundwood gross -- see the note on
        # wood_fuel_gross_value_by_country.
        gross = tp.wood_fuel_gross_value_by_country(faostat, countries, int(p.gep_base_year))
        rent = rent.merge(gross[['iso3_r250_id', 'wood_fuel_gross_value']],
                          on='iso3_r250_id', how='left')
        rent['fuelwood_gep_gross_at_export_price'] = rent['wood_fuel_gross_value']

        reference = pd.read_excel(str(p.get_path(p.fuelwood_reference_path)))
        reference.columns = ['iso3_r250_label', 'fuelwood_gep_reference_1000usd']
        reference['fuelwood_gep_reference'] = reference['fuelwood_gep_reference_1000usd'] * 1000.0
        out = rent.merge(reference[['iso3_r250_label', 'fuelwood_gep_reference']],
                         on='iso3_r250_label', how='left')
        out['year'] = int(p.gep_base_year)
        hb.df_write(out[utilities.GEP_COUNTRY_ATTR_COLS +
                        ['year', 'fuelwood_gep_from_forest_rent',
                         'fuelwood_gep_gross_at_export_price', 'fuelwood_gep_reference']],
                    p.fuelwood_gep_path)
        hb.log('fuelwood GEP: lambda-applied (share of CWoN rent) %.6g; gross at export price, no '
               'lambda %.6g; the author\'s committed reference %.6g USD'
               % (out['fuelwood_gep_from_forest_rent'].sum(),
                  out['fuelwood_gep_gross_at_export_price'].sum(),
                  out['fuelwood_gep_reference'].sum()))
    return True


# ---------------------------------------------------------------------------------------------
# Timber as a scenario ES shock, alongside carbon on FRS
#
# Carbon and timber both reach GTAP as afeall on forestry, and only one of them is what forestry
# land actually yields. Carbon is a proxy: it is a zonal mean density by land-cover class, so it
# moves when composition changes rather than when remaining forest becomes more or less productive,
# and it tells the model that land grows less WOOD when what changed is how much CARBON it holds.
# Timber is the direct measure -- biomass yield x log price, net of transport, on land the Lesiv map
# marks as actively managed.
#
# These tasks add timber BESIDE carbon rather than replacing it, so the two can be compared on the
# same scenarios and the size of the proxy's error is measurable instead of argued. Carbon's seam is
# untouched; a run that does not build the timber tree behaves exactly as before.
#
# The machinery is carbon's, deliberately. generate_carbon_density_raster and dynamic_shock_rows
# take a DENSITY LOOKUP (lulc_id x zone_id -> value) and know nothing about carbon, so timber needs
# its own lookup table and nothing else. Reimplementing them here would fork a verified path to
# change one input raster.
# ---------------------------------------------------------------------------------------------

def timber_value_density_table(p):
    """Mean timber value density per (land-cover class, zone), the lookup the shock reads.

    The same construction as terrestrial_carbon's density table, over the timber value raster
    instead of the carbon density raster, so the two shocks differ in their layer and in nothing
    else.
    """
    publish_inputs(p)
    p.timber_value_density_lookup_table_path = os.path.join(
        p.cur_dir, 'timber_value_density_lookup_table.csv')
    if not p.run_this:
        return True
    if hb.path_exists(p.timber_value_density_lookup_table_path):
        return True

    from global_invest.terrestrial_carbon import terrestrial_carbon_tasks as tct

    # Keyed on SEALS7 classes, because that is what the scenario maps carry. Carbon's shock lookup
    # (carbon_density_lookup_seals7_spawn.csv) is built the same way; its GEP lookup is keyed on the
    # 37 ESA classes and applied to a SEALS7 map matches zero cells -- the library refuses that
    # outright, which is how the first attempt at this table was caught.
    #
    # The base-year SEALS7 map is the one the ES shocks form their 2023 denominator from, under the
    # project's own fine_processed_inputs. The timber value raster and the carbon zones are on its
    # grid already (10 arcsec global), so nothing is resampled.
    utilities.hydrate_es_parameters(p, 'terrestrial_carbon', log=hb.log)
    base_year = int(getattr(p, 'es_shock_base_year', None) or p.key_base_year)
    seals7_base_path = os.path.join(p.intermediate_dir, 'fine_processed_inputs', 'lulc', 'esa', 'seals7',
                                    'lulc_esa_seals7_%d.tif' % base_year)
    zones_path = p.terrestrial_quantity_input_path
    value_path = p.timber_provision_value_raster_path
    for label, path in (('SEALS7 base map', seals7_base_path), ('carbon zones', zones_path),
                        ('timber value raster', value_path)):
        if not hb.path_exists(path):
            raise NameError('%s not found at %s' % (label, path))
    shapes = {label: rasterio.open(path).shape for label, path in
              (('base', seals7_base_path), ('zones', zones_path), ('value', value_path))}
    if len(set(shapes.values())) != 1:
        raise ValueError('timber density table needs one grid, got %s' % shapes)

    summary = tct.stack_layers_summary(
        group_layer1_path=seals7_base_path,
        group_layer2_path=zones_path,
        value_layer_path=value_path,
        group1_name='lulc_id',
        group2_name='carbon_zone_id',
        value_name='carbon_density')
    summary.to_csv(p.timber_value_density_lookup_table_path, index=False)
    hb.log('  timber value density table: %d (lulc x zone) rows -> %s'
           % (len(summary), p.timber_value_density_lookup_table_path))
    return True


def _timber_value_raster_path(p, scenario, year):
    return os.path.join(p.cur_dir, 'timber_value_on_forest_%s_%d.tif' % (scenario, year))


def _timber_summary_path(p, scenario, year):
    return os.path.join(p.cur_dir, 'timber_value_by_zone_%s_%d.csv' % (scenario, year))


def _write_eligible_value(value_path, base_lulc_path, out_path):
    """The base-year timber value on the fixed eligible area (managed AND forest in the base map)."""
    lulc_ndv = hb.get_ndv_from_path(base_lulc_path)
    def eligible(value_block, lulc_block):
        return tp.timber_eligible_value(value_block, lulc_block, lulc_ndv=lulc_ndv)
    hb.raster_calculator_flex([value_path, base_lulc_path], eligible, out_path, datatype=6, ndv=np.nan)


def _log_eligible_coverage(p, value_path, eligible_path):
    """How much base-year managed value the intersection with the base-year map keeps and excludes."""
    total = kept = 0.0
    with rasterio.open(value_path) as v, rasterio.open(eligible_path) as e:
        for _, window in v.block_windows(1):
            a = v.read(1, window=window).astype('float64'); b = e.read(1, window=window).astype('float64')
            total += float(np.nansum(np.where(a > 0, a, 0.0))); kept += float(np.nansum(np.where(b > 0, b, 0.0)))
    excluded = 100.0 * (1.0 - kept / total) if total > 0 else float('nan')
    hb.log('  timber eligible area: base-year managed value %.4g; %.4g (%.1f%%) on cells the base-year map classes '
           'as forest and kept; %.1f%% excluded where the two maps disagree' % (total, kept, 100 - excluded, excluded))
    p.timber_eligible_value_total = kept
    p.timber_managed_value_total = total
    p.timber_excluded_share_pct = excluded


def _summary_complete(summary_path, min_zones=100):
    """A zonal summary counts as done only if it holds the zones and a positive total: a summary
    written by a worker that was killed (or read from a torn raster) has a header and nothing
    else, or zeros, and a guard that trusts its existence keeps it forever."""
    if not hb.path_exists(summary_path):
        return False
    try:
        d = pd.read_csv(summary_path)
    except Exception:
        return False
    return len(d) >= min_zones and 'total' in d.columns and float(d['total'].sum()) > 0


def _value_one_timber_map(job):
    """Worker: the timber value raster of one scenario map and its zonal summary (both cached).
    An incomplete summary is rebuilt together with its raster, because a torn raster is what
    usually produced it."""
    scenario, year, lulc_path, eligible_path, raster_path, summary_path, boundary_path, id_col = job
    from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf
    if hb.path_exists(summary_path) and not _summary_complete(summary_path):
        for stale in (summary_path, summary_path[:-4] + '_zone_ids.tif', raster_path):
            if os.path.exists(stale):
                os.remove(stale)
    if not hb.path_exists(raster_path):
        lulc_ndv = hb.get_ndv_from_path(lulc_path)
        def on_forest(eligible_block, lulc_block):
            return tp.timber_value_on_forest(eligible_block, lulc_block, lulc_ndv=lulc_ndv)
        hb.raster_calculator_flex([eligible_path, lulc_path], on_forest, raster_path, datatype=6, ndv=np.nan)
    if not hb.path_exists(summary_path):
        tcf.summarize_raster_by_region(raster_path, boundary_path, summary_path, year=year, id_column=id_col)
    return summary_path


def _biomass_one_map(job):
    """Worker for the aboveground-carbon measure (D19): the forest biomass carbon a scenario map
    carries -- observed stock kept while forest, the zone's mean forest density where forest was
    gained -- and its zonal summary (both cached, torn summaries rebuilt)."""
    scenario, year, lulc_path, base_forest_path, new_forest_path, raster_path, summary_path, boundary_path, id_col = job
    from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf
    if hb.path_exists(summary_path) and not _summary_complete(summary_path):
        for stale in (summary_path, summary_path[:-4] + '_zone_ids.tif', raster_path):
            if os.path.exists(stale):
                os.remove(stale)
    if not hb.path_exists(raster_path):
        lulc_ndv = hb.get_ndv_from_path(lulc_path)
        def on_map(base_block, new_block, lulc_block):
            return tp.forest_biomass_carbon_on_map(base_block, new_block, lulc_block, lulc_ndv=lulc_ndv)
        hb.raster_calculator_flex([base_forest_path, new_forest_path, lulc_path], on_map, raster_path, datatype=6, ndv=np.nan)
    if not hb.path_exists(summary_path):
        tcf.summarize_raster_by_region(raster_path, boundary_path, summary_path, year=year, id_column=id_col)
    return summary_path


def _write_biomass_base_and_new_forest(p, base_lulc_path, base_forest_path, new_forest_path):
    """Once per project: the base-year forest biomass carbon per cell (NaN off the base forest) and
    the carbon zone's mean forest density x cell area the mature-density assumption credits to
    forest gained later. The lookup is built from the aboveground layer on the base forest."""
    from global_invest.terrestrial_carbon import terrestrial_carbon_tasks as tct
    density_path = p.timber_provision_biomass_carbon_density_path
    zones_path = p.terrestrial_quantity_input_path
    ha_path = getattr(p, 'ha_per_cell_10sec_path', None) or p.get_path(*utilities.HA_PER_CELL_10SEC_REF_PARTS)
    for label, path in (('aboveground biomass carbon', density_path), ('carbon zones', zones_path), ('ha per cell', ha_path)):
        if not hb.path_exists(path):
            raise NameError('%s not found at %s' % (label, path))
    shapes = {label: rasterio.open(path).shape for label, path in (('density', density_path), ('zones', zones_path), ('ha', ha_path), ('base map', base_lulc_path))}
    if len(set(shapes.values())) != 1:
        raise ValueError('the biomass measure needs one grid, got %s' % shapes)
    lulc_ndv = hb.get_ndv_from_path(base_lulc_path)
    if not hb.path_exists(base_forest_path):
        def base(density_block, ha_block, lulc_block):
            return tp.forest_biomass_carbon_base(density_block, ha_block, lulc_block, lulc_ndv=lulc_ndv)
        hb.raster_calculator_flex([density_path, ha_path, base_lulc_path], base, base_forest_path, datatype=6, ndv=np.nan)
    lookup_path = os.path.join(p.cur_dir, 'forest_agbc_density_by_carbon_zone.csv')
    if not hb.path_exists(lookup_path):
        summary = tct.stack_layers_summary(group_layer1_path=base_lulc_path, group_layer2_path=zones_path,
                                           value_layer_path=density_path, group1_name='lulc_id',
                                           group2_name='carbon_zone_id', value_name='agbc_density')
        summary = summary[summary['lulc_id'] == tp.SEALS7_FOREST_ID]
        summary.to_csv(lookup_path, index=False)
    lookup = pd.read_csv(lookup_path)
    mean_col = [c for c in lookup.columns if c.startswith('agbc_density') and ('mean' in c or c == 'agbc_density')][0]
    by_zone = dict(zip(lookup['carbon_zone_id'].astype(int), lookup[mean_col].astype(float)))
    hb.log('  biomass measure: forest AGB carbon density lookup, %d carbon zones, world median %.1f Mg C/ha (%s)'
           % (len(by_zone), float(np.median(list(by_zone.values()))) if by_zone else float('nan'), os.path.basename(density_path)))
    if not hb.path_exists(new_forest_path):
        def new_forest(zone_block, ha_block):
            dens = np.vectorize(lambda z: by_zone.get(int(z), 0.0), otypes=['float32'])(zone_block)
            ha = np.where(np.asarray(ha_block) > 0, ha_block, 0.0).astype('float32')
            return (dens * ha).astype('float32')
        hb.raster_calculator_flex([zones_path, ha_path], new_forest, new_forest_path, datatype=6, ndv=np.nan)


TIMBER_SHOCK_SIGNATURE = 'timber_provision_shock_signature.json'   # beside the table, in the shared es_shocks dir


def timber_provision_shock(p):
    """Per-scenario 300 m LULC -> a timber ES-productivity shock on FRS.

    Timber value per zone is the base-year managed net return summed over a FIXED eligible area --
    the cells that are managed in the base year (Lesiv mask, positive net return) and forest in the
    model's base-year map -- retained where the scenario map still says forest and removed where it
    has become non-forest (timber_provision_functions). Forest gained outside that area is not
    credited; a scenario cell without land-cover data is not treated as a conversion. The zonal
    step and the shock arithmetic are carbon's (same zones, same anchor years, same change from
    the base year), so the two tables stay comparable row for row. The per-map value rasters are
    written in parallel (p.num_workers). The task logs how much base-year managed value the
    intersection with the base-year map excludes.

    The class x zone mean lookup this task read before 21 Sep 2026 (timber_value_density_table) is
    no longer used: it diluted forest by unmanaged forest and gave non-forest classes value where
    the mask overlapped them, so afforestation lowered the measure and deforestation raised it.

    NOT wired into any GTAP pass by default; the run file routes it.
    """
    if not getattr(p, 'timber_provision_shock_output_path', None):
        p.timber_provision_shock_output_path = os.path.join(
            getattr(p, 'es_shock_dir', None) or p.project_dir, 'timber_provision_interpolated.csv')
    if not p.run_this:
        return
    import geopandas as gpd

    from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf
    from global_invest.terrestrial_carbon import terrestrial_carbon_tasks as tct

    # Carbon's export keys: same zones, same boundary, same activity (FRS). Borrowing them is what
    # makes the two seams comparable row for row rather than merely similar.
    utilities.hydrate_es_parameters(p, 'terrestrial_carbon', log=hb.log)
    base_scenario = utilities.required_base_scenario(p, 'terrestrial_carbon')
    es_shock_base_year = int(p.es_shock_base_year)
    anchor_years = sorted(y for y in map(int, p.es_shock_years) if y > es_shock_base_year)

    # Same resolution as carbon, from the same helper: the maps a project names by template.
    scenarios = tct._resolve_scenario_lulc_paths(p, base_scenario, anchor_years)

    reference_lulc_path = p.scenario_lulc_paths[base_scenario][anchor_years[-1]]
    tct._align_zones_to_lulc_grid(p, reference_lulc_path)
    tct._inject_base_year_map(p, base_scenario, es_shock_base_year, reference_lulc_path)

    zone_labels = tcf.zone_labels_from_boundary(
        gpd.read_file(p.region_boundary_path, engine='pyogrio'),
        p.terrestrial_carbon_shock_id_col, p.terrestrial_carbon_shock_endw_col,
        p.terrestrial_carbon_shock_reg_col, p.terrestrial_carbon_shock_endw_format)

    # WHICH RESOURCE the shock measures (D17 'net_return' is the default; D19 'aboveground_carbon'
    # is the alternative proxy construction, set by the variant's caller).
    measure = getattr(p, 'timber_provision_resource_measure', None) or 'net_return'
    if measure not in ('net_return', 'aboveground_carbon'):
        raise ValueError("p.timber_provision_resource_measure must be 'net_return' or 'aboveground_carbon', got %r" % measure)
    value_path = p.timber_provision_value_raster_path
    if measure == 'net_return' and not hb.path_exists(value_path):
        raise NameError('timber value raster not found at %s' % value_path)
    if es_shock_base_year not in p.scenario_lulc_paths.get(base_scenario, {}):
        raise NameError('timber needs the base-year map (%d) among the baseline maps to fix the eligible area' % es_shock_base_year)
    base_lulc_path = p.scenario_lulc_paths[base_scenario][es_shock_base_year]
    # Reuse the table when it was made from these very maps and settings; the per-map guards below
    # still cost minutes of verification, and a pass that finds the table must return in seconds.
    shock_maps = sorted({m for by_year in p.scenario_lulc_paths.values() for m in by_year.values()})
    shock_outputs = [p.timber_provision_shock_output_path]
    reason = utilities.reuse_reason(p, 'timber_provision', shock_outputs, TIMBER_SHOCK_SIGNATURE, light_inputs=shock_maps)
    if reason is None:
        hb.log('  timber shock: reusing %s (same maps, same settings)' % p.timber_provision_shock_output_path)
        return True
    hb.log('  timber shock: computing because %s' % reason)
    jobs = []
    if measure == 'net_return':
        # The fixed eligible area, once: managed in the base year AND forest in the base-year map.
        eligible_path = os.path.join(p.cur_dir, 'timber_eligible_value_%d.tif' % es_shock_base_year)
        if not hb.path_exists(eligible_path):
            _write_eligible_value(value_path, base_lulc_path, eligible_path)
        _log_eligible_coverage(p, value_path, eligible_path)
        worker = _value_one_timber_map
        # The baseline may also be listed among the scenarios; each map is valued exactly once, or two
        # workers would write the same raster at the same time.
        for scenario in dict.fromkeys([base_scenario] + list(scenarios)):
            years = list(anchor_years) + ([es_shock_base_year] if es_shock_base_year in p.scenario_lulc_paths.get(scenario, {}) else [])
            for year in years:
                jobs.append((scenario, year, p.scenario_lulc_paths[scenario][year], eligible_path,
                             _timber_value_raster_path(p, scenario, year), _timber_summary_path(p, scenario, year),
                             p.region_boundary_path, p.terrestrial_carbon_shock_id_col))
    else:
        # D19: aboveground forest biomass carbon on the forest of each map; the footprint moves.
        base_forest_path = os.path.join(p.cur_dir, 'forest_agbc_base_%d.tif' % es_shock_base_year)
        new_forest_path = os.path.join(p.cur_dir, 'forest_agbc_new_forest_zone_mean.tif')
        _write_biomass_base_and_new_forest(p, base_lulc_path, base_forest_path, new_forest_path)
        worker = _biomass_one_map
        for scenario in dict.fromkeys([base_scenario] + list(scenarios)):
            years = list(anchor_years) + ([es_shock_base_year] if es_shock_base_year in p.scenario_lulc_paths.get(scenario, {}) else [])
            for year in years:
                jobs.append((scenario, year, p.scenario_lulc_paths[scenario][year], base_forest_path, new_forest_path,
                             _timber_value_raster_path(p, scenario, year), _timber_summary_path(p, scenario, year),
                             p.region_boundary_path, p.terrestrial_carbon_shock_id_col))
    todo = [j for j in jobs if not _summary_complete(j[-3])]
    n_workers = max(1, min(int(getattr(p, 'num_workers', 1) or 1), len(todo)))
    hb.log('  timber (%s): %d scenario-year maps, %d to value, %d workers' % (measure, len(jobs), len(todo), n_workers))
    if todo:
        if n_workers > 1:
            import multiprocessing
            with multiprocessing.Pool(n_workers) as pool:
                pool.map(worker, todo)
        else:
            for j in todo:
                worker(j)

    def zone_value(scenario, year):
        return hb.df_read(_timber_summary_path(p, scenario, year)).set_index('region_id')[p.terrestrial_carbon_shock_value_col]
    baseline_by_year = {y: zone_value(base_scenario, y) for y in anchor_years}
    baseline_at_base_year = zone_value(base_scenario, es_shock_base_year)

    rows = []
    for scenario in scenarios:
        rows += tcf.dynamic_shock_rows(
            {y: zone_value(scenario, y) for y in anchor_years},
            baseline_by_year, baseline_at_base_year, zone_labels, es_shock_base_year,
            p.terrestrial_carbon_shock_acts, scenario)

    out = pd.DataFrame(rows)
    out = utilities.filter_to_model_domain(out, p.timber_provision_shock_output_path,
                                           'timber_provision', log=hb.log)
    # The run exports shock_pct_v3 (the scenario's own trajectory from the base year), which the
    # eligible-area measure bounds in [-100, 0]; the contemporaneous ratio can blow up where the
    # baseline's eligible value in a small zone goes to nearly nothing, and is not fed.
    utilities.assert_shock_table_sound(out, scenarios, 'timber_provision', column='shock_pct_v3')
    v3 = out['shock_pct_v3'].dropna()
    if measure == 'net_return' and len(v3) and (v3.max() > 1e-6 or v3.min() < -100 - 1e-6):
        raise ValueError('timber shock_pct_v3 outside [-100, 0]: min %.4g max %.4g -- the eligible-area measure '
                         'can only lose value' % (v3.min(), v3.max()))
    if measure == 'aboveground_carbon' and len(v3) and v3.min() < -100 - 1e-6:
        raise ValueError('biomass shock_pct_v3 below -100: min %.4g' % v3.min())
    out['timber_resource_measure'] = measure
    out.to_csv(p.timber_provision_shock_output_path, index=False)
    utilities.write_reuse_signature(p, 'timber_provision', shock_outputs, TIMBER_SHOCK_SIGNATURE, light_inputs=shock_maps)
    hb.log('  timber shock: %d rows, %d scenarios -> %s'
           % (len(out), out['scenario'].nunique() if rows else 0,
              p.timber_provision_shock_output_path))
    return True
