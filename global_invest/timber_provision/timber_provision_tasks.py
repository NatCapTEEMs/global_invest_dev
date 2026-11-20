"""Timber-provision GEP tasks, and the timber scenario ES shock.

The account's figure, `timber_provision_gep`, is CWoN's forest rent, computed here from CWoN's
timber revenue and rental rate and checked against the rent CWoN publishes. Beside it is OUR
spatial estimate, `timber_provision_gep_spatial`: the pipeline's value raster (verified equal to
the net-return layer masked and floored, see timber_provision_functions) summed per country on
the 10-arcsecond country-id raster. The committed Forestry CSV stays as the test anchor the
spatial sum is compared against, never as the output."""

import json
import os
import numpy as np
import pandas as pd
import rasterio
import hazelbean as hb
from global_invest import utilities
from global_invest.timber_provision import timber_provision_functions as tp


GRID_COVERED_ZONES = 'grid_covered_zones.json'   # measured once per run, beside the workspace
EXCLUDED_REGIONS = 'timber_provision_excluded_regions.json'   # beside the table, read at the export boundary


def _grid_covered_zones(p, summary_path):
    """The zone ids this run's GRID actually contains, from the zone-id raster the zonal step used.

    The D19 coverage assertion requires every zone in the boundary file, which is right for a global
    run and impossible for a bounded one: the zaf validation fixture covers 26 of the boundary's 470
    zones and the remaining 444 are absent because the model does not reach them, not because a
    total went missing. `intentional_exclusions` is the assertion's own mechanism for that
    distinction, and this supplies the set to declare.

    Read from the RASTER, never from the summaries. A zone the grid covers but a summary lost is
    still a failure -- that is the defect the assertion exists to catch, and deriving coverage from
    the summaries would make it vacuous. Zone geometry does not vary by scenario or year, so this is
    measured once per run and cached: one pass over the zone raster rather than one per
    scenario-year, which matters because the global grid is 129,600 x 64,800.

    Args:
        p: the ProjectFlow object, for `cur_dir`.
        summary_path (str): a scenario-year summary; its `_zone_ids.tif` sibling is what is read.

    Returns:
        set | None: the covered zone ids, or None when there is no zone raster to read, in which
        case the caller requires every boundary zone.
    """
    cache_path = os.path.join(p.cur_dir, GRID_COVERED_ZONES)
    if hb.path_exists(cache_path):
        return {int(z) for z in json.loads(open(cache_path, encoding='utf-8').read())}
    zone_ids_path = os.path.splitext(summary_path)[0] + '_zone_ids.tif'
    if not hb.path_exists(zone_ids_path):
        return None
    covered = sorted({int(v) for v in hb.unique_raster_values_path(zone_ids_path) if int(v) > 0})
    hb.write_to_file(json.dumps(covered), cache_path)
    hb.log('  timber: the run grid covers %d zone(s); measured once from %s'
           % (len(covered), os.path.basename(zone_ids_path)))
    return set(covered)


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


def cwon_forest_rent(p, countries):
    """CWoN's forest rent on the given country rows: revenue times rental rate, from the CWoN 2024
    package, with CWoN's assumed zeros for the countries es_parameters names.

    The rent CWoN publishes is read only to check the result, and the check is logged.
    """
    year = int(p.gep_base_year)
    # The World Bank writes a missing value as "..".
    revenue, rate = (pd.read_excel(str(p.get_path(path)), na_values=['..'])
                     for path in (p.timber_provision_cwon_forest_revenue_path,
                                  p.timber_provision_cwon_forest_rental_rate_path))
    rent = tp.cwon_forest_rent_from_revenue_and_rate(
        revenue, rate, countries, year, p.timber_provision_cwon_assumed_zero_countries)

    # pd.read_stata, like extractive_energy and fisheries: hb.df_read is csv-only.
    published = tp.cwon_forest_rent_by_country(
        pd.read_stata(str(p.get_path(p.timber_provision_cwon_forest_rent_path))),
        countries[['iso3_r250_label']], year).set_index('iso3_r250_label')['timber_provision_gep_cwon_rent']
    computed = rent.set_index('iso3_r250_label')['timber_provision_gep_cwon_rent']
    both = computed.notna() & published.notna()
    only_published = published.notna() & computed.isna()
    hb.log('CWoN check, %d: revenue x rate against the published rent, %d countries, max |diff| '
           '%.3g USD, %d assumed zeros. Published but not computed: %d (%s), all of them zero: %s'
           % (year, int(both.sum()), (computed[both] - published[both]).abs().max(),
              int(rent['timber_provision_cwon_rent_assumed_zero'].sum()), int(only_published.sum()),
              ', '.join(sorted(published.index[only_published])),
              bool(published[only_published].eq(0).all())))
    return rent


def gep_calculation(p):
    """GEP valuation for timber provision: CWoN's forest rent per country (revenue times rental
    rate), with the value-raster sum, the fuelwood split and the roundwood gross-value diagnostic
    beside it."""
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'timber_provision')
    if already_done:
        return

    if not p.run_this:
        return p
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

    # Both valuations, side by side: the issues document recommends the CWoN rent, which is the
    # account's figure (below), and publishing it alone would hide how far it sits from the
    # spatial estimate.
    df_gep = cwon_forest_rent(p, df_gep)
    # And the proxy diagnostic: FAOSTAT roundwood priced at the country's own industrial-roundwood
    # export unit value where one is reported, the world value otherwise. Not a third estimate: a
    # land share is a fraction of the gross value of the wood it comes from, so a country whose GEP
    # exceeds this column is flagged per country for a look. Export unit values are a proxy for
    # domestic prices, so an exceedance is a flag, not proof of an impossible value.
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
    if not p.run_this:
        return p
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
    - `fuelwood_gep_gross_at_export_price`  FAOSTAT wood fuel production times the world wood-fuel
      export unit value, with NO lambda: a gross value, the shape the reference has.
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
        rent = cwon_forest_rent(p, countries)
        rent = tp.fuelwood_share_of_forest_rent(faostat, rent, int(p.gep_base_year))
        rent['fuelwood_gep_from_forest_rent'] = (
            rent['timber_provision_gep_cwon_rent'] * rent['fuelwood_share_of_rent'])

        # the no-lambda gross value, which is the shape the reference has. Wood fuel priced at
        # the world wood-fuel export unit value, not as the fuelwood share of roundwood gross --
        # see the note on wood_fuel_gross_value_by_country.
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
# separate; a run that does not build the timber tree is unaffected by it.
#
# The machinery is carbon's, deliberately: the zonal summaries and the shock arithmetic
# (summarize_raster_by_region, dynamic_shock_rows, summed_shock_rows) know nothing about carbon,
# so timber supplies its own layer and nothing else. Reimplementing them here would fork a
# verified path to change one input raster.
# ---------------------------------------------------------------------------------------------

def timber_value_density_table(p):
    """Mean timber value density per (land-cover class, zone).

    The same construction as terrestrial_carbon's density table, over the timber value raster
    instead of the carbon density raster. timber_provision_shock does not read it: its docstring
    says why a class x zone mean is the wrong measure for timber.
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
    # outright.
    #
    # The base-year SEALS7 map is the one the ES shocks form their 2023 denominator from, under the
    # project's own fine_processed_inputs. The timber value raster and the carbon zones are on its
    # grid already (10 arcsec global), so nothing is resampled.
    utilities.hydrate_es_parameters(p, 'terrestrial_carbon', log=hb.log)
    base_year = int(getattr(p, 'es_shock_base_year', None) or p.key_base_year)
    seals7_base_path = os.path.join(p.intermediate_dir, 'fine_processed_inputs', 'lulc', 'esa', 'seals7',
                                    'lulc_esa_seals7_%d.tif' % base_year)
    zones_path = p.terrestrial_quantity_input_path
    value_path = getattr(p, 'timber_provision_value_raster_path', None)
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
            if hb.path_exists(stale):
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
            if hb.path_exists(stale):
                os.remove(stale)
    if not hb.path_exists(raster_path):
        lulc_ndv = hb.get_ndv_from_path(lulc_path)
        def on_map(base_block, new_block, lulc_block):
            return tp.forest_biomass_carbon_on_map(base_block, new_block, lulc_block, lulc_ndv=lulc_ndv)
        hb.raster_calculator_flex([base_forest_path, new_forest_path, lulc_path], on_map, raster_path, datatype=6, ndv=np.nan)
    if not hb.path_exists(summary_path):
        tcf.summarize_raster_by_region(raster_path, boundary_path, summary_path, year=year, id_column=id_col)
    return summary_path


# The displacement two grids may differ by before they are treated as different grids, in PIXELS.
# Explicit and absolute: a relative tolerance on coordinates would scale with longitude and quietly
# permit more slop at the edges of a global raster than at its centre.
GRID_TOLERANCE_PIXELS = 1e-7


def _assert_one_grid(grids, tolerance_pixels=GRID_TOLERANCE_PIXELS):
    """Every raster must lie on one grid. Shape and CRS exactly; geometry to a pixel tolerance.

    Exact equality of the affine transform is too strict: a base map can carry 90.00000000000001 as
    its y origin instead of 90.0 -- a difference of 1e-14 degrees, about a nanometre on the ground and
    4e-12 of a pixel. Representation noise is not misalignment.

    Comparing origins alone is not enough either. A tiny difference in pixel size or a rotation
    term leaves the origin identical and accumulates across 129,600 columns into a real offset, so
    the test is the displacement at ALL FOUR CORNERS, in pixel units. That catches translation,
    scale and rotation with one criterion, and it fails where it should: a half-pixel shift, or a
    resolution difference of one part in 10^9, both exceed the tolerance by the far corner.

    Args:
        grids (dict): label -> (shape, affine transform, CRS).
        tolerance_pixels (float): maximum corner displacement, in pixels.

    Raises:
        ValueError: naming the labels that disagree and the displacement measured.
    """
    labels = list(grids)
    reference_label = labels[0]
    ref_shape, ref_transform, ref_crs = grids[reference_label]
    height, width = ref_shape
    corners = ((0, 0), (0, width), (height, 0), (height, width))
    for label in labels[1:]:
        shape, transform, crs = grids[label]
        if shape != ref_shape:
            raise ValueError('the biomass measure needs one grid: %s is %s but %s is %s'
                             % (label, shape, reference_label, ref_shape))
        if crs != ref_crs:
            raise ValueError('the biomass measure needs one grid: %s has CRS %s, %s has %s'
                             % (label, crs, reference_label, ref_crs))
        worst, where = 0.0, None
        for row, column in corners:
            ax, ay = ref_transform * (column, row)
            bx, by = transform * (column, row)
            displacement = max(abs(ax - bx) / abs(ref_transform.a),
                               abs(ay - by) / abs(ref_transform.e))
            if displacement > worst:
                worst, where = displacement, (row, column)
        if worst > tolerance_pixels:
            raise ValueError(
                'the biomass measure needs one grid: %s is displaced from %s by %.3g pixels at '
                'corner (row %d, col %d), above the %g pixel tolerance. %s transform %s, %s %s'
                % (label, reference_label, worst, where[0], where[1], tolerance_pixels,
                   label, transform, reference_label, ref_transform))


_GDAL_TYPE_BY_DTYPE = {'byte': 1, 'uint8': 1, 'int16': 3, 'uint16': 2, 'int32': 5, 'uint32': 4,
                       'float32': 6, 'float64': 7}


def _aligned_to_reference(p, source_path, reference_path, label):
    """A copy of `source_path` on the reference's grid, when the two differ only by extent.

    Returns source_path unchanged when the grids already match. Refuses rather than resampling
    when the pixel sizes differ or the reference's origin does not fall on a source pixel
    boundary: that case is a genuine regridding decision, not a crop, and it must not be made
    silently inside a shock measurement.

    Args:
        p (ProjectFlow): supplies cur_dir for the cached clip.
        source_path (str): the raster to clip, typically a global layer.
        reference_path (str): the raster whose grid the run works on.
        label (str): used in the cached filename and in error messages.

    Returns:
        str: a path on the reference grid.

    Raises:
        ValueError: if the grids are not an exact window of one another.
    """
    with rasterio.open(source_path) as src, rasterio.open(reference_path) as ref:
        if src.shape == ref.shape and src.transform == ref.transform:
            return source_path
        st, rt = src.transform, ref.transform
        if abs(st.a - rt.a) > 1e-12 or abs(st.e - rt.e) > 1e-12:
            raise ValueError('%s is %.10f deg per pixel but the run grid is %.10f; clipping would '
                             'be a regridding choice, not a crop' % (label, st.a, rt.a))
        col = (rt.c - st.c) / st.a
        row = (rt.f - st.f) / st.e
        if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
            raise ValueError('%s does not align with the run grid: offset (%.6f, %.6f) is not whole '
                             'pixels, so a crop would shift the values' % (label, col, row))
        source_dtype, source_ndv = src.dtypes[0], src.nodata
    aligned_path = os.path.join(p.cur_dir, 'timber_%s_aligned.tif' % label)
    if not hb.path_exists(aligned_path):
        hb.resample_to_match(source_path, reference_path, aligned_path, resample_method='near',
                             output_data_type=_GDAL_TYPE_BY_DTYPE[source_dtype],
                             src_ndv=source_ndv, ndv=source_ndv)
    return aligned_path


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
    # The base map and carbon zones are on the run's grid, which for a sub-global AOI is a window
    # of the global pyramid; the biomass density and ha-per-cell rasters are global. Same
    # resolution and an aligned grid, so nearest is a lossless clip rather than a resample -- the
    # same reasoning terrestrial_carbon's _align_zones_to_lulc_grid records. Verified for the ZAF
    # window: identical 10-arcsec pixels, and its origin falls on global pixel (70560, 40320).
    # The clipped copy keeps the SOURCE's data type and nodata, never the reference's. Taking them
    # from a match raster truncates values its type cannot hold: carbon zone ids 606-620 in a Byte
    # raster become nodata, zone 0.
    density_path = _aligned_to_reference(p, density_path, base_lulc_path, 'biomass_carbon_density')
    ha_path = _aligned_to_reference(p, ha_path, base_lulc_path, 'ha_per_cell')
    grids = {}
    for label, path in (('density', density_path), ('zones', zones_path), ('ha', ha_path), ('base map', base_lulc_path)):
        with rasterio.open(path) as src:
            grids[label] = (src.shape, src.transform, src.crs)
    _assert_one_grid(grids)
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

    The class x zone mean lookup (timber_value_density_table) is not read here: it dilutes forest
    by unmanaged forest and gives non-forest classes value where the mask overlaps them, so
    afforestation lowers the measure and deforestation raises it.

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
    if measure not in ('net_return', 'aboveground_carbon', 'harvestable_flow'):
        raise ValueError("p.timber_provision_resource_measure must be 'net_return', 'aboveground_carbon' or "
                         "'harvestable_flow', got %r" % measure)
    value_path = getattr(p, 'timber_provision_value_raster_path', None)
    if measure == 'net_return' and not hb.path_exists(value_path):
        raise NameError('timber value raster not found at %s' % value_path)
    if es_shock_base_year not in p.scenario_lulc_paths.get(base_scenario, {}):
        raise NameError('timber needs the base-year map (%d) among the baseline maps to fix the eligible area' % es_shock_base_year)
    base_lulc_path = p.scenario_lulc_paths[base_scenario][es_shock_base_year]
    if measure == 'harvestable_flow':
        return _harvestable_flow_shock(p, base_scenario, scenarios, anchor_years, es_shock_base_year, zone_labels,
                                       base_lulc_path)
    # Reuse the table when it was made from these very maps and settings; the per-map guards below
    # still cost minutes of verification, and a pass that finds the table must return in seconds.
    shock_maps = sorted({m for by_year in p.scenario_lulc_paths.values() for m in by_year.values()})
    if measure == 'aboveground_carbon':
        settings, inputs = utilities._service_settings_and_inputs(p, 'timber_provision')
        inputs = {k: v for k, v in inputs.items() if not k.endswith('_output_path')}
        settings.update(resource_measure=measure, aggregation='regional_sums', anchors=anchor_years,
                        scenarios=list(scenarios), base_year=es_shock_base_year)
        signed = shock_maps + [p.timber_provision_biomass_carbon_density_path,
            p.terrestrial_quantity_input_path, p.region_boundary_path,
            getattr(p, 'ha_per_cell_10sec_path', None) or p.get_path(*utilities.HA_PER_CELL_10SEC_REF_PARTS),
            __file__, tp.__file__, tcf.__file__]
        inputs.update({str(path): utilities.file_fingerprint(path) for path in signed})
        # The value rasters and their zone-id rasters are INPUTS to the summarisation, not outputs
        # of it. The signature hashes whole source files, so any edit to this module or to the shock
        # arithmetic invalidates the workspace wholesale -- including rasters whose entire producing
        # chain (forest_biomass_carbon_on_map, raster_calculator_flex, the cython kernel) is
        # untouched. Naming them here is the guard's own mechanism for a prepared input, and it is
        # what allows a summariser fix to be re-run over rasters that cost hours and did not change.
        # It is NOT a way to reuse a raster whose producer moved: verify that before adding to this.
        prepared = [p.terrestrial_quantity_input_path, base_lulc_path,
                    os.path.join(p.cur_dir, 'forest_agbc_base_%d.tif' % es_shock_base_year),
                    os.path.join(p.cur_dir, 'forest_agbc_new_forest_zone_mean.tif'),
                    os.path.join(p.cur_dir, 'forest_agbc_density_by_carbon_zone.csv')]
        for scenario in dict.fromkeys([base_scenario] + list(scenarios)):
            for year in list(anchor_years) + ([es_shock_base_year] if es_shock_base_year in p.scenario_lulc_paths.get(scenario, {}) else []):
                summary_path = _timber_summary_path(p, scenario, year)
                prepared += [_timber_value_raster_path(p, scenario, year),
                             os.path.splitext(summary_path)[0] + '_zone_ids.tif']
        utilities.require_workspace_signature(p.cur_dir, utilities._signature(settings, inputs),
            os.path.join(p.cur_dir, 'd19_workspace_signature.json'),
            prepared_inputs=tuple(prepared))
    shock_outputs = [p.timber_provision_shock_output_path]
    reason = utilities.reuse_reason(p, 'timber_provision', shock_outputs, TIMBER_SHOCK_SIGNATURE, light_inputs=shock_maps)
    if reason is None and measure != 'aboveground_carbon':
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

    # D19 sums require every boundary zone, not merely a plausible row count.
    # Include AEZ0 in physical regional totals as in D19; economic rows are filtered later.
    if measure == 'aboveground_carbon':
        for job in jobs:
            scenario, year = job[:2]
            summary = pd.read_csv(job[-3]).assign(scenario=scenario, year=year)
            covered = _grid_covered_zones(p, job[-3])
            absent = None if covered is None else [
                (scenario, year, zone) for zone in zone_labels if int(zone) not in covered]
            utilities.assert_zonal_coverage_complete(summary, [scenario], [year], set(zone_labels),
                                                    'D19 biomass', intentional_exclusions=absent,
                                                    log=hb.log)
            if (summary['total'] < 0).any():
                raise ValueError('Negative D19 biomass total')

    def zone_value(scenario, year):
        return hb.df_read(_timber_summary_path(p, scenario, year)).set_index('region_id')[p.terrestrial_carbon_shock_value_col]

    def zone_total(scenario, year):
        """The zone's SUMMED quantity. The configured value column is a zone MEAN, which is what a
        per-zone ratio needs (the cell count cancels) and exactly what must not be added across
        zones of different size: a regional total formed from means understates large zones and
        overstates small ones. The summed measure therefore reads the summary's own total."""
        return hb.df_read(_timber_summary_path(p, scenario, year)).set_index('region_id')['total']
    baseline_by_year = {y: zone_value(base_scenario, y) for y in anchor_years}
    baseline_at_base_year = zone_value(base_scenario, es_shock_base_year)

    rows = []
    if measure == 'aboveground_carbon':
        # SUMS, NOT MEANS OF RATIOS (D19). This measure credits forest gained after the base
        # year, so a zone holding almost no forest in the base year and afforested later has an
        # unbounded ratio -- China AEZ18 goes from 116 to 532,000 Mg C, a ratio of +459,000 %, and
        # eighteen such zones pull the land-payment-weighted world mean to +1.2 % while the
        # aggregate biomass change is -2.8 %. The region's shock is therefore
        # the change in its SUMMED biomass: 100 x (sum scenario / sum baseline - 1). Zones with no
        # base-year biomass keep contributing their gain to the numerator; no ratio is formed for
        # them, nothing is capped and nothing is excluded. A region whose own total base-year
        # biomass is zero or negligible has no denominator and is reported, not shocked.
        excluded_regions = []
        rows += tcf.summed_shock_rows(
            {scenario: {y: zone_total(scenario, y) for y in anchor_years} for scenario in scenarios},
            zone_total(base_scenario, es_shock_base_year), zone_labels, es_shock_base_year,
            p.terrestrial_carbon_shock_acts, log=hb.log,
            covered_zones=_grid_covered_zones(p, _timber_summary_path(p, base_scenario, es_shock_base_year)),
            excluded_out=excluded_regions)
    else:
        for scenario in scenarios:
            rows += tcf.dynamic_shock_rows(
                {y: zone_value(scenario, y) for y in anchor_years},
                baseline_by_year, baseline_at_base_year, zone_labels, es_shock_base_year,
                p.terrestrial_carbon_shock_acts, scenario, log=hb.log)

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
    if measure == 'aboveground_carbon':
        out['timber_provision_accounting'] = 'aboveground_carbon_summed_v1'
    out.to_csv(p.timber_provision_shock_output_path, index=False)

    # DECLARE the regions this run could not measure, beside the table. write_afeall_hars fills
    # np.zeros over the whole AFEL set, so a region merely missing from the table reaches GTAP as a
    # zero shock -- indistinguishable from a measured no-change. es_shock_export reads this file and
    # refuses to export a table that carries one, so an unmeasured region cannot become a zero.
    # A global run measures every region, writes an empty declaration, and exports normally.
    declared = sorted(set(locals().get('excluded_regions') or ()))
    if declared:
        present = set(out['REG'].astype(str)) if len(out) else set()
        leaked = sorted(r for r in declared if r in present)
        if leaked:
            raise ValueError('timber shock: %d region(s) were excluded as unmeasured yet appear in '
                             'the table: %s' % (len(leaked), leaked))
        hb.log('  timber shock: DECLARED %d unmeasured region(s), which this run must not export: %s'
               % (len(declared), ', '.join(declared)))
    hb.write_to_file(json.dumps({'excluded_regions': declared,
                                 'reason': 'not fully covered by the run grid'}, indent=2),
                     os.path.join(os.path.dirname(p.timber_provision_shock_output_path), EXCLUDED_REGIONS))
    utilities.write_reuse_signature(p, 'timber_provision', shock_outputs, TIMBER_SHOCK_SIGNATURE, light_inputs=shock_maps)
    hb.log('  timber shock: %d rows, %d scenarios -> %s'
           % (len(out), out['scenario'].nunique() if rows else 0,
              p.timber_provision_shock_output_path))
    return True


# --- Age-aware harvestable flow (resource measure 'harvestable_flow') ----------------------------
# Selected only by p.timber_provision_resource_measure = 'harvestable_flow'; the other measures and
# the default never reach this code. Every setting below is required, set by the run that selects
# the measure, and none has a default. The state chain runs inside timber_provision_shock, as the
# aboveground-carbon rasters do, so it shares that task's scenario maps, base map and zones.
#
# The initial state is an input with this contract; no library task produces it, because building
# it (forest types, ages, management, access) is project data:
#   - one GeoTIFF on the run grid with float32 bands: either six in FOREST_STATE_BANDS order, one
#     cohort taking every forest cell whole, or COHORT_BANDS repeated once per cohort layer, where
#     share is the cohort's fraction of the cell (a regional plantation mixture is one layer per
#     genus at the region's area shares, each with its own ages);
#   - every cell the base-year map calls forest has at least one typed cohort, with a forest_type_id
#     of the growth table, and every other cell has none (the first step refuses any disagreement);
#   - on a typed cohort: age and since_cut in years (finite), stock in m3/ha within [0, vmax] of the
#     type (0 for an unmeasured type), managed and accessible as 0 or 1, share in (0, 1]; the typed
#     shares of a cell sum to 1, and an untyped cohort has share 0.
# The input provenance table says, for each band and for the growth table, whether it is observed,
# assumed or synthetic, and from what source. It travels into the outputs beside the shock.

HARVESTABLE_FLOW_SETTINGS = ('growth_parameters_path', 'initial_state_path', 'input_provenance_path', 'new_forest',
                             'harvest_rule', 'recovery_rule', 'salvage_fraction', 'reference_flow')
REFERENCE_FLOWS = ('base_map_first_interval',)
REFERENCE_LABEL = 'base_map_reference'
ZONE_QUANTITIES = ('forest_ha', 'timberland_open_ha', 'timberland_ha', 'opening_m3', 'growth_m3', 'harvest_m3', 'removed_m3',
                   'salvage_m3', 'closing_m3', 'eligible_m3')


def _harvestable_flow_settings(p):
    """The measure's explicit choices, read from p. A missing one raises; nothing is filled in."""
    missing = ['p.timber_provision_' + k for k in HARVESTABLE_FLOW_SETTINGS
               if getattr(p, 'timber_provision_' + k, None) in (None, '')]
    if missing:
        raise NameError('the harvestable_flow measure needs %s; none of them has a default' % ', '.join(missing))
    settings = {k: getattr(p, 'timber_provision_' + k) for k in HARVESTABLE_FLOW_SETTINGS}
    if settings['reference_flow'] not in REFERENCE_FLOWS:
        raise ValueError('timber_provision_reference_flow must be one of %s, got %r'
                         % (REFERENCE_FLOWS, settings['reference_flow']))
    tp.new_forest_cohorts(settings['new_forest'])
    return settings


def _volume_basis_weights(p):
    """The fixed reference weights that combine volume bases, or None when the run configures none.

    Optional, unlike the settings above: without it a region holding more than one volume basis is
    declared, the conservative result rather than a default value. The table and its contract are in
    timber_provision_functions.volume_basis_weights_from_table.
    """
    path = getattr(p, 'timber_provision_volume_basis_weights_path', None)
    return None if path in (None, '') else tp.volume_basis_weights_from_table(hb.df_read(path))


def _state_cohorts(bands, n_bands):
    """The cohort layers of a block of a state raster: six bands are one cohort taking every forest
    cell whole; otherwise COHORT_BANDS repeat once per cohort layer."""
    if n_bands == len(tp.FOREST_STATE_BANDS):
        layer = dict(zip(tp.FOREST_STATE_BANDS, bands))
        layer['share'] = (layer['type'] > 0).astype('float64')
        return [layer]
    n = len(tp.COHORT_BANDS)
    return [dict(zip(tp.COHORT_BANDS, bands[i:i + n])) for i in range(0, n_bands, n)]


def _advance_forest_state_raster(previous_path, lulc_path, ha_path, zone_ids_path, n_zones, n_years, settings,
                                 params, out_path, base_lulc_path=None):
    """One step of a scenario's forest-state chain on the run grid, block by block.

    Reads the previous state raster (see the initial-state contract above), advances every cohort
    n_years to the map at lulc_path, writes the new state as COHORT_BANDS per cohort layer and returns
    the step's totals per zone and forest type. A map cell without land-cover data is not a
    conversion: the cell keeps its forest status. When base_lulc_path is given (the step that starts
    from the configured initial state), the state must type exactly the cells the base map calls
    forest, with valid cohorts, or the step refuses.

    Quantities that happen before the year-end map (timberland_open_ha, opening, growth, harvest,
    removed, salvage and eligible volume) are counted under each cohort's type before the step;
    forest_ha, timberland_ha and the closing stock under its type after it. Area is the cohort's share
    of the cell's hectares, so the stock ledger closes per zone and type.

    Returns:
        pd.DataFrame: region_id, forest_type_id and volume_basis plus ZONE_QUANTITIES, in ha and m3,
        one row per zone and forest type of the growth table.
    """
    paths = [previous_path, lulc_path, ha_path, zone_ids_path] + ([base_lulc_path] if base_lulc_path else [])
    sources = [rasterio.open(path) for path in paths]
    previous, lulc, ha, zones = sources[:4]
    n_cohort_bands = len(tp.COHORT_BANDS)
    if previous.count != len(tp.FOREST_STATE_BANDS) and previous.count % n_cohort_bands:
        raise ValueError('%s has %d bands; a forest state has %d (%s) or %d per cohort layer (%s)'
                         % (previous_path, previous.count, len(tp.FOREST_STATE_BANDS), ', '.join(tp.FOREST_STATE_BANDS),
                            n_cohort_bands, ', '.join(tp.COHORT_BANDS)))
    _assert_one_grid({os.path.basename(s.name): (s.shape, s.transform, s.crs) for s in sources})
    layers_in = 1 if previous.count == len(tp.FOREST_STATE_BANDS) else previous.count // n_cohort_bands
    layers_out = max(layers_in, len(tp.new_forest_cohorts(settings['new_forest'])))
    codes = sorted(params)
    totals = np.zeros((len(ZONE_QUANTITIES), len(codes), n_zones + 1))
    before_step = ('timberland_open_ha', 'opening_m3', 'growth_m3', 'harvest_m3', 'removed_m3', 'salvage_m3', 'eligible_m3')
    mismatch = np.zeros(3, dtype=np.int64)
    worst_residual = 0.0
    profile = dict(previous.profile, count=layers_out * n_cohort_bands, dtype='float32', nodata=None)
    partial = out_path + '.partial.tif'
    try:
        with rasterio.open(partial, 'w', **profile) as out:
            for _, window in previous.block_windows(1):
                cohorts = _state_cohorts(previous.read(window=window).astype('float64'), previous.count)
                flags = np.ones(cohorts[0]['type'].shape, dtype=bool)
                typed_cell = np.zeros(flags.shape, dtype=bool)
                for c in cohorts:
                    flags &= np.isin(c['managed'], (0, 1)) & np.isin(c['accessible'], (0, 1))
                    c['managed'], c['accessible'] = c['managed'] > 0, c['accessible'] > 0
                    typed_cell |= c['type'] > 0
                if base_lulc_path:
                    base_map = sources[4].read(1, window=window)
                    known = (base_map != sources[4].nodata) if sources[4].nodata is not None else np.ones(base_map.shape, dtype=bool)
                    base_forest = np.isin(base_map, [tp.SEALS7_FOREST_ID])
                    invalid = ~flags | tp.cohort_share_violations(cohorts)
                    for c in cohorts:
                        vmax = np.full(base_map.shape, np.inf)
                        for code, prm in params.items():
                            vmax[c['type'] == code] = prm.get('vmax_m3_per_ha', 0.0) * (1 + 1e-6)   # float32 storage
                        valid = (np.isfinite(c['age']) & (c['age'] >= 0) & np.isfinite(c['since_cut'])
                                 & (c['since_cut'] >= 0) & np.isfinite(c['stock']) & (c['stock'] >= 0) & (c['stock'] <= vmax))
                        invalid |= (c['type'] > 0) & ~valid
                    mismatch += [int((known & base_forest & ~typed_cell).sum()), int((known & ~base_forest & typed_cell).sum()),
                                 int(invalid.sum())]
                lulc_block = lulc.read(1, window=window)
                unknown = (lulc_block == lulc.nodata) if lulc.nodata is not None else np.zeros(lulc_block.shape, dtype=bool)
                forest_now = np.where(unknown, typed_cell, np.isin(lulc_block, [tp.SEALS7_FOREST_ID]))
                new, flows = tp.advance_forest_cohorts(cohorts, n_years, forest_now, settings['new_forest'], params,
                                                       settings['harvest_rule'], settings['recovery_rule'],
                                                       float(settings['salvage_fraction']))
                for f in flows:
                    worst_residual = max(worst_residual, float(np.abs(tp.stock_ledger_residual(f)).max(initial=0.0)))
                out.write(np.stack([np.asarray(c[b], dtype='float32') for c in new for b in tp.COHORT_BANDS]), window=window)
                cell_ha = ha.read(1, window=window).astype('float64')
                cell_ha = np.where(np.isfinite(cell_ha) & (cell_ha > 0), cell_ha, 0.0)
                zone = zones.read(1, window=window).astype(np.int64)
                zone = np.where((zone > 0) & (zone <= n_zones), zone, 0).ravel()
                cohorts += [tp.empty_cohort(forest_now.shape) for _ in range(len(new) - len(cohorts))]
                for before, after, f in zip(cohorts, new, flows):
                    area_before, area_after = before['share'] * cell_ha, after['share'] * cell_ha
                    values = dict(timberland_open_ha=before['managed'] & before['accessible'], opening_m3=f['opening'],
                                  growth_m3=f['growth'], harvest_m3=f['harvest'], removed_m3=f['removed'],
                                  salvage_m3=f['salvage'], eligible_m3=f['eligible'], forest_ha=after['type'] > 0,
                                  timberland_ha=after['managed'] & after['accessible'], closing_m3=f['closing'])
                    for j, code in enumerate(codes):
                        is_before, is_after = (before['type'] == code).ravel(), (after['type'] == code).ravel()
                        for i, q in enumerate(ZONE_QUANTITIES):
                            pre = q in before_step
                            mask = is_before if pre else is_after
                            weights = (values[q] * (area_before if pre else area_after)).ravel()[mask]
                            totals[i, j] += np.bincount(zone[mask], weights=weights, minlength=n_zones + 1)
    finally:
        for s in sources:
            s.close()
    if mismatch.any():
        os.remove(partial)
        raise ValueError('the initial forest state breaks its contract: %d base-map forest cell(s) carry no forest '
                         'type, %d typed cell(s) are not forest on the base map, and %d cell(s) have an invalid age, '
                         'since_cut, stock, share or 0/1 flag. Build the initial state on the model\'s own base map; it '
                         'is never relabelled here.' % tuple(mismatch))
    if worst_residual > 1e-6:
        os.remove(partial)
        raise ValueError('forest stock ledger does not close: largest residual %.3g m3/ha' % worst_residual)
    os.replace(partial, out_path)
    tables = [pd.DataFrame(totals[:, j, :].T, columns=list(ZONE_QUANTITIES)).assign(
        region_id=np.arange(n_zones + 1), forest_type_id=code, volume_basis=params[code]['volume_basis'])
        for j, code in enumerate(codes)]
    table = pd.concat(tables, ignore_index=True)
    table = table[table['region_id'] > 0].sort_values(['region_id', 'forest_type_id'])
    return table[['region_id', 'forest_type_id', 'volume_basis'] + list(ZONE_QUANTITIES)].reset_index(drop=True)


def _forest_state_chain(p, label, maps_by_year, start, ha_path, zone_ids_path, n_zones, settings, params,
                        signed_settings, base_lulc_path):
    """Advance the initial state through one scenario's maps, reusing a step only when its provenance matches.

    Each step's signature holds the measure's settings, the growth table by content, the previous
    state's digest (recorded when that state was written, never recomputed) and the step's map, cell
    areas and zones by size and mtime. A change anywhere upstream therefore rebuilds every later
    step, and the large rasters are not re-read to decide reuse.

    Returns:
        dict: year -> (per-zone table, interval length in years).
    """
    previous_path, previous_year, previous_identity = start
    out = {}
    for year in sorted(maps_by_year):
        state_path = os.path.join(p.cur_dir, 'timber_forest_state_%s_%d.tif' % (label, year))
        table_path = os.path.join(p.cur_dir, 'timber_harvestable_flow_by_zone_%s_%d.csv' % (label, year))
        signature_path = state_path + '.signature.json'
        signature = utilities._signature(
            dict(signed_settings, scenario=label, year=int(year), previous_year=int(previous_year)),
            {'previous_state': previous_identity,
             'growth_parameters': utilities.file_fingerprint(settings['growth_parameters_path'])},
            light_inputs=[maps_by_year[year], base_lulc_path, ha_path, zone_ids_path])
        reason = utilities.outputs_reuse_reason([state_path, table_path], signature, signature_path)
        if reason is None:
            recorded = json.loads(open(signature_path, encoding='utf-8').read())['output']
            if recorded['state'] != utilities.file_fingerprint(state_path, content=False):
                reason = 'the state raster changed after it was written'
            elif recorded['table'] != utilities.file_fingerprint(table_path):
                reason = 'the zone table changed after it was written'
        if reason is None:
            table = hb.df_read(table_path)
        else:
            hb.log('  harvestable flow: %s %d computing because %s' % (label, year, reason))
            if hb.path_exists(signature_path):
                os.remove(signature_path)
            table = _advance_forest_state_raster(previous_path, maps_by_year[year], ha_path, zone_ids_path, n_zones,
                                                 int(year) - int(previous_year), settings, params, state_path,
                                                 base_lulc_path if previous_path == settings['initial_state_path'] else None)
            table.to_csv(table_path, index=False)
            signature['output'] = {'sha256': utilities.sha256_file(state_path),
                                   'state': utilities.file_fingerprint(state_path, content=False),
                                   'table': utilities.file_fingerprint(table_path)}
            utilities.write_outputs_signature(signature, signature_path)
        recorded = json.loads(open(signature_path, encoding='utf-8').read())['output']
        out[int(year)] = (table, int(year) - int(previous_year))
        previous_path, previous_year = state_path, year
        previous_identity = {'path': state_path, 'sha256': recorded['sha256']}
    return out


def _harvestable_flow_shock(p, base_scenario, scenarios, anchor_years, es_shock_base_year, zone_labels,
                            base_lulc_path):
    """The age-aware timber shock: per-hectare change in the harvest the forest state yields.

    f is a region's mean annual harvest over the interval ending at an anchor year, per hectare of
    the timberland (managed, accessible forest) present during that interval. A map takes effect at
    the end of its year: an interval's harvest is cut from the timberland present at its start, so
    with annual anchors a change in the 2030 map first changes the 2031 harvest. Every chain, the
    reference, the baseline and each scenario, starts from the same initial state and management.

    - shock_pct and shock_pct_contemp are 100 (f / f_bau - 1), against the baseline scenario in the
      same interval. The harvest timing of the shared age structure cancels, so identical maps give
      zero.
    - shock_pct_v3 is 100 (f / f_ref - 1), against the reference flow the setting names. It also
      carries that harvest timing, which timing_per_ha_factor (f_bau / f_ref) reports alone.
    - shock_pct_fixedbase is 100 (f - f_bau) / f_ref, as in the other measures.

    Only per-hectare factors are written as shocks. The area and total factors, against the
    reference and against the baseline scenario, are written beside them in the regional table.
    Feeding the per-hectare factor keeps a forest-area change from counting twice only if the
    economic model's own forest-land change matches the timberland change the maps carry, in
    definition and in hectares. Nothing here checks that condition. This is an experimental
    construction, not a calibrated aoall shock, and the annual values interpolate interval means
    between anchors.

    Harvest and timberland are taken per volume basis, and volumes on different bases are never added.
    A region holding timberland or harvest on more than one basis, in any chain at any anchor, combines
    its within-basis factors with the fixed reference weights of p.timber_provision_volume_basis_weights_path
    (combine_basis_factors). Without that table it is declared. With it, a region is also declared when a
    basis it holds has no weight or when it holds only part of its reference value. The combined
    region's area factors add hectares across bases, its volume columns stay empty, and the by-basis
    table reports each basis's harvest, timberland, timing and factors. Unmeasured types (known area, unknown
    yield) are not in the per-hectare denominator; the regional table reports their timberland
    beside the measured timberland and the total harvest, and the ledger reports every quantity by
    zone and forest type. Forest area comes only from the maps, so no economic output, frs or any
    other, creates or removes forest here.

    A region whose reference or same-interval baseline has no flow or no timberland at any anchor,
    or which itself has no timberland at an anchor, has no defined factor. It is left out of the
    table for every scenario and year and named in the excluded-regions file. Its effect is
    undefined, and a consumer must not read the missing rows as a zero shock.
    """
    settings = _harvestable_flow_settings(p)
    params = tp.growth_parameters_from_table(hb.df_read(settings['growth_parameters_path']))
    provenance_df = hb.df_read(settings['input_provenance_path'])
    provenance = tp.input_provenance_from_table(provenance_df)
    ha_path = getattr(p, 'ha_per_cell_10sec_path', None) or p.get_path(*utilities.HA_PER_CELL_10SEC_REF_PARTS)
    ha_path = _aligned_to_reference(p, ha_path, base_lulc_path, 'ha_per_cell')
    area_summary_path = os.path.join(p.cur_dir, 'timber_cell_area_by_zone.csv')
    zone_ids_path = os.path.splitext(area_summary_path)[0] + '_zone_ids.tif'
    zone_signature = utilities._signature({'id_column': p.terrestrial_carbon_shock_id_col},
                                          {'boundary': utilities.file_fingerprint(p.region_boundary_path)},
                                          light_inputs=[ha_path, base_lulc_path])
    zone_signature_path = area_summary_path + '.signature.json'
    if utilities.outputs_reuse_reason([area_summary_path, zone_ids_path], zone_signature, zone_signature_path) is not None:
        # hazelbean reuses a zone-id raster it finds on disk, and the covered-zone cache follows it.
        for stale in (zone_signature_path, area_summary_path, zone_ids_path, os.path.join(p.cur_dir, GRID_COVERED_ZONES)):
            if hb.path_exists(stale):
                os.remove(stale)
        utilities.summarize_raster_by_region(ha_path, p.region_boundary_path, area_summary_path,
                                             year=es_shock_base_year, id_column=p.terrestrial_carbon_shock_id_col)
        utilities.write_outputs_signature(zone_signature, zone_signature_path)
    n_zones = int(max(zone_labels))
    signed_settings = {k: (v if k != 'growth_parameters_path' else os.path.basename(str(v)))
                       for k, v in settings.items() if k != 'input_provenance_path'}
    signed_settings.update(measure='harvestable_flow', accounting='harvestable_flow_age_v2', base_year=es_shock_base_year)
    start = (settings['initial_state_path'], es_shock_base_year,
             utilities.file_fingerprint(settings['initial_state_path'], content=False))
    chains = {REFERENCE_LABEL: _forest_state_chain(p, REFERENCE_LABEL, {anchor_years[0]: base_lulc_path}, start, ha_path,
                                                   zone_ids_path, n_zones, settings, params, signed_settings, base_lulc_path)}
    for scenario in dict.fromkeys([base_scenario] + list(scenarios)):
        maps = {y: p.scenario_lulc_paths[scenario][y] for y in anchor_years}
        chains[scenario] = _forest_state_chain(p, scenario, maps, start, ha_path, zone_ids_path, n_zones, settings,
                                               params, signed_settings, base_lulc_path)

    covered = _grid_covered_zones(p, area_summary_path)
    zones_by_region = {}
    for zone_id, (endw, reg) in zone_labels.items():
        zones_by_region.setdefault(reg, []).append(zone_id)
    in_scope = {reg: ids for reg, ids in zones_by_region.items() if covered is None or all(int(z) in covered for z in ids)}
    declared = {reg: 'not fully covered by the run grid' for reg in zones_by_region if reg not in in_scope}

    quantities = ('harvest_m3', 'timberland_open_ha', 'forest_ha', 'closing_m3', 'timberland_ha')
    bases = sorted({prm['volume_basis'] for prm in params.values()})
    measured = [b for b in bases if b != tp.UNMEASURED_BASIS]

    def regional(label, year):
        """Per region and volume basis: annual harvest, timberland during the interval, forest, stock and timberland at the anchor."""
        table, n_years = chains[label][year]
        by_basis = table.groupby(['volume_basis', 'region_id'])[list(quantities)].sum()
        out = {}
        for reg, ids in in_scope.items():
            out[reg] = {}
            for b in bases:
                rows_b = by_basis.loc[b].reindex(ids) if b in by_basis.index.get_level_values(0) else None
                out[reg][b] = tuple(0.0 if rows_b is None else float(rows_b[c].sum()) / (n_years if c == 'harvest_m3' else 1)
                                    for c in quantities)
        return out
    regionals = {label: {y: regional(label, y) for y in chain} for label, chain in chains.items()}
    # Volumes measured on different bases are never added. A region holding one basis is shocked on it.
    # A region holding more is combined with the configured fixed reference weights, or declared when
    # none are configured. With weights configured, a region whose reference value the model holds only
    # in part is declared too, so a partial cover never stands for the region's whole forestry.
    weights = _volume_basis_weights(p)
    held = {}
    for reg in in_scope:
        held[reg] = [b for b in measured if any(max(by_reg[reg][b][:2]) > 0 for chain in regionals.values() for by_reg in chain.values())]
        if reg in declared or not held[reg]:
            continue
        if weights is None:
            if len(held[reg]) > 1:
                declared[reg] = ('timberland on more than one volume basis (%s): volumes on different bases are not added, '
                                 'and no volume-basis weights are configured' % ', '.join(held[reg]))
            continue
        reference = weights.get(reg)
        if not reference:
            declared[reg] = 'no reference weights for this region in the volume-basis weights table'
        elif any(b not in reference for b in held[reg]):
            declared[reg] = ('timberland on volume basis %s, which has no reference weight'
                             % ', '.join(b for b in held[reg] if b not in reference))
        elif sum(reference[b] for b in held[reg]) < sum(reference.values()) * (1.0 - 1e-12):
            declared[reg] = ('the model holds %.4g of the reference value: no timberland on %s'
                             % (sum(reference[b] for b in held[reg]) / sum(reference.values()),
                                ', '.join(b for b in reference if b not in held[reg])))
    zero = (0.0,) * len(quantities)
    all_years = list(range(es_shock_base_year, anchor_years[-1] + 1))
    rows, side_by_side, by_basis_rows = [], [], []
    for scenario in scenarios:
        for reg, ids in in_scope.items():
            if reg in declared:
                continue
            per_basis = {}
            for b in held[reg] or [None]:
                def on_basis(label, year):
                    return regionals[label][year][reg][b] if b else zero
                by_year = {y: on_basis(scenario, y) for y in anchor_years}
                bau = {y: on_basis(base_scenario, y) for y in anchor_years}
                flow_ref, area_ref = on_basis(REFERENCE_LABEL, anchor_years[0])[:2]
                flow, area = (np.array([by_year[y][i] for y in anchor_years]) for i in (0, 1))
                bau_flow, bau_area = (np.array([bau[y][i] for y in anchor_years]) for i in (0, 1))
                f = tp.harvestable_flow_factors(flow, area, flow_ref, area_ref)
                f_bau = tp.harvestable_flow_factors(bau_flow, bau_area, flow_ref, area_ref)
                vs_bau = tp.harvestable_flow_factors(flow, area, bau_flow, bau_area)
                on = ' on %s' % b if len(held[reg]) > 1 else ''
                if not np.isfinite(f['per_ha_factor']).all():
                    declared[reg] = ('no defined per-hectare factor%s: reference flow %.6g m3/yr, reference timberland '
                                     '%.6g ha, timberland at the anchors %s ha' % (on, flow_ref, area_ref, area.tolist()))
                    break
                if not np.isfinite(vs_bau['per_ha_factor']).all():
                    declared[reg] = ('no defined contemporaneous per-hectare factor%s: baseline harvest %s m3/yr and baseline '
                                     'timberland %s ha at the anchors' % (on, bau_flow.tolist(), bau_area.tolist()))
                    break
                per_basis[b] = dict(by_year=by_year, flow=flow, area=area, bau_flow=bau_flow, bau_area=bau_area,
                                    flow_ref=flow_ref, area_ref=area_ref, f=f, f_bau=f_bau, vs_bau=vs_bau)
            if reg in declared:
                continue
            w = {b: 1.0 for b in per_basis} if len(per_basis) == 1 else {b: weights[reg][b] for b in per_basis}
            share = {b: w[b] / sum(w.values()) for b in per_basis}

            def combined(factors, column):
                return tp.combine_basis_factors({b: v[factors][column] for b, v in per_basis.items()}, w)
            per_ha, timing, contemp = combined('f', 'per_ha_factor'), combined('f_bau', 'per_ha_factor'), combined('vs_bau', 'per_ha_factor')
            v3 = 100.0 * (per_ha - 1.0)
            columns = {'shock_pct_v3': v3, 'shock_pct_fixedbase': 100.0 * (per_ha - timing),
                       'shock_pct_contemp': 100.0 * (contemp - 1.0)}
            annual = {c: utilities.anchored_series(all_years, [es_shock_base_year] + list(anchor_years), [0.0] + list(v))
                      for c, v in columns.items()}
            for zone_id in ids:
                endw = zone_labels[zone_id][0]
                for i, year in enumerate(all_years):
                    rows.append({'ENDW': endw, 'ACTS': p.terrestrial_carbon_shock_acts, 'REG': reg, 'scenario': scenario,
                                 'year': year, 'shock_pct': annual['shock_pct_contemp'][i],
                                 'shock_pct_fixedbase': annual['shock_pct_fixedbase'][i],
                                 'shock_pct_contemp': annual['shock_pct_contemp'][i], 'shock_pct_v3': annual['shock_pct_v3'][i]})
            # Hectares add across bases; volumes do not, so a combined region's volume columns stay empty
            # here and are reported per basis in the by-basis table.
            one = per_basis[next(iter(per_basis))] if len(per_basis) == 1 else None
            area, area_ref, bau_area = (sum(v[k] for v in per_basis.values()) for k in ('area', 'area_ref', 'bau_area'))
            total, total_vs_bau = combined('f', 'total_factor'), combined('vs_bau', 'total_factor')
            for i, year in enumerate(anchor_years):
                every_basis = regionals[scenario][year][reg]
                side_by_side.append({'scenario': scenario, 'REG': reg, 'year': year, 'volume_basis': '+'.join(per_basis),
                                     'basis_weights': ' '.join('%s:%.6g' % (b, share[b]) for b in per_basis),
                                     'forest_ha': sum(v[2] for v in every_basis.values()),
                                     'timberland_during_interval_ha': area[i],
                                     'unmeasured_timberland_during_interval_ha': every_basis.get(tp.UNMEASURED_BASIS, zero)[1],
                                     'timberland_at_anchor_ha': sum(v['by_year'][year][4] for v in per_basis.values()),
                                     'stock_m3': one['by_year'][year][3] if one else np.nan,
                                     'harvest_m3_per_year': one['flow'][i] if one else np.nan,
                                     'reference_harvest_m3_per_year': one['flow_ref'] if one else np.nan,
                                     'reference_timberland_ha': area_ref,
                                     'area_factor': area[i] / area_ref, 'per_ha_factor': per_ha[i],
                                     'total_factor': total[i], 'timing_per_ha_factor': timing[i],
                                     'area_factor_vs_baseline_scenario': area[i] / bau_area[i],
                                     'per_ha_factor_vs_baseline_scenario': contemp[i],
                                     'total_factor_vs_baseline_scenario': total_vs_bau[i],
                                     'shock_pct_contemp': columns['shock_pct_contemp'][i], 'shock_pct_v3': v3[i]})
                for b, v in per_basis.items():
                    by_basis_rows.append({'scenario': scenario, 'REG': reg, 'year': year, 'volume_basis': b, 'weight': share[b],
                                          'timberland_during_interval_ha': v['area'][i], 'harvest_m3_per_year': v['flow'][i],
                                          'stock_m3': v['by_year'][year][3], 'baseline_timberland_ha': v['bau_area'][i],
                                          'baseline_harvest_m3_per_year': v['bau_flow'][i],
                                          'reference_timberland_ha': v['area_ref'], 'reference_harvest_m3_per_year': v['flow_ref'],
                                          'per_ha_factor': v['f']['per_ha_factor'][i], 'timing_per_ha_factor': v['f_bau']['per_ha_factor'][i],
                                          'area_factor_vs_baseline_scenario': v['vs_bau']['area_factor'][i],
                                          'per_ha_factor_vs_baseline_scenario': v['vs_bau']['per_ha_factor'][i],
                                          'total_factor_vs_baseline_scenario': v['vs_bau']['total_factor'][i]})
    rows = [r for r in rows if r['REG'] not in declared]
    side_by_side = [r for r in side_by_side if r['REG'] not in declared]
    by_basis_rows = [r for r in by_basis_rows if r['REG'] not in declared]
    for reg in sorted(declared):
        hb.log('  harvestable flow: %s declared, not shocked: %s' % (reg, declared[reg]))
    ledger = pd.concat([table.assign(scenario=label, year=year, interval_years=n_years)
                        for label, chain in chains.items() for year, (table, n_years) in chain.items()])
    ledger['ledger_residual_m3'] = ledger['closing_m3'] - (ledger['opening_m3'] + ledger['growth_m3']
                                                           - ledger['harvest_m3'] - ledger['removed_m3'])
    ledger.to_csv(os.path.join(p.cur_dir, 'timber_harvestable_flow_ledger_by_zone.csv'), index=False)
    pd.DataFrame(side_by_side).to_csv(os.path.join(p.cur_dir, 'timber_harvestable_flow_by_region.csv'), index=False)
    pd.DataFrame(by_basis_rows).to_csv(os.path.join(p.cur_dir, 'timber_harvestable_flow_by_region_and_basis.csv'), index=False)

    out = pd.DataFrame(rows, columns=['ENDW', 'ACTS', 'REG', 'scenario', 'year', 'shock_pct', 'shock_pct_fixedbase',
                                      'shock_pct_contemp', 'shock_pct_v3'])
    out = utilities.filter_to_model_domain(out, p.timber_provision_shock_output_path, 'timber_provision', log=hb.log)
    utilities.assert_shock_table_sound(out, scenarios, 'timber_provision', column='shock_pct_v3')
    out['timber_resource_measure'] = 'harvestable_flow'
    out['timber_provision_accounting'] = 'harvestable_flow_age_v2'
    out['harvest_basis'] = settings['harvest_rule']
    out['volume_basis'] = out['REG'].map({reg: '+'.join(bases_held) for reg, bases_held in held.items()})
    out['reference_flow'] = settings['reference_flow']
    out['input_provenance'] = ' '.join('%s:%s' % item for item in provenance.items())
    provenance_df.to_csv(os.path.join(p.cur_dir, 'timber_harvestable_flow_input_provenance.csv'), index=False)
    shock_signature = os.path.join(os.path.dirname(p.timber_provision_shock_output_path), TIMBER_SHOCK_SIGNATURE)
    if hb.path_exists(shock_signature):
        os.remove(shock_signature)
    out.to_csv(p.timber_provision_shock_output_path, index=False)
    hb.write_to_file(json.dumps({'excluded_regions': sorted(declared), 'reason': declared}, indent=2),
                     os.path.join(os.path.dirname(p.timber_provision_shock_output_path), EXCLUDED_REGIONS))
    # Signed with the measure and its settings, so a later run of another measure cannot reuse this table.
    shock_maps = sorted({m for by_year in p.scenario_lulc_paths.values() for m in by_year.values()})
    utilities.write_reuse_signature(p, 'timber_provision', [p.timber_provision_shock_output_path],
                                    TIMBER_SHOCK_SIGNATURE, light_inputs=shock_maps)
    hb.log('  timber shock (harvestable_flow): %d rows, %d declared region(s) -> %s'
           % (len(out), len(declared), p.timber_provision_shock_output_path))
    return True
