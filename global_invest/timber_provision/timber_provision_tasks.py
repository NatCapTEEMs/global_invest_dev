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
    # ⚠ The account's timber value is CWoN's rent, so `timber_provision_gep` -- the shared key
    # every other service writes and the account reads -- IS the CWoN rent. The spatial estimate
    # is kept beside it as `timber_provision_gep_spatial`: it is the only forestry layer this
    # library has.
    df_gep = df_gep.rename(columns={'timber_provision_gep': 'timber_provision_gep_spatial',
                                    'timber_provision_gep_cwon_rent': 'timber_provision_gep'})
    # ⚠ The fuelwood decomposition. timber_provision_gep is CWoN's forest rent, which is built
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
    hb.log('  ⚠ of which fuelwood: %s (%.1f%%); industrial roundwood: %s. A separate fuelwood '
           'service would be a SUBSET of the timber figure, not an addition.'
           % (f'{fuel:,.2f}', 100 * fuel / (fuel + industrial), f'{industrial:,.2f}'))
    for column, label in (('timber_provision_gep_spatial', 'spatial'),
                          ('timber_provision_gep', 'CWoN rent')):
        priced = df_gep[df_gep['timber_roundwood_gross_value'].gt(0) & df_gep[column].gt(0)]
        over = priced[priced[column] > priced['timber_roundwood_gross_value']]
        if len(over):
            hb.log('  ⚠ %s exceeds its own gross roundwood value in %d countries: %s'
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

    ⚠ Fuelwood is NOT an additional service on top of timber. CWoN's forest rent is built from
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

    ⚠ The reference applies no ecosystem share at all: across the 178 countries FAOSTAT also
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
