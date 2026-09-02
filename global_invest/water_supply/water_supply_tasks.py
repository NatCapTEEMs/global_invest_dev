"""Water-supply GEP tasks. First component: hydropower (CWoN resource-rent method).

The hydropower rent derives from CWoN 2024's capitalized wealth (see the functions module for
the identified method and its anchor); the agriculture and household components join here when
their science surfaces.
"""
import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities
from global_invest.water_supply import water_supply_functions as wf


def publish_inputs(p):
    """Every GEP task's first line: the water_supply es_config row and the CWoN data reference
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'water_supply', log=hb.log)
    utilities.hydrate_es_parameters(p, 'water_supply', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def hydropower_rent(p):
    """CWoN capitalized hydropower wealth -> the implied constant annual rent per country."""
    publish_inputs(p)
    p.hydropower_rent_path = os.path.join(p.cur_dir, 'hydropower_rent.csv')
    if not p.run_this:
        return
    if not hb.path_exists(p.hydropower_rent_path):
        wealth = pd.read_stata(p.water_supply_cwon_hydro_wealth_path)
        # The rate is configuration rather than a module constant, so the one number the
        # hydropower figure turns on is visible beside every other service's parameters.
        wf.hydropower_rent_from_wealth(
            wealth,
            capitalization_rate=float(p.water_supply_hydropower_capitalization_rate),
        ).to_csv(p.hydropower_rent_path, index=False)
    return True


def gep_calculation(p):
    """GEP valuation for water_supply: the hydropower component on the r250 country list,
    one row per country. water_supply_gep currently equals the hydropower component; the
    agriculture and household components add columns here when they arrive."""
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'water_supply')
    if already_done:
        return

    hydropower = hb.df_read(p.hydropower_rent_path)
    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = utilities.collapse_countries_to_r250(p.df_countries)[attr_cols]
    df_gep = wf.water_supply_gep_by_country(hydropower, countries)
    if hb.path_exists(getattr(p, 'water_use_components_path', None)):
        components = hb.df_read(p.water_use_components_path)
        df_gep = df_gep.merge(components.drop(columns=['iso3_r250_label']),
                              on='iso3_r250_id', how='left')
    else:
        df_gep['water_use_agriculture_gep'] = float('nan')
        df_gep['water_use_all_sector_gep'] = float('nan')
    df_gep['year'] = int(p.gep_base_year)
    # water_supply_gep stays the hydropower component alone: how (and whether) the water-use
    # components combine with it is the account's subgroup question, flagged on the deck.
    df_gep['water_supply_gep'] = df_gep['hydropower_gep']
    utilities.write_gep_by_country(
        p, df_gep[attr_cols + ['year', 'hydropower_gep', 'hydropower_gep_reference_variant',
                               'water_use_irrigation_gep', 'water_use_domestic_gep',
                               'water_use_agriculture_gep', 'water_use_all_sector_gep',
                               'water_supply_gep']],
        service_results['gep_by_country_base_year'])

    ours = df_gep['hydropower_gep'].sum()
    reference = df_gep['hydropower_gep_reference_variant'].sum()
    n_extra = int(df_gep['hydropower_gep'].notna().sum()
                  - df_gep['hydropower_gep_reference_variant'].notna().sum())
    hb.log(f'Total water_supply GEP (hydropower component) for base year {p.gep_base_year}: '
           f'{ours:,.2f}')
    hb.log(f'  reference-matching variant {reference:,.2f} over {n_extra} fewer countries; '
           f'the gap is the reference exclusions, which we no longer apply to the reported value')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)


def water_use_components(p):
    """The water-use calculation computed from the raw inputs (script-01 cleaning, then script-02
    efficiency x withdrawal products at the survey years), reported as OUR run's components.
    The drive's committed per-country tables are NOT this calculation's outputs -- a newer AQUASTAT
    vintage plus the appendix's deflate-to-2015 step for the all-sector total, a separate
    crop-water calculation for agriculture (see the functions module) -- so they are the comparison
    anchors, logged and pinned in the test suite, never the reported values."""
    publish_inputs(p)
    p.water_use_efficiency_path = os.path.join(p.cur_dir, 'aquastat_water_efficiency_cleaned.csv')
    p.water_use_gep_path = os.path.join(p.cur_dir, 'water_use_gep_by_country_year.csv')
    p.water_use_components_path = os.path.join(p.cur_dir, 'water_use_components.csv')
    if not p.run_this:
        return
    if not hb.path_exists(p.water_use_efficiency_path):
        raw = pd.read_csv(p.water_use_efficiency_input_path, encoding='utf-8-sig')
        wf.clean_aquastat_water_efficiency(raw).to_csv(
            p.water_use_efficiency_path, index=False, encoding='utf-8-sig')
    if not hb.path_exists(p.water_use_gep_path):
        efficiency = pd.read_csv(p.water_use_efficiency_path, encoding='utf-8-sig')
        withdrawal = pd.read_csv(p.water_use_withdrawal_path, encoding='utf-8-sig')
        df_gep = wf.water_use_gep_by_country_year(efficiency, withdrawal)
        df_gep.to_csv(p.water_use_gep_path, index=False, encoding='utf-8-sig')
        sector_cols = ['gep_water_agricultural', 'gep_water_industrial', 'gep_water_municipal']
        hb.log('water_use chain: %d country-year rows, %.4g USD summed over survey years '
               'and sectors' % (len(df_gep), df_gep[sector_cols].sum().sum()))
    if not hb.path_exists(p.water_use_components_path):
        df_gep = pd.read_csv(p.water_use_gep_path, encoding='utf-8-sig')
        name_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name', 'name_long']
        countries = p.df_countries[[c for c in name_cols if c in p.df_countries.columns]].drop_duplicates('iso3_r250_id')
        out = wf.water_use_components_from_chain(df_gep, countries)
        out.to_csv(p.water_use_components_path, index=False, encoding='utf-8-sig')
        hb.log('water_use components (OUR chain): agriculture %.4g USD (%d countries), all-sector %.4g USD '
               '(%d countries); %d chain countries without an r250 match' % (
                   out['water_use_agriculture_gep'].sum(), out['water_use_agriculture_gep'].notna().sum(),
                   out['water_use_all_sector_gep'].sum(), out['water_use_all_sector_gep'].notna().sum(),
                   out['iso3_r250_label'].isna().sum()))
        committed_ag = hb.df_read(p.water_use_agriculture_path)
        committed_all = hb.df_read(p.water_use_all_sector_path)
        hb.log('committed anchors: agriculture %.4g USD, all-sector %.4g USD -- different source '
               'chains (see the functions module), compared in the test suite' % (
                   committed_ag['wateruse_ag_gep'].sum(), committed_all['wateruse_gep'].sum()))
    return True
