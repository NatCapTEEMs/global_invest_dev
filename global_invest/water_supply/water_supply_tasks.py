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
        wf.hydropower_rent_from_wealth(wealth).to_csv(p.hydropower_rent_path, index=False)
    return True


def gep_calculation(p):
    """GEP valuation for water_supply: the hydropower component on the r250 country list,
    one row per country. water_supply_gep currently equals the hydropower component; the
    agriculture and household components add columns here when they arrive."""
    publish_inputs(p)
    service_results = {}
    p.results['water_supply'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, 'gep_by_country_base_year.csv')

    if hb.path_all_exist(list(service_results.values())):
        hb.log('All results already exist. Skipping GEP calculation for water_supply.')
        return
    hb.log('Starting GEP calculation for water_supply.')

    hydropower = pd.read_csv(p.hydropower_rent_path)
    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = p.df_countries[attr_cols].drop_duplicates('iso3_r250_id')
    df_gep = wf.water_supply_gep_by_country(hydropower, countries)
    if hb.path_exists(getattr(p, 'water_use_components_path', None)):
        components = pd.read_csv(p.water_use_components_path)
        df_gep = df_gep.merge(components.drop(columns=['iso3_r250_label']),
                              on='iso3_r250_id', how='left')
    else:
        df_gep['water_use_agriculture_gep'] = float('nan')
        df_gep['water_use_all_sector_gep'] = float('nan')
    df_gep['year'] = int(p.gep_base_year)
    # water_supply_gep stays the hydropower component alone: how (and whether) the water-use
    # components combine with it is the account's subgroup question, flagged on the deck.
    df_gep['water_supply_gep'] = df_gep['hydropower_gep']
    hb.df_write(df_gep[attr_cols + ['year', 'hydropower_gep', 'water_use_agriculture_gep',
                                    'water_use_all_sector_gep', 'water_supply_gep']],
                service_results['gep_by_country_base_year'])

    hb.log(f'Total water_supply GEP (hydropower component) for base year {p.gep_base_year}: '
           f'{df_gep["hydropower_gep"].sum():,.2f}')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)


def water_use_components(p):
    """The water-use chain computed from the raw inputs (script-01 cleaning, then script-02
    efficiency x withdrawal products at the survey years), and the components' per-country
    values from the drive's committed outputs. The committed per-country tables are NOT the
    chain's outputs -- a newer AQUASTAT vintage plus the appendix's deflate-to-2015 step for
    the all-sector total, a separate crop-water chain for agriculture (see the functions
    module) -- so they stay the adopted values; the chain's replication anchors are its own
    two committed intermediates, pinned in the test suite."""
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
        agriculture = pd.read_csv(p.water_use_agriculture_path)
        all_sector = pd.read_csv(p.water_use_all_sector_path)
        attr_cols = ['iso3_r250_id', 'iso3_r250_label']
        countries = p.df_countries[attr_cols].drop_duplicates('iso3_r250_id')
        out = wf.water_use_components_by_country(agriculture, all_sector, countries)
        out.to_csv(p.water_use_components_path, index=False)
        hb.log('water_use components: agriculture %.4g USD (%d countries), all-sector %.4g USD '
               '(%d countries)' % (out['water_use_agriculture_gep'].sum(),
                                   out['water_use_agriculture_gep'].notna().sum(),
                                   out['water_use_all_sector_gep'].sum(),
                                   out['water_use_all_sector_gep'].notna().sum()))
    return True
