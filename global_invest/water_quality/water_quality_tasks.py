"""Water-quality GEP tasks: retention priced at the authors' benefit-transfer treatment cost.

This layer owns every file read and write; the science it calls lives in water_quality_functions,
which never opens a file.
"""
import pandas as pd

import hazelbean as hb
from global_invest import utilities
from global_invest.water_quality import water_quality_functions as wq


def publish_inputs(p):
    """Every GEP task's first line: the water_quality es_config row and the data references
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'water_quality', log=hb.log)
    utilities.hydrate_es_parameters(p, 'water_quality', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def read_world_bank_indicator(path):
    """A World Bank indicator download (the API CSV, four metadata lines above the header) as a
    country-code by year table with int year columns."""
    df = pd.read_csv(path, skiprows=4).set_index('Country Code')
    years = [c for c in df.columns if str(c).isdigit()]
    df = df[years]
    df.columns = [int(c) for c in years]
    return df


def gep_calculation(p):
    """GEP valuation for water quality: the authors' price pipeline over the retention estimates,
    one row per country."""
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'water_quality')
    if already_done:
        return

    if not p.run_this:
        return p
    base_year = int(p.gep_base_year)
    raw = pd.read_excel(p.water_quality_raw_prices_path)
    deflator = read_world_bank_indicator(p.water_quality_gdp_deflator_path)
    exchange_rate = read_world_bank_indicator(p.water_quality_exchange_rate_path)
    gdppc = read_world_bank_indicator(p.water_quality_gdp_per_capita_path)[base_year]

    observed = wq.observed_prices_2019_usd(raw, deflator, exchange_rate, base_year)
    observed[f'gdppc_usd_{base_year}'] = observed['iso3_r250_label'].map(gdppc)
    coefficients, n_observations = wq.pooled_price_model(observed, base_year)
    hb.log('Water purification price model: %d observations, b0 %.8f, b_gdppc %.8f, b_p %.8f'
           % ((n_observations,) + coefficients))

    retention = hb.df_read(p.water_quality_retention_path)
    prices = wq.predicted_prices(gdppc.reindex(retention['iso3_r250_label']), coefficients)
    df_values = wq.water_purification_gep(retention, prices)

    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = utilities.collapse_countries_to_r250(p.df_countries)[attr_cols]
    df_gep = countries.merge(df_values, on='iso3_r250_label', how='left')
    df_gep['year'] = base_year
    value_cols = ['price_n_usd_per_kg', 'price_p_usd_per_kg', 'nitrogen_retention_kg',
                  'phosphorus_retention_kg', 'nitrogen_gep_usd', 'phosphorus_gep_usd', 'water_quality_gep']
    hb.df_write(df_gep[attr_cols + ['year'] + value_cols], service_results['gep_by_country_base_year'])

    gep = df_gep['water_quality_gep']
    hb.log(f'Total water_quality GEP for base year {base_year}: {gep.sum():,.2f} USD over '
           f'{int((gep > 0).sum())} positive, {int((gep == 0).sum())} zero and '
           f'{int(gep.isna().sum())} missing countries.')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.render_service_results(p)
