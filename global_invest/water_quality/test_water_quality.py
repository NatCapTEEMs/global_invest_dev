"""Unit tests for the water_quality port of the authors' price pipeline (`analysis.do`).

The conversion, regression and valuation steps are pinned on hand-built rows, and the whole
chain on the staged inputs is checked against the authors' own exported table. The retention
quantities are an InVEST NDR estimation taken as given, so nothing here tests them beyond the
identity that builds the domestic share from the total.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest.water_quality import water_quality_functions as wq


def _configured_base_data_path(parameter):
    """The base_data file an es_parameters row names, found the way the task finds it."""
    import tempfile
    import hazelbean as hb
    params = pd.read_csv(os.path.join(os.path.dirname(wq.__file__), '..', 'input_template',
                                      'es_parameters.csv'), encoding='utf-8-sig')
    value = params.loc[params['parameter'] == parameter, 'value'].iloc[0]
    project = hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors'))
    return os.path.join(project.base_data_dir, value)


def _rates():
    """Toy World Bank tables: local currency per USD and the GDP deflator, by country and year."""
    years = [2007, 2008, 2009, 2010, 2015, 2019]
    fx = pd.DataFrame({y: [1.0, 0.7, 3.0, 2.0, 11.0, 0.7, 0.9, 6.0] for y in years},
                      index=['USA', 'EMU', 'POL', 'DEU_unused', 'EST', 'DEU', 'FIN', 'CHN'])
    fx.loc['EMU', 2019] = 0.9
    fx.loc['DEU', 2019] = 0.9
    fx.loc['EST', 2019] = 0.9
    fx.loc['POL', 2019] = 4.0
    fx.loc['CHN', 2019] = 7.0
    for country in ('LVA', 'LTU'):
        fx.loc[country] = 1.0
    deflator = pd.DataFrame({y: 100.0 for y in years}, index=fx.index)
    deflator[2019] = 120.0
    return fx, deflator


def test_reported_prices_become_2019_usd_per_kg():
    fx, deflator = _rates()
    raw = pd.DataFrame({
        'iso3_r250_label': ['USA', 'POL', 'DEU', 'EST', 'CHN'],
        'year': [2015, 2008, 2008, 2008, 2015],
        'price_n_reported': [1.0, 12.0, 15.0, 12.0, 1750.0],
        'price_p_reported': [np.nan, 41.0, 61.0, 41.0, 2800.0],
        'currency': ['usd', 'euro', 'euro', 'euro', 'cny'],
        'per_unit': ['lb', 'kg', 'kg', 'kg', 'ton']})
    out = wq.observed_prices_2019_usd(raw, deflator, fx).set_index('iso3_r250_label')
    level = 120.0 / 100.0
    # USA: per lb to per kg, already local currency, deflated, 2019 rate 1.
    assert out.loc['USA', 'price_n_usd_kg_2019'] == pytest.approx(1.0 * 2.20462262 * level)
    # Poland reports in euro: to zloty through the euro's dollar rate in 2008, then 2019 zloty per USD.
    assert out.loc['POL', 'price_n_usd_kg_2019'] == pytest.approx(12.0 * 3.0 / 0.7 * level / 4.0)
    # Germany reports in its own currency: no conversion before the 2019 rate.
    assert out.loc['DEU', 'price_p_usd_kg_2019'] == pytest.approx(61.0 * level / 0.9)
    # Estonia before the euro: euro to kroon in 2008, deflated, kroon to euro at the 2010 rate, to USD.
    kroon_per_euro = fx.loc['EST', 2010] / fx.loc['EMU', 2010]
    assert out.loc['EST', 'price_n_usd_kg_2019'] == pytest.approx(
        12.0 * 11.0 / 0.7 * level / kroon_per_euro / 0.9)
    # China: yuan per tonne to per kg, already local currency.
    assert out.loc['CHN', 'price_p_usd_kg_2019'] == pytest.approx(2800.0 / 1000.0 * level / 7.0)
    assert np.isnan(out.loc['USA', 'price_p_usd_kg_2019'])


def test_the_pooled_model_recovers_a_known_line():
    gdppc = np.array([1e3, 5e3, 2e4, 6e4, 3e3, 4e4])
    b0, b_gdppc, b_p = -3.0, 0.6, 1.0
    df = pd.DataFrame({'gdppc_usd_2019': gdppc,
                       'price_n_usd_kg_2019': np.exp(b0 + b_gdppc * np.log(gdppc)),
                       'price_p_usd_kg_2019': np.exp(b0 + b_gdppc * np.log(gdppc) + b_p)})
    df.loc[0, 'price_p_usd_kg_2019'] = np.nan
    coefficients, n = wq.pooled_price_model(df)
    assert n == 11
    assert np.allclose(coefficients, (b0, b_gdppc, b_p))
    prices = wq.predicted_prices(pd.Series([1e4], index=['AAA']), coefficients)
    assert prices.loc['AAA', 'price_p_usd_per_kg'] / prices.loc['AAA', 'price_n_usd_per_kg'] == pytest.approx(np.e)


@pytest.mark.parametrize('case', ['empty', 'non-finite', 'one nutrient', 'one income level'])
def test_a_fit_that_cannot_identify_its_coefficients_is_refused(case):
    """Least squares answers all four: zeros for an empty sample (a $1/kg price everywhere), a
    minimum-norm guess for the rank-deficient ones, inf or nan for a non-finite input."""
    gdppc = np.array([1e3, 5e3, 2e4, 6e4])
    df = pd.DataFrame({'gdppc_usd_2019': gdppc, 'price_n_usd_kg_2019': [1.0, 2.0, 3.0, 4.0],
                       'price_p_usd_kg_2019': [3.0, 5.0, 8.0, 9.0]})
    if case == 'empty':
        df[['price_n_usd_kg_2019', 'price_p_usd_kg_2019']] = np.nan
    elif case == 'non-finite':
        df.loc[1, 'price_n_usd_kg_2019'] = np.inf
    elif case == 'one nutrient':
        df['price_p_usd_kg_2019'] = np.nan
    else:
        df['gdppc_usd_2019'] = 2e4
    with pytest.raises(ValueError):
        wq.pooled_price_model(df)


def _retention(n_total, p_total, fraction=0.5):
    return pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB', 'CCC'],
                         'n_TotalRetention(kg)': n_total, 'p_TotalRetention(kg)': p_total,
                         'DomesticWaterUseFraction': fraction,
                         'n_EstimatedRetention(kg)': np.array(n_total) * fraction,
                         'p_EstimatedRetention(kg)': np.array(p_total) * fraction})


def test_negative_retention_is_missing_and_a_true_zero_is_kept():
    prices = pd.DataFrame({'price_n_usd_per_kg': [2.0, 2.0, 2.0], 'price_p_usd_per_kg': [5.0, 5.0, 5.0]},
                          index=['AAA', 'BBB', 'CCC'])
    out = wq.water_purification_gep(_retention([100.0, -40.0, 0.0], [10.0, 10.0, 0.0]), prices)
    out = out.set_index('iso3_r250_label')
    assert out.loc['AAA', 'water_quality_gep'] == pytest.approx(2.0 * 50.0 + 5.0 * 5.0)
    assert np.isnan(out.loc['BBB', 'nitrogen_retention_kg'])
    assert np.isnan(out.loc['BBB', 'water_quality_gep'])        # one missing term, as in the script
    assert out.loc['CCC', 'water_quality_gep'] == 0.0


def test_a_broken_quantity_identity_raises():
    retention = _retention([100.0, 40.0, 10.0], [10.0, 10.0, 10.0])
    retention.loc[0, 'n_EstimatedRetention(kg)'] *= 2
    prices = pd.DataFrame({'price_n_usd_per_kg': [1.0] * 3, 'price_p_usd_per_kg': [1.0] * 3},
                          index=['AAA', 'BBB', 'CCC'])
    with pytest.raises(ValueError, match='domestic fraction'):
        wq.water_purification_gep(retention, prices)


def test_the_port_reproduces_the_authors_table_from_the_staged_inputs():
    """The whole chain on the authors' own inputs against the table their script exported. The
    residual is Stata storing the prices in single precision."""
    from global_invest.water_quality import water_quality_tasks as wt
    paths = {k: _configured_base_data_path(f'water_quality_{k}_path') for k in
             ('raw_prices', 'gdp_deflator', 'exchange_rate', 'gdp_per_capita', 'retention', 'reference_gep')}
    if not all(os.path.exists(v) for v in paths.values()):
        pytest.skip('the staged water purification price inputs are not on this machine')

    raw = pd.read_excel(paths['raw_prices'])
    deflator = wt.read_world_bank_indicator(paths['gdp_deflator'])
    exchange_rate = wt.read_world_bank_indicator(paths['exchange_rate'])
    gdppc = wt.read_world_bank_indicator(paths['gdp_per_capita'])[2019]
    observed = wq.observed_prices_2019_usd(raw, deflator, exchange_rate)
    observed['gdppc_usd_2019'] = observed['iso3_r250_label'].map(gdppc)
    coefficients, n = wq.pooled_price_model(observed)
    assert n == 32
    # The log's `scalar list b0 b_gdppc b_p` (analysis_log.log:545-547), 8 significant figures.
    # Stata keeps the generated prices in single precision (about 6e-8 relative), and the fit
    # carries that through: the largest gap here is 4.9e-7 relative, on the intercept. 2e-6
    # admits that and the log's rounding, and nothing a real change to the pipeline would move.
    assert np.allclose(coefficients, (-3.6374268, 0.59032049, 1.0098805), rtol=2e-6, atol=0)

    retention = pd.read_csv(paths['retention'])
    prices = wq.predicted_prices(gdppc.reindex(retention['iso3_r250_label']), coefficients)
    ours = wq.water_purification_gep(retention, prices).set_index('iso3_r250_label')
    theirs = pd.read_excel(paths['reference_gep']).set_index('iso3_r250_label').reindex(ours.index)
    for mine, his in (('price_n_usd_per_kg', 'price_n'), ('price_p_usd_per_kg', 'price_p'),
                      ('water_quality_gep', 'water_purification_gep')):
        assert (ours[mine].isna() == theirs[his].isna()).all(), mine
        both = ours[mine].notna()
        assert np.allclose(ours.loc[both, mine], theirs.loc[both, his], rtol=2e-6), mine
    gep = ours['water_quality_gep']
    assert gep.sum() == pytest.approx(theirs['water_purification_gep'].sum(), rel=1e-7)
    assert (int((gep > 0).sum()), int((gep == 0).sum()), int(gep.isna().sum())) == (178, 10, 62)
    assert sorted(ours.index[ours['nitrogen_retention_kg'].isna() & retention.set_index(
        'iso3_r250_label')['n_EstimatedRetention(kg)'].lt(0)]) == ['BLZ', 'KWT', 'SOM']


def test_es_config_and_parameters_rows_hydrate_water_quality(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'water_quality', log=lambda *a: None)
    assert p.gep_base_year == 2019
    utilities.hydrate_es_parameters(p, 'water_quality', log=lambda *a: None)
    assert p.water_quality_retention_path.endswith('price_transfer/retention_estimates.csv')
    assert p.water_quality_raw_prices_path.endswith('EXAMPLE_SERVICE_prices_raw_data.xlsx')
    assert not hasattr(p, 'water_quality_international_path')
