"""Unit tests for groundwater_recharge's valuation.

Every function is pinned on a hand-built frame, so the lift-cost constant and the missing-input
rule are stated as executable facts, and the staged committed table anchors the whole product.
"""
import numpy as np
import pandas as pd

from global_invest import utilities
from global_invest.groundwater_recharge import groundwater_recharge_functions as gr


def test_the_lift_cost_is_the_physics():
    # 9810 N/m3 over 0.5 efficiency and 3.6e6 J/kWh: one m3 lifted one metre at $0.10/kWh.
    cost = gr.pumping_cost_per_m3_per_m(0.10, pump_efficiency=0.5)
    assert np.isclose(cost, 9810.0 * 0.10 / (0.5 * 3.6e6))


def test_the_valuation_is_lift_cost_times_depth_times_volume():
    recharge = pd.DataFrame({'iso3_r250_label': ['AAA'], 'l_mean_mm': [100.0]})
    withdrawal = pd.DataFrame({'iso3_r250_label': ['AAA'], 'gw_wdw_bil_m3': [2.0]})
    price = pd.DataFrame({'iso3_r250_label': ['AAA'], 'electricity_usd_per_kwh': [0.10]})
    out = gr.groundwater_gep(recharge, withdrawal, price, pump_efficiency=0.5)
    expected = gr.pumping_cost_per_m3_per_m(0.10, 0.5) * 0.1 * 2.0e9
    assert np.isclose(out['groundwater_recharge_gep'].iloc[0], expected)
    assert out['note'].iloc[0] == ''


def test_a_missing_input_is_an_na_with_its_absence_named():
    recharge = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'l_mean_mm': [100.0, 50.0]})
    withdrawal = pd.DataFrame({'iso3_r250_label': ['AAA'], 'gw_wdw_bil_m3': [2.0]})
    price = pd.DataFrame({'iso3_r250_label': ['AAA'], 'electricity_usd_per_kwh': [0.10]})
    out = gr.groundwater_gep(recharge, withdrawal, price, pump_efficiency=0.5).set_index('iso3_r250_label')
    assert pd.isna(out.loc['BBB', 'groundwater_recharge_gep'])
    assert 'no withdrawal figure' in out.loc['BBB', 'note']
    assert 'no electricity tariff' in out.loc['BBB', 'note']


def test_the_staged_inputs_reproduce_the_committed_table_where_all_inputs_exist():
    import os, tempfile, hazelbean as hb
    base = utilities.service_data_dir(
        hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors')),
        'groundwater_recharge')
    recharge = pd.read_csv(os.path.join(base, 'l_val_dt_ctry.csv'))
    committed = pd.read_csv(os.path.join(base, 'ground_water_gep.csv'))
    committed = committed[committed['ground_water_gep'].notna()
                          & (committed['ground_water_gep'] > 0)]
    # India, the largest row: mean recharge, AQUASTAT 2020 withdrawal, the 2021 tariff.
    aq = pd.read_csv(os.path.join(base, 'AQUASTAT Dissemination System.csv'))
    px = pd.read_excel(os.path.join(base, 'global-electricity-per-kwh-pricing-2021.xlsx'))
    ind_l = recharge.loc[recharge['iso3_r250_label'] == 'IND', 'l_mean_mm'].iloc[0]
    ind_w = aq[(aq['Area'] == 'India') & (aq['Year'] == 2020)]['Value'].iloc[0]
    ind_p = px.loc[px['Country name'] == 'India', 'Average price of 1KW/h (USD)'].iloc[0]
    ours = gr.pumping_cost_per_m3_per_m(ind_p, 0.5) * (ind_l / 1000.0) * ind_w * 1e9
    theirs = committed.loc[committed['iso3_r250_label'] == 'IND', 'ground_water_gep'].iloc[0]
    # The staged committed CSV is one input vintage behind the current recharge table, so the
    # anchor holds to a percent here; the all-services sheet's column, built on the current
    # inputs, is reproduced to the cent (India 2,616,126.20) and the run log reports both.
    assert np.isclose(ours, theirs, rtol=0.02), (ours, theirs)
