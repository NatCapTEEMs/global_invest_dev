# -*- coding: utf-8 -*-
"""Groundwater-recharge science: recharge valued as the pumping energy it saves.

Nothing here opens a file. The task layer reads the recharge, withdrawal and price tables,
hands the frames in and writes back what it gets, so every step can be pinned on a hand-built
input in the test suite.

The valuation: recharge keeps the water table higher, so every cubic metre withdrawn is lifted
a shorter distance. Lifting one cubic metre by one metre takes gamma / (eta * 3.6e6) kilowatt
hours (gamma the specific weight of water in newtons per cubic metre, eta the pump efficiency,
3.6e6 joules per kilowatt hour), so the service value is that energy, priced at the country's
electricity tariff, times the year's mean recharge depth, times the year's withdrawal volume.
The construction values ONE year's recharge as a one-year head increment, which is why the
totals are small against services that value a stock or a harvest.
"""
import pandas as pd

WATER_SPECIFIC_WEIGHT_N_PER_M3 = 9810.0
JOULES_PER_KWH = 3.6e6


def pumping_cost_per_m3_per_m(electricity_price_usd_per_kwh, pump_efficiency):
    """The cost of lifting one cubic metre of water by one metre, in USD.

    Args:
        electricity_price_usd_per_kwh: the country's average electricity tariff.
        pump_efficiency (float): the fraction of electrical energy that becomes lift.

    Returns:
        The cost, elementwise over whatever the price argument is.
    """
    return (WATER_SPECIFIC_WEIGHT_N_PER_M3 * electricity_price_usd_per_kwh) / (
        pump_efficiency * JOULES_PER_KWH)


def groundwater_gep(recharge_df, withdrawal_df, price_df, pump_efficiency):
    """The avoided pumping cost per country: price of lift times recharge depth times withdrawal.

    Args:
        recharge_df (pd.DataFrame): iso3_r250_label and l_mean_mm, the country's mean local
            recharge depth from the seasonal water yield model.
        withdrawal_df (pd.DataFrame): iso3_r250_label and gw_wdw_bil_m3, fresh groundwater
            withdrawal in billions of cubic metres.
        price_df (pd.DataFrame): iso3_r250_label and electricity_usd_per_kwh.
        pump_efficiency (float): the fraction of electrical energy that becomes lift.

    Returns:
        pd.DataFrame: one row per country appearing in recharge_df, carrying the inputs and
        `groundwater_recharge_gep` in USD. A country missing any input carries NA there, never
        a silent zero: no recharge estimate, no withdrawal figure and no tariff are three
        different absences and the note column names which.
    """
    out = recharge_df[['iso3_r250_label', 'l_mean_mm']].copy()
    out['l_mean_m'] = out['l_mean_mm'] / 1000.0
    out = out.merge(withdrawal_df[['iso3_r250_label', 'gw_wdw_bil_m3']],
                    on='iso3_r250_label', how='left')
    out = out.merge(price_df[['iso3_r250_label', 'electricity_usd_per_kwh']],
                    on='iso3_r250_label', how='left')
    out['groundwater_recharge_gep'] = (
        pumping_cost_per_m3_per_m(out['electricity_usd_per_kwh'], pump_efficiency)
        * out['l_mean_m'] * out['gw_wdw_bil_m3'] * 1e9)
    absences = {'l_mean_mm': 'no recharge estimate', 'gw_wdw_bil_m3': 'no withdrawal figure',
                'electricity_usd_per_kwh': 'no electricity tariff'}
    out['note'] = out.apply(
        lambda row: '; '.join(absences[c] for c in absences if pd.isna(row[c])), axis=1)
    return out
