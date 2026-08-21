"""Local climate regulation (urban cooling) science, from the repo's v04 valuation chain.

The pipeline: InVEST urban cooling gives each city-month a vegetation cooling effect; the
avoided cooling energy is priced at the national electricity price (kwh_diff x price =
total_savings_usd, exact in the committed per-country city-month files); country sums give the
December 2025 version of the accounting table. The CURRENT final applies v04's correction,
written explicitly in its code: divide out the old GLOBAL marginal-consumption slope and
multiply by each country's own consumption per cooling-degree-day (cooling energy per capita
over the IEA-CMCC 2019 humid-CDD total).

What is verified with staged data: the kwh x price identity (exact), the merged-sum identity
against the December 2025 version (exact on every country tested), the current final's parse
and totals. What is NOT yet reproducible: the v04 correction's own inputs -- the IEA-CMCC CDD
file and the v2_nkd energy workbook live only on the author's machine (the staged VERSION4
energy CSV is a different vintage and does not validate the inversion) -- the two named asks.
"""
import numpy as np
import pandas as pd

# v04's constant: the OLD global marginal-consumption slope embedded in the v02 values,
# divided out before the country-specific consumption-per-CDD is applied (its code, verbatim).
URBAN_COOLING_OLD_GLOBAL_MC = 1.71737465916468
URBAN_COOLING_CDD_BASE_C = 18.0     # the humid-CDD base the chain uses (CDDhum18)
URBAN_COOLING_SMOOTH_WINDOW_C = 1.0


def smoothstep01(t):
    """v04's cubic smoothstep on [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def temp_degc_to_cdd_smooth(temp_degc, window=URBAN_COOLING_SMOOTH_WINDOW_C):
    """v04's smoothed exceedance: 0 below base-window, linear-in-the-limit above base+window,
    cubic-smooth across the window -- the annualized cooling-degree measure for one
    temperature."""
    t = (np.asarray(temp_degc, dtype=float) - (URBAN_COOLING_CDD_BASE_C - window)) / (2 * window)
    return 2 * window * smoothstep01(t) + np.maximum(
        np.asarray(temp_degc, dtype=float) - (URBAN_COOLING_CDD_BASE_C + window), 0.0)


def parse_dollar_column(series):
    """The committed accounting CSVs store values as dollar-formatted strings."""
    return pd.to_numeric(series.astype(str).str.replace(r'[\$,]', '', regex=True),
                         errors='coerce')


def city_savings_identity(merged_df):
    """The committed city-month files' own identity: total_savings_usd = kwh_diff x price.
    Raises if it breaks."""
    recomputed = merged_df['kwh_diff'] * merged_df['price_usd_per_kwh']
    if not np.allclose(recomputed, merged_df['total_savings_usd'], rtol=1e-6):
        raise ValueError('the committed city-month savings no longer equal kwh x price -- '
                         're-identify before trusting the chain.')
    return float(merged_df['total_savings_usd'].sum())


def apply_country_marginal_consumption(values, consumption_per_cdd):
    """v04's correction, verbatim: the old global slope divided out, the country's own
    consumption per CDD applied."""
    return values / URBAN_COOLING_OLD_GLOBAL_MC * consumption_per_cdd


def local_climate_gep_by_country(final_df, countries_df):
    """The committed accounting values joined onto the r250 country list by label."""
    df = final_df.copy()
    df.columns = [c.strip() for c in df.columns]
    df['local_climate_regulation_gep'] = parse_dollar_column(df['local_climate_regulation'])
    out = countries_df.merge(df[['iso3_r250_label', 'local_climate_regulation_gep']],
                             on='iso3_r250_label', how='left')
    return out
