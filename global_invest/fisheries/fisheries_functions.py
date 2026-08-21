"""Fisheries ES science: read the pre-computed marine-fisheries shock headers.

Marine-fisheries productivity is mapped by RCP (DBEM vs BOATS provenance is a paper question, see task
#16) and is never derived from the terrestrial SEALS maps, so unlike carbon and pollination it is READ
from a pre-computed HAR (cwon_shocks.har, headers FI26/FI45/FI85, one per RCP) rather than recomputed.

The headers carry a FULL ANNUAL series (Y2017..Y2050, 50 regions x 34 years). For the current file the
series is a 2017->2018 step that then holds flat, so it is already constant over the 2023-2050 solve
window; the read returns the whole annual series regardless, so the task reads each year directly and a
genuinely dynamic future source (DBEM/Fish-MIP, #45) needs no read change.
"""


def read_fisheries_headers(cwon_path, headers):
    """Per-region ANNUAL fisheries shock (%) per RCP header -> {header: {reg: {year_int: value}}}."""
    from gtappy.harpy.har_file import HarFileObj
    h = HarFileObj(filename=cwon_path)
    out = {}
    for hdr in headers:
        arr = h[hdr].array
        regs = [s.strip() for s in h[hdr].setElements[0]]
        years = [int(s.strip().lstrip('Yy')) for s in h[hdr].setElements[1]]
        out[hdr] = {reg: {yr: arr[i, j] for j, yr in enumerate(years)}
                    for i, reg in enumerate(regs)}
    return out


# =============================================================================
# GEP valuation (commercial capture fisheries, CWoN method). Ported from the
# source repo's 2026 script (gep_commcapturefisheries_cwonmethod_20260720.R):
# CWoN 2024 (FR_WLD_2024_195) fisheries economic rent, deflated to 2019 USD,
# per-country OLS trend over 2009-2018 predicting 2019, floored at zero.
# =============================================================================
FISHERIES_CPI_YEARS = (2009, 2019)          # CPI + rent window read from CWoN
FISHERIES_TREND_YEARS = (2009, 2018)        # the OLS window
FISHERIES_TREND_MIDPOINT = 2013.5           # mean of the OLS window years
FISHERIES_GEP_YEAR = 2019
# Country exclusions from the source script (CWoN data quality drops).
FISHERIES_RENT_EXCLUDED = ('CYP', 'MAF', 'PSE')
FISHERIES_RENT_EXCLUDED_COUNTRY_YEARS = (('CUW', 2009), ('CUW', 2010))


def clean_cwon_cpi(cpi_df):
    """CWoN cpi2019 table -> (wb_code, year, cpi2019) for the window, with a missing
    country-year CPI imputed by that YEAR's cross-country mean (the source's imputation)."""
    import pandas as pd
    df = cpi_df[(cpi_df['year'] >= FISHERIES_CPI_YEARS[0])
                & (cpi_df['year'] <= FISHERIES_CPI_YEARS[1])].copy()
    df['cpi2019'] = df['cpi2019'].fillna(df.groupby('year')['cpi2019'].transform('mean'))
    df = df.rename(columns={'countrycode': 'wb_code'})
    return df[['wb_code', 'year', 'cpi2019']]


def clean_cwon_econ_rent(rent_df):
    """CWoN economic-rent table filtered to the window with the source's exclusions."""
    df = rent_df.rename(columns={'Year': 'year'})
    df = df[(df['year'] >= FISHERIES_CPI_YEARS[0]) & (df['year'] <= FISHERIES_CPI_YEARS[1])]
    df = df[~df['wb_code'].isin(FISHERIES_RENT_EXCLUDED)]
    for wb_code, year in FISHERIES_RENT_EXCLUDED_COUNTRY_YEARS:
        df = df[~((df['wb_code'] == wb_code) & (df['year'] == year))]
    return df[['year', 'wb_code', 'FAOEconRent']].copy()


def deflate_rent_to_2019usd(rent_df, cpi_df):
    """Real 2019-USD economic rent: nominal rent / cpi2019 x 100."""
    df = rent_df.merge(cpi_df, on=['wb_code', 'year'], how='left')
    df['resrent_2019usd'] = df['FAOEconRent'] / df['cpi2019'] * 100
    return df


def fisheries_rent_trends(deflated_df):
    """Per-country 2019 rent estimate: mean real rent over the OLS window plus the OLS
    time-slope times the distance from the window midpoint to 2019, floored at zero.
    A country with fewer than two years of data gets no slope and no estimate (NaN),
    exactly as in the source."""
    import numpy as np
    import pandas as pd
    window = deflated_df[(deflated_df['year'] >= FISHERIES_TREND_YEARS[0])
                         & (deflated_df['year'] <= FISHERIES_TREND_YEARS[1])]
    rows = []
    for wb_code, group in window.groupby('wb_code'):
        valid = group[np.isfinite(group['resrent_2019usd'])]
        n_years = valid['year'].nunique()
        mean_rent = valid['resrent_2019usd'].mean() if len(valid) else np.nan
        beta = (np.polyfit(valid['year'].astype(float), valid['resrent_2019usd'], 1)[0]
                if n_years >= 2 else np.nan)
        rows.append({'wb_code': wb_code, 'n_years': n_years,
                     'mean_resrent_2009_2018': mean_rent, 'beta_hat': beta})
    trends = pd.DataFrame(rows)
    trends['resrent_2019_hat'] = (trends['mean_resrent_2009_2018']
                                  + trends['beta_hat'] * (FISHERIES_GEP_YEAR - FISHERIES_TREND_MIDPOINT))
    trends['positive_resrent_2019_hat'] = trends['resrent_2019_hat'].clip(lower=0)
    return trends


def commfish_gep_by_country(trends_df, countries_df):
    """Join the rent estimates onto the r250 country list by iso3 label (the CWoN wb_code IS
    the iso3_r250_label; the source joins the same way). Countries without an estimate stay
    NaN rather than zero -- no data is not zero rent."""
    trends = trends_df.rename(columns={'wb_code': 'iso3_r250_label'})
    df = countries_df.merge(trends[['iso3_r250_label', 'positive_resrent_2019_hat']],
                            on='iso3_r250_label', how='left')
    return df.rename(columns={'positive_resrent_2019_hat': 'commfish_provision'})


# =============================================================================
# Subsistence fisheries GEP (Lynch et al. 2024, USGS data release). Ported from
# the gep-subsistence-fisheries repo; the committed output CSV is the anchor.
# =============================================================================
def subsistence_fisheries_by_country(lynch_df, countries_df):
    """One TCUV (total consumptive-use value) per country from the Lynch et al. data
    release, name-joined onto the canonical r250 country rows by brk_name.

    Faithful-port notes: the source keeps the FIRST TCUV per admin, and 4 of the 81 admins
    carry more than one distinct TCUV, so keep-first is order-dependent -- the committed
    anchor pins the choice. The name join leaves countries absent from the release NaN:
    no data is not zero value."""
    data = lynch_df[['admin', 'TCUV']].drop_duplicates(subset=['admin'], keep='first')
    df = countries_df.merge(data, how='left', left_on='brk_name', right_on='admin')
    df = df.rename(columns={'TCUV': 'subsistence_fisheries_gep'})
    return df.drop(columns=['admin'])
