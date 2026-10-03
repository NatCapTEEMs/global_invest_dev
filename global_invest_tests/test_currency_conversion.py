"""The account's two dollar conversions, pinned on hand-built frames.

Every temporal deflation runs through one CPI series and every PPP conversion through one
price-level series, so no two services can convert differently.
"""
import pandas as pd
import numpy as np

from global_invest import utilities


def test_the_deflation_factor_is_the_annual_mean_cpi_ratio():
    cpi = pd.DataFrame({
        'observation_date': ['2019-01-01', '2019-07-01', '2021-01-01', '2021-07-01'],
        'CPIAUCSL': [100.0, 102.0, 110.0, 112.0]})
    factor = utilities.usd_deflation_factor(cpi, from_year=2021, to_year=2019)
    assert np.isclose(factor, 101.0 / 111.0)


def test_international_dollars_convert_by_the_country_price_level():
    df = pd.DataFrame({'iso3_r250_label': ['USA', 'BGR', 'XXX'], 'v': [100.0, 100.0, 100.0]})
    pli = pd.DataFrame({'iso3': ['USA', 'BGR'], 'year': [2018, 2018],
                        'price_level_ratio': [1.0, 0.41]})
    out = utilities.international_to_usd(df, pli, 2018, 'v')
    assert out.loc[0, 'v'] == 100.0
    assert np.isclose(out.loc[1, 'v'], 41.0)
    assert pd.isna(out.loc[2, 'v'])
