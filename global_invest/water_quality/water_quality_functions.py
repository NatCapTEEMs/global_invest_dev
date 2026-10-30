"""Water-quality (purification) science: nutrient retention valued at a benefit-transfer treatment cost.

Nothing here opens a file. The task layer reads the authors' price observations, the World Bank
series and the retention estimates and hands the frames in, so every step can be pinned on a
hand-built input in the test suite.

The calculation follows the authors' current write-up and their `analysis.do`, step by step:

1. Quantity. Per country and nutrient (nitrogen, phosphorus), the retention allocated to
   domestic use: EstimatedRetention(kg) = TotalRetention(kg) x the AQUASTAT domestic share of
   withdrawals. The share is an allocation proxy, not a measured delivery to domestic supply.
   The retention totals are an InVEST NDR estimation taken as given.
2. Observed prices. Each reported treatment cost becomes US dollars per kg at the 2019 price
   level: units to kg, the reported currency to local currency in the observation year, the GDP
   deflator to 2019, then the 2019 exchange rate to USD. The Baltic observations are reported in
   euro before those countries adopted it: each goes to the old local currency in its year, is
   deflated there, and returns to euro at the old-currency-per-euro rate of the year before
   adoption.
3. Benefit transfer. One pooled regression over both nutrients,
   ln(price) = b0 + b_gdppc ln(GDP per capita, 2019) + b_p phosphorus,
   and each country's predicted price exp(...), the conditional median.
4. Valuation. A negative retention is missing and a true zero is kept, and
   GEP = price_n x N + price_p x P, missing when either term is.
"""
import numpy as np
import pandas as pd

WATER_QUALITY_NUTRIENTS = ('n', 'p')   # nitrogen, phosphorus

# The factor that turns a price per reported unit into a price per kg.
PRICE_PER_KG_FACTOR = {'kg': 1.0, 'kilogram': 1.0, 'kilograms': 1.0,
                       'ton': 1e-3, 'tonne': 1e-3, 'metric ton': 1e-3, 'mt': 1e-3,
                       'lb': 2.20462262, 'lbs': 2.20462262, 'pound': 2.20462262, 'pounds': 2.20462262}

# Countries whose observations are reported in euro although the euro was not their currency:
# those prices go to local currency through the euro's own dollar rate. Germany, Finland and
# Italy report in their own currency and need no conversion.
EURO_REPORTED_OUTSIDE_THE_EURO = ('POL', 'DNK', 'SWE', 'RUS')

# The first euro year of the three Baltic states. A price reported in euro before it is converted
# to the old local currency for its observation year and, after deflating, back to euro at the
# old-currency-per-euro rate of the year before adoption: World Bank LCU per USD over the euro's
# own USD rate.
BALTIC_EURO_ADOPTION_YEAR = {'EST': 2011, 'LVA': 2014, 'LTU': 2015}


def _country_year_lookup(table, labels, years):
    """The value of a country-by-year table for each (label, year) pair, NaN where either is absent."""
    out = []
    for label, year in zip(labels, years):
        if pd.notna(year) and label in table.index and int(year) in table.columns:
            out.append(table.at[label, int(year)])
        else:
            out.append(np.nan)
    return pd.Series(out, index=labels.index, dtype='float64')


def observed_prices_2019_usd(raw_df, deflator_df, exchange_rate_df, base_year=2019):
    """Each reported treatment cost in US dollars per kg at the base year's price level.

    Args:
        raw_df (pd.DataFrame): the authors' observations, one row per country and year, with
            iso3_r250_label, year, price_n_reported, price_p_reported, currency and per_unit.
        deflator_df (pd.DataFrame): the World Bank GDP deflator, indexed by country code, one
            column per year (int).
        exchange_rate_df (pd.DataFrame): World Bank local currency per USD, same layout, with the
            euro area as the `EMU` row.
        base_year (int): the price level and exchange-rate year.

    Returns:
        pd.DataFrame: raw_df with price_n_usd_kg_2019 and price_p_usd_kg_2019 added, missing where
            the observation, its unit or a needed rate is missing.
    """
    df = raw_df.copy()
    label, year = df['iso3_r250_label'], df['year']
    fx_year = _country_year_lookup(exchange_rate_df, label, year)
    deflator_year = _country_year_lookup(deflator_df, label, year)
    euro_rate = year.map(lambda y: exchange_rate_df.at['EMU', int(y)]
                         if pd.notna(y) and int(y) in exchange_rate_df.columns else np.nan)
    fx_base = label.map(exchange_rate_df[base_year])
    deflator_base = label.map(deflator_df[base_year])

    usd_to_local = (df['currency'] == 'usd') & (label != 'USA')
    adoption = label.map(BALTIC_EURO_ADOPTION_YEAR)
    before_euro = adoption.notna() & (year < adoption)
    euro_to_local = (df['currency'] == 'euro') & (label.isin(EURO_REPORTED_OUTSIDE_THE_EURO) | before_euro)
    old_currency_per_euro = pd.Series(1.0, index=df.index)
    for country, first_euro_year in BALTIC_EURO_ADOPTION_YEAR.items():
        rows = (label == country) & (year < first_euro_year)
        if rows.any():
            old_currency_per_euro[rows] = (exchange_rate_df.at[country, first_euro_year - 1]
                                           / exchange_rate_df.at['EMU', first_euro_year - 1])

    for nutrient in WATER_QUALITY_NUTRIENTS:
        per_kg = df[f'price_{nutrient}_reported'] * df['per_unit'].map(PRICE_PER_KG_FACTOR)
        local = per_kg.where(~usd_to_local, per_kg * fx_year)
        local = local.where(~euro_to_local, per_kg * fx_year / euro_rate)
        # The base year's price level, still in the observation year's local currency; then the
        # Baltic old currencies to euro, so the unit matches the base year's exchange rate.
        local_base = local * (deflator_base / deflator_year) / old_currency_per_euro
        df[f'price_{nutrient}_usd_kg_{base_year}'] = local_base / fx_base
    return df


def pooled_price_model(prices_df, base_year=2019):
    """The pooled benefit-transfer regression over both nutrients.

    ln(price) = b0 + b_gdppc ln(gdppc) + b_p phosphorus, by ordinary least squares over every
    observation with a positive price and a positive GDP per capita.

    Args:
        prices_df (pd.DataFrame): observed_prices_2019_usd's output with gdppc_usd_<base_year>.
        base_year (int): the year of the prices and of GDP per capita.

    Returns:
        tuple: ((b0, b_gdppc, b_p), number of observations).

    Raises:
        ValueError: if a usable price or GDP per capita is not finite, or the observations
            cannot identify all three coefficients (none at all, one nutrient only, or one GDP
            level only). Least squares returns numbers in those cases -- zeros for an empty
            sample, which predict a price of $1 per kg everywhere -- so the fit is refused
            instead.
    """
    gdppc = prices_df[f'gdppc_usd_{base_year}']
    long = pd.concat([pd.DataFrame({'price': prices_df[f'price_{nutrient}_usd_kg_{base_year}'],
                                    'phosphorus': float(nutrient == 'p'), 'gdppc': gdppc})
                      for nutrient in WATER_QUALITY_NUTRIENTS])
    long = long[(long['price'] > 0) & (long['gdppc'] > 0)]
    if not (np.isfinite(long['price']).all() and np.isfinite(long['gdppc']).all()):
        raise ValueError('the price observations carry a non-finite price or GDP per capita.')
    design = np.column_stack([np.ones(len(long)), np.log(long['gdppc']), long['phosphorus']])
    if len(long) == 0 or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError('the %d usable price observations cannot identify the intercept, the '
                         'income slope and the phosphorus term; the fit needs both nutrients '
                         'and more than one GDP level.' % len(long))
    coefficients, *_ = np.linalg.lstsq(design, np.log(long['price'].to_numpy()), rcond=None)
    return tuple(float(c) for c in coefficients), len(long)


def predicted_prices(gdppc, coefficients):
    """Each country's predicted price per kg, the conditional median exp(fitted ln price).

    Args:
        gdppc (pd.Series): GDP per capita in the base year, indexed by country label.
        coefficients (tuple): (b0, b_gdppc, b_p) from pooled_price_model.

    Returns:
        pd.DataFrame: price_n_usd_per_kg and price_p_usd_per_kg, indexed like gdppc, missing where
            GDP per capita is missing or not positive.
    """
    b0, b_gdppc, b_p = coefficients
    ln_gdppc = np.log(gdppc.where(gdppc > 0))
    return pd.DataFrame({'price_n_usd_per_kg': np.exp(b0 + b_gdppc * ln_gdppc),
                         'price_p_usd_per_kg': np.exp(b0 + b_gdppc * ln_gdppc + b_p)},
                        index=gdppc.index)


def water_purification_gep(retention_df, prices_df):
    """The per-country valuation: masked retention times the predicted prices.

    Args:
        retention_df (pd.DataFrame): iso3_r250_label, <nutrient>_TotalRetention(kg),
            <nutrient>_EstimatedRetention(kg) and DomesticWaterUseFraction.
        prices_df (pd.DataFrame): predicted_prices' output, indexed by iso3_r250_label.

    Returns:
        pd.DataFrame: one row per retention row with the prices, the retention used
            (nitrogen_retention_kg, phosphorus_retention_kg; negative estimates missing), each
            nutrient's value and water_quality_gep, their sum.

    Raises:
        ValueError: if an estimated retention differs from its total times the domestic
            fraction, which pins the quantity to the components it is built from.
    """
    df = retention_df[['iso3_r250_label']].copy()
    df = df.join(prices_df, on='iso3_r250_label')
    for nutrient, name in (('n', 'nitrogen'), ('p', 'phosphorus')):
        estimated = retention_df[f'{nutrient}_EstimatedRetention(kg)']
        rebuilt = retention_df[f'{nutrient}_TotalRetention(kg)'] * retention_df['DomesticWaterUseFraction']
        both = estimated.notna() & rebuilt.notna()
        if both.any() and not np.allclose(estimated[both], rebuilt[both], rtol=1e-6, atol=1.0):
            raise ValueError(f'{nutrient}_EstimatedRetention(kg) no longer equals the total times the '
                             f'domestic fraction -- re-identify the retention table before valuing it.')
        df[f'{name}_retention_kg'] = estimated.where(estimated >= 0)
        df[f'{name}_gep_usd'] = df[f'price_{nutrient}_usd_per_kg'] * df[f'{name}_retention_kg']
    df['water_quality_gep'] = df['nitrogen_gep_usd'] + df['phosphorus_gep_usd']
    return df
