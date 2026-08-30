# -*- coding: utf-8 -*-
import logging
import pandas as pd


CORAL_REEF_VALUE_YEAR = 2011
COASTAL_PROTECTION_BASE_YEAR = 2019


def _read_excel_sheet(path: str, sheet_name: str, source_label: str):
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, engine='openpyxl')
        logging.info(
            f"Loaded {source_label} from {path} ({df.shape[0]} rows)."
        )
        return df
    except Exception as e:
        logging.error(f"Failed to read {source_label} file '{path}': {e}")
        raise


def _read_world_bank_indicator(path: str, value_name: str):
    """Read a World Bank wide indicator file and return country-year values.

    World Bank CSV downloads carry four metadata rows before their header.  Values are
    coerced with ``errors='coerce'`` so blank cells become NaN while true zeroes remain zero.
    """
    try:
        if path.lower().endswith(('.xlsx', '.xls')):
            df_raw = pd.read_excel(path, engine='openpyxl')
        else:
            with open(path, encoding='utf-8-sig') as source_file:
                first_line = source_file.readline().lstrip()
            skiprows = 4 if first_line.startswith(('Data Source', '"Data Source"')) else 0
            df_raw = pd.read_csv(path, skiprows=skiprows, encoding='utf-8-sig')
        logging.info(f"Loaded World Bank indicator from {path} ({df_raw.shape[0]} rows).")
    except Exception as e:
        logging.error(f"Failed to read World Bank indicator file '{path}': {e}")
        raise

    year_columns = [column for column in df_raw.columns if str(column).isdigit()]
    if 'Country Code' not in df_raw.columns or not year_columns:
        raise ValueError(
            f"World Bank indicator file '{path}' must contain Country Code and year columns."
        )

    df = df_raw.melt(
        id_vars=['Country Code'],
        value_vars=year_columns,
        var_name='year',
        value_name=value_name,
    )
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df[value_name] = pd.to_numeric(df[value_name], errors='coerce')
    return df


def read_world_bank_year_values(path: str, year, value_name: str):
    """Read one World Bank indicator for requested years, preserving missing values."""
    df = _read_world_bank_indicator(path, value_name)
    years = [year] if isinstance(year, int) else list(year)
    available_years = set(df['year'].dropna().astype(int))
    missing_years = sorted(set(years) - available_years)
    if missing_years:
        raise ValueError(
            f"World Bank indicator file '{path}' lacks requested years {missing_years}."
        )
    return df.loc[df['year'].isin(years), ['Country Code', 'year', value_name]].copy()


def read_world_bank_latest_values(path: str, target_year: int, value_name: str):
    """Read latest nonmissing World Bank value at or before ``target_year``."""
    df = _read_world_bank_indicator(path, value_name)
    df = df.loc[
        df['year'].le(target_year) & df[value_name].notna(),
        ['Country Code', 'year', value_name],
    ].sort_values(['Country Code', 'year'])
    return (
        df.drop_duplicates('Country Code', keep='last')
        .rename(columns={'year': f'{value_name}_source_year'})
        .reset_index(drop=True)
    )


def read_exchange_rate_fallbacks(path: str, years=None):
    """Read documented non-World-Bank exchange-rate fallback records."""
    df = pd.read_csv(path, encoding='utf-8-sig')
    required_columns = {
        'Country Code', 'year', 'exchange_rate_lcu_per_usd'
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Exchange-rate fallback file '{path}' lacks columns "
            f"{sorted(missing_columns)}."
        )
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df['exchange_rate_lcu_per_usd'] = pd.to_numeric(
        df['exchange_rate_lcu_per_usd'], errors='coerce'
    )
    df = df.dropna(subset=['Country Code', 'year', 'exchange_rate_lcu_per_usd'])
    if years is not None:
        df = df.loc[df['year'].isin(years)]
    return df


def apply_exchange_rate_fallbacks(df: pd.DataFrame, fallback_df: pd.DataFrame):
    """Append documented rates and keep World Bank values when available."""
    if fallback_df.empty:
        return df
    fallback_values = fallback_df[
        ['Country Code', 'year', 'exchange_rate_lcu_per_usd']
    ].rename(columns={'exchange_rate_lcu_per_usd': 'fx_lcu_per_usd'})
    combined = pd.concat([df, fallback_values], ignore_index=True)
    return (
        combined.sort_values(['Country Code', 'year'])
        .groupby(['Country Code', 'year'], as_index=False, dropna=False)[
            'fx_lcu_per_usd'
        ]
        .first()
    )


def _multiply_preserve_zeros(value, factor):
    """Multiply values while carrying exact source zeroes through missing factors."""
    value = pd.to_numeric(value, errors='coerce')
    factor = pd.to_numeric(factor, errors='coerce')
    factor = factor.where(factor > 0)
    result = value * factor
    return result.mask(value.eq(0), 0.0)


def _divide_preserve_zeros(value, divisor):
    """Divide values while carrying exact numerator zeroes through missing divisors."""
    value = pd.to_numeric(value, errors='coerce')
    divisor = pd.to_numeric(divisor, errors='coerce')
    divisor = divisor.where(divisor > 0)
    result = value / divisor
    return result.mask(value.eq(0), 0.0)


def read_mangrove_values(path: str):
    df_mangrove_value = _read_excel_sheet(
        path, 'Sheet1', 'mangrove coastal protection values'
    )
    df_mangrove_value.rename(columns={'countrycode': 'ee_r264_label'}, inplace=True)
    df_mangrove_value['annual_value_2019'] = pd.to_numeric(
        df_mangrove_value['annual_value_2019'], errors='coerce'
    )
    df_mangrove_value['mangrove_value_2019_usd'] = df_mangrove_value['annual_value_2019']
    df_mangrove_value['year'] = pd.to_numeric(
        df_mangrove_value['year'], errors='coerce'
    ).astype('Int64')

    logging.info(f"Reshaped to long format ({df_mangrove_value.shape[0]} rows).")
    return df_mangrove_value


def read_coral_reef_values(path: str):
    df_coral_reef_value = _read_excel_sheet(
        path, 'Sheet1', 'coral-reef coastal protection values'
    )
    df_coral_reef_value['coral_reef_value'] = pd.to_numeric(
        df_coral_reef_value['coral_reef_value'], errors='coerce'
    )
    df_coral_reef_value['year'] = pd.to_numeric(
        df_coral_reef_value['year'], errors='coerce'
    ).astype('Int64')

    logging.info(f"Finished cleaning up ({df_coral_reef_value.shape[0]} rows).")


    return df_coral_reef_value


def read_gdp_inflation_deflator(path: str):
    """
    Read World Bank GDP-deflator index data from CSV or Excel.
    https://data.worldbank.org/indicator/NY.GDP.DEFL.ZS

    """

    return _read_world_bank_indicator(path, 'value')


def get_gdp_deflator_factor(path, source_year: int, target_year: int):
    """Return GDP-deflator index ratio carrying source-year LCU into target-year LCU.

    ``NY.GDP.DEFL.ZS`` is an index, not an annual inflation-rate series.  The correct
    conversion is ``deflator[target_year] / deflator[source_year]``.
    """
    df = _read_world_bank_indicator(path, 'GDP_deflator')
    levels = (
        df.loc[df['year'].isin([source_year, target_year])]
        .pivot_table(index='Country Code', columns='year', values='GDP_deflator', aggfunc='first')
    )
    source = levels.get(source_year)
    target = levels.get(target_year)
    if source is None or target is None:
        raise ValueError(
            f"GDP-deflator file '{path}' lacks required years {source_year} and {target_year}."
        )

    factor = (target / source).where((source > 0) & (target > 0))
    return pd.DataFrame({
        'Country Code': factor.index,
        'deflator_source': source.reindex(factor.index).to_numpy(),
        'deflator_target': target.reindex(factor.index).to_numpy(),
        'deflator_factor': factor.to_numpy(),
    })


def get_inflation_deflator_multiplier(path, start_year, end_year):

    """
    Backward-compatible name for a GDP-deflator index ratio.

    ``start_year`` is the first inflation year; source-value year is
    ``start_year - 1``.
    """

    # Backward-compatible wrapper.  Existing callers pass the first inflation year;
    # for an index series that means source-year = start_year - 1.
    df = get_gdp_deflator_factor(path, start_year - 1, end_year)
    return df.rename(columns={'deflator_factor': 'deflator_multiplier'})


def merge_world_bank_factors(df: pd.DataFrame, *factor_dfs):
    """Left-join country-level World Bank factors onto source values."""
    for factor_df in factor_dfs:
        df = (
            df.merge(
                factor_df,
                how='left',
                left_on='ee_r264_label',
                right_on='Country Code',
                validate='many_to_one',
            )
            .drop(columns=['Country Code'])
        )
    return df


def convert_usd_to_2019_int_dollars(
    value_usd, exchange_rate, ppp_factor, deflator_factor=None
):
    """Convert source-year USD to 2019 LCU and 2019 PPP-adjusted international dollars.

    ``exchange_rate`` must match source-value year.  Supply ``deflator_factor`` only
    when source LCU must first be expressed in 2019 LCU.  Returns source-year LCU,
    2019 LCU, and 2019 international dollars, in that order.
    """
    source_lcu = _multiply_preserve_zeros(value_usd, exchange_rate)
    target_lcu = source_lcu
    if deflator_factor is not None:
        target_lcu = _multiply_preserve_zeros(source_lcu, deflator_factor)
    int_dollar = _divide_preserve_zeros(target_lcu, ppp_factor)
    return source_lcu, target_lcu, int_dollar


def group_sum_preserving_nan(df: pd.DataFrame, groupby_cols, value_cols):
    """Aggregate values without turning all-missing groups into zero."""
    return (
        df.groupby(groupby_cols, as_index=False, dropna=False)[value_cols]
        .agg(lambda values: values.sum(skipna=False))
    )

def group_countries(df: pd.DataFrame):
    """
    Aggregate total GEP across countries by year.

    Missing country values are excluded when at least one country is valid;
    an all-missing year remains NaN.
    """
    df_gep_by_year = (
        df.groupby('year', as_index=False, dropna=False)['Value']
        .sum(min_count=1)
    )
    df_gep_by_year.sort_values('year', inplace=True)
    logging.info(f"Grouped total by year ({df_gep_by_year.shape[0]} rows).")
    return df_gep_by_year
