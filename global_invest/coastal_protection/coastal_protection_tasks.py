import os
import subprocess
import sys

import hazelbean as hb
import pandas as pd

from global_invest.coastal_protection import coastal_protection_functions


SERVICE_LABEL = 'coastal_protection'
COUNTRY_OUTPUT_NAME = 'gep_by_country_base_year.csv'
YEAR_OUTPUT_NAME = 'gep_by_year.csv'
CORAL_FALLBACK_OUTPUT_NAME = 'missing_coral_reef_values.csv'
COUNTRY_VALUE_COLUMN = 'Value (2019 int$)'

COUNTRY_INFO_COLUMNS = [
    'iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
    'ee_r264_id', 'ee_r264_label', 'ee_r264_name',
    'country', 'continent', 'region_un', 'region_wb', 'income_grp',
    'subregion', 'area_code_M49', 'area_code',
]
MANGROVE_OUTPUT_COLUMNS = [
    'mangrove_value_2019_usd',
    'mangrove_value_2019_lcu',
    'mangrove_value_2019_int_dollar',
    'coastal_protection_gep_mangrove',
]
CORAL_OUTPUT_COLUMNS = [
    'coral_reef_value_2011_usd',
    'coral_reef_value_2011_lcu',
    'coral_reef_value_2019_lcu',
    'coral_reef_value_2019_int_dollar',
    'coastal_protection_gep_coral_reef',
]
NUMERIC_COUNTRY_OUTPUT_COLUMNS = [
    'ee_r264_id', 'iso3_r250_id', 'year', 'area_code_M49', 'area_code',
] + MANGROVE_OUTPUT_COLUMNS + CORAL_OUTPUT_COLUMNS + [COUNTRY_VALUE_COLUMN]


def coastal_protection(p):
    """Set source paths and valuation years for coastal protection."""
    data_dir = os.path.join(p.base_data_dir, SERVICE_LABEL)
    p.base_year = getattr(
        p, 'base_year', coastal_protection_functions.COASTAL_PROTECTION_BASE_YEAR
    )
    p.coral_reef_value_year = coastal_protection_functions.CORAL_REEF_VALUE_YEAR

    p.cwon_input_ref_path = os.path.join(data_dir, 'data_mangroves_2019.xlsx')
    p.coral_reef_ref_path = os.path.join(
        data_dir, 'coral_reefs_annual_expected_benefit_nfamara.xlsx'
    )
    p.wb_exchange_rate_ref_path = os.path.join(
        data_dir, 'API_PA.NUS.FCRF_DS2_en_csv_v2_32.csv'
    )
    p.wb_ppp_ref_path = os.path.join(
        data_dir, 'API_PA.NUS.PPP_DS2_en_csv_v2_38116.csv'
    )
    p.wb_deflator_ref_path = os.path.join(
        data_dir, 'API_NY.GDP.DEFL.ZS_DS2_en_csv_v2_24026.csv'
    )
    p.treasury_exchange_rate_fallback_path = os.path.join(
        data_dir, 'treasury_exchange_rate_fallbacks.csv'
    )
    # Compatibility alias for callers using the former attribute name.
    p.df_gdp_inflation_deflator_path = p.wb_deflator_ref_path


def _merge_source_values(countries, source_values, join_column, year):
    """Join source values to country correspondence and retain one source year."""
    df = hb.df_merge(countries, source_values, how='inner', on=join_column)
    return df.loc[df['year'].eq(year)].copy()


def _group_service_values(df, value_columns):
    """Aggregate service values to R250 country-year without hiding missing values."""
    df = df.dropna(subset=['iso3_r250_label'])
    return coastal_protection_functions.group_sum_preserving_nan(
        df, ['iso3_r250_label', 'year'], value_columns
    )


def _coerce_numeric_output_columns(df, columns):
    """Write numeric CSV fields as numeric tokens; preserve missing values as NaN."""
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def _country_outputs_have_current_schema(country_output, geopackage_output):
    """Check whether cached country outputs use current columns and total rule."""
    fallback_output = os.path.join(
        os.path.dirname(os.path.dirname(country_output)),
        'gep_result', CORAL_FALLBACK_OUTPUT_NAME,
    )
    if not os.path.exists(fallback_output):
        return False
    try:
        country_df = hb.df_read(country_output)
        country_gdf = hb.read_vector(geopackage_output)
        fallback_df = hb.df_read(fallback_output)
    except Exception:
        return False
    if list(fallback_df.columns) != [
        'iso3_r250_name', 'value_2011_usd',
        'value_2019_int_dollar', 'missing_components',
    ]:
        return False

    component_columns = [
        'coastal_protection_gep_mangrove',
        'coastal_protection_gep_coral_reef',
    ]
    required_columns = (
        component_columns
        + [COUNTRY_VALUE_COLUMN]
        + MANGROVE_OUTPUT_COLUMNS
        + CORAL_OUTPUT_COLUMNS
    )
    if not all(column in country_df.columns for column in required_columns):
        return False
    if not all(column in country_gdf.columns for column in required_columns):
        return False
    expected_columns = [
        column for column in (
            COUNTRY_INFO_COLUMNS
            + ['year']
            + MANGROVE_OUTPUT_COLUMNS
            + CORAL_OUTPUT_COLUMNS
            + [COUNTRY_VALUE_COLUMN]
        )
        if column in country_df.columns
    ]
    if list(country_df.columns) != expected_columns:
        return False

    components = country_df[component_columns]
    expected = components.fillna(0.0).sum(axis=1).where(
        components.notna().any(axis=1)
    )
    actual = country_df[COUNTRY_VALUE_COLUMN]
    difference = (actual - expected).abs()
    matches = (
        (actual.isna() & expected.isna())
        | difference.le((expected.abs() + 1.0) * 1e-12)
    )
    return bool(matches.all())


def gep_calculation(p):
    """Convert mangrove and coral-reef values to 2019 PPP-adjusted int$."""
    service_results = p.results.setdefault(SERVICE_LABEL, {})
    # Match crop_provision: calculation outputs live in current task directory.
    country_output = os.path.join(p.cur_dir, COUNTRY_OUTPUT_NAME)
    year_output = os.path.join(p.cur_dir, YEAR_OUTPUT_NAME)
    geopackage_output = os.path.splitext(country_output)[0] + '.gpkg'
    service_results.update({
        'gep_by_country_base_year': country_output,
        'gep_by_year': year_output,
    })
    p.coastal_protection_geopackage_path = geopackage_output

    required_outputs = list(service_results.values()) + [geopackage_output]
    if (
        hb.path_all_exist(required_outputs)
        and _country_outputs_have_current_schema(country_output, geopackage_output)
    ):
        hb.log('All coastal-protection results exist. Skipping GEP calculation.')
        return

    hb.log('Starting GEP calculation for coastal protection.')
    p.gdf_countries = hb.read_vector(p.gdf_countries_vector_path)
    base_year = p.base_year
    coral_value_year = p.coral_reef_value_year

    # Read source values once. Exchange-rate data serves both source years.
    df_mangrove_value = coastal_protection_functions.read_mangrove_values(
        p.cwon_input_ref_path
    )
    df_coral_reef_value = coastal_protection_functions.read_coral_reef_values(
        p.coral_reef_ref_path
    )
    df_fx = coastal_protection_functions.read_world_bank_year_values(
        p.wb_exchange_rate_ref_path,
        [coral_value_year, base_year],
        'fx_lcu_per_usd',
    )
    treasury_fx_fallbacks = coastal_protection_functions.read_exchange_rate_fallbacks(
        p.treasury_exchange_rate_fallback_path,
        [coral_value_year, base_year],
    )
    df_fx = coastal_protection_functions.apply_exchange_rate_fallbacks(
        df_fx, treasury_fx_fallbacks
    )
    df_fx = (
        df_fx.pivot_table(
            index='Country Code',
            columns='year',
            values='fx_lcu_per_usd',
            aggfunc='first',
        )
        .rename(columns={
            coral_value_year: 'fx_source_year_lcu_per_usd',
            base_year: 'fx_base_year_lcu_per_usd',
        })
        .rename_axis(columns=None)
        .reset_index()
    )
    # Mangrove values are already in 2019 USD; retain existing latest-prior
    # PPP behavior for this separate 2019-USD source.
    df_mangrove_ppp = coastal_protection_functions.read_world_bank_latest_values(
        p.wb_ppp_ref_path,
        base_year,
        'ppp_base_year_lcu_per_int_dollar',
    )
    # Coral values are in 2011 USD. Use only the requested 2019 PPP observation;
    # missing PPP is handled below with deflated 2011 USD instead of prior PPP.
    df_coral_ppp = coastal_protection_functions.read_world_bank_year_values(
        p.wb_ppp_ref_path,
        base_year,
        'ppp_base_year_lcu_per_int_dollar',
    ).drop(columns='year')
    df_deflator = coastal_protection_functions.get_gdp_deflator_factor(
        p.wb_deflator_ref_path, coral_value_year, base_year
    )
    df_deflator['deflator_fallback_used'] = df_deflator[
        'deflator_factor'
    ].isna()
    world_deflator_factor = df_deflator['deflator_factor'].mean()
    if pd.isna(world_deflator_factor):
        raise ValueError('No usable World Bank GDP-deflator factors found.')
    df_deflator['deflator_factor'] = df_deflator[
        'deflator_factor'
    ].fillna(world_deflator_factor)

    # Mangrove: 2019 USD -> 2019 LCU -> 2019 int$.
    df_mangrove = _merge_source_values(
        p.gdf_countries, df_mangrove_value, 'ee_r264_label', base_year
    )
    df_mangrove = coastal_protection_functions.merge_world_bank_factors(
        df_mangrove, df_fx, df_mangrove_ppp
    )
    _, mangrove_lcu, mangrove_int_dollar = (
        coastal_protection_functions.convert_usd_to_2019_int_dollars(
            df_mangrove['mangrove_value_2019_usd'],
            df_mangrove['fx_base_year_lcu_per_usd'],
            df_mangrove['ppp_base_year_lcu_per_int_dollar'],
        )
    )
    df_mangrove['mangrove_value_2019_lcu'] = mangrove_lcu
    df_mangrove['mangrove_value_2019_int_dollar'] = mangrove_int_dollar
    df_mangrove = _group_service_values(
        df_mangrove,
        [
            'mangrove_value_2019_usd',
            'mangrove_value_2019_lcu',
            'mangrove_value_2019_int_dollar',
        ],
    )
    df_mangrove['coastal_protection_gep_mangrove'] = df_mangrove[
        'mangrove_value_2019_int_dollar'
    ]

    # Coral: 2011 USD -> 2011 LCU -> 2019 LCU -> 2019 int$.
    df_coral = _merge_source_values(
        p.gdf_countries, df_coral_reef_value, 'ee_r264_name', coral_value_year
    )
    # Correspondence contains alias rows for China, India, and Pakistan.
    # Source workbook has one value per country name; do not count aliases twice.
    df_coral['_primary_correspondence'] = df_coral[
        'ee_r264_label'
    ].eq(df_coral['iso3_r250_label'])
    df_coral = (
        df_coral.sort_values('_primary_correspondence', ascending=False)
        .drop_duplicates(subset=['iso3_r250_label', 'year', 'coral_reef_value'])
        .drop(columns='_primary_correspondence')
    )
    df_coral = coastal_protection_functions.merge_world_bank_factors(
        df_coral, df_fx, df_deflator, df_coral_ppp
    )
    treasury_fallback_codes = set(
        treasury_fx_fallbacks['Country Code'].astype(str)
    )
    df_coral['treasury_fx_fallback_used'] = df_coral[
        'ee_r264_label'
    ].isin(treasury_fallback_codes)
    missing_deflator = (
        df_coral['deflator_factor'].isna()
        | df_coral['deflator_fallback_used'].fillna(False)
    )
    df_coral['deflator_fallback_used'] = missing_deflator
    df_coral['deflator_factor'] = df_coral['deflator_factor'].fillna(
        world_deflator_factor
    )
    df_coral['coral_reef_value_2011_usd'] = df_coral['coral_reef_value']
    coral_source_lcu, coral_target_lcu, coral_int_dollar = (
        coastal_protection_functions.convert_usd_to_2019_int_dollars(
            df_coral['coral_reef_value_2011_usd'],
            df_coral['fx_source_year_lcu_per_usd'],
            df_coral['ppp_base_year_lcu_per_int_dollar'],
            df_coral['deflator_factor'],
        )
    )
    # If 2019 PPP is missing, report the original 2011 USD converted to 2019 USD
    # using the country deflator, or the world-average deflator when unavailable.
    direct_usd_deflated = (
        df_coral['coral_reef_value_2011_usd']
        * df_coral['deflator_factor']
    )
    missing_ppp = df_coral[
        'ppp_base_year_lcu_per_int_dollar'
    ].isna()
    df_coral['ppp_direct_usd_used'] = missing_ppp
    coral_int_dollar = coral_int_dollar.fillna(direct_usd_deflated)
    df_coral['coral_reef_value_2019_int_dollar'] = coral_int_dollar

    def _missing_components(row):
        reasons = []
        if row['treasury_fx_fallback_used']:
            reasons.append('PA.NUS.FCRF (2011)')
        if row['deflator_fallback_used']:
            reasons.append('NY.GDP.DEFL.ZS (2011-2019)')
        if row['ppp_direct_usd_used']:
            reasons.append('PA.NUS.PPP (2019)')
        return '; '.join(reasons)

    df_coral['missing_components'] = df_coral.apply(
        _missing_components, axis=1
    )
    fallback_columns = [
        'iso3_r250_name',
        'coral_reef_value_2011_usd', 'coral_reef_value_2019_int_dollar',
        'missing_components',
    ]
    coral_fallback_table = df_coral.loc[
        df_coral['missing_components'].ne(''), fallback_columns
    ].rename(columns={
        'coral_reef_value_2011_usd': 'value_2011_usd',
        'coral_reef_value_2019_int_dollar': 'value_2019_int_dollar',
    })
    coral_fallback_table = coral_fallback_table.sort_values('iso3_r250_name')
    coral_fallback_table = _coerce_numeric_output_columns(
        coral_fallback_table,
        ['value_2011_usd', 'value_2019_int_dollar'],
    )
    fallback_output_path = os.path.join(
        p.intermediate_dir, SERVICE_LABEL, 'gep_result',
        CORAL_FALLBACK_OUTPUT_NAME,
    )
    hb.create_directories(fallback_output_path)
    hb.df_write(coral_fallback_table, fallback_output_path)
    hb.log(
        f'Coral fallback factors used: Treasury FX rows={len(treasury_fx_fallbacks)}, '
        f'world-average deflator rows={missing_deflator.sum()}, '
        f'direct-USD-deflated rows={missing_ppp.sum()}.'
    )
    df_coral['coral_reef_value_2011_lcu'] = coral_source_lcu
    df_coral['coral_reef_value_2019_lcu'] = coral_target_lcu
    df_coral['coral_reef_value_2019_int_dollar'] = coral_int_dollar
    df_coral['year'] = base_year
    df_coral = _group_service_values(
        df_coral,
        [
            'coral_reef_value_2011_usd',
            'coral_reef_value_2011_lcu',
            'coral_reef_value_2019_lcu',
            'coral_reef_value_2019_int_dollar',
        ],
    )
    df_coral['coastal_protection_gep_coral_reef'] = df_coral[
        'coral_reef_value_2019_int_dollar'
    ]

    # Missing both components means missing total. A missing single component counts as zero.
    df_gep_by_country_year = df_mangrove.merge(
        df_coral,
        how='outer',
        on=['iso3_r250_label', 'year'],
        validate='one_to_one',
    )
    component_values = df_gep_by_country_year[
        ['coastal_protection_gep_mangrove', 'coastal_protection_gep_coral_reef']
    ]
    df_gep_by_country_year['coastal_protection_gep'] = (
        component_values.fillna(0.0).sum(axis=1)
        .where(component_values.notna().any(axis=1))
    )
    df_gep_by_country_year['Value'] = df_gep_by_country_year[
        'coastal_protection_gep'
    ]

    metadata_columns = [
        'ee_r264_id', 'iso3_r250_id', 'ee_r264_label', 'iso3_r250_label',
        'ee_r264_name', 'iso3_r250_name', 'continent', 'region_un',
        'region_wb', 'income_grp', 'subregion', 'area_code_M49',
        'area_code', 'country',
    ]
    country_metadata = p.gdf_countries.loc[
        p.gdf_countries['ee_r264_label'].eq(p.gdf_countries['iso3_r250_label']),
        [column for column in metadata_columns if column in p.gdf_countries.columns],
    ].drop_duplicates('iso3_r250_label')
    df_gep_by_country_year = df_gep_by_country_year.merge(
        country_metadata,
        how='left',
        on='iso3_r250_label',
        validate='many_to_one',
    )

    df_gep_by_country_base_year = df_gep_by_country_year.loc[
        df_gep_by_country_year['year'].eq(base_year)
    ].copy()
    df_gep_by_year = coastal_protection_functions.group_countries(
        df_gep_by_country_year
    )

    # Keep output schema stable and readable: country metadata, mangrove, coral reef, total.
    df_gep_by_country_base_year[COUNTRY_VALUE_COLUMN] = (
        df_gep_by_country_base_year['coastal_protection_gep']
    )
    country_output_columns = [
        column for column in (
            COUNTRY_INFO_COLUMNS
            + ['year']
            + MANGROVE_OUTPUT_COLUMNS
            + CORAL_OUTPUT_COLUMNS
            + [COUNTRY_VALUE_COLUMN]
        )
        if column in df_gep_by_country_base_year.columns
    ]
    df_gep_by_country_base_year = df_gep_by_country_base_year[
        country_output_columns
    ]
    df_gep_by_country_base_year = _coerce_numeric_output_columns(
        df_gep_by_country_base_year,
        NUMERIC_COUNTRY_OUTPUT_COLUMNS,
    )
    hb.df_write(df_gep_by_country_base_year, country_output)
    hb.df_write(
        _coerce_numeric_output_columns(df_gep_by_year, ['year', 'Value']),
        year_output,
    )

    gdf_gep_by_country_base_year = hb.df_merge(
        p.gdf_countries_vector_simplified_path,
        df_gep_by_country_base_year,
        how='outer',
        on='ee_r264_id',
    )
    gdf_gep_by_country_base_year.to_file(geopackage_output, driver='GPKG')

    value_gep_base_year = df_gep_by_country_base_year[
        COUNTRY_VALUE_COLUMN
    ].sum(min_count=1)
    hb.log(f'Total GEP value for base year {base_year}: {value_gep_base_year}')
    return value_gep_base_year


def gep_result(p):
    """Render each service report and fail if Quarto fails."""
    os.environ['QUARTO_PYTHON'] = sys.executable
    module_root = hb.get_projectflow_module_root()

    for service_label in p.results:
        source_qmd_path = os.path.join(
            module_root, service_label, f'{service_label}_results.qmd'
        )
        project_qmd_path = os.path.join(
            p.cur_dir, f'{service_label}_results.qmd'
        )
        hb.create_directories(project_qmd_path)
        hb.path_copy(source_qmd_path, project_qmd_path)
        try:
            env = os.environ.copy()
            env['QUARTO_LOG_LEVEL'] = 'DEBUG'
            cmd = ['quarto', 'render', project_qmd_path, '--verbose']
            hb.log(f"Running {' '.join(cmd)}")
            subprocess.run(cmd, check=True, env=env)
        finally:
            hb.path_remove(project_qmd_path)


def gep_results_distribution(p):
    """Copy registered GEP results to the project output directory."""
    hb.log('Distributing GEP results...')
    for key, value in p.results[SERVICE_LABEL].items():
        output_path = os.path.join(p.output_dir, os.path.basename(value))
        hb.path_copy(value, output_path)
        hb.log(f'Distributed {key} to {output_path}')
    hb.log('GEP results distribution complete.')
