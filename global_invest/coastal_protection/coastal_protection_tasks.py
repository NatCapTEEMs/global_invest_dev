import os
import subprocess
import sys

import hazelbean as hb
import pandas as pd

from global_invest.coastal_protection import coastal_protection_functions


SERVICE_LABEL = 'coastal_protection'
COUNTRY_OUTPUT_NAME = 'gep_by_country_base_year.csv'
YEAR_OUTPUT_NAME = 'gep_by_year.csv'
COUNTRY_VALUE_COLUMN = 'Value (2019 USD)'

COUNTRY_INFO_COLUMNS = [
    'iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
    'ee_r264_id', 'ee_r264_label', 'ee_r264_name',
    'country', 'continent', 'region_un', 'region_wb', 'income_grp',
    'subregion', 'area_code_M49', 'area_code',
]
MANGROVE_OUTPUT_COLUMNS = [
    'mangrove_value_2019_usd',
    'coastal_protection_gep_mangrove',
]
CORAL_OUTPUT_COLUMNS = [
    'coral_reef_value_2011_usd',
    'coral_reef_value_2019_usd',
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
    p.wb_deflator_ref_path = os.path.join(
        data_dir, 'API_NY.GDP.DEFL.ZS_DS2_en_csv_v2_24026.csv'
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
    try:
        country_df = hb.df_read(country_output)
        country_gdf = hb.read_vector(geopackage_output)
    except Exception:
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
    """Harmonize mangrove and coral-reef values in 2019 USD."""
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

    # Read source monetary values (mangrove in 2019 USD; coral in 2011 USD).
    df_mangrove_value = coastal_protection_functions.read_mangrove_values(
        p.cwon_input_ref_path
    )
    df_coral_reef_value = coastal_protection_functions.read_coral_reef_values(
        p.coral_reef_ref_path
    )
    us_inflation_factor = coastal_protection_functions.get_us_inflation_factor(
        p.wb_deflator_ref_path, coral_value_year, base_year
    )

    # Mangrove values are already 2019 USD.
    df_mangrove = _merge_source_values(
        p.gdf_countries, df_mangrove_value, 'ee_r264_label', base_year
    )
    df_mangrove = _group_service_values(
        df_mangrove, ['mangrove_value_2019_usd']
    )
    df_mangrove['coastal_protection_gep_mangrove'] = df_mangrove[
        'mangrove_value_2019_usd'
    ]

    # Coral: 2011 USD -> 2019 USD using the US GDP deflator.
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
    df_coral['coral_reef_value_2011_usd'] = df_coral['coral_reef_value']
    df_coral['coral_reef_value_2019_usd'] = (
        coastal_protection_functions.convert_usd_to_target_year(
            df_coral['coral_reef_value_2011_usd'], us_inflation_factor
        )
    )
    df_coral['year'] = base_year
    df_coral = _group_service_values(
        df_coral,
        ['coral_reef_value_2011_usd', 'coral_reef_value_2019_usd'],
    )
    df_coral['coastal_protection_gep_coral_reef'] = df_coral[
        'coral_reef_value_2019_usd'
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
