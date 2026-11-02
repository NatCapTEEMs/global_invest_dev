
import pandas as pd
import hazelbean as hb
from global_invest import utilities

from global_invest.coastal_protection import coastal_protection_functions

# The mangrove workbook ships its table on a sheet with this name. The coral workbook's sheet of
# this name is the authors' country table, the anchor the study units reproduce.
SOURCE_SHEET_NAME = 'Sheet1'
# The coral workbook's study-unit sheet, one row per Beck et al. (2018) study unit, which the coral
# value is summed from.
CORAL_REEF_UNIT_SHEET_NAME = 'Sheet4'


def read_mangrove_values(path):
    """The CWoN mangrove workbook read and renamed. See clean_mangrove_values."""
    return coastal_protection_functions.clean_mangrove_values(
        pd.read_excel(path, sheet_name=SOURCE_SHEET_NAME, engine='openpyxl'))


def publish_inputs(p):
    """Every task's first line: the CWoN coastal-protection valuation's es_config row (defaults layer -- a caller-set value wins)
    plus the shared country references and the results registry."""
    utilities.hydrate_es_config(p, 'coastal_protection', log=hb.log)
    utilities.hydrate_es_parameters(p, 'coastal_protection', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    # Auxiliary science input beside the quantity row: the coral-reef workbook (really a second
    # sheet subgroup).
    p.coral_reef_ref_path = p.get_path(p.coastal_protection_coral_reef_path)
    if not hasattr(p, 'results'):
        p.results = {}
    return p

def coastal_protection(p):
    """
    Parent task for mangrove coastal protection. Inputs resolve in publish_inputs.
    """
    publish_inputs(p)


def gep_preprocess(p):
    """Placeholder in the full tree. This service has no preprocessing step: its two valuation
    tables, the mangrove workbook and the coral-reef workbook, are staged in base_data and read
    directly by gep_calculation."""
    publish_inputs(p)
    if not p.run_this:
        return p
    pass  # intentionally empty: the service has no preprocessing step



def gep_calculation(p):
    """ GEP calculation task for coastal protection."""
    publish_inputs(p)
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.   
    service_results, already_done = utilities.begin_gep_calculation(p, 'coastal_protection')
    if already_done:
        return

    base_year = coastal_protection_functions.COASTAL_PROTECTION_BASE_YEAR
    if not p.run_this:
        return p
    p.gdf_countries = hb.read_vector(p.gdf_countries_vector_path)

    df_mangrove_value = read_mangrove_values(p.gep_quantity_input_path)
    df_coral_reef_units = pd.read_excel(p.coral_reef_ref_path, sheet_name=CORAL_REEF_UNIT_SHEET_NAME,
                                        engine='openpyxl')
    df_coral_reef_value = coastal_protection_functions.coral_reef_values_from_study_units(
        df_coral_reef_units)
    hb.log('  coral values summed from %d study units: %d countries, %s USD %.2f'
           % (len(df_coral_reef_units), len(df_coral_reef_value),
              coastal_protection_functions.CORAL_REEF_VALUE_YEAR,
              df_coral_reef_value['coral_reef_value'].sum()))
    # The coral table is in CORAL_REEF_VALUE_YEAR USD, carried to the base year by the shared
    # CPI series every temporal dollar conversion in the account uses.
    cpi = pd.read_csv(str(p.get_path(p.coastal_protection_us_cpi_input_path)))
    coral_deflation = utilities.usd_deflation_factor(
        cpi, coastal_protection_functions.CORAL_REEF_VALUE_YEAR, base_year)
    hb.log('  coral values carried from %s to %s USD: the shared CPI factor %.6f'
           % (coastal_protection_functions.CORAL_REEF_VALUE_YEAR, base_year, coral_deflation))

    df_mangrove = coastal_protection_functions.mangrove_gep_by_country(
        p.gdf_countries, df_mangrove_value)
    df_coral_reef = coastal_protection_functions.coral_reef_gep_by_country(
        p.gdf_countries, df_coral_reef_value, coral_deflation, base_year)

    # Countries the mangrove source removed from its analysis: their mangrove component is
    # missing, not the zero area its processing writes for them.
    excluded = hb.df_read(p.coastal_protection_mangrove_excluded_path)['iso3_r250_label']
    unknown = sorted(set(excluded) - set(p.gdf_countries['iso3_r250_label']))
    if unknown:
        raise ValueError(f'mangrove exclusion list names countries outside the correspondence: {unknown}')
    hb.log('  %d countries the mangrove source excluded publish a missing mangrove component: %s'
           % (len(excluded), ', '.join(excluded)))

    df_gep_by_country_year = coastal_protection_functions.combine_coastal_components(
        df_mangrove, df_coral_reef, mangrove_excluded_labels=excluded)
    df_gep_by_country_year = coastal_protection_functions.attach_country_attributes(
        df_gep_by_country_year, p.gdf_countries)

    df_gep_by_country_base_year = df_gep_by_country_year.loc[
        df_gep_by_country_year['year'] == base_year].copy()
    # `Value` is the copy of coastal_protection_gep that combine_coastal_components adds for the
    # shared reporting templates. Dropping it here rather than at the write means the map file below
    # carries the account's name as well.
    df_gep_by_country_base_year = df_gep_by_country_base_year.drop(columns=['Value'], errors='ignore')

    # The frame keeps its r264 columns for the map merge below; the published table does not.
    utilities.write_gep_by_country(
        p, df_gep_by_country_base_year[utilities.published_country_columns(
            df_gep_by_country_base_year, 'coastal_protection')],
        p.results['coastal_protection']['gep_by_country_base_year'])

    # Map only: the r264-expanded boundaries, each sub-region carrying its country's value.
    gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_vector_simplified_path, df_gep_by_country_base_year, how='outer', on='ee_r264_id')
    gdf_gep_by_country_base_year.to_file(p.results['coastal_protection']['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    value_gep_base_year = df_gep_by_country_base_year['coastal_protection_gep'].sum()
    value_gep_both_components = df_gep_by_country_base_year['coastal_protection_gep_both_components'].sum()

    hb.log(f"Total GEP value for base year {base_year}: {value_gep_base_year}")
    hb.log(f"  total over countries with both components measured: {value_gep_both_components}")
    hb.log('  countries by coverage: %s' % df_gep_by_country_base_year[
        coastal_protection_functions.COVERAGE_COLUMN].value_counts(dropna=False).to_dict())
    return value_gep_base_year

def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.render_service_results(p)

def gep_results_distribution(p):
    """Copy this service's results into the output directory. Shared implementation in
    utilities, which is also where the service key stops being written out by hand."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.distribute_results(p, 'coastal_protection')
