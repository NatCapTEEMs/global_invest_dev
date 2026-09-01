import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities

from global_invest.crop_provision import crop_provision_initialize
from global_invest.crop_provision import crop_provision_functions

# Project content/config (was crop_provision_defaults.py; the template keeps content in the task module).


def read_crop_values(path, items, aggregate_areas, value_column='crop_provision_gep'):
    """The FAOSTAT bulk file read and cleaned. See clean_crop_values for the science.

    The file ships Latin-1 encoded, so it is read as such rather than as UTF-8. Both components
    read the same table, and each names the value for itself.
    """
    return utilities.clean_faostat_values(
        pd.read_csv(path, encoding='ISO-8859-1'), items, value_column, aggregate_areas)


def read_crop_coefs_raw(path):
    """The CWoN rental-rate table as shipped. It is semicolon-delimited.

    Both components read this file and key it differently -- the commercial one on the FAO area
    code, the subsistence one on ISO3 -- so the reshape happens in each, not here.
    """
    return pd.read_csv(path, delimiter=';', encoding='utf-8')


def read_crop_coefs(path):
    """The CWoN rental-rate table read and reshaped. See build_rental_rate_lookup for the science."""
    return utilities.build_rental_rate_lookup(read_crop_coefs_raw(path))


def publish_inputs(p):
    """Every task's first line: the FAO crop-production valuation's es_config row (defaults layer -- a caller-set value wins)
    plus the shared country references and the results registry."""
    utilities.hydrate_es_config(p, 'crop_provision', log=hb.log)
    utilities.hydrate_es_parameters(p, 'crop_provision', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    if not hasattr(p, 'results'):
        p.results = {}
    return p

def crop_provision(p):
    """
    Parent task for commercial agriculture.
    """
    publish_inputs(p)

def gep_preprocess(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.crop_provision_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    publish_inputs(p)
    pass # NYI

def gep_calculation(p):
    """ GEP calculation task for commercial agriculture."""
    publish_inputs(p)
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.   
    # The extra tables this service writes are registered up front, so the skip check
    # covers all of them and the results page can find every one on a skipped rerun.
    service_results, already_done = utilities.begin_gep_calculation(
        p, 'crop_provision', extra_results={
         'gep_by_country_year_crop': os.path.join(p.cur_dir, 'gep_by_country_year_crop.csv'),
         'gep_by_country_year': os.path.join(p.cur_dir, 'gep_by_country_year.csv'),
         'gep_by_year': os.path.join(p.cur_dir, 'gep_by_year.csv')})
    if already_done:
        return

    if not getattr(p, 'crop_provision_subservices', None):
        p.commercial_attribute_subservices = utilities.read_column(
            p.crop_provision_default_items_path, 'item_fao')

    df_crop_value = read_crop_values(
        p.fao_input_path, p.commercial_attribute_subservices,
        utilities.read_column(p.faostat_aggregate_areas_path, 'area_fao'))
    df_crop_coefs = read_crop_coefs(p.cwon_crop_coefficients_path)

    df_gep_by_country_year_crop = utilities.apply_rental_rates(
        df_crop_value, df_crop_coefs, 'crop_provision_gep')
    df_gep_by_country_year_crop = utilities.normalize_m49_codes(df_gep_by_country_year_crop)
    df_gep_by_country_year_crop = crop_provision_functions.attach_countries_in_usd(
        df_gep_by_country_year_crop, p.df_countries)

    df_gep_by_country_year = utilities.sum_items_to_country_year(
        df_gep_by_country_year_crop, 'crop_provision_gep')

    df_gep_by_year = utilities.sum_countries_to_year(df_gep_by_country_year, 'crop_provision_gep')

    base_year = int(p.gep_base_year)
    df_gep_by_country_base_year = df_gep_by_country_year.loc[
        df_gep_by_country_year['year'] == base_year].copy()

    # Write to CSVs
    hb.df_write(df_gep_by_country_year_crop, p.results['crop_provision']['gep_by_country_year_crop'])
    hb.df_write(df_gep_by_country_year, p.results['crop_provision']['gep_by_country_year'])
    utilities.write_gep_by_country(
        p, df_gep_by_country_base_year[utilities.published_country_columns(
            df_gep_by_country_base_year, 'crop_provision')],
        p.results['crop_provision']['gep_by_country_base_year'])
    hb.df_write(df_gep_by_year, p.results['crop_provision']['gep_by_year'], handle_quotes='all')
    hb.df_write(df_gep_by_year, hb.replace_ext(p.results['crop_provision']['gep_by_year'], 'xlsx'), handle_quotes='all')
    
    # Map only: the r264-expanded boundaries, each sub-region carrying its country's value.
    gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_simplified, df_gep_by_country_base_year, how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
    gdf_gep_by_country_base_year.to_file(p.results['crop_provision']['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    value_gep_base_year = df_gep_by_country_base_year['crop_provision_gep'].sum()

    hb.log(f"Total GEP value for base year {base_year}: {value_gep_base_year}")

    return value_gep_base_year

def crop_subsistence_gep(p):
    """Subsistence-crop GEP: the reference pipeline ported, reproduced, and its unit error undone.

    A separate component from the commercial figure, written to its own table and never summed into
    it, exactly as the Lynch subsistence value sits beside the commercial rent in fisheries.

    Three numbers come out, and the reason there are three is the point. `crop_subsistence_gep` is
    the reference reproduced, so a disagreement about it is a disagreement about arithmetic we have
    both run. `crop_subsistence_gep_corrected` is the same chain with FAOSTAT's thousands of
    hectares and Lowder's percentage read as their sources label them, which is ten times larger.
    And the delivered table is on the account's country list by the house collapse, which keeps
    every valued country rather than the sixteen the reference's own join delivers.
    """
    publish_inputs(p)
    p.crop_subsistence_gep_path = os.path.join(p.cur_dir, 'subsistence_gep_by_country.csv')
    p.crop_subsistence_panel_path = os.path.join(p.cur_dir, 'subsistence_panel_by_country_year.csv')
    p.crop_subsistence_own_consumption_path = os.path.join(
        p.cur_dir, 'subsistence_own_consumption.csv')
    outputs = [p.crop_subsistence_gep_path, p.crop_subsistence_panel_path,
               p.crop_subsistence_own_consumption_path]
    if not p.run_this:
        return
    if hb.path_all_exist(outputs):
        return True

    read = lambda path, **kw: pd.read_csv(path, encoding='utf-8', **kw)
    df_iso = read(p.crop_subsistence_iso3_path)

    # Step 01: own consumption per country and survey year, the reference's arithmetic and the
    # unit-corrected one side by side.
    df_own = crop_provision_functions.subsistence_own_consumption(
        read(p.crop_subsistence_rulis_path, delimiter=';'),
        read(p.crop_subsistence_wb_income_history_path, delimiter=';'),
        read(p.crop_subsistence_area_value_path, delimiter=';'),
        read(p.crop_subsistence_wb_income_group_path, delimiter=';'),
        read(p.crop_subsistence_lowder_path, delimiter=';'),
        read(p.crop_subsistence_gross_production_path, delimiter=';')[
            ['Country', 'Year', 'Value', 'Unit']].query("Unit == '1000 USD'"),
        df_iso)
    hb.df_write(df_own, p.crop_subsistence_own_consumption_path)

    df_gross_usd = read(p.crop_subsistence_gross_production_usd_path)[
        ['Country', 'Year', 'Value']].rename(columns={'Value': 'Value_gross_prof'})
    df_gdp = read(p.crop_subsistence_gdp_per_capita_path, delimiter=';')
    year_columns = [c for c in df_gdp.columns if c.isdigit()
                    and crop_provision_functions.SUBSISTENCE_FIRST_YEAR <= int(c)
                    <= crop_provision_functions.SUBSISTENCE_LAST_YEAR]
    df_gdp = df_gdp[['Country'] + year_columns].melt(
        id_vars=['Country'], var_name='Year', value_name='GDP_capita')
    df_gdp['Year'] = df_gdp['Year'].astype(int)

    base_year = int(p.gep_base_year)
    for value_column in ('own_con', 'own_con_corrected'):
        # Both figures travel the same four remaining stages, so the difference between them is the
        # units and nothing else.
        df_panel = df_own[['Country', 'Year', 'alpha-3', value_column]].rename(
            columns={value_column: 'own_con'})
        df_interpolated = crop_provision_functions.interpolate_missing_years(
            df_panel, df_gross_usd, df_gdp)
        df_extrapolated, feature = crop_provision_functions.extrapolate_to_unsurveyed(
            df_interpolated, read(p.crop_subsistence_wb_income_history_path, delimiter=';'),
            read(p.crop_subsistence_covariates_path), df_iso)
        df_valued = crop_provision_functions.apply_subsistence_rental_rate(
            df_extrapolated, read_crop_coefs_raw(p.cwon_crop_coefficients_path))
        df_deflated = crop_provision_functions.deflate_to_base_year(
            df_valued, read(p.crop_subsistence_cpi_path, delimiter=';'), base_year)
        if value_column == 'own_con':
            df_reference, df_reference_delivered = df_deflated, (
                crop_provision_functions.subsistence_on_country_list(
                    df_deflated, p.df_countries, base_year))
        else:
            corrected = crop_provision_functions.subsistence_on_country_list(
                df_deflated, p.df_countries, base_year)
            df_reference_delivered['crop_subsistence_gep_corrected'] = (
                corrected['crop_subsistence_gep'].values)

    hb.df_write(df_reference, p.crop_subsistence_panel_path)
    utilities.write_gep_by_country(
        p, df_reference_delivered[utilities.published_country_columns(
            df_reference_delivered, 'crop_subsistence')],
        p.crop_subsistence_gep_path)
    p.results.setdefault('crop_provision', {})['subsistence_gep_by_country'] = (
        p.crop_subsistence_gep_path)

    reproduced = df_reference_delivered['crop_subsistence_gep'].sum()
    corrected = df_reference_delivered['crop_subsistence_gep_corrected'].sum()
    hb.log('Crop subsistence GEP for base year %d: %.9g USD reproducing the reference across %d '
           'countries, and %.9g USD with the units corrected.'
           % (base_year, reproduced,
              int(df_reference_delivered['crop_subsistence_gep'].notna().sum()), corrected))
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

def gep_results_distribution(p):
    """Distribute the results of the GEP calculation."""
    publish_inputs(p)
    # This task is intended to copy the results to the output directory.
    hb.log("Distributing GEP results...")
    
    for key, value in p.results['crop_provision'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")
    
    hb.log("GEP results distribution complete.")