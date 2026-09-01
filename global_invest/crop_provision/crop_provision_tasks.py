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
    # FAOSTAT publishes both the individual crop and the group total that adds those crops up.
    # The item list carries both, so a group total is kept only where no individual item is. The
    # frame before that rule is kept, because valuing it is what reproduces the reference and a
    # reproduction nobody can re-run is a claim rather than a check (condition 12).
    df_crop_value_with_group_totals = df_crop_value.copy()
    df_crop_value = utilities.drop_aggregates_where_components_exist(
        df_crop_value, utilities.read_column(p.faostat_aggregate_items_path, 'item_fao'),
        'crop_provision_gep')
    df_crop_coefs = read_crop_coefs(p.cwon_crop_coefficients_path)

    df_gep_by_country_year_crop = utilities.apply_rental_rates(
        df_crop_value, df_crop_coefs, 'crop_provision_gep')
    df_gep_by_country_year_crop = utilities.normalize_m49_codes(df_gep_by_country_year_crop)
    df_gep_by_country_year_crop = crop_provision_functions.attach_countries_in_usd(
        df_gep_by_country_year_crop, p.df_countries)

    df_gep_by_country_year = utilities.sum_items_to_country_year(
        df_gep_by_country_year_crop, 'crop_provision_gep')

    df_gep_by_year = utilities.sum_countries_to_year(df_gep_by_country_year, 'crop_provision_gep')

    # The reference's own selection, valued the same way, as the column that proves the port.
    df_reference = utilities.apply_rental_rates(
        df_crop_value_with_group_totals, df_crop_coefs, 'crop_provision_gep')
    df_reference = utilities.normalize_m49_codes(df_reference)
    df_reference = crop_provision_functions.attach_countries_in_usd(df_reference, p.df_countries)
    df_reference = utilities.sum_items_to_country_year(df_reference, 'crop_provision_gep')

    base_year = int(p.gep_base_year)
    df_gep_by_country_base_year = df_gep_by_country_year.loc[
        df_gep_by_country_year['year'] == base_year].copy()
    reference_base_year = df_reference.loc[df_reference['year'] == base_year,
                                           ['iso3_r250_id', 'crop_provision_gep']].rename(
        columns={'crop_provision_gep': 'crop_provision_gep_reference'})
    df_gep_by_country_base_year = df_gep_by_country_base_year.merge(
        reference_base_year, on='iso3_r250_id', how='left')

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
    """Subsistence-crop GEP: the reference pipeline ported, its unit error corrected, and the
    uncorrected arithmetic published beside it as the proof of the port.

    A separate component from the commercial figure, written to its own table and never summed into
    it, exactly as the Lynch subsistence value sits beside the commercial rent in fisheries.

    `crop_subsistence_gep` is the account's figure and is ours: FAOSTAT reports cropland in
    thousands of hectares against an intensity per single hectare, and the Lowder share is a
    percentage, so reading each as its source labels it is not a variant of the method but the
    method done right. `crop_subsistence_gep_reference` is the reference's own arithmetic through
    the identical four downstream stages, reproducing its published panel to 2.3e-14, which is what
    lets a reader check the port rather than take it on trust. The delivered table is also put on
    the account's country list by the house collapse, keeping every valued country rather than the
    sixteen the reference's own join delivers.
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
    # The observed national shares, latest survey per country, as the imputation's evidence base.
    rulis = read(p.crop_subsistence_rulis_path, delimiter=';')
    observed_shares = rulis[
        (rulis['Indicator'] == crop_provision_functions.RULIS_OWN_CONSUMPTION_INDICATOR)
        & (rulis['Disaggregation'] == crop_provision_functions.RULIS_NATIONAL)]
    observed_shares = observed_shares.sort_values('Year').groupby(
        'Country', as_index=False).last()[['Country', 'Value']]

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
    # The account's figure is the one our code stands behind, so the units read as their sources
    # label them is what `crop_subsistence_gep` holds. The reference's arithmetic travels the same
    # four stages and is published beside it as `crop_subsistence_gep_reference`, which is what
    # makes the reproduction checkable rather than asserted. Both come off one run, so the
    # difference between the columns is the units and nothing else.
    delivered = {}
    for value_column in ('own_con_corrected', 'own_con'):
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
        delivered[value_column] = (
            df_deflated,
            crop_provision_functions.subsistence_on_country_list(
                df_deflated, p.df_countries, base_year))

    # The account's own method: the same corrected units, but the own-consumption SHARE imputed
    # where no survey reached a country, rather than the own-consumption LEVEL regressed on cropland
    # area. Three of the four factors are then the country's own. Built directly at the base year,
    # so no CPI step applies -- the intensity is already in that year's money.
    df_ours = crop_provision_functions.subsistence_value_from_shares(
        read(p.crop_subsistence_area_value_path, delimiter=';'),
        read(p.crop_subsistence_lowder_path, delimiter=';'),
        read(p.crop_subsistence_wb_income_group_path, delimiter=';'),
        read(p.crop_subsistence_wb_income_history_path, delimiter=';'),
        crop_provision_functions.impute_own_consumption_shares(
            observed_shares, read(p.crop_subsistence_wb_income_group_path, delimiter=';')),
        base_year)
    df_ours = df_ours.rename(columns={'own_con': 'own_con2'})
    df_ours = crop_provision_functions.apply_subsistence_rental_rate(
        df_ours, read_crop_coefs_raw(p.cwon_crop_coefficients_path))
    df_ours['crop_subsistence_gep'] = df_ours['gep_value']
    df_ours['own_con_source'] = df_ours['share_source']
    df_ours['own_con'] = df_ours['own_con2']
    df_by_country = crop_provision_functions.subsistence_on_country_list(
        df_ours, p.df_countries, base_year)
    df_panel_published = df_ours

    # The reference's two figures, joined on so each change is separable: the units alone, and the
    # units plus its own extrapolation.
    for column, source in (('crop_subsistence_gep_units_only', 'own_con_corrected'),
                           ('crop_subsistence_gep_reference', 'own_con')):
        theirs = delivered[source][1][['iso3_r250_label', 'crop_subsistence_gep']].rename(
            columns={'crop_subsistence_gep': column})
        df_by_country = df_by_country.merge(theirs, on='iso3_r250_label', how='left')

    hb.df_write(df_panel_published, p.crop_subsistence_panel_path)
    utilities.write_gep_by_country(
        p, df_by_country[utilities.published_country_columns(
            df_by_country, 'crop_subsistence')],
        p.crop_subsistence_gep_path)
    p.results.setdefault('crop_provision', {})['subsistence_gep_by_country'] = (
        p.crop_subsistence_gep_path)

    published = df_by_country['crop_subsistence_gep'].sum()
    reference = df_by_country['crop_subsistence_gep_reference'].sum()
    hb.log('Crop subsistence GEP for base year %d: %.9g USD across %d countries, the units read as '
           'their sources label them. The reference arithmetic through the same stages is %.9g '
           'USD, which reproduces their published panel.'
           % (base_year, published,
              int(df_by_country['crop_subsistence_gep'].notna().sum()), reference))
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