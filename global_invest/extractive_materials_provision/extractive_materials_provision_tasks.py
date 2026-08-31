"""Extractive-materials GEP tasks: the mineral-rent share of GDP, valued on the r250 rows.

This layer owns every file read and write. The science it calls lives in
extractive_materials_provision_functions, which never opens a file.
"""
import os

import hazelbean as hb
import pandas as pd

from global_invest import utilities
from global_invest.extractive_materials_provision import extractive_materials_provision_initialize
from global_invest.extractive_materials_provision import extractive_materials_provision_functions as emf

# Applied to (mineral rents share x GDP) in the valuation. Provenance UNDOCUMENTED as of 2026-08-16:
# no source in the code, the drive submission, or its raw_data notes -- open question for the service
# owner. Do not change without an owner-blessed source; the staged reference output embeds it.
MINERAL_RENT_GEP_FACTOR = 0.49
# The World Bank CSVs ship UTF-8 with a byte-order mark, which -sig strips (EE spec).
WORLD_BANK_CSV_ENCODING = 'utf-8-sig'


def publish_inputs(p):
    """Every task's first line: the mineral-rents valuation's es_config row (defaults layer -- a caller-set value wins)
    plus the shared country references and the results registry."""
    utilities.hydrate_es_config(p, 'extractive_materials_provision', log=hb.log)
    utilities.hydrate_es_parameters(p, 'extractive_materials_provision', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    if not hasattr(p, 'results'):
        p.results = {}
    return p

def extractive_materials_provision(p):
    """
    Parent task for extractive materials provision.
    """
    publish_inputs(p)
    pass  # Inputs resolve in publish_inputs.

def gep_preprocess(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.extractive_materials_provision_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    publish_inputs(p)
    pass # NYI

def gep_calculation(p):
    """GEP calculation task for extractive materials provision.

    Mineral rents (percent of GDP) times GDP times the attribution factor, per country and
    year, with the base-year rows split out for the map and the report.
    """
    publish_inputs(p)
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.
    service_results = {}
    p.results['extractive_materials_provision'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, "gep_by_country_base_year.csv")

    # Optional additional results.
    service_results['gep_by_country_year_mineral'] = os.path.join(p.cur_dir, "gep_by_country_year_mineral.csv")
    service_results['gep_by_country_year'] = os.path.join(p.cur_dir, "gep_by_country_year.csv")
    service_results['gep_by_year'] = os.path.join(p.cur_dir, "gep_by_year.csv")

    reason = utilities.reuse_reason(p, 'extractive_materials_provision',
                                    list(service_results.values()))
    if reason is None:
        hb.log('extractive_materials_provision reuses its gep outputs: the signature is unchanged.')
        return
    hb.log('extractive_materials_provision recomputes its gep, %s' % reason)
    hb.log("Starting GEP calculation for extractive materials provision.")

    base_year = int(p.gep_base_year)
    df_mineral_values = emf.world_bank_wide_to_long(
        pd.read_csv(p.gep_attribution_input_path, encoding=WORLD_BANK_CSV_ENCODING), 'mineral_rent')
    df_gdp_values = emf.world_bank_wide_to_long(
        pd.read_csv(p.gep_quantity_input_path, encoding=WORLD_BANK_CSV_ENCODING), 'GDP_currentUSD')

    df_gep_by_country_year_mineral = df_mineral_values.merge(
        df_gdp_values, on=['Country Code', 'year'], how='left')
    df_gep_by_country_year_mineral['extractive_materials_provision_gep'] = emf.mineral_rent_gep(
        df_gep_by_country_year_mineral['mineral_rent'],
        df_gep_by_country_year_mineral['GDP_currentUSD'], MINERAL_RENT_GEP_FACTOR)
    # 'Value' is the column group_countries aggregates and the results report plots.
    df_gep_by_country_year_mineral['Value'] = df_gep_by_country_year_mineral['extractive_materials_provision_gep']
    df_gep_by_country_year_mineral.drop_duplicates(subset=['Country Code', 'year'], inplace=True)

    # One row per country: r264 splits large countries, so the correspondence is
    # collapsed before the join.
    ee_r264_to_250 = utilities.collapse_countries_to_r250(p.df_countries)
    df_gep_by_country_year_mineral = hb.df_merge(ee_r264_to_250, df_gep_by_country_year_mineral, how='left', left_on='iso3_r250_label', right_on='Country Code')

    df_gep_by_country_year = df_gep_by_country_year_mineral.copy()
    df_gep_by_country_base_year = df_gep_by_country_year.loc[df_gep_by_country_year['year'] == base_year].copy()
    df_gep_by_year = emf.group_countries(df_gep_by_country_year)

    # Write to CSVs
    hb.df_write(df_gep_by_country_year_mineral, service_results['gep_by_country_year_mineral'])
    hb.df_write(df_gep_by_country_year, service_results['gep_by_country_year'])
    hb.df_write(df_gep_by_country_base_year[utilities.published_country_columns(
        df_gep_by_country_base_year, 'extractive_materials_provision')],
        service_results['gep_by_country_base_year'])
    hb.df_write(df_gep_by_year, service_results['gep_by_year'], handle_quotes='all')
    hb.df_write(df_gep_by_year, hb.replace_ext(service_results['gep_by_year'], 'xlsx'), handle_quotes='all')
    utilities.write_reuse_signature(p, 'extractive_materials_provision', list(service_results.values()))

    # Map only: the r264-expanded boundaries, each sub-region carrying its country's value.
    gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_simplified, df_gep_by_country_base_year, how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
    gdf_gep_by_country_base_year.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    value_gep_base_year = df_gep_by_country_base_year['extractive_materials_provision_gep'].sum()
    hb.log(f"Total GEP value for base year {base_year}: {value_gep_base_year}")
    return value_gep_base_year

def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

def gep_results_distribution(p):
    """Distribute the results of the GEP calculation."""
    publish_inputs(p)
    # This task is intended to copy the results to the output directory.
    hb.log("Distributing GEP results...")

    for key, value in p.results['extractive_materials_provision'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")

    hb.log("GEP results distribution complete.")
