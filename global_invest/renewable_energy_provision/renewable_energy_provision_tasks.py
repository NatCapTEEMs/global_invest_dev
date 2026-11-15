import os

import hazelbean as hb
from global_invest import utilities

from global_invest.renewable_energy_provision import renewable_energy_provision_functions as rf


def publish_inputs(p):
    """Every task's first line: the renewable-energy valuation's es_config and es_parameters rows (defaults layer --
    a caller-set value wins) plus the shared country references and the results registry."""
    utilities.hydrate_es_config(p, 'renewable_energy_provision', log=hb.log)
    utilities.hydrate_es_parameters(p, 'renewable_energy_provision', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    if not hasattr(p, 'results'):
        p.results = {}
    return p

def renewable_energy_provision(p):
    """
    Parent task for renewable energy provision. Inputs resolve in publish_inputs.
    """
    publish_inputs(p)

def gep_preprocess(p):
    """
    Preprocessing step, not meant to be run by users: its output becomes a model input and is promoted
    to the base_data_dir that users receive. Not yet implemented.
    """
    publish_inputs(p)
    if not p.run_this:
        return p
    pass # NYI

def write_published(df, path, one_row_per_country=False):
    """Write a country table in the published shape, renaming on the way out only: the frame
    keeps the source's `Year` because the split and the map merge read it, while the published
    table uses the lowercase `year` every other service writes.

    With one_row_per_country, the per-technology rows collapse to the account's shape, one row
    per country with the technologies summed; otherwise any downstream join on iso3 multiplies
    each country by its technology count. The collapse groups on the country keys only, because
    the technology, its price and its generation differ per row by construction. The
    per-technology detail is what the three subservice tables are for.
    """
    published = df.rename(columns={'Year': 'year'})
    published = published[utilities.published_country_columns(
        published, 'renewable_energy_provision')]
    if one_row_per_country:
        keys = [c for c in utilities.GEP_COUNTRY_ATTR_COLS + ['year']
                if c in published.columns]
        # min_count=1: a country whose every technology is missing stays missing. A bare
        # sum() maps an all-NaN group to 0, which would publish unmeasured countries as zero
        # renewable value.
        published = (published.groupby(keys, dropna=False, as_index=False)
                     ['renewable_energy_provision_gep'].sum(min_count=1))
    hb.df_write(published, path, index=False)


def gep_calculation(p):
    """GEP valuation for renewable energy: each resource's generation priced at the country-year
    electricity price and scaled by nature's share of the resource rent, one CSV per resource."""
    publish_inputs(p)
    service_results = {'gep_by_country_base_year': os.path.join(
        p.cur_dir, "renewable_energy_provision_gep_by_country_base_year.csv")}
    p.results['renewable_energy_provision'] = service_results
    subservices = {subservice: {'gep_by_country_base_year': os.path.join(
        p.cur_dir, f"{subservice}_gep_by_country_base_year.csv")}
        for subservice in rf.SUBSERVICE_TECHNOLOGIES}
    service_results['subservices'] = subservices

    if not p.run_this:
        return p
    output_paths = ([service_results['gep_by_country_base_year']]
                    + [s['gep_by_country_base_year'] for s in subservices.values()])
    if hb.path_all_exist(output_paths):
        hb.log("All results already exist. Skipping GEP calculation for renewable energy provision.")
        return
    hb.log("Starting GEP calculation for renewable energy provision.")

    generation_frames = rf.generation_by_technology(hb.df_read(p.gep_quantity_input_path))
    df_price = rf.price_in_usd_per_gwh(hb.df_read(p.gep_price_input_path))
    priced_frames = rf.merge_price_onto_generation(df_price, generation_frames)
    # The region map rides the IRENA table itself, so the fill needs no second source.
    regions = hb.df_read(p.gep_quantity_input_path)[['Country', 'Sub-region', 'Region']]
    df_valued = rf.valued_generation(priced_frames, hb.df_read(p.gep_attribution_input_path),
                                     country_regions=regions)
    df_gep = rf.base_year_valued_rows(df_valued, int(p.gep_base_year))

    # The source is keyed by ISO3 strings, so the join matches on the r250 label. One row per
    # country: r264 splits large countries, so the correspondence is collapsed before the join.
    ee_r264_to_250 = utilities.collapse_countries_to_r250(p.df_countries)
    valued_rows_before_join = int(df_gep['renewable_energy_provision_gep'].notna().sum())
    df_gep = hb.df_merge(ee_r264_to_250, df_gep,
                         left_on='iso3_r250_label', right_on='iso3_r250_label', how='left')
    utilities.assert_join_coverage(df_gep, 'renewable_energy_provision_gep',
                                   valued_rows_before_join, 'renewable_energy_provision', log=hb.log)

    write_published(df_gep, service_results['gep_by_country_base_year'],
                    one_row_per_country=True)
    by_resource = rf.split_by_resource(df_gep)
    for subservice, technology in rf.SUBSERVICE_TECHNOLOGIES.items():
        write_published(by_resource[technology],
                        subservices[subservice]['gep_by_country_base_year'])

    # Map only: the r264-expanded boundaries, each sub-region carrying its country's value.
    gdf = hb.df_merge(p.gdf_countries_simplified, df_gep, how='outer',
                      left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    value_gep_base_year = df_gep['renewable_energy_provision_gep'].sum()
    hb.log(f"Total GEP value for base year {int(p.gep_base_year)}: {value_gep_base_year}")
    return value_gep_base_year

def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.render_service_results(p)

def gep_results_distribution(p):
    """Copy this service's results into the output directory. Shared implementation in
    utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.distribute_results(p, 'renewable_energy_provision')
