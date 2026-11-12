"""
Global InVEST: Terrestrial Carbon GEP Pipeline

For each carbon-density variant:
  1. Build Spawn-derived lookup tables.
  2. Generate carbon-density rasters for selected LULC year(s).
  3. Summarize AGB, BGB, and total carbon by ee264 region.
  4. Aggregate results to iso250.
  5. Calculate terrestrial-carbon GEP at configured preferred rental SCC.
  6. Distribute GEP calculation outputs to the configured results directory.
"""

import logging
import os

import hazelbean as hb

from global_invest.terrestrial_carbon import (
    terrestrial_carbon_functions as tcf,
)
from global_invest.terrestrial_carbon import (
    terrestrial_carbon_initialization as tci,
)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # PROJECT CONFIGURATION
    # -------------------------------------------------------------------------
    p = hb.ProjectFlow()
    p.project_name = "gep_terrestrial_carbon"
    p.project_dir = os.path.join(
        os.path.expanduser("~"),
        "Files",
        "global_invest",
        "projects",
        p.project_name,
    )
    p.base_data_dir = (
        "/Users/long/Library/CloudStorage/"
        "GoogleDrive-yxlong@umn.edu/Shared drives/"
        "NatCapTEEMs/Files/base_data/submissions"
    )
    p.set_project_dir(p.project_dir)

    # -------------------------------------------------------------------------
    # INPUT DATA
    # -------------------------------------------------------------------------
    p.lulc_folder_path = p.get_path("lulc", "esa")
    p.base_year_lulc_path = p.get_path(
        "lulc",
        "esa",
        "lulc_esa_2010.tif",
    )
    p.carbon_zones_path = p.get_path(
        "carbon",
        "johnson_2019",
        "decision_tree_combined_carbon",
        "carbon_zones_rasterized.tif",
    )

    # -------------------------------------------------------------------------
    # COUNTRY BOUNDARIES
    # -------------------------------------------------------------------------
    p.df_countries_csv_path = p.get_path(
        "cartographic",
        "ee",
        "ee_r264_correspondence.csv",
    )
    p.gdf_countries_vector_path = p.get_path(
        "cartographic",
        "ee",
        "ee_r264_correspondence.gpkg",
    )
    p.gdf_countries_vector_simplified_path = p.get_path(
        "cartographic",
        "ee",
        "ee_r264_simplified300sec.gpkg",
    )
    p.zone_col = "ee_r264_id"

    # -------------------------------------------------------------------------
    # VARIANTS AND ANALYSIS YEAR
    # -------------------------------------------------------------------------
    variants = [0] # this is the number of latitudinal bands to use for the carbon-density variants. 0 means no bands, 1 means one band, etc.

    p.analysis_year = 2019  # Set None for all LULC years.

    # -------------------------------------------------------------------------
    # GEP CONFIGURATION
    # -------------------------------------------------------------------------
    # 2019 international-dollar annual rental SCC, per tC/year.
    p.gep_enabled = True
    p.gep_iso_col = "iso3_r250_id"
    p.gep_expected_iso_count = 250
    p.gep_missing_coverage_as_na = True
    p.gep_preferred_price_label = "r2pct"
    p.gep_preferred_price_column = "rental scc r2%"
    p.gep_preferred_rental_scc_2019_int_dollars_per_tc_year = 13.19
    p.gep_value_column = "terrestrial_carbon_gep"
    p.gep_results_dir = os.path.join(p.project_dir, "gep_results")

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    p.dry_run = False

    tcf.setup_logging(p.project_dir)
    tcf.run_shared_preprocessing(p)

    for n_lat_bands in variants:
        tcf.run_variant(
            p,
            n_lat_bands,
            tci.build_gep_service_task_tree,
        )

    logging.info("Done! All variants completed.")
    print("Done! All variants completed.")
