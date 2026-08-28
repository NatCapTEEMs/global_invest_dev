import os

import hazelbean as hb

from global_invest.flood import flood_initialize
import flood_paths as FP


def set_flood_paths(p):
    """
    Set every p.flood_* attribute the flood module reads. Factored out of
    __main__ so the MSI array/stage runners (msi/run_flood_slice.py) share one
    definition of the project layout instead of duplicating it.
    """
    # -------------------------------------------------------------------
    # Project-wide paths
    # -------------------------------------------------------------------
    p.flood_root = str(FP.ROOT)
    p.flood_input_dir = os.path.join(p.flood_root, "inputs")
    # The jajohns group storage quota has been full -- write model outputs to
    # scratch. Point this back at {flood_root}/outputs once quota is resolved.
    # Outputs live in the project directory, not scratch. /scratch.global is
    # purged on a schedule and took a full set of Step 4B/4C/4D results with it
    # during development. Home has ample space (745 TB free on the volume).
    p.flood_output_dir = str(FP.ROOT) + "/outputs"

    p.flood_country_boundary_path = os.path.join(
        p.flood_input_dir, "country_vector", "country_boundary_r250_with_iso3.gpkg")

    # Return periods actually on disk: RP10, RP20, RP50, RP500.
    #   inputs/floodplain_depth/aligned_to_lulc/JRC_flood_depth_rp{RP}y__matchLULC.tif
    #   inputs/floodplain_depth/masks_aligned_to_lulc/JRC_flood_mask_rp{RP}y__matchLULC.tif
    # JRC also publishes RP100/RP200 at the same URL pattern; omitting them
    # biases EAD high by ~1% (see the note above RETURN_PERIODS in
    # flood_functions.py). Add them to this list once downloaded.
    p.flood_return_periods = [10, 20, 50, 100, 200, 500]

    # -------------------------------------------------------------------
    # Section A -- input preparation (depth, LULC->SDA mapping, SDA, SPA QA)
    # -------------------------------------------------------------------
    p.flood_lulc_path = os.path.join(p.flood_input_dir, "lulc", "lulc_esa_2019_int_reproj.tif")
    # Hazard release. v1 (2016) is floodplain_depth; v2 (2024) resolves basins
    # down to 500 km2 rather than 5,000, and finds about three times the wet
    # area at less than half the mean depth. Nothing upstream of the valuation
    # changes: the service-flow ratio and the amplification rasters are built
    # from rainfall and flow direction, not from the hazard maps, so the 37
    # mosaic layers carry over unchanged.
    p.flood_depth_aligned_dir = os.path.join(
        p.flood_input_dir, "floodplain_depth_v2_nowater", "aligned_to_lulc")
    p.flood_sda_mapping_json = os.path.join(
        p.flood_input_dir, "lulc_to_sda_mapping", "lulc_to_sda_mapping.json")
    p.flood_global_sda_raster_path = os.path.join(
        p.flood_input_dir, "sda", "sda_esa300m_artif_crop_pasture.tif")
    p.flood_spa_path = os.path.join(p.flood_input_dir, "global_spa_ben", "global_prr_spa.tif")
    p.flood_spa_ratio_path = os.path.join(
        str(FP.INPUTS / "counterfactual_mosaic"),
        "global_upstream_spa_ratio.tif")

    # Depths at or below this (metres) contribute no damage.
    p.flood_depth_threshold_m = 0.1
    # Grassland as an optional pasture proxy (JRC default is conservative).
    p.flood_include_pasture = True
    # The JRC depth zips are ~45-50 MB each; set True once they are on disk.
    p.flood_skip_depth_download = False

    # -------------------------------------------------------------------
    # Section B -- SDA delineation per ISO3 x RP
    # -------------------------------------------------------------------
    p.flood_iso3_list = ""      # blank = every country in the Admin0 layer
    p.flood_iso3_start = 0
    p.flood_iso3_n = 0          # 0 = all remaining
    p.flood_skip_done = True
    p.flood_all_touched = False

    # Roads as SDA class 4. NOTE: leaving this False means road pixels are not
    # counted as exposed assets anywhere in the pipeline. If you set it True,
    # the damage table must carry a 'roads' sda_type or those pixels get zero
    # damage in Section D -- see README "Known issues".
    p.flood_use_roads = False
    p.flood_roads_path = os.path.join(p.flood_input_dir, "roads", "roads_mask_match_depth.tif")

    p.flood_with_pop = False
    p.flood_pop_path = os.path.join(p.flood_input_dir, "pop", "GlobPOP_Count_30arc_2020_I32.tif")

    # -------------------------------------------------------------------
    # Section C -- SPA -> SDA service flow
    # -------------------------------------------------------------------
    p.flood_write_service_flow_rasters = True
    p.flood_include_existing_in_summary = True

    # -------------------------------------------------------------------
    # Section D -- monetary valuation (4A -> 4B -> 4C -> 4D)
    # -------------------------------------------------------------------
    p.flood_damage_dir = os.path.join(p.flood_input_dir, "flood_damage")
    p.flood_canonical_eur_csv = os.path.join(
        p.flood_damage_dir, "country_landtype_flood_damage_JRC_EUR_m2.csv")
    p.flood_currency_factors_csv = os.path.join(
        p.flood_damage_dir, "_currency_audit",
        "currency_conversion_factors_EUR2010_to_USD2019.csv")
    p.flood_set_pasture_equal_crop = True

    # Correct pixel area for Web Mercator distortion. EPSG:3857 inflates area
    # by 1/cos^2(lat) EXACTLY, so the transform's pixel area is the equatorial
    # value: 4.00x too large at 60N, 5.60x at 65N. Set False only to reproduce
    # the uncorrected historical numbers.
    p.flood_latitude_correct_area = True
    p.flood_write_damage_rasters = True
    p.flood_damage_depth_mode = "interpolated"  # "interpolated" or "banded" (INCA nine-band step)
    p.flood_ead_tail_mode = "flat"      # hold D constant from p=1/RPmax to p=0
    p.flood_ead_add_p1_zero = False      # anchor (p=1, D=0)
    p.flood_ead_enforce_monotone = False
    p.flood_ead_write_points = True

    p.flood_global_export_dir = os.path.join(p.flood_output_dir, "_global")
    p.flood_region_col = "region_wb"
    p.flood_fill_missing_zero = True
    p.flood_compute_ead_rasters = False
    p.flood_mosaic_global_raster = False

    # Join Section C's service-flow fractions onto the damage series and carry
    # an attributed EAD alongside the gross one. Read the note above
    # VAL_APPLY_SERVICE_FLOW in flood_functions.py before using that number:
    # it attributes residual damage to naturally-served floodplains, it is NOT
    # avoided damage and NOT directly comparable to the erosion module's GEP.
    p.flood_apply_service_flow = True

    # --- Counterfactual scenarios: the route to a real service value --------
    # Two degraded worlds are run and reported side by side:
    #   degraded_insitu  UMRB scenario-2 CN values (ecosystems degraded in place)
    #                    -> conservative; use for the flood manuscript
    #   degraded_bare    TR-55 fallow bare soil (77/86/91/94)
    #                    -> same baseline as InVEST SDR's RKLS, so THIS is the
    #                       column for any combined GEP table with erosion
    #
    # Amplification rasters are built externally, one per scenario x RP, by
    # counterfactual/build_amplification_routed.py.
    p.flood_amplification_dir = os.path.join(
        str(FP.INPUTS / "counterfactual_mosaic"))
    # The exponent is not in the pattern, so one run gives one exponent. For
    # the sweep, change this to "..._f0p3.tif" / "..._f0p5.tif" together with
    # flood_gep_csv, and run three times.
    p.flood_amplification_pattern = "global_amplification_{scenario}_rp{rp}.tif"
    p.flood_cn_table = os.path.join(
        p.flood_amplification_dir, "esa_cci_CN_three_scenarios.csv")

    # --- Flood protection truncation (Vallecillo Eq.7) ----------------------
    # Reported as a SENSITIVITY, not the default: FLOPROS is largely GDP-inferred
    # where unreported and the damage curves are also GDP-scaled, so truncation
    # errors correlate across countries. Set the CSV (iso3, protection_rp) to
    # get NC / NC+ columns alongside the untruncated total.
    p.flood_protection_csv = os.path.join(
        p.flood_input_dir, "protection", "flopros_merged_by_iso3.csv")
    p.flood_report_protection_split = True

    # Restrict truncation to the 37 countries with DOCUMENTED river protection
    # standards (FLOPROS design & policy layers). They cover 83.1% of global EAD.
    # Countries whose protection is GDP-inferred are reported UNTRUNCATED rather
    # than truncated on a guess -- important because the Huizinga damage curves
    # are GDP-scaled too, so inferring protection from GDP would correlate the
    # error with the quantity being truncated.
    p.flood_protection_documented_only = True
    p.flood_protection_evidence_csv = os.path.join(
        p.flood_input_dir, "protection", "flopros_documented_iso3.csv")

    # -------------------------------------------------------------------
    # Section E -- maps & figures
    # -------------------------------------------------------------------
    p.flood_figures_dir = os.path.join(p.flood_global_export_dir, "figures_2024hazard_waterremoved")
    p.flood_map_k_classes = 5
    p.flood_top_n = 20
    p.flood_money_unit_label = "2019 USD million"
    p.flood_exclude_iso3 = {"ATA"}

    # Step 4A must NOT regenerate the damage table. The canonical EUR CSV it
    # reads has all-zero Residential depth curves and is missing PSE/SRB/XKX;
    # the working table was rebuilt by counterfactual/rebuild_damage_table.py
    # from country_landtype_depth_damage.xlsx. Regenerating silently restores
    # the defect that zeroed all urban damage.
    p.flood_skip_damage_tables = True

    p.results = {}
    return p


if __name__ == '__main__':

    # ProjectFlow object
    p = hb.ProjectFlow()
    p.project_name = 'gep_flood'
    p.project_dir = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name)
    p.set_project_dir(p.project_dir)

    # Task tree
    flood_initialize.build_flood_gep_task_tree(p)

    set_flood_paths(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    result = 'Done!'


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    flood_initialize.build_gep_service_task_tree(p)


def run_project(p):
    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)
    p.skip_tasks(p.tasks_to_skip)
    hb.log('Created ProjectFlow object at ' + p.project_dir)
    p.execute()
    return p
