"""Standard task-tree seam for the erosion-control ES model (mirrors add_pollination/carbon/fisheries).

add_erosion_tasks dispatches STATIC (read the pre-computed erosion dependency table) vs DYNAMIC (recompute
from the SEALS maps via InVEST SDR, #26) on whether 'erosion' is listed in p.dynamic_es. Consumers
(ngfs_pnas) set the shared es_shock_* config on p, then call add_erosion_tasks(p) alongside the other seams.

THE GEP VALUATION IS FOLDED IN (2026-08-16, per the recipe that used to live here): the
prevention-share valuation from `global_erosion_gep` -- InVEST SDR -> on-farm/upstream prevention
shares -> per-country GEP -> maps/figures. Its functions/tasks were appended to
erosion_functions.py / erosion_tasks.py (zero name collisions; the one duplicate, SPAM_ALIAS_MAP,
kept OUR corrected exact-FAO-name version), erosion_utils.py arrived as a new file, and every
import-time side effect was made lazy (output-dir mkdirs, natcap imports, the root-logging/gdal
env block -- see the fold separator in erosion_functions.py). Its builders are exposed below under
the template names (build_gep_service_*); its two-line add_erosion_tasks alias was dropped --
add_erosion_tasks here stays the ES-shock seam. ⚠ The GEP chain is cluster-scale (global InVEST
SDR): folded and import-clean, NOT yet number-verified -- see the tracker.
"""
from global_invest.erosion import erosion_tasks


def initialize_paths(p):
    """Resolve the erosion GEP inputs on p via get_path REFERENCE paths (the configure_* functions
    read these attrs at run time; their built-in defaults point at the source repo's cluster layout
    and are never used once this ran). Section-A (InVEST SDR) inputs are fully staged in base_data
    at the 6.45 km analysis grid. The three section-B valuation CSVs (FAO GPV / FAO prices / WB GDP
    2019) and the upstream-prevention rasters are the service owner's artifacts, not yet in
    base_data: resolved tolerantly so section A runs; the valuation crashes loudly until they are
    staged (requested via the erosion submission).
    """
    import os
    # Section A -- InVEST SDR.
    p.erosion_dem_path = p.get_path('global_invest', 'erosion', 'global_dem_reproj.tif')
    p.erosion_sdr_input_dir = os.path.dirname(p.erosion_dem_path)
    p.erosion_lulc_path = p.get_path('global_invest', 'erosion', 'lulc_esa_2019_reproj_6p45km.tif')
    p.erosion_biophysical_table_path = p.get_path('global_invest', 'erosion', 'expanded_biophysical_table_gura.csv')
    p.erosion_erodibility_path = p.get_path('soil', 'erodibility_30s.tif')
    p.erosion_erosivity_path = p.get_path('soil', 'erosivity_30s.tif')
    p.erosion_watersheds_path = p.get_path('global_invest', 'erosion', 'hybas_global_lev06_v1c.gpkg')
    # Section B -- prevention shares + valuation.
    p.erosion_yield_stack_path = p.get_path('global_invest', 'erosion', 'spam2020_yield_stack_TA.tif')
    p.erosion_area_stack_path = p.get_path('global_invest', 'erosion', 'spam2020_harvested_area_stack_TA.tif')
    p.erosion_bandmap_csv_path = p.get_path('global_invest', 'erosion', 'spam2020_bandmap.csv')
    p.erosion_elasticity_csv_path = p.get_path('global_invest', 'erosion', 'elasticity_crops_fao_revised.csv')
    p.erosion_elevation_path = p.erosion_dem_path
    p.erosion_country_boundary_path = p.get_path('cartographic', 'ee', 'ee_r250.gpkg')
    p.erosion_fao_gpv_iso3_csv_path = p.get_path('global_invest', 'erosion', 'faostat_gpv_2019_iso3.csv',
                                                 raise_error_if_fail=False)
    p.erosion_fao_prices_csv_path = p.get_path('global_invest', 'erosion', 'faostat_prices_2019_completed_revised.csv',
                                               raise_error_if_fail=False)
    p.erosion_gdp_csv_path = p.get_path('global_invest', 'erosion', 'worldbank_gdp_2019.csv',
                                        raise_error_if_fail=False)
    return p



def add_erosion_tasks(p, parent=None):
    """Graft the erosion ES-shock tasks onto p, dispatching STATIC vs DYNAMIC on p.dynamic_es.

    STATIC (the default): read the pre-computed dependency table -> task_compute_erosion_shock_static.
    DYNAMIC ('erosion' in p.dynamic_es): recompute per scenario x year from our
    SEALS maps -- SDR -> upstream (D8) -> exposure -> shock. The shock task emits the
    shock the same two ways as carbon/pollination, as ABSOLUTE differences of the productivity-share
    level (the level is already a fraction of output, so an absolute change IS the productivity %;
    dividing would give a change-of-a-fraction): contemporaneous (scn_Y - base_Y) and fixed-base
    (scn_Y - base_0). Resolution follows p.modality (local -> 6.45 km, sc/msi -> native 300 m).
    Dynamic build tracked in #26.

    Caller sets only the shared es_shock_* config (see run_ngfs_pnas STEP 6). Everything erosion-specific
    defaults in the task (output CSV) or here (the SDR inputs, out of base_data).
    """
    dynamic = 'erosion' in getattr(p, 'dynamic_es', [])
    if not dynamic:
        p.compute_erosion_shock_task = p.add_task(erosion_tasks.task_compute_erosion_shock_static, parent=parent)
        return p
    # DYNAMIC-only inputs. Unlike the other services, the SDR chain needs a dozen rasters/tables, so they
    # resolve here rather than in every consumer's run file. Already in base_data:
    p.erosion_erosivity_path = p.get_path('soil', 'erosivity_30s.tif')
    p.erosion_erodibility_path = p.get_path('soil', 'erodibility_30s.tif')
    p.erosion_watersheds_path = p.get_path('global_invest', 'erosion', 'hybas_global_lev06_v1c.gpkg')
    p.erosion_biophysical_table_path = p.get_path('global_invest', 'erosion', 'expanded_biophysical_table_gura.csv')
    # (zone boundary: erosion reads the shared p.region_boundary_path, defaulted in-task like carbon/pollination)
    # Provisioned into base_data/global_invest/sdr/ (erosion-specific 6.45 km grid + DEM + country
    # boundary + SPAM2020 yield/area stacks + bandmap + crop-coefficient table):
    p.erosion_analysis_grid_path = p.get_path('global_invest', 'erosion', 'erosion_analysis_grid_6p45km.tif')
    p.erosion_dem_path = p.get_path('global_invest', 'erosion', 'global_dem_reproj.tif')
    p.erosion_country_boundary_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.gpkg')  # standard per-country set (as carbon)
    p.erosion_yield_stack_path = p.get_path('global_invest', 'erosion', 'spam2020_yield_stack_TA.tif')
    p.erosion_area_stack_path = p.get_path('global_invest', 'erosion', 'spam2020_harvested_area_stack_TA.tif')
    p.erosion_bandmap_csv_path = p.get_path('global_invest', 'erosion', 'spam2020_bandmap.csv')
    p.erosion_elasticity_csv_path = p.get_path('global_invest', 'erosion', 'elasticity_crops_fao_revised.csv')
    # skip_existing=1 on the three EXPENSIVE steps makes the chain resumable: InVEST SDR and the D8
    # routing each cost minutes per scenario-year and their outputs are deterministic, so re-running them
    # on every relaunch wastes the whole iteration. The final shock task deliberately does NOT skip --
    # it is cheap, it is the step still being iterated on, and it must pick up any change to the
    # coefficients, the crop-sector map or the method selector.
    # ⚠ Consequence: a task killed MID-WRITE leaves a dir that now looks complete and will be skipped.
    # If a run dies inside sdr/upstream/exposure, delete that task's dir before relaunching.
    p.erosion_sdr_task      = p.add_task(erosion_tasks.task_erosion_sdr, parent=parent, skip_existing=1)
    p.erosion_upstream_task = p.add_task(erosion_tasks.task_erosion_upstream, parent=parent, skip_existing=1)
    p.erosion_exposure_task = p.add_task(erosion_tasks.task_erosion_exposure, parent=parent, skip_existing=1)
    p.erosion_shock_task    = p.add_task(erosion_tasks.task_erosion_shock, parent=parent)
    return p


# ---------------------------------------------------------------------------------------------
# GEP task trees (folded from global_erosion_gep; template names, cf. terrestrial_carbon).
# ---------------------------------------------------------------------------------------------
def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: InVEST SDR run + prevention-share per-country GEP valuation.
    skip_existing=1 on the SDR task (dir present -> paths published, work skipped); the valuation
    registers plain and skips on its registered result, like every service's gep_calculation."""
    p.task_run_invest_sdr = p.add_task(erosion_tasks.task_run_invest_sdr, skip_existing=1)
    p.task_compute_prevention_shares = p.add_task(erosion_tasks.task_compute_prevention_shares)
    return p


def build_gep_service_results_task_tree(p):
    """Results-only: render maps/figures from an existing prevention-share run."""
    p.task_generate_maps_and_figures = p.add_task(erosion_tasks.task_generate_maps_and_figures, skip_existing=1)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: SDR + valuation + maps/figures."""
    p = build_gep_service_calculation_task_tree(p)
    p.task_generate_maps_and_figures = p.add_task(erosion_tasks.task_generate_maps_and_figures, skip_existing=1)
    return p
