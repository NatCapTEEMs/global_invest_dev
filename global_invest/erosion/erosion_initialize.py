"""Standard task-tree seam for the erosion-control ES model (mirrors add_pollination/carbon/fisheries).

add_erosion_tasks dispatches STATIC (read the pre-computed erosion dependency table) vs DYNAMIC
(recompute from the SEALS maps via InVEST SDR) on whether 'erosion' is listed in p.dynamic_es.
Consumers set the shared es_shock_* config on p, then call add_erosion_tasks(p) alongside the
other seams. The GEP valuation (InVEST SDR -> on-farm/upstream prevention shares -> per-country
GEP -> maps/figures) is exposed through the template builders (build_gep_service_*); that calculation
is cluster-scale (global InVEST SDR).
"""
from global_invest.erosion import erosion_tasks


# ---------------------------------------------------------------------------------------------
# GEP task trees (folded from global_erosion_gep; template names, cf. terrestrial_carbon).
# ---------------------------------------------------------------------------------------------
def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: InVEST SDR, the upstream prevention share routed from its outputs,
    then the per-country GEP valuation.

    skip_existing=1 on the SDR and routing tasks (dir present -> paths published, work skipped),
    since both cost minutes and are deterministic; the valuation registers plain and skips on its
    registered result, like every service's gep_calculation."""
    p.invest_sdr = p.add_task(erosion_tasks.invest_sdr, skip_existing=1)
    p.upstream_prevention_share = p.add_task(erosion_tasks.upstream_prevention_share, skip_existing=1)
    p.prevention_shares = p.add_task(erosion_tasks.prevention_shares)
    p.gep_calculation = p.add_task(erosion_tasks.gep_calculation)
    return p


def build_gep_service_results_task_tree(p):
    """Results-only: render maps/figures from an existing prevention-share run."""
    p.maps_and_figures = p.add_task(erosion_tasks.maps_and_figures, skip_existing=1)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: SDR + valuation + maps/figures + the results report."""
    p = build_gep_service_calculation_task_tree(p)
    p.maps_and_figures = p.add_task(erosion_tasks.maps_and_figures, skip_existing=1)
    p.erosion_gep_result_task = p.add_task(erosion_tasks.gep_result)
    return p


# ---------------------------------------------------------------------------------------------
# ES-shock wiring (the consumer seam). Everything above builds the GEP task trees; this builds
# the ES-shock one.
# ---------------------------------------------------------------------------------------------
def add_erosion_tasks(p, parent=None):
    """Graft the erosion ES-shock tasks onto p, dispatching STATIC vs DYNAMIC on p.dynamic_es.

    STATIC (the default): read the pre-computed dependency table -> erosion_shock_static.
    DYNAMIC ('erosion' in p.dynamic_es): recompute per scenario x year from our
    SEALS maps -- SDR -> upstream (D8) -> exposure -> shock. The shock task emits the
    shock the same two ways as carbon/pollination, as ABSOLUTE differences of the productivity-share
    level (the level is already a fraction of output, so an absolute change IS the productivity %;
    dividing would give a change-of-a-fraction): contemporaneous (scn_Y - base_Y) and fixed-base
    (scn_Y - base_0). Resolution follows the erosion_native_resolution row (false -> 6.45 km, true -> native 300 m).
    Dynamic build tracked in #26.

    Caller sets only the shared es_shock_* config (see run_ngfs_pnas STEP 6). Everything erosion-specific
    defaults in the task (output CSV) or here (the SDR inputs, out of base_data).
    """
    dynamic = 'erosion' in getattr(p, 'dynamic_es', [])
    if not dynamic:
        p.erosion_shock_task = p.add_task(erosion_tasks.erosion_shock_static, parent=parent)
        return p
    # The SDR data references live in es_parameters (erosion rows), hydrated by publish_inputs
    # in each task -- a builder constructs the tree and configures nothing.
    # skip_existing=1 on the three EXPENSIVE steps makes the chain resumable: InVEST SDR and the D8
    # routing each cost minutes per scenario-year and their outputs are deterministic, so re-running them
    # on every relaunch wastes the whole iteration. The final shock task deliberately does NOT skip --
    # it is cheap, it is the step still being iterated on, and it must pick up any change to the
    # coefficients, the crop-sector map or the method selector.
    # ⚠ Consequence: a task killed MID-WRITE leaves a dir that now looks complete and will be skipped.
    # If a run dies inside sdr/upstream/exposure, delete that task's dir before relaunching.
    p.erosion_sdr_task      = p.add_task(erosion_tasks.erosion_sdr, parent=parent, skip_existing=1)
    p.erosion_upstream_task = p.add_task(erosion_tasks.erosion_upstream, parent=parent, skip_existing=1)
    p.erosion_exposure_task = p.add_task(erosion_tasks.erosion_exposure, parent=parent, skip_existing=1)
    p.erosion_shock_task    = p.add_task(erosion_tasks.erosion_shock, parent=parent)
    return p
