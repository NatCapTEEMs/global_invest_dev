"""Standard task-tree seam for the erosion-control ES model (mirrors add_pollination/carbon/fisheries).

add_erosion_tasks dispatches STATIC (read the pre-computed erosion dependency table) vs DYNAMIC (recompute
from the SEALS maps via InVEST SDR, #26) on the number of SEALS map years. Consumers (ngfs_pnas) set the
erosion_shock_* inputs on p, then call add_erosion_tasks(p) alongside the other ES seams.
"""
from global_invest.erosion import erosion_tasks


def add_erosion_tasks(p, parent=None):
    """Graft the erosion ES-shock tasks onto p, dispatching by data availability.

    STATIC (default, <2 SEALS map years): read the pre-computed dependency table ->
    task_compute_erosion_shock.
    DYNAMIC (>=2 SEALS map years in p.seals_years, the erosion_* inputs wired): recompute per scenario x year from our
    SEALS maps -- SDR -> upstream (D8) -> prevention shares -> valuation. The valuation emits the
    shock the same two ways as carbon/pollination, as ABSOLUTE differences of the productivity-share
    level (the level is already a fraction of output, so an absolute change IS the productivity %;
    dividing would give a change-of-a-fraction): contemporaneous (scn_Y - base_Y) and fixed-base
    (scn_Y - base_0). Resolution follows p.modality (local -> 6.45 km, sc/msi -> native 300 m).
    Dynamic build tracked in #26.

    Caller sets on p before calling: erosion_shock_scenarios, erosion_shock_base_year,
    erosion_shock_end_year, erosion_shock_output_path (and, for dynamic, p.seals_years + the
    erosion input rasters/tables).
    """
    # gate on the number of SEALS map years (p.seals_years is a space-separated string, e.g. "2020 2050")
    dynamic = len(str(getattr(p, 'seals_years', '') or '').split()) >= 2
    if not dynamic:
        p.compute_erosion_shock_task = p.add_task(erosion_tasks.task_compute_erosion_shock, parent=parent)
        return p
    p.erosion_sdr_task        = p.add_task(erosion_tasks.task_erosion_sdr, parent=parent)
    p.erosion_upstream_task   = p.add_task(erosion_tasks.task_erosion_upstream, parent=parent)
    p.erosion_prevention_task = p.add_task(erosion_tasks.task_erosion_prevention, parent=parent)
    p.erosion_valuation_task  = p.add_task(erosion_tasks.task_erosion_valuation, parent=parent)
    return p
