"""Standard task-tree seam for the erosion-control ES model (mirrors add_pollination/carbon/fisheries).

STATIC for now (reads the pre-computed erosion dependency table); the DYNAMIC InVEST-SDR version
(Nfamara's global_erosion_gep, re-run on each SEALS scenario x year map) is the heavy upgrade tracked in
#26. Consumers (ngfs_pnas) set the erosion_shock_* inputs on p, then call add_erosion_tasks(p) alongside
the other ES seams.
"""
from global_invest.erosion import erosion_tasks


def add_erosion_tasks(p, parent=None):
    """Graft the (static) erosion ES-shock task onto p.

    Caller sets on p before calling: erosion_shock_scenarios, erosion_shock_base_year,
    erosion_shock_end_year, erosion_shock_output_path. Writes the per-region 8-sector shock CSV at
    erosion_shock_output_path.
    """
    p.compute_erosion_shock_task = p.add_task(erosion_tasks.task_compute_erosion_shock, parent=parent)
    return p
