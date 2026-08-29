"""Standard task-tree seam for the flood-control ES model (mirrors erosion/carbon/pollination).

add_flood_tasks is the consumer seam: a pipeline grafts it alongside the other services and
configures nothing flood-specific. The GEP valuation -- SDA delineation, the SPA-to-SDA service
flow, the paired counterfactual and the per-country valuation -- is exposed through the template
builders (build_gep_service_*), which is what a caller managing several services reaches for.

The flood-specific builders below the template ones exist because the account has stages that no
other service does: the counterfactual amplification is built from rainfall and flow direction
rather than from the hazard maps, so it survives a change of hazard product and is worth being
able to skip. build_flood_accounting_task_tree stops before the valuation for that reason.
"""

import hazelbean as hb

from global_invest.flood import flood_tasks


def add_flood_tasks(p):
    """
    Standard entry point (per Justin's global_invest convention: each service
    module exposes at least one add_<service>_tasks(p)) for building the full
    flood-control task tree: input preparation -> SDA delineation ->
    SPA-to-SDA service flow -> monetary valuation -> maps & figures.
    This is what run_flood.py calls by default.
    """
    return build_gep_service_task_tree(p)


def build_flood_task_tree(p):
    """The full pipeline: inputs, SDA, service flow, valuation, the counterfactual GEP, reporting.

    Every variant below builds this same tree and disables what it does not want. Omitting a task
    would also omit the paths it publishes, and the later sections read those: the valuation task
    names the per-country directory the GEP task writes its degraded scenarios into, and the SDA
    task names the directory the service flow reads.
    """
    p.task_prepare_flood_inputs = p.add_task(flood_tasks.task_prepare_flood_inputs)
    p.task_build_sda = p.add_task(flood_tasks.task_build_sda)
    p.task_compute_service_flow = p.add_task(flood_tasks.task_compute_service_flow)
    p.task_compute_flood_damages = p.add_task(flood_tasks.task_compute_flood_damages)
    p.task_compute_flood_gep = p.add_task(flood_tasks.task_compute_flood_gep)
    p.task_generate_maps_and_figures = p.add_task(flood_tasks.task_generate_maps_and_figures)
    return p


def build_flood_calculation_task_tree(p):
    """Calculation only: everything except maps and figures."""
    build_flood_task_tree(p)
    p.skip_tasks(['task_generate_maps_and_figures'])
    return p


def build_flood_accounting_task_tree(p):
    """The SEEA-EA physical account on its own: SDA delineation and the SPA-to-SDA service flow,
    no monetary valuation. Useful when the JRC damage tables are not available or not wanted."""
    build_flood_task_tree(p)
    p.skip_tasks(['task_compute_flood_damages', 'task_compute_flood_gep',
                  'task_generate_maps_and_figures'])
    return p


def build_flood_valuation_task_tree(p):
    """Valuation only, against SDA rasters an earlier run produced."""
    build_flood_task_tree(p)
    p.skip_tasks(['task_prepare_flood_inputs', 'task_build_sda', 'task_compute_service_flow',
                  'task_compute_flood_gep', 'task_generate_maps_and_figures'])
    return p


def build_flood_gep_task_tree(p):
    """The paired counterfactual, the only tree that produces a service value rather than gross
    exposure: gep_flood = ead_bare - ead_current, plus the maps."""
    build_flood_task_tree(p)
    p.skip_tasks(['task_prepare_flood_inputs', 'task_build_sda', 'task_compute_service_flow',
                  'task_compute_flood_damages'])
    return p


def build_flood_results_task_tree(p):
    """Maps and figures from an existing run."""
    build_flood_task_tree(p)
    p.skip_tasks(['task_prepare_flood_inputs', 'task_build_sda', 'task_compute_service_flow',
                  'task_compute_flood_damages', 'task_compute_flood_gep'])
    return p


# ---------------------------------------------------------------------------------------------
# Template builders. Every service exposes these three under the same names, so a caller managing
# the whole set does not have to know which service it is invoking.
# ---------------------------------------------------------------------------------------------
def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: inputs, SDA, service flow, then the counterfactual valuation.

    No maps or figures. This is the tree a pipeline runs when it wants the numbers and will do its
    own reporting."""
    return build_flood_gep_task_tree(p)


def build_gep_service_results_task_tree(p):
    """Results-only: render maps and figures from an existing valuation run."""
    return build_flood_results_task_tree(p)


def build_gep_service_task_tree(p):
    """Full GEP run: calculation, then maps and figures."""
    return build_flood_task_tree(p)
