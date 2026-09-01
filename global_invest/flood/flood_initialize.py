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

    Every variant below builds this same tree and turns off what it does not want, by setting run=0
    on the task itself. Omitting a task would also omit the paths it publishes, and the later
    sections read those: the valuation task names the per-country directory the GEP task writes its
    degraded scenarios into, and the SDA task names the directory the service flow reads. Setting
    the flag directly rather than by name means a typo raises here instead of leaving an expensive
    task quietly enabled.
    """
    p.prepare_flood_inputs_task = p.add_task(flood_tasks.task_prepare_flood_inputs)
    p.build_sda_task = p.add_task(flood_tasks.task_build_sda)
    p.compute_service_flow_task = p.add_task(flood_tasks.task_compute_service_flow)
    p.compute_flood_damages_task = p.add_task(flood_tasks.task_compute_flood_damages)
    p.compute_flood_gep_task = p.add_task(flood_tasks.task_compute_flood_gep)
    return p


def build_flood_calculation_task_tree(p):
    """Calculation only: the whole chain, which no longer draws anything."""
    build_flood_task_tree(p)
    return p


def build_flood_accounting_task_tree(p):
    """The SEEA-EA physical account on its own: SDA delineation and the SPA-to-SDA service flow,
    no monetary valuation. Useful when the JRC damage tables are not available or not wanted."""
    build_flood_task_tree(p)
    p.compute_flood_damages_task.run = 0
    p.compute_flood_gep_task.run = 0
    return p


def build_flood_valuation_task_tree(p):
    """Valuation only, against SDA rasters an earlier run produced."""
    build_flood_task_tree(p)
    p.prepare_flood_inputs_task.run = 0
    p.build_sda_task.run = 0
    p.compute_service_flow_task.run = 0
    p.compute_flood_gep_task.run = 0
    return p


def build_flood_gep_task_tree(p):
    """The paired counterfactual, the only tree that produces a service value rather than gross
    exposure: gep_flood = ead_bare - ead_current."""
    build_flood_task_tree(p)
    p.prepare_flood_inputs_task.run = 0
    p.build_sda_task.run = 0
    p.compute_service_flow_task.run = 0
    p.compute_flood_damages_task.run = 0
    return p


def build_flood_results_task_tree(p):
    """Render the report from an existing run, the way every other service's results tree does.

    It used to mean the four publication figures, which nothing read. gep_result was defined all
    along and wired into no tree, so flood was the one service whose results tree did not render
    its results page.
    """
    p.flood_gep_result_task = p.add_task(flood_tasks.gep_result)
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
    """Results-only: render the results page from an existing valuation run."""
    return build_flood_results_task_tree(p)


def build_gep_service_task_tree(p):
    """Full GEP run: the whole calculation chain."""
    return build_flood_task_tree(p)
