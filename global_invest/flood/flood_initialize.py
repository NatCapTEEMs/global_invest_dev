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
    return build_flood_task_tree(p)


def build_flood_task_tree(p):
    """Default task tree: inputs, SDA, service flow, valuation, then reporting."""
    p.task_prepare_flood_inputs = p.add_task(flood_tasks.task_prepare_flood_inputs)
    p.task_build_sda = p.add_task(flood_tasks.task_build_sda)
    p.task_compute_service_flow = p.add_task(flood_tasks.task_compute_service_flow)
    p.task_compute_flood_damages = p.add_task(flood_tasks.task_compute_flood_damages)
    p.task_generate_maps_and_figures = p.add_task(flood_tasks.task_generate_maps_and_figures)
    return p


def build_flood_calculation_task_tree(p):
    """Calculation-only variant: everything except maps/figures."""
    p.task_prepare_flood_inputs = p.add_task(flood_tasks.task_prepare_flood_inputs)
    p.task_build_sda = p.add_task(flood_tasks.task_build_sda)
    p.task_compute_service_flow = p.add_task(flood_tasks.task_compute_service_flow)
    p.task_compute_flood_damages = p.add_task(flood_tasks.task_compute_flood_damages)
    return p


def build_flood_accounting_task_tree(p):
    """
    Biophysical-accounting-only variant: SDA delineation + SPA-to-SDA service
    flow, no monetary valuation. This is the SEEA-EA physical account on its
    own, useful when the JRC damage tables are not available or not wanted.
    """
    p.task_prepare_flood_inputs = p.add_task(flood_tasks.task_prepare_flood_inputs)
    p.task_build_sda = p.add_task(flood_tasks.task_build_sda)
    p.task_compute_service_flow = p.add_task(flood_tasks.task_compute_service_flow)
    return p


def build_flood_valuation_task_tree(p):
    """
    Valuation-only variant: run the monetary chain against existing SDA
    rasters. Assumes Sections A and B already produced their outputs.
    """
    p.task_compute_flood_damages = p.add_task(flood_tasks.task_compute_flood_damages)
    return p


def build_flood_gep_task_tree(p):
    """
    Paired-counterfactual variant: the only tree that produces an actual
    ecosystem service value rather than gross exposure. Assumes Sections A-C
    have already run.

    ⚠ Until 2026-08-29 no stage reached this tree. The MSI runner's STAGE_TREES
    mapped inputs, sda, flow, valuation, maps, accounting and calculation, so the
    one tree that computes gep_flood = ead_bare - ead_current could only be run by
    asking for `all`, which redoes Sections A and B as well. That is why the
    service had a gross-exposure number and no service value: not because the
    counterfactual damages were missing -- they are on disk per country -- but
    because nothing invoked the step that differences them.
    """
    p.task_compute_flood_gep = p.add_task(flood_tasks.task_compute_flood_gep)
    p.task_generate_maps_and_figures = p.add_task(flood_tasks.task_generate_maps_and_figures)
    return p


def build_flood_results_task_tree(p):
    """Results-only variant: just render maps/figures from an existing run."""
    p.task_generate_maps_and_figures = p.add_task(flood_tasks.task_generate_maps_and_figures)
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
