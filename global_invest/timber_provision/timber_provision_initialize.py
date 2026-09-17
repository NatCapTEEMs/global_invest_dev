"""Timber-provision wiring: GEP task trees (global_invest module layout).

GEP side: build_gep_service_task_tree(p) gives the static country valuation the account reads.

Shock side: consumers with SEALS maps call add_timber_provision_shock_tasks(p, parent=...) after
their downscaling, exactly as they call add_terrestrial_carbon_tasks. It writes
timber_provision_interpolated.csv beside the carbon seam, so forestry's two candidate shocks --
carbon density, which is a proxy, and timber value, which is what the land actually yields -- can be
compared on the same scenarios. Nothing routes it to GTAP; producing the comparison is the point.
"""
from global_invest.timber_provision import timber_provision_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: the committed timber table -> r250 one-row-per-country."""
    p.gep_calculation_task = p.add_task(timber_provision_tasks.gep_calculation)
    p.fuelwood_gep_task = p.add_task(timber_provision_tasks.fuelwood_gep)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.timber_provision_gep_result_task = p.add_task(timber_provision_tasks.gep_result)
    return p


def add_timber_provision_shock_tasks(p, parent=None):
    """The scenario ES shock for timber, to be grafted beside the carbon seam.

    Named and shaped like the other seams' graft functions so it can hang under the same es_shocks
    parent, which is what places its output in intermediate/es_shocks/ alongside
    terrestrial_carbon_interpolated.csv rather than in the project root.

    Separate from build_gep_service_task_tree on purpose: the GEP account wants the static country
    valuation, and only a coupled run wants the per-scenario shock. Grafting both into one tree
    would make every GEP run build SEALS-dependent tasks it has no maps for.
    """
    kwargs = {'parent': parent} if parent is not None else {}
    p.timber_value_density_table_task = p.add_task(
        timber_provision_tasks.timber_value_density_table, **kwargs)
    p.timber_provision_shock_task = p.add_task(
        timber_provision_tasks.timber_provision_shock, **kwargs)
    return p
