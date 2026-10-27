"""Register country-level water valuations and the optional report."""
from global_invest.water_supply import water_supply_tasks


def add_water_supply_tasks(p, parent=None):
    """Add separate hydropower, domestic and agricultural calculations."""
    p.hydropower_rent_task = p.add_task(water_supply_tasks.hydropower_rent, parent=parent)
    p.water_use_components_task = p.add_task(water_supply_tasks.water_use_components, parent=parent)
    p.agriculture_water_task = p.add_task(water_supply_tasks.agriculture_water, parent=parent)
    p.water_supply_gep_calculation_task = p.add_task(water_supply_tasks.gep_calculation, parent=parent)
    return p


def build_gep_service_calculation_task_tree(p, parent=None):
    return add_water_supply_tasks(p, parent=parent)


def build_gep_service_task_tree(p, parent=None):
    add_water_supply_tasks(p, parent=parent)
    p.water_supply_gep_result_task = p.add_task(water_supply_tasks.gep_result, parent=parent)
    return p
