import hazelbean as hb

from global_invest.coastal_protection import coastal_protection_initialization


if __name__ == '__main__':
    p = coastal_protection_initialization.create_projectflow()
    coastal_protection_initialization.build_gep_service_calculation_task_tree(p)
    coastal_protection_initialization.initialize_project_inputs(p)

    hb.log(
        'Created ProjectFlow object at '
        + p.project_dir
        + '\n    from script '
        + p.calling_script
        + '\n    with base_data set at '
        + p.base_data_dir
    )
    p.execute()
