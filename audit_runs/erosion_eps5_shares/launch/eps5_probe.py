"""The publish-only pass of the EPS-5 run, shared by the run file and the preflights.

Every task is skipped, so each only publishes its paths, and the pass then checks what
prevention_shares will read: the pinned settings, the staged rasters and the original stacks.
"""
import importlib.util
import os

SHARED_LAUNCH = os.path.join('audit_runs', 'erosion_spam_repair', 'shares', 'launch', 'shares_launch.py')


def load_shared_launch(snapshot):
    """The committed gates of the repaired-stack package, from the library snapshot."""
    spec = importlib.util.spec_from_file_location('shares_launch', os.path.join(snapshot, SHARED_LAUNCH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_paths(sl, base):
    """The attribute each input must resolve to: staged SDR rasters, original stacks and band map, under base data."""
    expected = {attribute: os.path.join(base, sl.SDR_REF_DIR, name) for attribute, name in sl.SDR.items()}
    expected['erosion_area_stack_path'] = os.path.join(base, 'global_invest', 'erosion', 'spam2020_harvested_area_stack_TA.tif')
    expected['erosion_yield_stack_path'] = os.path.join(base, 'global_invest', 'erosion', 'spam2020_yield_stack_TA.tif')
    expected['erosion_bandmap_csv_path'] = os.path.join(base, sl.BANDMAP_REF)
    return expected


def configure(p, base, settings):
    """What this run sets before hydration, so it wins over es_parameters: base data, the high threshold and the labels."""
    p.base_data_dir = base
    high = next(r for r in settings if r['setting'] == 'threshold_high')
    setattr(p, high['task_attribute'], float(high['value']))
    p.erosion_run_tag = 'eps5_onfarm_upstream_combined_library'
    p.erosion_scenario_name = 'EPS-5 - On-farm + Upstream (decomposed)'


def publish_only_pass(hb, run_erosion, sl, eg, project_dir, base, settings):
    """Run the tree with every task skipped and check the inputs and settings. Returns the task lists."""
    probe = hb.ProjectFlow(project_dir=project_dir, run_mode='check')
    run_erosion.build_task_tree(probe)
    every_task, all_but_shares = sl.task_lists(probe)
    p = hb.ProjectFlow(project_dir=project_dir, run_mode='check')
    configure(p, base, settings)
    p.tasks_to_skip = every_task
    p = run_erosion.run_project(p)
    for attribute, path in expected_paths(sl, base).items():
        if os.path.normpath(str(getattr(p, attribute))) != os.path.normpath(path):
            raise eg.Mismatch('publish-only pass: %s resolves to %s, not %s' % (attribute, getattr(p, attribute), path))
    eg.check_task_settings(p, settings)
    if os.path.exists(os.path.join(project_dir, 'intermediate', 'prevention_shares')):
        raise eg.Mismatch('the publish-only pass ran prevention_shares')
    return every_task, all_but_shares
