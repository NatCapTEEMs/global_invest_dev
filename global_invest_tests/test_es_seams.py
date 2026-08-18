"""Every ES service must still graft its shock tasks after any change to its module.

These modules are shared: the GEP valuation chains live beside the ES-shock tasks, and more will be
merged in (see the recipe in erosion_initialize's docstring). A merge that clobbers an entry point, or
appends a same-named function over one of ours, breaks the NGFS pipeline SILENTLY -- the task simply
never gets registered, no shock CSV is written, and GTAP runs with a zero where a shock should be.

This asserts the contract each consumer relies on: call add_<es>_tasks(p) and get the expected tasks
on the tree. No science, no rasters, no I/O -- a stub stands in for ProjectFlow.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from global_invest.terrestrial_carbon import terrestrial_carbon_initialize
from global_invest.pollination import pollination_initialize
from global_invest.erosion import erosion_initialize
from global_invest.fisheries import fisheries_initialize


class FakeProjectFlow:
    """Records add_task calls. Stands in for hazelbean's ProjectFlow, which needs a real project dir."""

    def __init__(self, dynamic_es=()):
        self.dynamic_es = list(dynamic_es)
        self.registered = []

    def add_task(self, fn, parent=None, **kwargs):
        self.registered.append(fn.__name__)
        return fn

    def get_path(self, *parts):
        return os.path.join('/nonexistent', *parts)


class TestESSeams(unittest.TestCase):
    """One entry point per service, and it must register the right tasks."""

    SEAMS = {
        'terrestrial_carbon': (terrestrial_carbon_initialize.add_terrestrial_carbon_tasks,
                               'terrestrial_carbon_shock'),
        'pollination':        (pollination_initialize.add_pollination_tasks,
                               'pollination_shock'),
        'erosion':            (erosion_initialize.add_erosion_tasks,
                               'erosion_shock'),
    }

    def test_dynamic_path_registers_the_shock_task(self):
        for es, (seam, expected) in self.SEAMS.items():
            with self.subTest(es=es):
                p = FakeProjectFlow(dynamic_es=[es])
                seam(p)
                self.assertIn(expected, p.registered,
                              f'{es}: add_{es}_tasks did not register {expected}. If a merge replaced '
                              f'the entry point, the NGFS pipeline silently produces no {es} shock.')

    def test_static_path_registers_the_static_task(self):
        for es, (seam, _) in self.SEAMS.items():
            with self.subTest(es=es):
                p = FakeProjectFlow(dynamic_es=[])          # service included, but not dynamic
                seam(p)
                self.assertTrue(any(n.endswith('_static') for n in p.registered),
                                f'{es}: the static fallback did not register. Registered: {p.registered}')

    def test_erosion_registers_its_whole_chain_in_order(self):
        """Erosion is the only multi-task service; the order is the dependency order."""
        p = FakeProjectFlow(dynamic_es=['erosion'])
        erosion_initialize.add_erosion_tasks(p)
        self.assertEqual(p.registered,
                         ['erosion_sdr', 'erosion_upstream',
                          'erosion_exposure', 'erosion_shock'],
                         'the erosion chain must be SDR -> upstream -> exposure -> shock')

    def test_fisheries_is_static_only(self):
        """Marine, so it never reads a SEALS map and has no dynamic branch -- but the seam is the same."""
        p = FakeProjectFlow(dynamic_es=['fisheries'])
        fisheries_initialize.add_fisheries_tasks(p)
        self.assertEqual(p.registered, ['fisheries_shock'])

    def test_every_seam_takes_the_same_signature(self):
        """Consumers call these uniformly: add_<es>_tasks(p, parent=...)."""
        import inspect
        for seam in [terrestrial_carbon_initialize.add_terrestrial_carbon_tasks,
                     pollination_initialize.add_pollination_tasks,
                     erosion_initialize.add_erosion_tasks,
                     fisheries_initialize.add_fisheries_tasks]:
            with self.subTest(seam=seam.__name__):
                params = list(inspect.signature(seam).parameters)
                self.assertEqual(params[:2], ['p', 'parent'],
                                 f'{seam.__name__} must be callable as (p, parent=...)')


if __name__ == '__main__':
    unittest.main()
