"""NTFP from an empty project directory, to settle condition 10 for this service.

The question is narrow: does the pipeline reproduce its published $13,852,553,748 over 196
countries when nothing is carried over from a previous run? `run_mode='full'` mints a fresh
timestamped project directory, so every raster -- the warps, the burned lines, the reach, the
country ids -- is built from the staged inputs alone, with `assert_raster_has_data` proving each
one holds data as it is built.
"""
import hazelbean as hb

from global_invest.ntfp.run_ntfp import run_project


if __name__ == '__main__':

    p = hb.ProjectFlow(project_name='gep_ntfp_coldstart', run_mode='full')
    run_project(p)

    print('COLD START PROJECT DIR: ' + p.project_dir)
    print('Compare the total against the published $13,852,553,748 over 196 countries.')
