"""Erosion from an empty project directory, to settle condition 10.

The question this answers is narrow: does the erosion pipeline reproduce its published
$18,240,930,373 when nothing is carried over from a previous run?

⚠⚠ `invest_sdr` is SKIPPED, and that is the whole point of this file rather than an optimisation.
The task reassigns `erosion_usle_path` to its own output whenever it runs:

    if p.run_this or hb.path_exists(usle):
        p.erosion_usle_path = usle

So a run that executes it stops reading the author's staged `usle_2019_revised_feb_13.tif` --
which is what the published number is computed from -- and reads its own SDR output instead. On
2026-09-02 a cold start that let it run produced $12,099,377,565, a third of the total missing,
with a clean exit and no error anywhere in the log. The valuation was never at fault: our own SDR
output reproduces the author's DECEMBER raster exactly on every cell he computed, and the pipeline
is configured against his FEBRUARY one.

Skipping is the correct treatment, not a workaround. Running InVEST SDR globally is a
`gep_preprocess`-shaped stage by the house convention -- "the output of a preprocess task is an
input to the actual model... we promote the data output by a preprocess task to the base_data_dir"
-- and its output is already promoted, as the `revised_feb_13` rasters in base_data. A pipeline
that recomputes its own staged input is not reproducing anything; it is producing something else.

Run it on a cluster. `prevention_shares` took 818 minutes on the laptop.
"""
import hazelbean as hb

from global_invest.erosion.run_erosion import run_project


if __name__ == '__main__':

    p = hb.ProjectFlow(project_name='gep_erosion_coldstart', run_mode='full')

    # The one thing this variant changes. Everything else -- the tree, the inputs, the thresholds
    # -- is the full run's, so a difference in the total is a difference in reproducibility and
    # not in configuration.
    p.tasks_to_skip = ['invest_sdr']

    run_project(p)

    print('COLD START PROJECT DIR: ' + p.project_dir)
    print('Compare the total against the published $18,240,930,373 over 250 countries.')
