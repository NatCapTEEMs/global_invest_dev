"""Resolve every input the erosion cold start reads, and exit non-zero if one is not on this machine.

    python global_invest/erosion/preflight_erosion_inputs.py

A 14-hour `prevention_shares` that dies on a missing table in hour three is the failure this
prevents. ⚠ Every path is resolved the way the tasks resolve it -- through `publish_inputs` -- so
what this reports is what the run will read, not what the CSV happens to say.

⚠⚠ `invest_sdr` is skipped in the cold start, so `erosion_usle_path` and
`erosion_avoided_erosion_path` must resolve to the AUTHOR'S STAGED `revised_feb_13` rasters rather
than to a project directory. That is the whole point of the variant, so it is asserted here rather
than assumed: a path under `intermediate/` means the skip did not hold and the run would produce
the December answer.

⚠ This has to be a FILE inside the repository rather than a heredoc in the sbatch. ProjectFlow
infers where project directories live from the calling script's git repo, and a script piped in on
stdin is not inside one.
"""
import sys

import hazelbean as hb

from global_invest.erosion import erosion_tasks

# The paths the valuation actually opens, from the prevention-shares manifest of a completed run.
INPUTS = ('erosion_usle_path', 'erosion_avoided_erosion_path',
          'erosion_upstream_prevention_share_path', 'erosion_country_boundary_path',
          'erosion_dem_path', 'erosion_yield_stack_path', 'erosion_area_stack_path',
          'erosion_bandmap_csv_path', 'erosion_elasticity_csv_path',
          'erosion_fao_gpv_iso3_csv_path', 'erosion_fao_prices_csv_path',
          'erosion_gdp_csv_path')

# These two must come from base_data, not from a project directory. See the module docstring.
MUST_BE_STAGED = ('erosion_usle_path', 'erosion_avoided_erosion_path')


def main():
    p = hb.ProjectFlow(project_name='gep_erosion_coldstart', run_mode='check')
    erosion_tasks.publish_inputs(p)

    problems = []
    for name in INPUTS:
        path = getattr(p, name, None)
        found = hb.path_exists(path)
        print('[preflight] %-40s %-8s %s' % (name, 'OK' if found else 'MISSING', path))
        if not found:
            problems.append('%s does not resolve' % name)
        elif name in MUST_BE_STAGED and 'intermediate' in str(path):
            problems.append('%s points into a project directory (%s), so invest_sdr was not '
                            'skipped and the run would read its own SDR output' % (name, path))

    if problems:
        sys.exit('[preflight] %d problem(s):\n  %s' % (len(problems), '\n  '.join(problems)))
    print('[preflight] every input resolves, and the SDR rasters are the staged ones')


if __name__ == '__main__':
    main()
