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

# ⚠⚠ EVERY `*_path` the service hydrates, read off `p` rather than listed here.
#
# The first version of this file listed the twelve paths a completed run's prevention-shares
# manifest recorded, and it passed on MSI while the job died 45 seconds later on
# `soil/erodibility_30s.tif` -- a file the manifest never mentions because it is an SDR input, and
# one that sits outside `global_invest/erosion/` so the staging rsync missed it too. A hand-picked
# list checks what somebody remembered; this checks what the code will actually resolve.
def _hydrated_paths(p):
    """Every attribute on `p` whose name ends in `_path`, which is what hydration produces."""
    return sorted(name for name in vars(p)
                  if name.endswith('_path') and not name.startswith('_'))

# These two must come from base_data, not from a project directory. See the module docstring.
MUST_BE_STAGED = ('erosion_usle_path', 'erosion_avoided_erosion_path')


def main():
    p = hb.ProjectFlow(project_name='gep_erosion_coldstart', run_mode='check')
    erosion_tasks.publish_inputs(p)

    problems = []
    for name in _hydrated_paths(p):
        path = getattr(p, name, None)
        # A blank or nan row hydrates to None and is not an input; an output path a task will
        # WRITE is not one either, and those live under the project dir rather than base_data.
        if path is None or 'nan' == str(path).lower():
            continue
        found = hb.path_exists(path)
        if not found and str(path).startswith(str(p.project_dir)):
            continue                       # a file this run will generate, not one it reads
        print('[preflight] %-42s %-8s %s' % (name, 'OK' if found else 'MISSING', path))
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
