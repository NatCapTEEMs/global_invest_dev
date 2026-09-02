"""Resolve every input the NTFP run reads, and exit non-zero if one is not on this machine.

Run before the job proper, so a 4.3 GB roads shapefile that was never staged costs seconds to
discover rather than the queue wait plus however far the warps get before reaching it.

    python global_invest/ntfp/preflight_ntfp_inputs.py

⚠ This has to be a FILE inside the repository rather than a heredoc in the sbatch. ProjectFlow
infers where project directories live from the calling script's git repo, and a script piped in
on stdin is not inside one, so the constructor raises before any input is checked.
"""
import sys

import hazelbean as hb

from global_invest import utilities

INPUTS = ('gep_lulc_input_path', 'ntfp_ndvi_mean_path', 'ntfp_roads_vector_path',
          'ntfp_rivers_path', 'ntfp_value_per_ha_path')


def main():
    p = hb.ProjectFlow(project_name='gep_ntfp', run_mode='check')
    # The same four calls the tasks make in publish_inputs, in the same order, so what this
    # resolves is what the run will resolve. ⚠ Writing them out here is what caught
    # initialize_pyramid_paths missing from the task's own publish_inputs: ha_per_cell was read
    # by the warps and published by nobody, and the run would have died on the first one.
    utilities.hydrate_es_config(p, 'ntfp', log=hb.log)
    utilities.hydrate_es_parameters(p, 'ntfp', log=hb.log)
    utilities.initialize_country_paths(p)
    utilities.initialize_pyramid_paths(p)

    missing = []
    for name in INPUTS:
        path = getattr(p, name, None)
        found = hb.path_exists(path)
        print('[preflight] %-24s %-8s %s' % (name, 'OK' if found else 'MISSING', path))
        if not found:
            missing.append(name)

    # The pyramid the whole run is now on. Its absence would surface much later, as a warp with
    # no template, so it is named here with the rest.
    print('[preflight] %-24s %-8s %s'
          % ('ha_per_cell_10sec_path', 'OK' if hb.path_exists(p.ha_per_cell_10sec_path)
             else 'MISSING', p.ha_per_cell_10sec_path))
    if not hb.path_exists(p.ha_per_cell_10sec_path):
        missing.append('ha_per_cell_10sec_path')

    if missing:
        sys.exit('[preflight] %d input(s) do not resolve on this machine: %s'
                 % (len(missing), ', '.join(missing)))
    print('[preflight] every input resolves')


if __name__ == '__main__':
    main()
