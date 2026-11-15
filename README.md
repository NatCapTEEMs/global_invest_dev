# Global InVEST Dev repo

Public development repo for the global_invest models used in the Gross Ecosystem Service (GEP) project. Upon publication, will be moved to global_invest (non _dev).

## Contributing

Create a directory for your service, along with run, tasks and functions python files in the directory. See [How to Contribute](https://justinandrewjohnson.com/earth_economy_devstack/version_control.html) for more.

## Running a service

Each service is a folder under `global_invest/` with four files: `run_<service>.py` (the entry
point), `<service>_tasks.py` (reads inputs, writes results), `<service>_functions.py` (the
maths; opens nothing), and `test_<service>.py`. Configuration lives in
`global_invest/input_template/es_parameters.csv` and `es_config.csv`, never in code.

Requirements: the `hazelbean_env` conda environment (hazelbean and its geospatial stack), and a
`base_data` directory holding the staged inputs; `~/.config/hazelbean/machine.env` names its
location (default `~/Files/base_data`). Paths in es_parameters are relative references resolved
against base_data — nothing in the repo points at a machine-specific location.

Run a service from the repository root, with the import pinned to this checkout:

    PYTHONPATH=$(pwd) python global_invest/<service>/run_<service>.py

The first run creates a project directory OUTSIDE the repository (at
`<repo_parent>/projects/gep_<service>`) and fills it task by task; completed outputs are skipped
on re-run. The per-country result lands in
`projects/gep_<service>/intermediate/gep_calculation/gep_by_country_base_year.csv`.

Verify an installation by running the test suite from the repository root (both directories, or
the cross-checks are skipped):

    PYTHONPATH=$(pwd) python -m pytest global_invest global_invest_tests -q

The tests pin every service's published totals against staged references, so a passing suite on
a fresh clone means the code in this repository reproduces the account's numbers from base_data.
