"""Shared GEP utilities and cross-service conventions.

AGGREGATE ON r250, NEVER ON r264. r250 is one row per country (250 countries and territories) and is the
only correct key for sums, totals and the GEP CSV. r264 splits large countries into sub-regions -- China
x6, India x6, France/Turkey/UK/Pakistan x2 -- so summing r264 rows that carry a per-country value counts
those countries once per sub-region. If you need an r264 crosswalk (e.g. for a choropleth), filter it with
``ee_r264_label == iso3_r250_label`` so it collapses to one canonical row per country (the five provisioning
services do this; terrestrial_carbon/coastal_carbon instead sum on r250 and keep r264 only for the map).

Diagnostic -- if a national total is inflated by this, the gap decomposes exactly into China x6, India x6
and the four 2-way splits. A gap of any other shape is a legitimate sum, not this bug. This is how the bug
reached terrestrial_carbon + coastal_carbon and not the other five GEP services.
"""

# The scenario-name mapping that used to live here (our label -> a service's frozen-table label) was
# NGFS-specific and has moved to the consumer: each service's static shock task now defaults to identity
# and warns loudly if a scenario is absent from its table, and ngfs_pnas sets p.<service>_scenario_map
# with its two non-identity entries. A general library should not hardcode one project's scenario names.

def initialize_country_paths(p, simplified='300sec'):
    """Shared country-boundary references every GEP service needs: the r264 correspondence
    (csv + gpkg + simplified gpkg, all as get_path reference paths) and the loaded df_countries.
    Each service's initialize_paths calls this and then adds only its service-specific inputs
    (this block used to be pasted into every module). A service with a different aggregation
    surface (e.g. coastal's marine r566) overrides the aliases after calling.
    """
    import pandas as pd

    p.df_countries_csv_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.csv')
    p.gdf_countries_vector_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.gpkg')
    p.gdf_countries_vector_simplified_path = p.get_path('cartographic', 'ee', f'ee_r264_simplified{simplified}.gpkg')

    p.df_countries = pd.read_csv(p.df_countries_csv_path)
    # The GDFs stay as path strings; hb.read_vector converts on demand.
    p.gdf_countries = p.gdf_countries_vector_path
    p.gdf_countries_simplified = p.gdf_countries_vector_simplified_path
    return p


def summarize_raster_by_region(value_raster_path, region_boundary_path, out_path, year, id_column):
    """Per-polygon total / mean / pixel count of a value raster, via hb.zonal_statistics_flex.

    Shared by every GEP service that aggregates a value raster to regions (terrestrial_carbon's
    stock chain and shock zones; pollination's USD value raster). Promoted here on its second
    caller. Heavy imports stay inside so importing utilities stays light.

    id_column: the vector column holding a unique integer id per polygon ('ee_r264_id' for the
    country valuation, 'ee_r50_aez18_id' for the shock zones). The output emits the boundary's own
    id so consumers join on a value, never on row position; background zone 0 and empty zones are
    dropped.
    """
    import os
    import geopandas as gpd
    import hazelbean as hb

    regions = gpd.read_file(region_boundary_path)
    # zones_raster_data_type=5 (Int32) so ids past 255 don't saturate (r264 runs to 264, r50xAEZ
    # higher); all_touched=True matches the old per-polygon masking. Returns a frame indexed by
    # zone id, with a zone 0 = everything outside every polygon.
    zone_ids_raster = os.path.splitext(out_path)[0] + '_zone_ids.tif'
    stats = hb.zonal_statistics_flex(
        value_raster_path, region_boundary_path, zone_ids_raster_path=zone_ids_raster,
        id_column_label=id_column, zones_raster_data_type=5, all_touched=True,
        stats_to_retrieve='sums_counts', assert_projections_same=False, verbose=False)
    stats = stats[(stats.index != 0) & (stats['counts'] > 0)]   # drop background + empty zones

    df = regions.assign(_zid=regions[id_column].astype('int64')).merge(
        stats, left_on='_zid', right_index=True, how='right').drop(columns=['_zid', 'geometry'])
    df = df.rename(columns={'sums': 'total', 'counts': 'count'})
    df['mean'] = df['total'] / df['count']
    df['year'] = year
    if 'ee_r50_aez18_id' in df.columns:
        df['region_id'] = df['ee_r50_aez18_id'].astype(int)
    df.to_csv(out_path, index=False)
    print(f"Summary written to: {out_path}")


def render_service_results(p):
    """Render each service's <service>_results.qmd via quarto, streaming output. Shared by every
    module's gep_result task (was duplicated with drift in terrestrial_carbon and coastal_carbon;
    promoted here -- keeping the superset of both -- when pollination became the third copy).

    Behaviour (the coastal variant's, the most complete): a missing qmd RAISES; bibliography/CSL
    sidecars next to the qmd are copied so citeproc resolves them in cur_dir; the render runs with
    the repo root on PYTHONPATH so the qmd can import global_invest; a non-zero quarto exit RAISES.
    The qmd copy and sidecars are removed afterwards so nobody edits a copy expecting the source
    to change.
    """
    import os
    import sys
    import subprocess
    import hazelbean as hb

    os.environ['QUARTO_PYTHON'] = sys.executable
    module_root = os.path.dirname(os.path.abspath(__file__))
    for service_label in list(p.results.keys()):
        results_qmd_path = os.path.join(module_root, service_label, f'{service_label}_results.qmd')
        results_qmd_project_path = os.path.join(p.cur_dir, f'{service_label}_results.qmd')
        if not os.path.exists(results_qmd_path):
            raise FileNotFoundError(f"Results QMD template not found: {results_qmd_path}")
        hb.create_directories(results_qmd_project_path)
        hb.path_copy(results_qmd_path, results_qmd_project_path)

        # Copy bibliography/CSL/template assets sitting next to the qmd source.
        qmd_src_dir = os.path.dirname(results_qmd_path)
        copied_sidecar_paths = []
        for sidecar_name in os.listdir(qmd_src_dir):
            if sidecar_name.endswith(('.bib', '.csl', '.bst', '.yml', '.yaml')):
                if sidecar_name in ('_quarto.yml', '_quarto.yaml'):
                    continue
                src = os.path.join(qmd_src_dir, sidecar_name)
                if os.path.isfile(src):
                    dst = os.path.join(p.cur_dir, sidecar_name)
                    hb.path_copy(src, dst)
                    copied_sidecar_paths.append(dst)

        env = os.environ.copy()
        repo_root = os.path.dirname(module_root)
        env['GLOBAL_INVEST_REPO_ROOT'] = repo_root
        existing_pythonpath = env.get('PYTHONPATH')
        env['PYTHONPATH'] = (repo_root if not existing_pythonpath
                             else repo_root + os.pathsep + existing_pythonpath)

        cmd = ['quarto', 'render', results_qmd_project_path, '--verbose']
        hb.log('Running quarto command: %s' % ' '.join(cmd))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env=env, text=True, bufsize=1, universal_newlines=True)
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
                sys.stdout.flush()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        hb.path_remove(results_qmd_project_path)
        for sidecar in copied_sidecar_paths:
            hb.path_remove(sidecar)


def resolve_raw_scenario(scenario_labels, scenario_map, our_scn, service, log=print):
    """Map our scenario name to the label its dependency table uses; shared by every ES static shock task.

    scenario_map defaults to identity (our_scn -> [our_scn]); the first candidate present in
    scenario_labels wins. If none is present, warn loudly -- naming the labels that ARE present -- and
    return None so the caller skips the scenario rather than emitting a silent zero into GTAP. log is the
    caller's logger (hb.log or print).
    """
    candidates = scenario_map.get(our_scn, [our_scn])
    raw = next((c for c in candidates if c in scenario_labels), None)
    if raw is None:
        log("  WARNING %s shock: scenario '%s' (tried %s) has no row in the dependency table "
            "(present: %s) -- skipping, so GTAP gets NO %s shock for it. Set p.%s_scenario_map "
            "if the table uses a different label."
            % (service, our_scn, candidates, sorted(set(scenario_labels)), service, service))
    return raw


def required_base_scenario(p, service):
    """Return p.es_shock_base_scenario, raising if the consumer never set it.

    The library carries NO default spelling: the frozen tables spell the nature-off baseline
    differently per service (baseline_ignore_dependencies vs baseline_ignore_damages), so any
    baked-in default is wrong for someone -- before this, seven call sites defaulted it three
    different ways. The consumer names its base once (e.g. from its scenarios CSV) and
    resolve_base_scenario matches it against each table's actual labels.
    """
    base_scn = getattr(p, 'es_shock_base_scenario', None)
    if not base_scn:
        raise ValueError(
            '%s shock: p.es_shock_base_scenario is not set. The consumer must name its base '
            'scenario explicitly (the library carries no default spelling).' % service)
    return base_scn


def resolve_base_scenario(scenario_labels, scenario_map, base_scn, service, log=print):
    """Resolve the BASE scenario name against a dependency table's labels, via the same candidate
    mechanism as resolve_raw_scenario (the consumer's scenario_map supplies alternate spellings,
    e.g. the frozen tables spell the nature-off baseline both 'baseline_ignore_dependencies' and
    'baseline_ignore_damages').

    Unlike a data scenario, an unresolvable base is FATAL rather than skippable: it is the
    subtraction reference, so without it every shock in the table is meaningless -- an exact-match
    miss here previously yielded an empty base, an empty output, and a silent GTAP zero.
    """
    raw = resolve_raw_scenario(scenario_labels, scenario_map, base_scn, service, log=log)
    if raw is None:
        raise ValueError(
            "%s shock: BASE scenario '%s' (tried %s) has no row in the dependency table "
            "(present: %s). The base is the subtraction reference -- refusing to compute shocks "
            "without it. Set p.%s_scenario_map with the table's spelling."
            % (service, base_scn, scenario_map.get(base_scn, [base_scn]),
               sorted(set(scenario_labels)), service))
    return raw


# example utility function

def convert_currency(value, from_currency, to_currency, exchange_rate):
    """
    Convert a value from one currency to another using the provided exchange rate.
    """

    pass


# ---------------------------------------------------------------------------------------------
# Silent-failure guard for the ES shock tables.
#
# A shock table that is WRONG looks exactly like one that is right: well-formed CSV, plausible
# numbers, no exception. Nothing downstream re-derives it, so a bad table reaches GTAP as a shock and
# is never questioned. Three real failures had that shape:
#
#   scenario silently dropped  a label absent from a frozen dependency table makes the building loop
#                              `continue`, so the scenario contributes NO rows and GTAP runs with a
#                              zero where a shock should be.
#   value contamination        SEALS once fed a coefficient of 1811.0 into an allocation because
#                              float('0_18_1_1') parses rather than raising. A shock two orders of
#                              magnitude out is the same class of thing and is equally quiet.
#   duplicated rows            a per-entity value replicated across sub-rows then summed. This is how
#                              the national GEP total came out 23.5% high.
#
# So state what must be true of the table, and check it where it is built.
# ---------------------------------------------------------------------------------------------

# Contamination bound, NOT a plausibility bound. Measured legitimate maxima (2026-08-15, from the
# frozen dependency tables under each static task's own formula, scenario - base at end year):
# terrestrial_carbon 84.0 (low_demand), pollination 133.1 (net_zero), erosion 8.0; dynamic-path
# observed maxima are far lower (carbon 16.8 on the ZAF test AOI). 500 clears the largest legitimate
# value ~3.7x and still catches order-of-magnitude contamination (a SEALS coefficient once reached
# 1811 because float('0_18_1_1') parses). Tighter per-service ceilings are a domain-owner call:
# carbon/pollination are unbounded percent-change ratios (a near-zero base-year denominator can make
# a large value legitimate), erosion/fisheries are bounded percentage points (<=8 / <=2 upstream).
SHOCK_ABS_MAX = 500.0


def assert_shock_table_sound(df, requested_scenarios, label, abs_max=SHOCK_ABS_MAX):
    """Raise if the ES shock table `df` violates what must hold before it is written.

    Called immediately before to_csv in each task_compute_<es>_shock*, so the failure surfaces where
    the table was built rather than as a wrong number in a GTAP solve days later.

    Checks, in the order they would bite:
      1. every requested scenario produced rows      -- a dropped scenario is a zero shock
      2. no duplicate rows on the identifying keys   -- duplicates multiply under any later sum
      3. |shock_pct| <= abs_max                      -- an absurd value is contamination, not signal

    Raises ValueError naming the offending scenarios/values. Deliberately does NOT warn-and-continue:
    a warning in a long log is how these get missed in the first place.
    """
    problems = []

    got = set(df['scenario'].unique()) if len(df) and 'scenario' in df.columns else set()
    missing = [s for s in requested_scenarios if s not in got]
    if missing:
        problems.append(
            'produced NO rows for %d of %d requested scenario(s): %s. Each is a zero shock in GTAP. '
            'Check the scenario map against the labels actually present in the dependency table.'
            % (len(missing), len(requested_scenarios), sorted(missing)))

    keys = [c for c in ('ENDW', 'ACTS', 'REG', 'scenario', 'year') if c in df.columns]
    if len(keys) >= 3 and len(df):
        n_dup = int(df.duplicated(subset=keys).sum())
        if n_dup:
            example = df[df.duplicated(subset=keys, keep=False)].head(2)[keys].to_dict('records')
            problems.append('has %d duplicate row(s) on %s -- any sum over these multiplies. e.g. %s'
                            % (n_dup, keys, example))

    if 'shock_pct' in df.columns and len(df):
        worst = float(df['shock_pct'].abs().max())
        if worst > abs_max:
            bad = df.loc[df['shock_pct'].abs() > abs_max].head(2)
            cols = [c for c in ('scenario', 'REG', 'year', 'shock_pct') if c in bad.columns]
            problems.append('has |shock_pct| up to %.6g, above the %.6g sanity bound -- that is '
                            'contamination, not signal. e.g. %s'
                            % (worst, abs_max, bad[cols].to_dict('records')))

    if problems:
        raise ValueError('%s shock table is unsound:\n  - %s' % (label, '\n  - '.join(problems)))
    return True
