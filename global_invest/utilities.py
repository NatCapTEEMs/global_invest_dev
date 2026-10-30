"""Shared GEP utilities and cross-service conventions.

AGGREGATE ON r250, NEVER ON r264. r250 is one row per country (250 countries and territories) and is the
only correct key for sums, totals and the GEP CSV. r264 splits large countries into sub-regions -- China
x6, India x6, France/Turkey/UK/Pakistan x2 -- so summing r264 rows that carry a per-country value counts
those countries once per sub-region. If you need an r264 crosswalk (e.g. for a choropleth), filter it with
``ee_r264_label == iso3_r250_label`` so it collapses to one canonical row per country (the five provisioning
services do this; terrestrial_carbon/coastal_carbon instead sum on r250 and keep r264 only for the map).

Diagnostic -- if a national total is inflated by this, the gap decomposes exactly into China x6, India x6
and the four 2-way splits. A gap of any other shape is a legitimate sum, not this double count.
"""
from osgeo import gdal
from osgeo import osr
import filecmp
import json
import mapclassify
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shutil
import subprocess
import tempfile
import sys
import urllib.request
import zipfile


# A general library does not hardcode one project's scenario names: each service's static shock
# task defaults to identity (plus the nature-off spelling alias below) and warns loudly when a
# scenario is absent from its table; a consumer supplies p.<service>_scenario_map for its own naming.

def initialize_country_paths(p, simplified='300sec'):
    """Shared country-boundary references every GEP service needs: the r264 correspondence
    (csv + gpkg + simplified gpkg, all as get_path reference paths) and the loaded df_countries.
    Called from each service's publish_inputs; the service then adds only its service-specific
    inputs.
    """

    if getattr(p, 'df_countries', None) is not None:
        return p          # country world already published (or caller-set): touch nothing
    # Two different jobs, two different homes. WHICH surface a service aggregates on (marine
    # r566, land r264) legitimately varies per service, so that lives in the es_config cell
    # gep_regions_input_path, read by the aggregating tasks. THIS function serves the other job,
    # shared by every service identically: at the end, collapse whatever was aggregated into ONE
    # ROW PER COUNTRY -- and that collapse always goes through the r264 correspondence, the
    # table that stops split countries (China x6, India x6) being counted once per sub-region.
    # No run ever legitimately wants a different table for that step, so it is code, not a cell.
    ref = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.gpkg')
    p.gdf_countries_vector_path = ref
    p.df_countries_csv_path = ref.replace('.gpkg', '.csv')
    simplified_ref = ref.replace('_correspondence.gpkg', f'_simplified{simplified}.gpkg')
    p.gdf_countries_vector_simplified_path = simplified_ref if hb.path_exists(simplified_ref) else ref
    p.df_countries = hb.df_read(p.df_countries_csv_path)
    # The GDFs stay as path strings; hb.read_vector converts on demand.
    p.gdf_countries = p.gdf_countries_vector_path
    p.gdf_countries_simplified = p.gdf_countries_vector_simplified_path
    return p


# Kept as its parts rather than a joined constant, because os is imported inside the functions
# here so that importing utilities stays light.
HA_PER_CELL_10SEC_REF_PARTS = ('pyramids', 'ha_per_cell_10sec.tif')


def initialize_pyramid_paths(p):
    """The shared pyramid rasters, published under one name for every service that needs them.

    Hectares per cell at 10 arc-seconds is fixed geometry, not a run choice, so like the country
    correspondence it is code rather than a cell in a CSV. Every service reads it from here under
    one attribute name, so no service drifts onto a different grid than another.

    Args:
        p (ProjectFlow): the project, which gains ha_per_cell_10sec_path.

    Returns:
        ProjectFlow: the same project.
    """

    p.ha_per_cell_10sec_path = p.get_path(
        os.path.join(*HA_PER_CELL_10SEC_REF_PARTS))
    return p


def summarize_raster_by_region(value_raster_path, region_boundary_path, out_path, year, id_column):
    """Per-polygon total / mean / pixel count of a value raster, via hb.zonal_statistics_flex.

    Shared by every GEP service that aggregates a value raster to regions (terrestrial_carbon's
    stock calculation and shock zones; pollination's USD value raster). Heavy imports stay inside so
    importing utilities stays light.

    id_column: the vector column holding a unique integer id per polygon ('ee_r264_id' for the
    country valuation, 'ee_r50_aez18_id' for the shock zones). The output emits the boundary's own
    id so consumers join on a value, never on row position; background zone 0 and empty zones are
    dropped.
    """

    regions = gpd.read_file(region_boundary_path)
    # zones_raster_data_type=5 (Int32) so ids past 255 don't saturate (r264 runs to 264, r50xAEZ
    # higher). Returns a frame indexed by zone id, with a zone 0 = everything outside every polygon.
    #
    # all_touched=False is the centre rule: a cell goes to whichever zone covers its centre point,
    # so it has exactly one owner and the answer does not depend on anything but geography. Under
    # all_touched=True every zone whose outline touches a cell claims it, and because the claims
    # are burned into one id raster the zone rasterised LAST keeps it, so a country's number moves
    # with its row order in the boundary file. Nothing is double counted either way; what is
    # arbitrary is the split between neighbours.
    #
    # Both boundaries have 100% of their vertices on the 10 arc-second grid, so a value raster on
    # that grid (terrestrial_carbon) has no partly covered cells and the two rules agree exactly. A
    # raster on a coarser grid (pollination, 0.05 degree) does differ between the rules, by up to
    # tens of percent for single countries.
    #
    # hazelbean reuses a zone ids raster it finds on disk, so a cached raster keeps the rule it was
    # built under until it is deleted.
    zone_ids_raster = os.path.splitext(out_path)[0] + '_zone_ids.tif'
    stats = hb.zonal_statistics_flex(
        value_raster_path, region_boundary_path, zone_ids_raster_path=zone_ids_raster,
        id_column_label=id_column, zones_raster_data_type=5, all_touched=False,
        stats_to_retrieve='sums_counts', assert_projections_same=False, verbose=False)
    # The ONLY test of emptiness is that nothing was measured in the zone: counts == 0. A zone whose
    # measured quantity comes to exactly zero is kept, with its zero, because for anything that later
    # forms a regional total or a ratio the alternative is fatal -- an absent row and a real zero are
    # then the same thing, and a zone that genuinely lost all its forest cannot be told from a zone
    # nobody measured. Zone 0 is the background outside every polygon.
    stats = stats[(stats.index != 0) & (stats['counts'] > 0)]
    measured_zeros = sorted(stats.index[stats['sums'] == 0].tolist())
    if measured_zeros:
        print('  %s: %d zone(s) hold valid pixels whose total is exactly zero, retained as measured '
              'zeros rather than dropped: %s' % (os.path.basename(out_path), len(measured_zeros), measured_zeros))

    df = regions.assign(_zid=regions[id_column].astype('int64')).merge(
        stats, left_on='_zid', right_index=True, how='right').drop(columns=['_zid', 'geometry'])
    df = df.rename(columns={'sums': 'total', 'counts': 'count'})
    df['mean'] = df['total'] / df['count']
    # Carried so a consumer can act on the distinction without re-deriving it, and so a zero in a
    # finished table says on its own face whether it was measured.
    df['measured_zero'] = df['total'] == 0
    df['year'] = year
    if 'ee_r50_aez18_id' in df.columns:
        df['region_id'] = df['ee_r50_aez18_id'].astype(int)
    df.to_csv(out_path, index=False)
    print(f"Summary written to: {out_path}")


def render_service_results(p):
    """Render each service's <service>_results.qmd via quarto, streaming output. Shared by every
    module's gep_result task.

    Behaviour: a missing qmd RAISES; bibliography/CSL sidecars next to the qmd are copied so
    citeproc resolves them in cur_dir; the render runs with the repo root on PYTHONPATH so the
    qmd can import global_invest; a non-zero quarto exit RAISES. The qmd copy and sidecars are
    removed afterwards so nobody edits a copy expecting the source to change.
    """

    # A cluster node has no quarto, and the render is a report over results that already
    # exist -- so its absence downgrades to a loud skip rather than failing a job whose
    # calculation succeeded. The report renders on any machine that has quarto, from the same
    # results.
    if shutil.which('quarto') is None:
        hb.log('render_service_results: quarto is not on PATH, so the results page is NOT '
               'rendered. The calculation results are unaffected; render on a machine with '
               'quarto.')
        return

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
        # Name the project being reported on. Without this every results page falls back to a
        # hardcoded home-directory path, so a report rendered from a test project or another
        # machine would show the stable project's numbers instead of this run's.
        env['PROJECTFLOW_ROOT'] = p.project_dir

        # --to html is explicit because quarto renders EVERY format a qmd declares when it is
        # omitted, and a qmd that also declares pdf or docx (coastal carbon) then needs a working
        # TeX installation. The results report is the HTML page in every service; the other
        # formats stay available to anyone who asks quarto for them directly.
        cmd = ['quarto', 'render', results_qmd_project_path, '--to', 'html', '--verbose']
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


# The frozen dependency tables spell the SAME nature-off baseline two ways (carbon:
# baseline_ignore_dependencies; pollination/erosion: baseline_ignore_damages). That is
# table-internal vocabulary, not one project's naming, so the equivalence is normalized here at
# the point of reading -- the library's own stated principle -- instead of re-encoded in every
# consumer's scenario_map. An explicit scenario_map entry still wins.
NATURE_OFF_SPELLINGS = ('baseline_ignore_dependencies', 'baseline_ignore_damages')


def resolve_raw_scenario(scenario_labels, scenario_map, our_scn, service, log=print):
    """Map our scenario name to the label its dependency table uses; shared by every ES static shock task.

    scenario_map defaults to identity (our_scn -> [our_scn]) -- except that the two nature-off
    spellings are mutual aliases by default, since the frozen tables themselves disagree on it.
    The first candidate present in scenario_labels wins. If none is present, warn loudly --
    naming the labels that ARE present -- and return None so the caller skips the scenario rather
    than emitting a silent zero into GTAP. log is the caller's logger (hb.log or print).
    """
    default_candidates = [our_scn] + [s for s in NATURE_OFF_SPELLINGS
                                      if our_scn in NATURE_OFF_SPELLINGS and s != our_scn]
    candidates = scenario_map.get(our_scn, default_candidates)
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
    baked-in default is wrong for someone. The consumer names its base once (e.g. from its
    scenarios CSV) and resolve_base_scenario matches it against each table's actual labels.
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
    miss here would yield an empty base, an empty output, and a silent GTAP zero.
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




# ---------------------------------------------------------------------------------------------
# Silent-failure guard for the ES shock tables.
#
# A shock table that is WRONG looks exactly like one that is right: well-formed CSV, plausible
# numbers, no exception. Nothing downstream re-derives it, so a bad table reaches GTAP as a shock and
# is never questioned. Three failures have that shape:
#
#   scenario silently dropped  a label absent from a frozen dependency table makes the building loop
#                              `continue`, so the scenario contributes NO rows and GTAP runs with a
#                              zero where a shock should be.
#   value contamination        a value orders of magnitude out, e.g. from float('0_18_1_1'), which
#                              parses to 1811.0 rather than raising. It is equally quiet.
#   duplicated rows            a per-entity value replicated across sub-rows then summed, which
#                              inflates every total built on it.
#
# So state what must be true of the table, and check it where it is built.
# ---------------------------------------------------------------------------------------------

# Contamination bound, NOT a plausibility bound. Measured legitimate maxima (from the frozen
# dependency tables under each static task's own formula, scenario - base at end year):
# terrestrial_carbon 84.0 (low_demand), pollination 133.1 (net_zero), erosion 8.0; dynamic-path
# observed maxima are far lower (carbon 16.8 on the ZAF test AOI). 500 clears the largest legitimate
# value ~3.7x and still catches order-of-magnitude contamination (float('0_18_1_1') parses to
# 1811). Tighter per-service ceilings are a domain-owner call:
# carbon/pollination are unbounded percent-change ratios (a near-zero base-year denominator can make
# a large value legitimate), erosion/fisheries are bounded percentage points (<=8 / <=2 upstream).
SHOCK_ABS_MAX = 500.0


# GTAP land endowments: exactly the AEZS set, and the land block of ENDW, in the model sets.har
# (AEZ1-AEZ18; ENDW adds the labour types, Capital and NatRes). The r50xAEZ18 shock boundary also
# carries an AEZ0 residual polygon that no GTAP element corresponds to, so a shock row on it cannot
# be consumed whatever its value. Membership, not magnitude, is what makes such a row invalid:
# an AEZ0 row can carry |shock| well under the contamination bound.
GTAP_LAND_ENDW = tuple("AEZ%d" % i for i in range(1, 19))


def filter_to_model_domain(df, output_path, label, valid_endw=GTAP_LAND_ENDW, endw_col='ENDW', log=print):
    """Drop shock rows GTAP has no element for, keeping them beside the table for diagnostics.

    Applied immediately before assert_shock_table_sound, so validation judges only what will actually
    be delivered. An out-of-domain row is neither a small value to tolerate nor a large one to
    reject: it is a row the model cannot consume, and it should not reach a bound check at all.

    Only ENDW is filtered. AEZ1-AEZ18 is fixed across GTAP-AEZ aggregations and can be asserted here;
    the ACTS and REG sets are aggregation-specific (s26/r50 for the current consumer) and pinning them in
    a shared science library would bake one consumer aggregation into every other.

    Args:
        df (pd.DataFrame): the assembled shock table.
        output_path (str): where the table itself is going. Excluded rows are written beside it as
            <stem>_out_of_domain.csv; a stale one is removed when nothing is excluded, so the file
            never outlives the run that produced it.
        label (str): service name, for the log line.
        valid_endw (tuple): the endowment elements GTAP accepts.
        endw_col (str): the endowment column. A table without one (fisheries is region-only) passes
            through untouched.
        log (callable): where to report what was excluded.

    Returns:
        pd.DataFrame: the rows GTAP can consume.
    """
    if endw_col not in df.columns or not len(df):
        return df
    excluded_path = "%s_out_of_domain.csv" % os.path.splitext(output_path)[0]
    in_domain = df[endw_col].astype(str).str.strip().isin(set(valid_endw))
    excluded = df[~in_domain]
    if not len(excluded):
        if os.path.exists(excluded_path):
            os.remove(excluded_path)
        return df
    excluded.to_csv(excluded_path, index=False)
    log('  %s shock: held back %d of %d row(s) outside the GTAP endowment domain (%s) -> %s'
        % (label, len(excluded), len(df), ', '.join(sorted(excluded[endw_col].astype(str).unique())),
           excluded_path))
    return df[in_domain].reset_index(drop=True)


# How many (scenario, year, zone) keys a failure message prints in full before it summarises.
# Generous, because the whole point is that the reader can act on the message without going back to
# the data: at 470 zones and 28 scenario-years a complete failure is ~13,000 keys, and a cap well
# above one scenario-year's worth keeps every ordinary failure fully enumerated.
KEY_ENUMERATION_LIMIT = 600


def _enumerate_keys(keys, limit=KEY_ENUMERATION_LIMIT):
    """Every key, or a bounded sample that SAYS it is one, plus the distinct zones and scenario-years.

    A count paired with an unlabelled excerpt reads as the whole set, so a truncated list says how
    many keys it leaves out. The distinct zone and scenario-year lists are always complete because
    they are bounded by the run's own dimensions, and they are what a reader actually acts on.
    """
    keys = list(keys)
    zones = sorted({str(k[2]) for k in keys})
    scenario_years = sorted({(str(k[0]), k[1]) for k in keys})
    if len(keys) <= limit:
        listed = '  all %d: %s' % (len(keys), keys)
    else:
        listed = ('  first %d of %d (%d NOT shown): %s'
                  % (limit, len(keys), len(keys) - limit, keys[:limit]))
    return ('%s\n  distinct zones (%d, complete): %s\n  distinct scenario-years (%d, complete): %s'
            % (listed, len(zones), zones, len(scenario_years), scenario_years))


def assert_zonal_coverage_complete(frame, expected_scenarios, expected_years, expected_zones, label,
                                   value_column='total', scenario_column='scenario',
                                   year_column='year', zone_column='region_id',
                                   intentional_exclusions=None, log=None):
    """Raise unless every expected scenario, year and zone carries a finite total.

    A zonal summary that silently loses a scenario-year, or carries NaN where a map failed to
    produce a value, reaches the economic model as a zero shock rather than as an error. Both
    failures look identical in a finished table, which is why they are asserted here rather than
    inspected later.

    Exclusions that are MEANT to be absent -- a zone the model does not cover, a scenario-year a
    variant deliberately omits -- are declared by the caller and reported on their own, so the
    distinction between "we chose to leave this out" and "this went missing" survives in the log.

    Args:
        frame (pd.DataFrame): the assembled zonal totals, one row per scenario, year and zone.
        expected_scenarios (iterable): every scenario that must appear.
        expected_years (iterable): every anchor year that must appear.
        expected_zones (iterable): every zone or country id that must appear.
        label (str): the service or table name, for the message.
        value_column (str): the column that must be present and finite.
        scenario_column, year_column, zone_column (str): the identifying columns.
        intentional_exclusions (iterable or None): (scenario, year, zone) tuples that are expected to
            be absent. Reported separately; never counted as missing.
        log (callable or None): where the exclusion report goes; print when None.

    Raises:
        ValueError: naming the missing combinations and the non-finite values, with examples.
    """
    import numpy as np
    import pandas as pd

    say = log or print
    key_columns = [scenario_column, year_column, zone_column]
    missing_columns = [c for c in key_columns + [value_column] if c not in frame.columns]
    if missing_columns:
        raise ValueError('%s: required columns absent: %s' % (label, missing_columns))
    if frame[key_columns].isna().any().any():
        raise ValueError('%s: missing identifying keys' % label)
    normalized_keys = frame[key_columns].copy()
    normalized_keys[scenario_column] = normalized_keys[scenario_column].astype(str)
    years_numeric = pd.to_numeric(normalized_keys[year_column], errors='coerce')
    if (~np.isfinite(years_numeric) | (years_numeric != np.floor(years_numeric))).any():
        raise ValueError('%s: year keys must be finite integers' % label)
    normalized_keys[year_column] = years_numeric.astype(int)
    duplicates = normalized_keys.duplicated(keep=False)
    if duplicates.any():
        raise ValueError('%s: duplicate scenario/year/zone keys: %s' %
                         (label, normalized_keys.loc[duplicates].head(5).to_dict('records')))
    declared = {(str(s), int(y), z) for s, y, z in (intentional_exclusions or ())}
    expected = {(str(s), int(y), z) for s in expected_scenarios for y in expected_years
                for z in expected_zones}
    required = expected - declared

    if len(frame):
        present = {(str(r[scenario_column]), int(r[year_column]), r[zone_column])
                   for _, r in frame[[scenario_column, year_column, zone_column]].iterrows()}
    else:
        present = set()

    problems = []
    missing = sorted(required - present, key=lambda t: (t[0], t[1], str(t[2])))
    if missing:
        problems.append('%s: %d of %d required (scenario, year, zone) totals are absent. A missing '
                        'total is a zero shock in the economic model, not a gap.\n%s'
                        % (label, len(missing), len(required), _enumerate_keys(missing)))

    if len(frame) and value_column in frame.columns:
        values = pd.to_numeric(frame[value_column], errors='coerce')
        bad = frame.loc[~np.isfinite(values)]
        if len(bad):
            example = bad.head(3)[[scenario_column, year_column, zone_column, value_column]].to_dict('records')
            problems.append('%s: %d row(s) carry a non-finite %s. NaN and inf reach the model as a '
                            'zero or an overflow rather than as a failure. e.g. %s'
                            % (label, len(bad), value_column, example))

    unexpected_exclusions = sorted(declared - expected, key=lambda t: (t[0], t[1], str(t[2])))
    if unexpected_exclusions:
        problems.append('%s: %d declared exclusion(s) are not in the expected set at all, so the '
                        'declaration is stale or wrong.\n%s'
                        % (label, len(unexpected_exclusions), _enumerate_keys(unexpected_exclusions)))

    if declared:
        say('  %s: %d (scenario, year, zone) total(s) excluded by declaration, not missing: %s%s'
            % (label, len(declared), sorted(declared)[:5],
               ' ...' if len(declared) > 5 else ''))

    if problems:
        raise ValueError('\n'.join(problems))
    say('  %s: %d of %d required zonal totals present and finite' % (label, len(required), len(required)))
    return True


def assert_shock_table_sound(df, requested_scenarios, label, abs_max=SHOCK_ABS_MAX, column='shock_pct'):
    """Raise if the ES shock table `df` violates what must hold before it is written.

    Called immediately before to_csv in each <es>_shock / <es>_shock_static, so the failure surfaces where
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

    # The bound applies to the measure the run exports (`column`); a table's other measures are
    # informational and can legitimately blow up (a contemporaneous ratio over a baseline that
    # went to nearly zero in a small zone).
    if column in df.columns and len(df):
        worst = float(df[column].abs().max())
        if worst > abs_max:
            bad = df.loc[df[column].abs() > abs_max].head(2)
            cols = [c for c in ('scenario', 'REG', 'year', column) if c in bad.columns]
            problems.append('has |%s| up to %.6g, above the %.6g sanity bound -- that is contamination, '
                            'not signal. e.g. %s' % (column, worst, abs_max, bad[cols].to_dict('records')))

    if problems:
        raise ValueError('%s shock table is unsound:\n  - %s' % (label, '\n  - '.join(problems)))
    return True


ANCHOR_INTERPOLATION = 'pchip'   # the selected specification


def anchored_series(x, anchor_x, anchor_y, method=ANCHOR_INTERPOLATION):
    """Expand anchor-year values to every year, shape-preserving by default.

    The selected specification is PCHIP: shape-preserving piecewise cubic Hermite interpolation
    between the 2023, 2030, 2040 and 2050 anchors. It retains the anchor values exactly and cannot
    overshoot between adjacent anchors, which piecewise-linear interpolation also does -- but linear
    interpolation of a LEVEL makes the implied annual increment jump discontinuously at the anchor
    years, which is the artefact PCHIP removes. Measured on the bounded carbon stocks, the mean
    jump at 2030/2040 falls from 2.9e6 to 6.3e5 Mg C/yr while cumulative change is unaltered.

    Every module interpolates anchors through this one function, so the specification changes in
    one place.

    Args:
        x (array-like): the years to evaluate.
        anchor_x (array-like): the anchor years, ascending, at least two.
        anchor_y (array-like): the value at each anchor.
        method (str): 'pchip' (the specification) or 'linear' (retained for sensitivity
            comparison and for quantities whose anchors must not be smoothed).

    Returns:
        numpy.ndarray: one value per entry of `x`.

    Raises:
        ValueError: on an unknown method, or fewer than two anchors.
    """
    import numpy as np
    # None means "the specification", so a caller that threads an optional method through does not
    # have to know what the default is.
    if method is None:
        method = ANCHOR_INTERPOLATION
    anchor_x = np.asarray(anchor_x, dtype=float)
    anchor_y = np.asarray(anchor_y, dtype=float)
    if anchor_x.size < 2:
        raise ValueError('anchored_series needs at least two anchors, got %d' % anchor_x.size)
    if method == 'linear':
        return np.interp(x, anchor_x, anchor_y)
    if method != 'pchip':
        raise ValueError("anchored_series method must be 'pchip' or 'linear', got %r" % method)
    # Two anchors make PCHIP the straight line between them, so scipy is not needed and the
    # dependency is not imposed on a run that only has endpoints (fisheries' ramp, for one).
    if anchor_x.size == 2:
        return np.interp(x, anchor_x, anchor_y)
    from scipy.interpolate import PchipInterpolator
    x = np.asarray(x, dtype=float)
    values = PchipInterpolator(anchor_x, anchor_y, axis=0)(x)
    # Evaluating the cubic Hermite at a knot is not bit-exact: a 0.0 anchor can come back as a tiny
    # negative, which a strict non-negativity check rejects. The anchors are data, not something to
    # re-derive, so they are restored exactly.
    for position, anchor in enumerate(anchor_x):
        values[x == anchor] = anchor_y[position]
    return values


def _signature(settings, inputs, light_inputs=()):
    """The signature dict: settings by value, inputs by content, light inputs by size and mtime."""
    fingerprints = dict(inputs)
    for path in light_inputs:
        fingerprints[str(path)] = file_fingerprint(path, content=False)
    return {'settings': settings, 'inputs': fingerprints}


def _service_settings_and_inputs(p, service):
    """The service's own es_parameters values and a content fingerprint of each `_path` among them."""
    prefix = service + '_'
    settings, inputs = {}, {}
    for name, value in sorted(vars(p).items()):
        if not name.startswith(prefix) or callable(value):
            continue
        if name.endswith('_path') and isinstance(value, str):
            inputs[name] = file_fingerprint(value)
        elif isinstance(value, (str, int, float, bool, list, tuple, type(None))):
            settings[name] = value
    return settings, inputs


def outputs_reuse_reason(outputs, signature, signature_path):
    """Why `outputs` cannot be reused, or None when they can.

    An existence check answers "is there an output?" when the question is "was it made from what
    we are running now?". Gating on existence alone silently republishes the old answer after a
    change; recomputing because nothing recorded what produced a finished table repeats hours of
    work. The recorded signature answers the real question.

    Args:
        outputs (list): the files reuse would republish.
        signature (dict): what would produce them now, from `_signature`.
        signature_path (str): where the previous run recorded its signature.

    Returns:
        str | None: the reason to recompute, naming what differs, or None to reuse.
    """
    missing = [o for o in outputs if not hb.path_exists(o)]
    if missing:
        return 'there is no %s to reuse' % os.path.basename(missing[0])
    if not hb.path_exists(signature_path):
        return ('the outputs carry no signature, so what produced them is unknown; they predate '
                'this check')
    try:
        old = json.loads(open(signature_path, encoding='utf-8').read())
    except Exception:
        return 'the signature beside the outputs cannot be read'
    settings, inputs = signature['settings'], signature['inputs']
    changed = sorted(k for k in set(old.get('settings', {})) | set(settings)
                     if old.get('settings', {}).get(k) != settings.get(k))
    moved = sorted(k for k in set(old.get('inputs', {})) | set(inputs)
                   if old.get('inputs', {}).get(k) != inputs.get(k))
    if changed or moved:
        return 'the signature changed in %s' % ', '.join(changed + moved)
    return None


def write_outputs_signature(signature, signature_path):
    """Record what produced a set of outputs, so the next run can tell whether it may reuse them."""
    hb.write_to_file(json.dumps(signature, indent=2, sort_keys=True, default=str), signature_path)


def reuse_reason(p, service, outputs, signature_name='run_signature.json', light_inputs=()):
    """Why a service task cannot reuse what is on disk, or None when it can.

    The signature is the service's own es_parameters values plus a fingerprint of every input path
    among them, so any configuration or input change invalidates it without each service listing
    its dependencies by hand. `light_inputs` are inputs the task reads that are not among its
    es_parameters -- the scenario land-cover maps, for the shock tasks -- fingerprinted by size and
    mtime because reading gigabytes of maps is the very cost being avoided; a `touch` on one of them
    costs a rebuild that was not needed, which is the safe direction.

    Args:
        p: the ProjectFlow object, already hydrated.
        service (str): the es_parameters service name.
        outputs (list): the files reuse would republish.
        signature_name (str): the signature filename, written beside the first output.
        light_inputs (iterable): paths fingerprinted by size and mtime.

    Returns:
        str | None: the reason to recompute, naming what differs, or None to reuse.
    """
    settings, inputs = _service_settings_and_inputs(p, service)
    return outputs_reuse_reason(outputs, _signature(settings, inputs, light_inputs),
                                os.path.join(os.path.dirname(outputs[0]), signature_name))


def write_reuse_signature(p, service, outputs, signature_name='run_signature.json', light_inputs=()):
    """Record what produced these outputs, so the next run can tell whether it may reuse them."""
    settings, inputs = _service_settings_and_inputs(p, service)
    write_outputs_signature(_signature(settings, inputs, light_inputs),
                            os.path.join(os.path.dirname(outputs[0]), signature_name))


def add_rows_missing_from_template(local_path, template_path, key_columns, log=print):
    """Bring the project's copy up to the template's schema. A value anyone has set always wins.

    A definitions CSV is a schema plus values: the template names every key the code may read,
    and the project's copy supplies this machine's values. Seeding copies the file only when it
    is absent, so a copy made before a key was added keeps shadowing the template forever, and
    the key reaches the run as a missing attribute rather than as its documented value.

    Three ways the template can be ahead, and all three are filled, because a template addition
    that does not reach `input/` is indistinguishable from one nobody made:

    - a row whose key the copy lacks, appended;
    - a column the copy lacks, added with the template's values;
    - a cell the copy leaves blank where the template has a value, filled.

    What is never touched is a cell the copy has filled in. That is the machine's own answer --
    an ssh host, a scratch path, a drive location -- and the template ships those blank precisely
    so the copy can own them.

    The third case is the one that fails without saying so: the row and the key are present but
    the cell is empty, so the run raises AttributeError on a value the template carries.

    Args:
        local_path (str): the project's input/ copy, modified in place when it is behind.
        template_path (str): the tracked template to take absent rows, columns and values from.
        key_columns (list): columns that together identify a row, e.g. ['service', 'parameter'].
        log (callable): where to report what was added.
    """
    if not os.path.exists(template_path) or not os.path.exists(local_path):
        return
    local, template = hb.df_read(local_path), hb.df_read(template_path)
    if any(c not in local.columns or c not in template.columns for c in key_columns):
        return

    def keys_of(df):
        return list(zip(*[df[c].astype(str) for c in key_columns])) if len(df) else []

    def is_blank(value):
        return value is None or (isinstance(value, float) and pd.isna(value)) \
            or (isinstance(value, str) and not value.strip()) or pd.isna(value)

    added_columns = [c for c in template.columns if c not in local.columns]
    for column in added_columns:
        local[column] = None

    present = set(keys_of(local))
    missing = template[[k not in present for k in keys_of(template)]]
    if not missing.empty:
        local = pd.concat([local, missing], ignore_index=True)

    by_key = {k: row for k, row in zip(keys_of(template), template.to_dict('records'))}
    filled = []
    for position, key in enumerate(keys_of(local)):
        source = by_key.get(key)
        if source is None:
            continue
        for column in template.columns:
            if column in key_columns or column not in local.columns:
                continue
            if is_blank(local.at[local.index[position], column]) and not is_blank(source[column]):
                local.at[local.index[position], column] = source[column]
                filled.append('%s.%s' % (':'.join(key), column))

    if missing.empty and not added_columns and not filled:
        return
    hb.df_write(local, local_path)
    parts = []
    if not missing.empty:
        named = ', '.join(':'.join(k) for k in keys_of(missing)[:6])
        parts.append(f'{len(missing)} row(s) ({named}{", ..." if len(missing) > 6 else ""})')
    if added_columns:
        parts.append(f'column(s) {", ".join(added_columns)}')
    if filled:
        parts.append(f'{len(filled)} blank cell(s) ({", ".join(filled[:6])}'
                     f'{", ..." if len(filled) > 6 else ""})')
    log(f'{os.path.basename(local_path)}: took {"; ".join(parts)} from the template, which the '
        f'project copy was behind on. Values already set were left alone.')


def seed_input_template(p, file_name, log=print, required=True, key_columns=None):
    """Return the project's input/ copy of a tracked input_template file, seeding it on first use.

    The house input calculation (same as ngfs/seals): the tracked template under
    global_invest/input_template/ is copied into the project's input/ if absent, and the run
    always reads the input/ copy -- edit that copy to configure a single project. file_name may
    be a relative path (e.g. the lulc test fixtures); the nesting is recreated under input/.

    key_columns names the columns identifying a row in a definitions CSV. Given it, an existing
    copy has any absent keys topped up from the template, so a copy predating a new key does not
    silently withhold it. Local values are never overwritten: a stale copy still shadows an
    updated template for keys it already carries, which is what lets a machine keep its own.

    required=False is for files that only OPTIONALLY ship as fixtures (a production machine
    resolves the same reference in base_data instead): a missing template is skipped, and the
    later get_path on the reference stays the loud failure if it resolves nowhere at all.
    """
    template_path = os.path.join(os.path.dirname(__file__), 'input_template', file_name)
    local_path = os.path.join(p.input_dir, file_name)
    if not os.path.exists(local_path):
        if not required and not os.path.exists(template_path):
            return local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy(template_path, local_path)
        log(f'Seeded {file_name} into {p.input_dir} from the tracked template.')
    elif key_columns:
        add_rows_missing_from_template(local_path, template_path, key_columns, log)
    return local_path


# What a definitions cell can say instead of a value. Blank already means "this attribute does
# not apply to this service", but blank cannot distinguish that from "nobody has filled this in
# yet", and the two read identically to anyone scanning the table. These words mean the same as
# blank to the code and something specific to the reader.
NOT_A_VALUE = ('computed', 'skip', 'n/a')


def is_not_a_value(value):
    """Whether a definitions cell is blank or says, in words, that it holds no value."""

    return pd.isna(value) or str(value).strip().lower() in ('',) + NOT_A_VALUE


def hydrate_es_config(p, service, log=print):
    """Fill a service's per-ES configuration onto p from es_config.csv (wide format:
    one row per service, one column per attribute -- one shared table for the whole library;
    a cell left empty means the attribute does not apply to that service and is skipped).

    DEFAULTS layer, never an override: an attribute already set on p (by a consumer pipeline
    or a caller) is left untouched, so the seam contract is unchanged. Attributes ending in
    _path resolve through p.get_path (base_data-relative references); values that parse as
    integers become ints; everything else stays a string.

    The csv follows the house input calculation (same as ngfs/seals): the tracked template at
    global_invest/input_template/es_config.csv is SEEDED into the project's input/ on first
    use, and the run always reads the project's own input/ copy -- edit that copy to configure
    a single project. Note the standard caveat: a stale input/ copy shadows an updated
    template; delete it (or use a fresh project) to pick up template changes.
    """
    df = hb.df_read(seed_input_template(p, 'es_config.csv', log, key_columns=['service']))
    rows = df[df['service'] == service]
    if rows.empty:
        log(f"es_config.csv has no row for service '{service}' -- nothing hydrated.")
        return p
    row = rows.iloc[0]
    for attribute in df.columns:
        if attribute == 'service':
            continue
        value = row[attribute]
        if is_not_a_value(value):
            continue
        if getattr(p, attribute, None) is not None:
            continue
        if attribute.endswith('_path'):
            # A fixture shipped under input_template at the same relative ref seeds into
            # input/ first (required=False: absent template -> get_path stays the loud gate),
            # so a cell can point at data the library carries -- example_service does.
            seed_input_template(p, str(value), log, required=False)
            # leave_ref_path_if_fail: hydration publishes paths for later tasks rather than
            # consuming them, so a config path this machine does not hold should fail in the task
            # that reads it, naming that file, rather than here, naming the service. Landslide's
            # raw input directory, for example, is a cluster asset that its results-only run,
            # reading a staged table, never touches.
            value = p.get_path(str(value), leave_ref_path_if_fail=True)
        else:
            try:
                value = int(float(value))
            except (TypeError, ValueError):
                value = str(value)
        setattr(p, attribute, value)
    return p


def hydrate_es_parameters(p, service, log=print):
    """Per-service parameters from es_parameters.csv (long key-value rows scoped by a service
    column) -- the ngfs parameters.csv pattern: machine-specific keys ship BLANK in the template
    and each machine fills its project's input/ copy (a blank value is skipped); method knobs
    ship with their defaults. es_config stays the GEP formula's roles; this file holds what a
    formula row cannot express -- run knobs, method constants promoted to configuration, and
    machine locations (e.g. erosion_gep_root, the MSI data root that
    configure_prevention_shares reads off p).

    DEFAULTS layer like its siblings: an attribute the caller already set on p wins. Values
    parse as JSON where they can (ints, lists, dicts, true/false), else stay strings; *_path
    keys resolve via get_path.
    """
    df = hb.df_read(seed_input_template(p, 'es_parameters.csv', log,
                                        key_columns=['service', 'parameter']))
    for _, row in df[df['service'] == service].iterrows():
        attribute, value = str(row['parameter']), row['value']
        if pd.isna(value) or str(value) == '':
            continue
        if getattr(p, attribute, None) is not None:
            continue
        if attribute.endswith('_path'):
            # Permissive: a task must not crash resolving a shipped path it never uses;
            # whatever consumes the path fails loudly at use.
            value = p.get_path(str(value), raise_error_if_fail=False)
        else:
            try:
                value = json.loads(str(value))
            except (ValueError, TypeError):
                value = str(value)
        setattr(p, attribute, value)
    return p


def hydrate_es_scenarios(p, log=print):
    """Set the shared es_shock_* seam attributes from a scenarios CSV -- the same derivation a
    consumer pipeline performs on its own scenarios file (run_ngfs_pnas.py STEP 6), so a
    standalone module run is data-driven exactly the way a pipeline run is. One shared CSV for
    the whole library because these attributes are identical for every service; per-service
    configuration stays in es_config.csv.

    Columns follow the standard seals scenarios.csv vocabulary (scenario_label, scenario_type,
    baseline_reference_label, key_base_year, years) plus the two map-reference columns; the
    shipped es_scenarios_test.csv uses the standard seals scenario names, so its maps are the
    ones a standard seals run writes. The derivation:

      es_shock_scenarios      scenario_label of every scenario_type == 'policy' row
      es_shock_base_scenario  scenario_label of the scenario_type == 'bau' row (the comparison base)
      es_shock_base_year      int(key_base_year)
      es_shock_years          the first non-baseline row's years, space-delimited ints
      es_shock_end_year       max(es_shock_years)
      es_lulc_path_template   dirname resolved via get_path, {scenario}/{year} pattern rejoined
      es_base_year_lulc_path  resolved via get_path
      aggregation_label       first non-null, when the column is present

    Mirroring es_config's empty-cell rule, a column that is absent (or has no values) is simply
    not derived: a static-table service like fisheries has no bau row and no map columns -- its
    file carries only labels, years and the aggregation -- and whatever a task then misses fails
    as a named AttributeError in that task, not silently.

    Map references also seed from input_template when a fixture with that relative path ships
    there (the tiny standard-seals test maps), so the standalone smoke test is self-contained;
    get_path finds input/ before base_data, and a production machine without fixtures resolves
    the same references in base_data.

    DEFAULTS layer, never an override: an attribute the caller already set on p wins, which is
    the same seam contract consumers rely on (they set these from their own scenarios CSV and
    never read this one). The CSV follows the same input calculation as es_config.csv (tracked
    template seeded into the project's input/; the run reads the input/ copy). Set
    p.es_scenario_definitions_filename to run a different scenarios file.
    """
    file_name = getattr(p, 'es_scenario_definitions_filename', None) or 'es_scenarios_test.csv'
    df = hb.df_read(seed_input_template(p, file_name, log))

    def unset(attribute):
        # The defaults-layer contract in one predicate: hydrate only what the caller
        # (e.g. a consumer pipeline) has not already set on p.
        return getattr(p, attribute, None) is None

    if unset('es_shock_scenarios'):
        p.es_shock_scenarios = [str(s) for s in df.loc[df['scenario_type'] == 'policy', 'scenario_label']]
        if not p.es_shock_scenarios:
            raise ValueError(f"{file_name} has no scenario_type == 'policy' row -- "
                             'the shock would silently compute nothing.')
    def column_values(label):
        # A column that is absent, or present with no values, is "not applicable" -- same
        # semantics as an empty es_config cell.
        return df[label].dropna() if label in df.columns else pd.Series(dtype=object)

    bau_labels = df.loc[df['scenario_type'] == 'bau', 'scenario_label']
    if unset('es_shock_base_scenario') and len(bau_labels):
        p.es_shock_base_scenario = str(bau_labels.iloc[0])
    if unset('es_shock_base_year'):
        p.es_shock_base_year = int(df['key_base_year'].dropna().iloc[0])
    if unset('es_shock_years'):
        year_cells = df.loc[df['scenario_type'] != 'baseline', 'years'].dropna()
        p.es_shock_years = [int(y) for y in str(year_cells.iloc[0]).split(' ')]
    if unset('es_shock_end_year'):
        p.es_shock_end_year = max(p.es_shock_years)
    if unset('es_lulc_path_template') and len(column_values('es_lulc_path_template')):
        ref = str(column_values('es_lulc_path_template').iloc[0])
        base_scenario = getattr(p, 'es_shock_base_scenario', None)
        for scenario in ([base_scenario] if base_scenario else []) + list(p.es_shock_scenarios):
            for year in p.es_shock_years:
                seed_input_template(p, ref.format(scenario=scenario, year=year), log, required=False)
        p.es_lulc_path_template = os.path.join(p.get_path(os.path.dirname(ref)), os.path.basename(ref))
    if unset('es_base_year_lulc_path') and len(column_values('es_base_year_lulc_path')):
        ref = str(column_values('es_base_year_lulc_path').iloc[0])
        seed_input_template(p, ref, log, required=False)
        p.es_base_year_lulc_path = p.get_path(ref)
    if unset('aggregation_label') and len(column_values('aggregation_label')):
        p.aggregation_label = str(column_values('aggregation_label').iloc[0])
    if unset('es_shock_climate_labels') and len(column_values('climate_label')):
        # scenario -> climate (rcp) label, for services whose science keys on the RCP rather
        # than the scenario (fisheries maps this to its FI headers) -- so scenario NAMES never
        # need a per-service translation in the library.
        non_baseline = df[df['scenario_type'] != 'baseline']
        p.es_shock_climate_labels = {
            str(row['scenario_label']): str(row['climate_label'])
            for _, row in non_baseline.iterrows() if pd.notna(row.get('climate_label'))}
    return p


def raster_sum(raster_path, block_rows=2048):
    """Nodata-safe sum of a raster's first band, read blockwise so global rasters fit in
    memory. Promotion candidate to hazelbean (no equivalent there)."""
    gdal.UseExceptions()
    ds = gdal.Open(raster_path)
    band = ds.GetRasterBand(1)
    ndv = band.GetNoDataValue()
    total = 0.0
    for y in range(0, band.YSize, block_rows):
        rows = min(block_rows, band.YSize - y)
        array = band.ReadAsArray(0, y, band.XSize, rows).astype('float64')
        if ndv is not None:
            array[array == ndv] = 0.0
        array[~np.isfinite(array)] = 0.0
        total += array.sum()
    return total


def assert_zonal_conservation(country_totals_sum, raster_path, service, lower=0.95, upper=1.001):
    """The conservation invariant for a value-raster country aggregation: the country sums must
    add up to the raster's own total. A shortfall beyond `lower` means value fell outside every
    polygon (or zones were dropped). An excess beyond `upper` means DOUBLE-COUNTING -- the exact
    failure the split-country guard exists for."""
    total = raster_sum(raster_path)
    if total == 0:
        raise ValueError(f'{service}: conservation check impossible, the raster sums to zero.')
    coverage = country_totals_sum / total
    hb.log(f'{service}: zonal conservation {coverage:.4%} of the raster total.')
    if not (lower <= coverage <= upper):
        raise ValueError(
            f'{service}: country sums are {coverage:.4%} of the raster total '
            f'(allowed {lower:.0%} to {upper:.1%}). Above 100% means double-counting; '
            f'far below means dropped value.')
    return coverage


def download_missing_inputs(p, service, log=print):
    """Fetch any of a service's inputs that are missing and have a recorded source.

    es_parameters carries one row per input path (`<name>_path`). An input that can be
    fetched carries companion rows:

      `<name>_source_url`             the public URL it comes from
      `<name>_source_archive_member`  the member to extract, when the URL is an archive
      `<name>_source_note`            the human instruction, when no URL can exist
                                      (an interactive export, or a file only a
                                      collaborator has)

    Only missing files are fetched, so a rerun downloads nothing. Inputs with a note, or
    with no source at all, are returned by name rather than silently skipped: that list is
    what a collaborator has to send.

    Everything lands in base_data, which is where a shared input belongs and what
    `run_ngfs_pnas.py` means by "if anything is missing, it will download it": one base_data
    across projects, so a file fetched for one run is there for the next.

    This is a deliberate step, not part of publish_inputs. Only files that are absent are
    fetched, so it cannot swap a vintage; what it must not become is a check on every path in
    every task, which would cost a run more than it saves.

    Returns:
        (downloaded, needs_a_person): the paths written, and {input name: reason} for the
        inputs a person has to supply.
    """

    df = hb.df_read(seed_input_template(p, 'es_parameters.csv', log,
                                        key_columns=['service', 'parameter']))
    rows = df[df['service'] == service]

    def companions(suffix):
        return {str(r['parameter'])[:-len(suffix)]: str(r['value'])
                for _, r in rows.iterrows()
                if str(r['parameter']).endswith(suffix) and not pd.isna(r['value'])}

    urls, members, notes = companions('_source_url'), companions('_source_archive_member'), companions('_source_note')

    downloaded, needs_a_person = [], {}
    for _, row in rows.iterrows():
        attribute, ref_path = str(row['parameter']), row['value']
        if not attribute.endswith('_path') or pd.isna(ref_path) or not str(ref_path).strip():
            continue
        # The destination comes from the CSV's reference path under base_data, not from p. An
        # input that resolves nowhere leaves its attribute unset, so reading p would skip exactly
        # the file that needs fetching -- and get_path, asked not to raise, answers with the
        # project dir, which puts a shared input somewhere no other project can see it.
        path = str(ref_path) if os.path.isabs(str(ref_path)) else os.path.join(
            p.base_data_dir, str(ref_path))
        if hb.path_exists(path):
            continue
        name = attribute[:-len('_path')]
        if name not in urls:
            needs_a_person[attribute] = notes.get(name, 'no recorded source')
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log(f'{service}: downloading {attribute} from {urls[name]}')
        if name in members:
            archive = path + '.download'
            urllib.request.urlretrieve(urls[name], archive)
            with zipfile.ZipFile(archive) as zf, open(path, 'wb') as out:
                shutil.copyfileobj(zf.open(members[name]), out)
            os.remove(archive)
        else:
            urllib.request.urlretrieve(urls[name], path)
        downloaded.append(path)

    for attribute, reason in needs_a_person.items():
        log(f'{service}: {attribute} needs a person ({reason})')
    return downloaded, needs_a_person


def download_inputs_task(service):
    """Build a ProjectFlow task that fetches a service's missing inputs.

    Deliberately opt-in: add it to a tree when a machine needs its inputs, and leave it out
    of routine runs, so no run silently refetches a file mid-analysis.

        p.add_task(utilities.download_inputs_task('extractive_energy'))
    """

    def download_inputs(p):
        if not p.run_this:
            return
        downloaded, needs_a_person = download_missing_inputs(p, service, log=hb.log)
        hb.log(f'{service}: {len(downloaded)} inputs downloaded, '
               f'{len(needs_a_person)} still need a person')
        return True

    download_inputs.__name__ = f'download_{service}_inputs'
    return download_inputs


def collapse_countries_to_r250(df_countries, keep_columns=()):
    """The r264 correspondence reduced to one canonical row per country.

    r264 splits large countries into sub-regions, so joining a per-country value against it
    repeats that country once per sub-region. Filtering to the rows whose r264 label equals
    the r250 label leaves exactly one row per country, which is what a country join needs.

    Args:
        df_countries (pd.DataFrame): the r264 correspondence (p.df_countries).
        keep_columns (iterable): extra columns to carry through, beyond the identifiers and
            the standard country attributes.

    Returns:
        pd.DataFrame: one row per country, holding the identifiers, the attributes, and
        whatever `keep_columns` names.
    """
    identifiers = ['ee_r264_id', 'iso3_r250_id', 'ee_r264_label', 'iso3_r250_label',
                   'ee_r264_name', 'iso3_r250_name']
    attributes = ['continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    wanted = identifiers + attributes + list(keep_columns)
    one_row_per_country = df_countries[df_countries['ee_r264_label'] == df_countries['iso3_r250_label']]
    return one_row_per_country[[c for c in wanted if c in one_row_per_country.columns]].copy()


def assert_join_coverage(joined_df, value_column, expected_rows, service, log=print):
    """Every source row must survive a country join, or the loss is named.

    A join on country labels drops any row whose label the correspondence does not carry,
    and the result still looks like a valid table. This compares the surviving valued rows
    against what went in and raises with the count when they disagree.
    """
    survived = int(joined_df[value_column].notna().sum())
    if survived < expected_rows:
        raise ValueError(
            f'{service}: {expected_rows - survived} of {expected_rows} valued rows did not '
            f'match a country in the correspondence. A label that does not match is dropped '
            f'silently, so the join key needs checking before this total is used.')
    log(f'{service}: all {expected_rows} valued rows matched a country.')


# The choropleth every service's report closes with. It is here rather than pasted into each
# results page so every page draws the same map, and a service needs no geopackage of its own to
# draw one: the shared country geometry plus the per-country table the service writes is enough.
CHOROPLETH_COLORMAP = 'OrRd'
CHOROPLETH_FIGSIZE = (15, 10)
CHOROPLETH_DPI = 300


def plot_gep_choropleth(df_by_country, value_column, countries_vector_path, out_png_path,
                        title='GEP by Country in Base Year', label=None):
    """Draw one service's per-country values on the shared country geometry.

    Values are shaded on a log scale, because a handful of countries carry most of every
    service's total and a linear ramp renders the rest indistinguishable. A country the table
    does not value is left unshaded rather than shaded as zero.

    Args:
        df_by_country (pd.DataFrame): one row per country, carrying iso3_r250_id and value_column.
        value_column (str): the column to map.
        countries_vector_path (str): the shared country geometry (any r250 or r264 gpkg).
        out_png_path (str): where the figure is written.
        title (str): the figure title.
        label (str): the colour-bar label; defaults to the value column.

    Returns:
        str: out_png_path, or None if the table had no positive value to scale a log ramp on.
    """
    mpl.use('Agg')

    gdf = gpd.read_file(countries_vector_path)
    join_column = 'iso3_r250_id' if 'iso3_r250_id' in gdf.columns else 'ee_r264_id'
    values = df_by_country[['iso3_r250_id', value_column]].dropna(subset=['iso3_r250_id'])
    values = values.groupby('iso3_r250_id', as_index=False)[value_column].sum(min_count=1)
    gdf = gdf.merge(values, how='left', left_on=join_column, right_on='iso3_r250_id')

    positive = gdf[value_column][gdf[value_column] > 0]
    if not len(positive):
        return None

    norm = mpl.colors.LogNorm(vmin=positive.min(), vmax=gdf[value_column].max())
    figure, axes = plt.subplots(figsize=CHOROPLETH_FIGSIZE)
    gdf.plot(column=value_column, cmap=CHOROPLETH_COLORMAP, legend=False, norm=norm, ax=axes)
    gdf.boundary.plot(ax=axes, color='black', linewidth=0.5)

    scalar_map = plt.cm.ScalarMappable(cmap=CHOROPLETH_COLORMAP, norm=norm)
    scalar_map._A = []
    colour_bar = figure.colorbar(scalar_map, ax=axes, orientation='vertical',
                                 fraction=0.03, pad=0.02)
    colour_bar.set_label(label or value_column)
    colour_bar.ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    axes.set_title(title)
    axes.set_axis_off()
    hb.create_directories(os.path.dirname(out_png_path))
    plt.savefig(out_png_path, bbox_inches='tight', dpi=CHOROPLETH_DPI)
    plt.close(figure)
    return out_png_path


import numpy as np


def sum_by_zone(value, zone_ids, n_zones):
    """Per-zone pixel sums of a value block, indexed by zone id.

    Shared because more than one service sums a raster inside country polygons block by
    block: timber over its value raster, stormwater over its retention volume.

    Args:
        value (np.ndarray): Value raster block (per-pixel value: dollars, cubic metres, whatever the raster holds).
        zone_ids (np.ndarray): Integer zone-id block, same shape; 0 = background.
        n_zones (int): Highest zone id; the output has n_zones + 1 entries.

    Returns:
        np.ndarray: float64 sums, index i = total for zone id i. Blockwise callers
        accumulate by adding successive blocks' arrays.
    """
    return np.bincount(zone_ids.ravel(), weights=value.astype(np.float64).ravel(),
                       minlength=n_zones + 1)


# ---------------------------------------------------------------------------------------------
# The two steps every service's gep_calculation shares: register a results dict naming
# gep_by_country_base_year and skip when it exists; collapse to r250, set the year, write the
# table and log the total.
#
# What is NOT here, deliberately: the merge that joins a service's values to the country list.
# ntfp joins on iso3_r250_label with how='left', stormwater on iso3_r250_id with how='right',
# and fisheries passes the country frame into its science function instead. That is real
# variation, so folding it in would mean a parameter per caller and a helper nobody can read.
# ---------------------------------------------------------------------------------------------

# The attributes every per-country table carries. One list, so a service cannot quietly ship a
# table with a column its siblings have.
GEP_COUNTRY_ATTRIBUTE_COLUMNS = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']

# Columns that reach a country table from an upstream source and say nothing the table does not
# already say, each mapped to the column it repeats. A column is dropped only when the one it
# duplicates is actually there: renewable_energy_provision carries `Year` and no lowercase `year`,
# so dropping it unconditionally would take the year out of the table and nothing would report an
# error. `Value` is a byte-for-byte copy of the service's own `_gep` column.
REDUNDANT_COUNTRY_COLUMNS = {'Value': '_gep', 'Year': 'year', 'Country': 'iso3_r250_name',
                             'Country_Name': 'iso3_r250_name', 'Country Code': 'iso3_r250_id',
                             'area_code_M49': 'iso3_r250_id'}


def is_redundant(column, df):
    """Whether a column repeats one the frame already has."""
    duplicates = REDUNDANT_COUNTRY_COLUMNS.get(column)
    if duplicates is None:
        return False
    if duplicates == '_gep':
        return any('_gep' in c for c in df.columns)
    return duplicates in df.columns


# The aggregations every results page reports beside the country table. A GEP account is read by
# region and by income group at least as often as by country, so a page that shows only the country
# table is missing the view most people open it for.
GEP_SUMMARY_GROUPINGS = ('income_grp', 'region_un', 'continent', 'subregion')


def usd_deflation_factor(cpi_df, from_year, to_year):
    """The US CPI annual-mean ratio turning nominal from_year USD into to_year USD.

    One series for every temporal dollar conversion in the account, so no two services
    deflate differently. The frame is FRED's CPIAUCSL as staged (monthly observations).

    Args:
        cpi_df (pd.DataFrame): observation_date and CPIAUCSL columns.
        from_year (int): the year the nominal values are denominated in.
        to_year (int): the year to express them in.

    Returns:
        float: the multiplicative factor (less than 1 when deflating a later year back).
    """
    cpi = cpi_df.copy()
    cpi['year'] = pd.to_datetime(cpi['observation_date']).dt.year
    annual = cpi.groupby('year')['CPIAUCSL'].mean()
    return float(annual.loc[to_year] / annual.loc[from_year])


def fill_base_year_gaps_from_adjacent_years(df, base_year, cpi_df, value_label,
                                            country_column='countrycode'):
    """A wide CWoN panel's base-year column with source gaps filled from the adjacent years.

    A gap is a base-year cell that is missing, or zero while an adjacent year (the year before
    or the year after) is positive: the CWoN current-dollar panels read zero or NaN for
    country-years the edition did not value, and a resource earning rent either side of the base
    year did not earn nothing in it. Everything else is preserved: a zero flanked by nonpositive
    neighbours is a genuine zero, a NONZERO base-year value is never replaced, and a gap with no
    positive neighbour stays missing -- flagged in the provenance table, never filled from
    another source such as WDI.

    The donor is the mean of the positive adjacent values, each converted into base-year dollars
    with the account's shared CPI (usd_deflation_factor); with one positive neighbour, that year
    alone. Symmetric, deterministic, and using only the panel's own local information.

    Args:
        df (pd.DataFrame): the CWoN panel with YR<year> columns.
        base_year (int): the account's base year.
        cpi_df (pd.DataFrame): the shared FRED CPIAUCSL frame.
        value_label (str): what the panel holds, for the provenance rows.
        country_column (str): the panel's country key.

    Returns:
        tuple: (the frame with the base-year column repaired, a provenance DataFrame with one
        row per repair or flag: country, value_label, original, reason, donor years, nominal
        donor values, conversion factors and replacement).
    """
    base = 'YR%d' % base_year
    years = [base_year - 1, base_year + 1]
    out = df.copy()
    for column in [base] + ['YR%d' % y for y in years]:
        out[column] = pd.to_numeric(out.get(column), errors='coerce')
    factors = {y: usd_deflation_factor(cpi_df, y, base_year) for y in years}
    records = []
    for i, row in out.iterrows():
        original = row[base]
        positives = [(y, float(row['YR%d' % y])) for y in years
                     if pd.notna(row['YR%d' % y]) and row['YR%d' % y] > 0]
        if pd.notna(original) and original != 0:
            continue
        if original == 0 and not positives:
            continue  # a genuine zero: nonpositive on both sides
        record = {'country': row[country_column], 'value': value_label,
                  'original': original,
                  'donor_years': ' '.join(str(y) for y, _ in positives),
                  'donor_nominal': ' '.join('%r' % v for _, v in positives),
                  'conversion_factors': ' '.join('%r' % factors[y] for y, _ in positives)}
        if not positives:
            record['reason'], record['replacement'] = 'missing_no_donor', np.nan
        else:
            record['reason'] = 'missing_2019' if pd.isna(original) else 'zero_with_positive_neighbour'
            record['replacement'] = float(np.mean([v * factors[y] for y, v in positives]))
            out.loc[i, base] = record['replacement']
        records.append(record)
    provenance = pd.DataFrame(records, columns=['country', 'value', 'original', 'reason',
                                                'donor_years', 'donor_nominal',
                                                'conversion_factors', 'replacement'])
    return out, provenance


def international_to_usd(df, pli_df, year, value_col, iso3_col='iso3_r250_label'):
    """Convert per-country international-dollar (PPP) values into market USD of the same year.

    The price level ratio is market-exchange-rate GDP over PPP GDP from the World Bank's two
    headline series, so the USA is exactly 1 by construction. A country without a ratio for
    that year carries NA afterwards, never a silent pass-through, and the note names it.

    Args:
        df (pd.DataFrame): per-country values in international dollars.
        pli_df (pd.DataFrame): iso3, year, price_level_ratio (the staged series).
        year (int): the year the PPP values are denominated in.
        value_col (str): the column to convert.
        iso3_col (str): the country key in df.

    Returns:
        pd.DataFrame: df with value_col converted and `price_level_ratio` carried beside it.
    """
    ratios = pli_df[pli_df['year'] == year][['iso3', 'price_level_ratio']]
    out = df.merge(ratios.rename(columns={'iso3': iso3_col}), on=iso3_col, how='left')
    out[value_col] = out[value_col] * out['price_level_ratio']
    return out


def rasterize_id_column(vector_path, ref_raster_path, id_column, output_path):
    """Vector id column -> Int32 raster on the reference grid (country ids), ALL_TOUCHED.

    Args:
        vector_path: the polygon file carrying the id column.
        ref_raster_path: the raster whose grid (extent, resolution, projection) the ids land on.
        id_column: the integer attribute to burn.
        output_path: the raster written, 0 where no polygon touches.
    """
    ref = gdal.Open(ref_raster_path)
    driver = gdal.GetDriverByName('GTiff')
    out = driver.Create(output_path, ref.RasterXSize, ref.RasterYSize, 1, gdal.GDT_Int32,
                        options=['COMPRESS=DEFLATE', 'TILED=YES'])
    out.SetGeoTransform(ref.GetGeoTransform())
    out.SetProjection(ref.GetProjection())
    band = out.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.Fill(0)
    gdal.Rasterize(out, vector_path, options=gdal.RasterizeOptions(
        attribute=id_column, allTouched=True))
    out.FlushCache()


def report_dir():
    """The directory a results page is rendered into, and where its tables and figures belong.

    Quarto runs a qmd with the working directory set to the qmd's own location, which is the
    report task's directory. `p.cur_dir` is not that. A results page builds the calculation tree
    and executes it, so by the time the display code runs `p.cur_dir` is whichever task happened
    to run last, which differs from service to service: it is gep_calculation for most and
    fisheries_subsistence_gep for fisheries. Writing report artifacts there scatters them across
    task folders while the page itself is written somewhere else, and the page then reads a
    figure back from a path nothing wrote to.

    Returns:
        str: the report's own directory.
    """

    return os.getcwd()


def gep_summary_tables(df, value_column, out_dir, log=None):
    """The country table and the four grouped tables a results page displays.

    Writes each one beside the report as `gep_by_<grouping>_base_year_table.csv`. A grouping whose
    column the frame does not carry is skipped rather than raising: fire_protection covers 161
    countries and does not reach every income group.

    Args:
        df (pd.DataFrame): the service's country table.
        value_column (str): the column to sum, normally `<service>_gep`.
        out_dir (str): where the CSVs go, normally the report task's own directory.
        log (callable): optional logger.

    Returns:
        dict: 'country' plus one key per grouping present, each a DataFrame.
    """

    tables = {'country': df[['iso3_r250_name', value_column]]}
    hb.df_write(tables['country'], os.path.join(out_dir, 'gep_by_country_base_year_table.csv'))
    for grouping in GEP_SUMMARY_GROUPINGS:
        if grouping not in df.columns:
            if log:
                log('No %s column, so that summary table is not written.' % grouping)
            continue
        grouped = df.groupby(grouping, as_index=False)[value_column].sum()
        tables[grouping] = grouped[[grouping, value_column]]
        hb.df_write(tables[grouping],
                    os.path.join(out_dir, 'gep_by_%s_base_year_table.csv' % grouping))
    return tables


def published_country_columns(df, service):
    """The columns a published country table carries, in the order every service uses.

    Attributes first, then the year, then the account's own value columns, then whatever
    supporting quantities the service reports. Two kinds of column are left out: the
    `ee_r264_*` correspondence columns, which several services keep on the frame because the map
    merge joins on them but which are the source side of a collapse the table has already made;
    and the redundant columns above. Nothing else is, so a new value column a service adds still
    appears without anyone editing this list.

    Args:
        df (pd.DataFrame): the frame about to be written.
        service (str): the service label, used only to order its own columns first.

    Returns:
        list: the column names to write, in order.
    """
    attributes = [c for c in GEP_COUNTRY_ATTRIBUTE_COLUMNS if c in df.columns]
    rest = [c for c in df.columns
            if c not in attributes and c != 'year'
            and not c.startswith('ee_r264') and not is_redundant(c, df)]
    value_columns = [c for c in rest if '_gep' in c]
    value_columns.sort(key=lambda c: (not c.startswith(service), len(c)))
    supporting = [c for c in rest if '_gep' not in c]
    return attributes + (['year'] if 'year' in df.columns else []) + value_columns + supporting


def write_gep_by_country(p, df, path, log=None):
    """Write a service's per-country table on the account's country list, one row per country.

    Every country appears. A country the service produced no value for gets NA, not a missing row,
    because those say different things: NA is "we do not know", an absent row is nothing at all,
    and a reader summing the column cannot tell which countries were considered. It is the
    authors' items 1 and 4 in one place -- the r250 list, and true zero against missing.

    The country list is joined against the service's data, not the data against the list, so a
    country the source never mentions still gets its row.

    A true zero is the service's own business and is left alone: coastal protection writing 0 for
    a landlocked country is correct and must not become NA here.

    Args:
        p (ProjectFlow): the project, for `df_countries`.
        df (pd.DataFrame): the service's table, keyed on iso3_r250_label.
        path: where to write it.
        log (callable): where to report the reindex.

    Returns:
        str: the path written.
    """
    log = log or hb.log
    countries = collapse_countries_to_r250(p.df_countries)
    attributes = [c for c in ('iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name', 'continent',
                              'region_un', 'region_wb', 'income_grp', 'subregion')
                  if c in countries.columns]
    countries = countries[attributes].drop_duplicates('iso3_r250_label')
    if 'iso3_r250_label' not in df.columns:
        raise NameError(
            'The per-country table has no iso3_r250_label, so it cannot be put on the account '
            'country list; it carries %s.' % list(df.columns)[:12])
    values = df[[c for c in df.columns if c not in attributes or c == 'iso3_r250_label']]
    full = countries.merge(values, on='iso3_r250_label', how='left')
    # `year` is the table's metadata, not a value: a country added as NA still belongs to the
    # table's year, so it carries it when the table has exactly one.
    if 'year' in full.columns:
        years = values['year'].dropna().unique()
        if len(years) == 1:
            full['year'] = full['year'].fillna(years[0]).astype(type(years[0]))
    added = len(full) - df['iso3_r250_label'].nunique()
    if added:
        log('Country list: %d rows, %d added as NA that the service produced no value for.'
            % (len(full), added))
    return write_csv(full, path)


def register_result(p, service, name, path):
    """Register one output as a result of `service`, from whatever task writes it.

    `begin_gep_calculation`'s `extra_results` covers what `gep_calculation` writes itself. This
    covers the rest: a map is usually built in its own earlier task, and registering it there --
    beside the line that writes it -- is what keeps the two from drifting apart.

    Registration is not bookkeeping. `distribute_results` copies what this registry names and
    nothing else, and it is the only place a raster is converted into a POG. A map that is never
    registered never leaves `intermediate/`, so nobody outside a run can open it and condition 16
    has nothing to check.

    Args:
        p (ProjectFlow): the project.
        service (str): the service's key in p.results.
        name (str): the file name the result takes in the output directory, extension included.
        path (str): where the task writes it.

    Returns:
        str: path, so a caller can register and assign in one line.
    """
    if not hasattr(p, 'results'):
        p.results = {}
    service_results = p.results.setdefault(service, {})
    service_results[name] = path
    return path


def begin_gep_calculation(p, service, extra_results=None, log=None):
    """Register a service's results and say whether the work is already done.

    Args:
        p (ProjectFlow): the project, inside gep_calculation.
        service (str): the service's key in p.results.
        extra_results (dict): any further outputs this service registers, name to path.
        log (callable): where to log; hazelbean's log by default.

    Returns:
        tuple: (service_results, already_done). When already_done is True the caller returns
        without doing anything, which is what makes a rerun cheap.
    """
    log = log or hb.log
    service_results = p.results.setdefault(service, {})
    service_results['gep_by_country_base_year'] = os.path.join(
        p.cur_dir, 'gep_by_country_base_year.csv')
    for name, path in (extra_results or {}).items():
        service_results[name] = path
    if hb.path_all_exist(list(service_results.values())):
        log('All results already exist. Skipping GEP calculation for %s.' % service)
        return service_results, True
    log('Starting GEP calculation for %s.' % service)
    return service_results, False


def country_attributes(p, columns=None):
    """One row per country, with the shared attribute columns.

    The r264 correspondence splits large countries into territories, so joining a per-country
    value against it repeats that country once per sub-region. Going through
    collapse_countries_to_r250 is what stops that, and putting it here means every service does
    it.

    Args:
        p (ProjectFlow): the project, after publish_inputs.
        columns (list): the attribute columns wanted; the shared set by default.

    Returns:
        pd.DataFrame: one row per country.
    """
    wanted = list(columns) if columns else list(GEP_COUNTRY_ATTRIBUTE_COLUMNS)
    return collapse_countries_to_r250(p.df_countries)[wanted]


# =============================================================================
# The imports the helpers below need.
import hashlib
import os
import warnings
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import hazelbean as hb
import pandas as pd
import rasterio
from rasterio.windows import Window


# Raster, table and figure helpers. Not service-specific: any service needing them
# finds them here rather than keeping a copy.
# =============================================================================


# The plotting helpers carry no module-level map defaults: their settings are arguments, supplied
# by the caller from es_parameters.csv.


# -----------------------------------------------------------------------------
# Existence / assertions
# -----------------------------------------------------------------------------

def service_data_dir(p, service):
    """Where one service's inputs live under base data, from the ProjectFlow that knows.

    Replication anchors sit here with everything else the service reads: a `reference/`
    directory inside the repo would make them a special kind of input, and they are not, they
    are inputs.

    A missing local directory is seeded once from the machine's shared data roots
    (p.shared_data_dirs, the same tier get_path searches), which mirror base_data's layout.
    get_path caches one file per ref_path; this is the directory-shaped counterpart, because
    the callers ask for the directory. Only the shared directory's top-level files are
    copied: the anchors live flat, and the shared dirs also hold working subtrees
    (Archive/, Results/, ...) that are not inputs and can be large. No local copy and no
    shared copy still fails loudly, and a shared directory holding nothing copyable seeds
    nothing.

    Args:
        p: the ProjectFlow, for `base_data_dir` and `shared_data_dirs`.
        service (str): the service directory name.

    Raises:
        NameError: naming the directory, because reading the wrong one is worse than not reading.
    """
    path = os.path.join(p.base_data_dir, 'global_invest', service)
    if not os.path.isdir(path):
        for shared_root in getattr(p, 'shared_data_dirs', []):
            shared_dir = os.path.join(shared_root, 'global_invest', service)
            if not os.path.isdir(shared_dir):
                continue
            for filename in sorted(os.listdir(shared_dir)):
                shared_path = os.path.join(shared_dir, filename)
                if not os.path.isfile(shared_path):
                    continue
                if hb.is_google_native_file(shared_path):
                    # A Drive .gsheet/.gdoc is a ~100-byte JSON pointer, not data.
                    continue
                hb.cache_file_from_shared_root(shared_path, os.path.join(path, filename))
            if os.path.isdir(path):
                hb.log('service_data_dir: seeded %s from shared root %s' % (path, shared_root))
                break
    if not os.path.isdir(path):
        raise NameError('%s has no data directory at %s' % (service, path))
    return path


# ---------------------------------------------------------------------------------------------
# Rasterio read and write. Seven services open rasters by hand; hazelbean is GDAL-based and
# its as_array returns a Dataset, a Band and an array rather than the rasterio profile these
# writers need, so this pair is ours rather than a duplicate of hb.as_array.
# ---------------------------------------------------------------------------------------------

def read_raster(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a single-band GeoTIFF.

    Returns
    -------
    data : np.ndarray
        2-D float32 array.
    meta : dict
        Rasterio profile (used to write matching outputs).
    """
    path = str(path)

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()

    return data, meta


def write_raster(path, data, meta, nodata=None):
    """Write a single-band float32 GeoTIFF, creating the parent directory if needed.

    Args:
        path (Path): where to write.
        data (np.ndarray): the 2-D array.
        meta (dict): a rasterio profile, normally the one read_raster returned.
        nodata (float): overrides the profile's nodata when given.

    Returns:
        Path: the path written.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = dict(meta)
    profile.update(driver='GTiff', dtype='float32', count=1,
                   compress='deflate', predictor=2, tiled=True, zlevel=6)
    if nodata is not None:
        profile['nodata'] = nodata
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


# The files, not their directories: get_path searches the task's own directory first, so a
# directory reference resolves into the task's intermediate directory rather than into base data.



# The country attributes every GEP per-country CSV carries, in the order the CSV writes them. One
# list rather than one per service, because a service that spells them differently produces a
# table that will not stack with the others.
GEP_COUNTRY_ATTR_COLS = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                         'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']



def read_column(path, column, cast=str):
    """One column of a small reference table, as a list.

    The tables these read are CSVs rather than dictionaries and lists in the modules -- 38 ESA
    codes, 37 FLOPROS countries, 178 FAO crop names -- because a list of facts in a .py is a list
    nobody can open in a spreadsheet, diff usefully, or correct without a commit.
    """
    return [cast(v) for v in hb.df_read(path)[column].dropna().tolist()]


def read_lookup(path, key_column, value_column, key_cast=str, value_cast=str):
    """A two-column reference table as a dict."""
    df = hb.df_read(path)
    return {key_cast(k): value_cast(v)
            for k, v in zip(df[key_column], df[value_column]) if k == k}



def read_lookup_of_sets(path, first_key_column, second_key_column, value_column, cast=int):
    """A three-column reference table as {(first, second): set_of_values}.

    The land-cover class sets are the shape this exists for: which ids count as agricultural and
    which as natural, in each of two coding schemes.
    """
    df = hb.df_read(path)
    out = {}
    for first, second, value in zip(df[first_key_column], df[second_key_column], df[value_column]):
        out.setdefault((str(first), str(second)), set()).add(cast(value))
    return out


def assert_exists(path, hint: str = ""):
    """Fail naming the missing file and what needed it, rather than where the read happened."""
    if not hb.path_exists(path):
        raise FileNotFoundError(f"Missing: {path}\n{hint}")


def assert_same_grid(src_a, src_b, label_a: str = "A", label_b: str = "B", rtol: float = 1e-6):
    """
    Hard-lock the alignment principle used throughout the flood pipeline:
    depth rasters, LULC and SDA must share CRS + transform + shape. We never
    silently warp the accounting grid; if this fails, re-align the *input*
    to the LULC grid first.
    """
    problems = []
    if src_a.crs != src_b.crs:
        problems.append(f"CRS differs: {label_a}={src_a.crs} vs {label_b}={src_b.crs}")
    if (src_a.width, src_a.height) != (src_b.width, src_b.height):
        problems.append(
            f"Shape differs: {label_a}=({src_a.height},{src_a.width}) "
            f"vs {label_b}=({src_b.height},{src_b.width})"
        )
    ta, tb = src_a.transform, src_b.transform
    for name, va, vb in zip("abcdef", ta[:6], tb[:6]):
        if not np.isclose(va, vb, rtol=rtol, atol=1e-9):
            problems.append(f"Transform.{name} differs: {va} vs {vb}")
    if problems:
        raise ValueError(
            f"Grid mismatch between {label_a} and {label_b}:\n  " + "\n  ".join(problems)
        )
    return True


def raster_profile_string(ds) -> str:
    return (
        f"CRS: {ds.crs}\n"
        f"Transform: {ds.transform}\n"
        f"Width x Height: {ds.width} x {ds.height}\n"
        f"Res (approx): {ds.transform.a:.4f} x {abs(ds.transform.e):.4f}\n"
        f"Dtype: {ds.dtypes[0]}\n"
        f"Nodata: {ds.nodata}\n"
        f"Bounds: {ds.bounds}\n"
    )


def warn_if_geographic(ds, label: str = "raster"):
    """Pixel area from an affine transform is only m^2 in a projected CRS."""
    if ds.crs is not None and ds.crs.is_geographic:
        warnings.warn(
            f"[WARN] {label} CRS is geographic (degrees). Pixel area from the "
            f"transform is NOT m^2. Reproject to a projected CRS aligned to the "
            f"LULC grid before running valuation."
        )
        return True
    return False


def random_windows(width: int, height: int, n: int, wsize: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        col = int(rng.integers(0, max(1, width - wsize)))
        row = int(rng.integers(0, max(1, height - wsize)))
        yield Window(
            col_off=col, row_off=row,
            width=min(wsize, width - col), height=min(wsize, height - row),
        )


def save_raster_completely(final_path, profile: dict, array: np.ndarray, band: int = 1):
    """
    Write to <name>.tmp then rename, so a killed job never leaves a half-written GeoTIFF at the
    final path for a later skip-existing run to mistake for complete.
    """
    final_path = str(final_path)
    hb.create_directories(os.path.dirname(final_path))
    tmp = final_path + ".tmp"
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(array, band)
    os.replace(tmp, final_path)          # os.replace, not str.replace: this is the rename
    return final_path


def raster_ok(path) -> bool:
    """Cheap validity probe used by the smart-skip logic."""
    path = str(path)
    if not hb.path_exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with rasterio.open(path) as ds:
            _ = ds.profile
        return True
    except (OSError, rasterio.errors.RasterioIOError):
        return False


# -----------------------------------------------------------------------------
# Fingerprinting (smart-skip / provenance)
# -----------------------------------------------------------------------------
def sha256_file(path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_fingerprint(path, content: bool = True) -> dict:
    """What a file was, for deciding whether work that read it can be reused.

    Content by default, because size and mtime answer a different question than the one being
    asked. They say whether the file was touched; reuse turns on whether it changed. A restore
    from backup, an rsync without --times, a `touch`, or a task rewriting its own unchanged output
    all move the mtime while the bytes stay put, and every one of those invalidates hours of
    downstream work that did not need redoing.

    Hashing is cheap against what it prevents: at about 440 MB/s on MSI, flood's whole 1.7 GB
    input set costs about four seconds, where a false rebuild of Section D costs hours.

    Args:
        path: the file to fingerprint.
        content (bool): hash the bytes. False keys on size and mtime alone, for a file so large
            that reading it is itself the cost being avoided.

    Returns:
        dict: what the file was, comparable across runs.
    """
    path = str(path)
    if not hb.path_exists(path):
        return {"path": path, "exists": False}
    st = os.stat(path)
    found = {
        "path": str(path),
        "exists": True,
        "size": st.st_size,
    }
    if content and os.path.isfile(path):
        found["sha256"] = sha256_file(path)
    else:
        found["mtime"] = st.st_mtime
    return found


# -----------------------------------------------------------------------------
# Column detection (tolerant of underscore/space/case differences)
# -----------------------------------------------------------------------------
def norm_label(s: str) -> str:
    s = str(s).strip().lower().replace("_", " ")
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return " ".join(s.split())


def find_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    norm_map: Dict[str, str] = {norm_label(c): c for c in df.columns}
    for cand in candidates:
        k = norm_label(cand)
        if k in norm_map:
            return norm_map[k]
    for cand in candidates:  # contains-match fallback
        k = norm_label(cand)
        for kk, orig in norm_map.items():
            if k in kk:
                return orig
    return None


def to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def write_csv(df: pd.DataFrame, path):
    """hb.df_write, plus the parent directory, which it does not create.

    A write that would produce the bytes already on disk leaves the file alone, mtime included.
    Otherwise a fingerprint keyed on size and mtime (`file_fingerprint(..., content=False)`) tells
    everything downstream that its input changed, and triggers a rebuild that arrives at the file
    it already had. Rewriting unchanged output is never what a caller wants.
    """
    path = str(path)
    hb.create_directories(os.path.dirname(path))
    if os.path.exists(path):
        existing = tempfile.mktemp(suffix='.csv')
        hb.df_write(df, existing)
        try:
            if filecmp.cmp(existing, path, shallow=False):
                return path
            shutil.move(existing, path)
            return path
        finally:
            if os.path.exists(existing):
                os.remove(existing)
    hb.df_write(df, path)
    return path


# -----------------------------------------------------------------------------
# Numerics
# -----------------------------------------------------------------------------
def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.nanmean(x)) if x.size else float("nan")


# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------
def fmt_usd_millions(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if abs(x) >= 10:
        return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.1f}"
    if abs(x) >= 1:
        return f"{x:,.1f}"
    return f"{x:,.2f}"


def fmt_percent(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def fmt_usd(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"${x:,.0f}"


def build_interval_labels(edges: np.ndarray, label_format: str = "usd_millions") -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if label_format == "usd_millions":
            lo_txt, hi_txt = fmt_usd_millions(lo), fmt_usd_millions(hi)
        else:
            lo_txt, hi_txt = fmt_percent(lo), fmt_percent(hi)
        labels.append(f"{lo_txt} \u2013 {hi_txt}")
    return labels


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def savefig(path, dpi: int = 300, **kwargs):
    """Write the current figure, tightly cropped, and close it.

    The call below must stay plt.savefig: an unprefixed savefig resolves to this function, and
    the resulting TypeError on bbox_inches reads like a matplotlib version problem.
    """
    hb.create_directories(os.path.dirname(str(path)))
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def top_n(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    d = df[np.isfinite(pd.to_numeric(df[col], errors="coerce"))].copy()
    return d.sort_values(col, ascending=False).head(n)


def compute_classification(values: pd.Series, scheme: str = "fisher_jenks", k: int = 5):
    s = pd.to_numeric(values, errors="coerce")
    m = np.isfinite(s)
    clean = s[m]

    if clean.empty:
        return pd.Series(index=values.index, dtype="float64"), np.array([0.0, 1.0])

    try:

        scheme = (scheme or "fisher_jenks").lower()
        k_eff = max(min(k, int(clean.nunique())), 1)

        if scheme == "equal_interval":
            classifier = mapclassify.EqualInterval(clean.to_numpy(), k=k_eff)
        elif scheme == "quantiles":
            classifier = mapclassify.Quantiles(clean.to_numpy(), k=k_eff)
        else:
            classifier = mapclassify.FisherJenks(clean.to_numpy(), k=k_eff)

        edges = np.concatenate(([clean.min()], np.asarray(classifier.bins, dtype=float)))
        class_ids = pd.Series(np.nan, index=values.index)
        class_ids.loc[m] = classifier.yb
        return class_ids, edges

    except Exception:
        warnings.warn("mapclassify unavailable or failed; falling back to qcut quantiles.")
        q = min(k, max(1, int(clean.nunique())))
        cats = pd.qcut(clean, q=q, duplicates="drop")
        codes = pd.Series(np.nan, index=values.index)
        codes.loc[m] = cats.cat.codes.astype(float)
        intervals = cats.cat.categories
        edges = [intervals[0].left] + [iv.right for iv in intervals]
        return codes, np.asarray(edges, dtype=float)


def plot_publication_choropleth_categorical(
    world_joined: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    out_png,
    legend_title: str,
    scheme: str = "fisher_jenks",
    k: int = 5,
    value_unit: str = "raw",
    label_format: str = "usd_millions",
    legend_loc: str = "lower left",
    exclude_iso3=("ATA",),
    robinson_crs: str = "+proj=robin",
    usd_to_millions: float = 1e6,
):
    gdf = world_joined.copy()

    if "iso3" in gdf.columns:
        gdf = gdf[~gdf["iso3"].isin(set(exclude_iso3))].copy()
    gdf = gdf[gdf.geometry.notna()].copy()

    if value_col not in gdf.columns:
        warnings.warn(f"Column not found for map: {value_col}")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_axis_off()
        ax.set_title(f"{title}\n[missing column: {value_col}]", fontsize=16, pad=14)
        savefig(out_png, dpi=300)
        return

    if value_unit == "usd_millions":
        gdf["_plot_value"] = pd.to_numeric(gdf[value_col], errors="coerce") / usd_to_millions
    else:
        gdf["_plot_value"] = pd.to_numeric(gdf[value_col], errors="coerce")

    try:
        gdf = gdf.to_crs(robinson_crs)
    except Exception as e:
        warnings.warn(f"CRS transform failed ({e}). Plotting in native CRS.")

    minx, miny, maxx, maxy = gdf.total_bounds
    class_ids, edges = compute_classification(gdf["_plot_value"], scheme=scheme, k=k)

    valid_codes = pd.Series(class_ids).dropna()
    if valid_codes.empty:
        warnings.warn(f"No valid data for map: {value_col}")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_axis_off()
        ax.set_title(title, fontsize=16, pad=14)
        savefig(out_png, dpi=300)
        return

    n_classes = int(valid_codes.max()) + 1
    labels = build_interval_labels(edges[:n_classes + 1], label_format=label_format)

    gdf["_class_id"] = pd.Series(class_ids, index=gdf.index)
    gdf["_class_label"] = pd.Categorical(
        [labels[int(x)] if np.isfinite(x) and int(x) < len(labels) else np.nan
         for x in gdf["_class_id"]],
        categories=labels, ordered=True,
    )

    try:
        cmap = mpl.colormaps[mpl.rcParams["image.cmap"]].resampled(n_classes)
    except Exception:  # matplotlib < 3.6
        cmap = mpl.cm.get_cmap(mpl.rcParams["image.cmap"], n_classes)
    color_list = [mpl.colors.to_hex(cmap(i)) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_axis_off()
    gdf.plot(
        column="_class_label", ax=ax,
        cmap=mpl.colors.ListedColormap(color_list),
        legend=False, linewidth=0.35, edgecolor="white",
        missing_kwds={"color": "lightgrey", "edgecolor": "white"},
    )
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(title, fontsize=16, pad=14)

    handles = [Patch(facecolor=color_list[i], edgecolor="none", label=labels[i])
               for i in range(n_classes)]
    handles.append(Patch(facecolor="lightgrey", edgecolor="none", label="No data"))
    leg = ax.legend(
        handles=handles, title=legend_title, loc=legend_loc, frameon=True,
        fontsize=10, title_fontsize=11, borderpad=0.8, labelspacing=0.5,
        handlelength=1.6, handletextpad=0.6,
    )
    leg.get_frame().set_alpha(0.95)
    savefig(out_png, dpi=300)


# ---------------------------------------------------------------------------------------------
# Country columns and raster geometry, shared by erosion and flood so a boundary file carrying
# both ISO_A3 and ADM0_A3 is read the same way by both services. ISO_A3 wins: it is the official
# code, where ADM0_A3 is Natural Earth's own and fills gaps with invented ones.
# ---------------------------------------------------------------------------------------------
def pick_iso3_column(gdf):
    """The first ISO3-like column present, or None.

    Args:
        gdf (GeoDataFrame): any country layer.

    Returns:
        str or None: the column name.
    """
    for c in ("iso3", "ISO3", "iso_a3", "ISO_A3", "ADM0_A3", "adm0_a3", "iso3_r250_label"):
        if c in gdf.columns:
            return c
    return None


def pick_name_column(gdf):
    """The first country-name column present, or None."""
    for c in ("country_name", "NAME_EN", "ADMIN", "NAME_LONG", "NAME",
              "COUNTRY", "NAME_0", "ADM0_NAME", "GEOUNIT", "iso3_r250_name"):
        if c in gdf.columns:
            return c
    return None


def to_num(df, columns):
    """Coerce the named columns to numeric in place, leaving unparseable cells as NaN."""
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_columns(df):
    """A copy with column names stripped and lowercased."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def pixel_area_m2(transform) -> float:
    """Nominal pixel area from an affine transform, in square metres.

    Nominal: in a conformal projection this is the equatorial value. See mercator_area_scale.
    """
    return abs(float(transform.a) * float(transform.e))


def pixel_area_km2(transform) -> float:
    """Nominal pixel area from an affine transform, in square kilometres."""
    return pixel_area_m2(transform) / 1e6


def attach_income_group(df, df_countries, iso3_column="iso3", column="income_group"):
    """Join the World Bank income group from the shared country table.

    Every service that reports by income group reads the same column, so one country cannot sit in
    a different group in two accounts. A per-service dict in code silently drops every country
    missing from it.

    Args:
        df (DataFrame): rows carrying an ISO3 column.
        df_countries (DataFrame): the shared country table, from initialize_country_paths.
        iso3_column (str): the ISO3 column in `df`.
        column (str): the column to write.

    Returns:
        tuple: (the frame with `column` added, the groups present ordered poorest first).

    Raises:
        NameError: if the country table carries no income column, rather than leaving every row
            unlabelled and the figure quietly empty.
    """
    source = None
    for candidate in ("income_grp", "income_group", "incomegrp"):
        if candidate in df_countries.columns:
            source = candidate
            break
    if source is None:
        raise NameError(
            "No income column in the country table; looked for income_grp, income_group, "
            "incomegrp and found %s." % list(df_countries.columns))
    key = pick_iso3_column(df_countries)
    lookup = (df_countries[[key, source]].dropna()
              .assign(**{key: lambda d: d[key].astype(str).str.upper().str.strip()})
              .drop_duplicates(key).set_index(key)[source])
    out = df.copy()
    out[column] = out[iso3_column].astype(str).str.upper().str.strip().map(lookup)

    groups = [g for g in out[column].dropna().unique()]
    # Natural Earth prefixes each label with its rank ("1. High income: OECD" ... "5. Low income"),
    # so descending order puts the poorest first, which is how these read on a chart. Labels
    # without that prefix fall back to alphabetical.
    ranked = all(str(g)[:1].isdigit() for g in groups)
    return out, sorted(groups, reverse=ranked)


def income_group_colors(groups):
    """A red-to-green ramp over the income groups, poorest first."""
    ramp = mpl.colormaps["RdYlGn"].resampled(max(len(groups), 2))
    return {g: mpl.colors.to_hex(ramp(i)) for i, g in enumerate(groups)}


# =============================================================================================
# FAOSTAT Value of Production: the pipeline crop_provision, livestock_provision and
# extractive_materials_provision share. The value column is a parameter; items select by FAO item
# code as well as by name, and the value before the rental rate is carried alongside it.
# =============================================================================================
FAOSTAT_VALUE_UNIT = '1000 USD'
FAOSTAT_GROSS_PRODUCTION_VALUE_ELEMENT = 57
FAOSTAT_THOUSAND_USD = 1000.0
# The span of the staged bulk file, used by the test fixture to build a realistic frame. The
# reader takes its years from the file it is handed, because a FAO release adds years.
FAOSTAT_FIRST_YEAR = 1961
FAOSTAT_LAST_YEAR = 2023
FAOSTAT_TURKIYE_AREA_CODE = 223
CROP_ID_COLUMNS = ['area_code', 'area_code_M49', 'country', 'crop_code', 'crop']


def clean_faostat_values(df_raw, items, value_column, aggregate_areas):
    """One row per country-item-year of FAOSTAT gross production value, in thousand USD.

    The bulk file is wide (one column per year, each with a flag column beside it) and mixes
    elements, units, aggregate areas and items nobody asked for. This keeps the gross-production-
    value rows in USD, keeps the requested items, drops the aggregate areas, and melts the year
    columns into rows.

    Args:
        df_raw (pd.DataFrame): the FAOSTAT Value of Production bulk table as shipped.
        items (iterable): the items to keep. Integer entries select by FAO item code, which is
            robust to FAO's item-name revisions; strings select by name.
        value_column (str): what to call the value, e.g. 'crop_provision_gep'.

    Returns:
        pd.DataFrame: area_code, area_code_M49, country, crop_code, crop, year, <value_column>.
    """
    # The years the file actually carries, not a fixed span: a FAO release adds a year routinely,
    # and a fixed span drops the new year without a word.
    years = sorted(int(c[1:]) for c in df_raw.columns
                   if len(c) == 5 and c.startswith('Y') and c[1:].isdigit())
    if not years:
        raise NameError(
            'No Y<year> columns in the FAOSTAT frame; it carries %s. The bulk file is wide by '
            'year and this reads those columns, so a file without them is the wrong file.'
            % list(df_raw.columns)[:10])
    df = df_raw[(df_raw['Unit'] == FAOSTAT_VALUE_UNIT)
                & (df_raw['Element Code'] == FAOSTAT_GROSS_PRODUCTION_VALUE_ELEMENT)].copy()
    df = df.drop(columns=[col for col in df.columns if col.endswith('F')])

    old_names = ['Area Code', 'Area Code (M49)', 'Area', 'Item Code', 'Item'] + [f'Y{y}' for y in years]
    new_names = CROP_ID_COLUMNS + [str(y) for y in years]
    df = df.rename(columns=dict(zip(old_names, new_names)))

    codes = [i for i in items if isinstance(i, int)]
    names = [i for i in items if isinstance(i, str)]
    df = df[df['crop_code'].isin(codes) | df['crop'].isin(names)]
    df = df[~df['country'].isin(aggregate_areas)]

    df = pd.melt(df, id_vars=CROP_ID_COLUMNS, value_vars=[str(y) for y in years],
                 var_name='year', value_name=value_column)
    df['area_code'] = pd.to_numeric(df['area_code'], errors='coerce').astype(int)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    df.loc[df['area_code'] == FAOSTAT_TURKIYE_AREA_CODE, 'country'] = 'Turkey'
    hb.log('FAOSTAT values cleaned and reshaped to long (%d rows).' % df.shape[0])
    return df


def apply_rental_rates(df_values, df_coefs, value_column):
    """Production value attributed to land, country by country.

    Each year takes the rental rate of the most recent decade that has started, which is a
    backward as-of merge on year within a country. A country the CWoN table never covers keeps a
    missing rate, so its value becomes missing rather than being attributed in full.

    gross_production_value carries the value BEFORE the rate, because the attribution factor is
    still an open decision and comparing the two needs the unattributed figure.

    Args:
        df_values (pd.DataFrame): long values, with area_code, year and <value_column>.
        df_coefs (pd.DataFrame): the rental-rate lookup, with FAO, year and rental_rate.
        value_column (str): the value column to attribute.

    Returns:
        pd.DataFrame: with rental_rate and gross_production_value attached and <value_column>
        multiplied by the rate, sorted by country then year.
    """
    merged_parts = []
    for code, df_group in df_values.groupby('area_code', sort=True):
        lookup_sub = df_coefs[df_coefs['FAO'] == code]
        if lookup_sub.empty:
            df_group = df_group.copy()
            df_group['rental_rate'] = pd.NA
            merged_parts.append(df_group)
            continue
        merged = pd.merge_asof(
            left=df_group.sort_values('year'),
            right=lookup_sub.sort_values('year')[['year', 'rental_rate']],
            on='year', direction='backward')
        merged_parts.append(merged)

    df = pd.concat(merged_parts, ignore_index=True)
    df['gross_production_value'] = df[value_column]
    df[value_column] = df[value_column] * df['rental_rate']
    df = df.sort_values(by=['area_code', 'year'], ascending=[True, True])
    hb.log('Values merged with rental rates (%d rows).' % df.shape[0])
    return df


def sum_items_to_country_year(df, value_column):
    """Item rows summed to one row per country and year.

    A country-year whose every item value is missing stays MISSING: a bare sum maps an all-NaN
    group to zero, which publishes a country nobody valued as a zero. A group with any valued
    item keeps its sum, including a genuine zero.

    `gross_production_value` sums over the VALUED rows only, so it is the gross value behind
    the published value, never gross value the valuation skipped. For a caller whose valued
    rows share one rate per country-year and whose gross column is thousand USD against a USD
    value (crop provision), this makes the country identity value = rate x gross x 1000 hold;
    a caller with per-row rates or other units gets the valued-row restriction, not that
    identity. Summing every row instead pulls in dissolved-source-area rows (Ethiopia PDR
    beside Ethiopia) that carry gross value the rent never valued, which breaks crop's identity
    for ETH, SDN, SRB and CZE. The item table keeps those source rows untouched.
    """
    valued_mask = pd.to_numeric(df[value_column], errors='coerce').notna()
    agg_dict = {value_column: 'sum'}
    out = hb.df_groupby(df, ['iso3_r250_id', 'year'], agg_dict=agg_dict,
                        preserve='keep_all_valid')
    if 'gross_production_value' in df.columns:
        out = out.drop(columns=['gross_production_value'], errors='ignore')
        gpv = df[valued_mask].groupby(['iso3_r250_id', 'year'])['gross_production_value'] \
            .sum().rename('gross_production_value')
        out = out.merge(gpv, left_on=['iso3_r250_id', 'year'], right_index=True, how='left')
    out = out.sort_values(by=['iso3_r250_id', 'year'], ascending=[True, True])
    out[value_column] = pd.to_numeric(out[value_column], errors='coerce')
    n_valued = df[valued_mask].groupby(['iso3_r250_id', 'year']).size().rename('_n_valued')
    out = out.merge(n_valued, left_on=['iso3_r250_id', 'year'], right_index=True, how='left')
    out.loc[out['_n_valued'].isna(), value_column] = np.nan
    out = out.drop(columns='_n_valued')
    hb.log('Grouped by country-year (%d rows).' % out.shape[0])
    return out


def sum_countries_to_year(df, value_column):
    """Country-year rows summed to one global row per year."""
    out = hb.df_groupby(df, groupby_cols='year', agg_cols=value_column,
                        preserve='keep_all_valid')
    out.sort_values('year', inplace=True)
    hb.log('Grouped total by year (%d rows).' % out.shape[0])
    return out


# FAOSTAT keeps dissolved states under their own M49 codes. Each maps to the successor the
# country correspondence uses, so their production joins to a country instead of dropping.
M49_SUCCESSORS = {
    159: 156,   # China (mainland) -> China
    891: 688,   # Serbia and Montenegro -> Serbia
    200: 203,   # Czechoslovakia -> Czechia
    230: 231,   # Ethiopia PDR -> Ethiopia
    736: 729,   # Sudan (former) -> Sudan
}


def build_rental_rate_lookup(df_raw):
    """The CWoN rental rates as one row per country and decade start.

    The workbook is one column per decade ("1961-1970", "1971-1980", ...) keyed on the FAO area
    code. Melting it and keeping the decade's first year gives the lookup merge_crop_with_coefs
    reads as-of. Columns that are not a decade (the ISO3 label) carry no leading year, so they
    fall out with the rows whose decade start does not parse.

    Args:
        df_raw (pd.DataFrame): the CWoN coefficient table as shipped.

    Returns:
        pd.DataFrame: columns FAO, year, rental_rate.
    """
    df = df_raw.melt(id_vars=['Order', 'FAO', 'Country/territory'],
                     var_name='Decade', value_name='rental_rate')
    df['Decade_start'] = df['Decade'].str.extract(r'^(\d{4})').astype(float)
    df = df.dropna(subset=['Decade_start', 'FAO'])

    df = df[['FAO', 'Decade_start', 'rental_rate']].copy()
    df['FAO'] = df['FAO'].astype(int)
    df['Decade_start'] = df['Decade_start'].astype(int)
    df = df.rename(columns={'Decade_start': 'year'})
    hb.log(f'Prepared coef lookup ({df.shape[0]} rows).')
    return df


def normalize_m49_codes(df, column='area_code_M49', successors=None):
    """FAOSTAT's M49 area codes as integers, with dissolved states mapped to their successor.

    The codes arrive quoted ("'156"), so they are unquoted and cast before the mapping.

    Args:
        df (pd.DataFrame): a frame holding FAOSTAT area codes.
        column (str): the code column.
        successors (dict): code -> successor code, defaulting to M49_SUCCESSORS.

    Returns:
        pd.DataFrame: the frame with that column as integers, successors applied.
    """
    out = df.copy()
    out[column] = out[column].astype(str).str.replace("'", '', regex=False).astype(int)
    out[column] = out[column].replace(M49_SUCCESSORS if successors is None else successors)
    return out


def collapse_regions_to_countries(df_regions, attribute_columns, value_column, sum_column='total'):
    """Per-region totals summed to one row per country and year, with the country attributes back on.

    Summing the r264-expanded table as it stands would count a split country once per sub-region
    (China spans 6 r264 rows, India 6, France, Turkey, the UK and Pakistan 2), so the sum is taken
    on the r250 country id and the attributes are attached afterwards from one representative
    sub-region each.

    Args:
        df_regions (pd.DataFrame): the per-region table, carrying iso3_r250_id, year and
            `sum_column`.
        attribute_columns (list): the country attribute columns to carry through.
        value_column (str): what to call the summed value.
        sum_column (str): the column to sum.

    Returns:
        pd.DataFrame: one row per country and year, with the attributes attached.
    """
    totals = (df_regions.groupby(['iso3_r250_id', 'year'], as_index=False)[sum_column].sum()
              .rename(columns={sum_column: value_column}))
    attributes = df_regions[attribute_columns].drop_duplicates('iso3_r250_id')
    return totals.merge(attributes, how='left', on='iso3_r250_id')


def expand_country_values_to_regions(df_regions, df_by_country, value_column):
    """Each r264 region carrying its COUNTRY's value, for the map only.

    The result must never be summed: every sub-region of a split country carries the whole
    country's value, so a sum counts China six times.

    Args:
        df_regions (pd.DataFrame): the r264 regions.
        df_by_country (pd.DataFrame): one row per country, carrying iso3_r250_id and `value_column`.
        value_column (str): the value to attach.

    Returns:
        pd.DataFrame: df_regions with `value_column` attached.
    """
    return df_regions.merge(df_by_country[['iso3_r250_id', value_column]],
                            how='left', on='iso3_r250_id')



def faostat_item_group_members(membership_df, domain=None):
    """The FAOSTAT item-group membership as a two-column integer frame.

    Args:
        membership_df (pd.DataFrame): the staged Definitions and Standards Item Group listing,
            with `Domain Code`, `Item Group Code` and `Item Code` columns (or already-normalized
            `item_group_code`/`item_code`).
        domain (str): restrict to one FAOSTAT domain, e.g. 'QV' for Value of Production.

    Returns:
        pd.DataFrame: unique (item_group_code, item_code) integer pairs.
    """
    m = membership_df.copy()
    if domain is not None and 'Domain Code' in m.columns:
        m = m[m['Domain Code'] == domain]
    m = m.rename(columns={'Item Group Code': 'item_group_code', 'Item Code': 'item_code'})
    for column in ('item_group_code', 'item_code'):
        m[column] = pd.to_numeric(m[column], errors='coerce')
    m = m.dropna(subset=['item_group_code', 'item_code'])
    return (m[['item_group_code', 'item_code']].astype(int).drop_duplicates()
            .reset_index(drop=True))


def drop_aggregates_where_components_exist(df, aggregate_items, value_column, membership_df,
                                           country_column='area_code', log=None):
    """FAOSTAT group totals kept except where their OWN members are individually valued.

    FAOSTAT publishes crop and livestock value at two levels: the individual item, and the group
    total that adds those items up for you. "Oilcrops Primary" is oil palm plus coconuts plus the
    rest; "Livestock" is every animal product. A list holding both counts the same production
    twice, and it does so invisibly, because the result is a plausible number in the right units.

    Malaysia 2019 is the clearest case: oil palm $10.09bn plus coconuts $0.16bn is exactly
    Oilcrops Primary at $10.25bn, and counting both puts Malaysia at 179% of its own total crop
    production.

    Deleting every group total is not the fix, because this release stopped publishing individual
    items for many countries -- Argentina and India have no individual crop values at all in 2019,
    only group totals. And dropping a group because the country has ANY individual item is not the
    fix either: a country can carry "Roots and Tubers, Total" beside an individually valued Wheat,
    which the group never contained. So a group total is dropped exactly for a country-year where
    at least one of ITS OWN MEMBERS (per the FAOSTAT item-group definitions) is individually
    valued, and kept otherwise. A dropped group with only some members valued loses its unvalued
    members' share, which the log reports; whether to allocate such remainders is an accounting
    decision, not this function's.

    Args:
        df (pd.DataFrame): the long cleaned FAOSTAT frame, with `crop` and `crop_code`.
        aggregate_items (iterable): the group-total item names.
        value_column (str): the value column.
        membership_df (pd.DataFrame): the item-group membership, as staged or normalized
            (see faostat_item_group_members).
        country_column (str): the column identifying the country.
        log (callable): where to report what was dropped.

    Returns:
        pd.DataFrame: the frame with redundant group totals removed.
    """
    log = log or hb.log
    aggregates = set(aggregate_items)
    if 'crop' not in df.columns:
        raise NameError('The frame has no item column to identify group totals by; it carries %s.'
                        % list(df.columns)[:10])
    is_aggregate = df['crop'].isin(aggregates)
    if not is_aggregate.any():
        return df

    members = faostat_item_group_members(membership_df)
    out = df.reset_index(drop=True)
    is_aggregate = out['crop'].isin(aggregates)
    codes = pd.to_numeric(out['crop_code'], errors='coerce')
    valued = out[value_column].notna()

    valued_pairs = (out.loc[~is_aggregate & valued, [country_column, 'year']]
                    .assign(item_code=codes[~is_aggregate & valued])
                    .dropna(subset=['item_code']).drop_duplicates())
    valued_pairs['item_code'] = valued_pairs['item_code'].astype(int)
    group_rows = (out.loc[is_aggregate, [country_column, 'year']]
                  .assign(item_group_code=codes[is_aggregate])
                  .dropna(subset=['item_group_code']))
    group_rows['item_group_code'] = group_rows['item_group_code'].astype(int)
    group_rows['row'] = group_rows.index
    expanded = group_rows.merge(members, on='item_group_code')
    hits = expanded.merge(valued_pairs, on=[country_column, 'year', 'item_code'])
    redundant = out.index.isin(hits['row'])

    n_members_hit = hits.groupby('row').size()
    n_members = members.groupby('item_group_code').size()
    partial = sum(1 for row in n_members_hit.index
                  if n_members_hit[row] < n_members.get(int(codes[row]), 0))
    dropped_value = out.loc[redundant & valued, value_column].sum()
    kept = out.loc[is_aggregate & ~redundant, country_column].nunique()
    log('Group totals: %d rows dropped where their own members are individually valued '
        '(%.6g in the value column; %d of them with only PARTIAL member coverage, whose '
        'unvalued members\' share is lost with them), and kept in %d countries whose groups '
        'have no individually valued member.'
        % (int(redundant.sum()), dropped_value, partial, kept))
    return out[~redundant]


def warp_raster_to_pyramid_sum(src_path, dst_path, arcseconds):
    """Warp a per-cell quantity raster onto the global pyramid grid at `arcseconds`,
    conserving the global sum (GDAL's overlap-weighted sum resampling).

    For publishing a raster whose native grid is off the pyramid. Counts and currency are
    per-cell amounts, so average-style resampling would rescale the total by the cell-area
    ratio; sum keeps the map's total equal to the table's.

    Args:
        src_path (str): the native-grid raster.
        dst_path (str): where the pyramid-grid raster is written.
        arcseconds (float): a pyramid rung (1, 10, 30, 150, 300, 900, 1800, 3600, ...).

    Returns:
        str: dst_path.
    """
    if float(arcseconds) not in hb.pyramid_compatible_resolutions:
        # The table is keyed both ways, 150.0 and '150', so sorting it whole raises TypeError and
        # masks this ValueError with one that says nothing about the argument.
        rungs = sorted(k for k in hb.pyramid_compatible_resolutions if not isinstance(k, str))
        raise ValueError('arcseconds %r is not a pyramid rung (supported: %s)'
                         % (arcseconds, rungs))
    degrees = hb.pyramid_compatible_resolutions[float(arcseconds)]
    ndv = hb.get_ndv_from_path(src_path)

    # Exact overlap aggregation rather than gdal.Warp's sum resampling: measured on the
    # pollination value raster (180 arcseconds, global), the warp returned 109.6 percent of the
    # source total at 300 arcseconds and 81.3 percent at 900, so its weights do not conserve.
    # Each sparse matrix row holds the fraction of every source row (column) covered by one
    # target row (column); every source cell's fractions sum to one, so the global sum is
    # conserved by construction rather than by trust in a resampler.
    from scipy.sparse import coo_matrix

    def overlap(n_src, src_origin, src_step, n_dst, dst_origin, dst_step):
        dst_lo_all = min(dst_origin, dst_origin + n_dst * dst_step)
        step = abs(dst_step)
        rows, cols, vals = [], [], []
        for j in range(n_src):
            lo, hi = src_origin + j * src_step, src_origin + (j + 1) * src_step
            lo, hi = min(lo, hi), max(lo, hi)
            first = int(np.floor((lo - dst_lo_all) / step))
            last = int(np.ceil((hi - dst_lo_all) / step))
            for k in range(max(0, first), min(n_dst, last)):
                t_lo = dst_lo_all + k * step
                t_hi = t_lo + step
                cover = (min(hi, t_hi) - max(lo, t_lo)) / (hi - lo)
                if cover <= 0:
                    continue
                # k counts bands from the low coordinate; map it to the raster index,
                # which counts from dst_origin in the direction of dst_step.
                i = k if dst_step > 0 else n_dst - 1 - k
                rows.append(i); cols.append(j); vals.append(cover)
        return coo_matrix((vals, (rows, cols)), shape=(n_dst, n_src)).tocsr()

    src = gdal.Open(src_path)
    gt = src.GetGeoTransform()
    a = src.GetRasterBand(1).ReadAsArray().astype('float64')
    valid = np.isfinite(a)
    if ndv is not None:
        valid &= (a != ndv)
    a = np.where(valid, a, 0.0)

    n_dst_y, n_dst_x = int(round(180.0 / degrees)), int(round(360.0 / degrees))
    wy = overlap(src.RasterYSize, gt[3], gt[5], n_dst_y, 90.0, -degrees)
    wx = overlap(src.RasterXSize, gt[0], gt[1], n_dst_x, -180.0, degrees)
    out = wy @ a @ wx.T
    covered = (wy @ valid.astype('float64') @ wx.T) > 0
    out_ndv = float(ndv) if ndv is not None else -9999.0
    out = np.where(covered, out, out_ndv).astype('float32')

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(dst_path, n_dst_x, n_dst_y, 1, gdal.GDT_Float32,
                       options=['COMPRESS=DEFLATE', 'TILED=YES'])
    ds.SetGeoTransform((-180.0, degrees, 0.0, 90.0, 0.0, -degrees))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(out_ndv)
    band.WriteArray(out)
    # Sum-preserved per-cell quantities are extensive by construction; declaring it on the file
    # is what makes make_path_pog build sum overviews instead of defaulting to covariate means.
    ds.SetMetadataItem('POG_VARIABLE_CLASS', 'extensive')
    band.FlushCache()
    ds = None
    return dst_path


def publish_raster_as_pog(path, log=None):
    """Rewrite a published raster as a POG: a pyramidal COG.

    A result somebody opens should be readable without downloading all of it and should sit on the
    account's own grid. `hb.is_path_pog` is both of those in one call -- a pyramidal geotransform
    at a pyramid resolution, with overview levels [3, 15, 30, 90, 180, 360] and exact statistics
    stored internally, tiled and COG-valid.

    What is usually missing is NOT the grid but the overviews and the internal statistics,
    which is what `hb.make_path_pog` adds. A raster already on the pyramid still fails without
    them.

    Rewrites in place and is idempotent: a raster already a POG is left alone, so distributing
    twice costs nothing.

    Args:
        path (str): the raster to convert, already copied into the output directory.
        log (callable): where to report; hazelbean's log by default.

    Returns:
        bool: True if the file is a POG when this returns, False if it could not be made one --
        a raster off the pyramid entirely cannot be fixed by adding overviews, and saying so is
        more useful than raising inside a distribution step that runs last.
    """
    log = log or hb.log
    if hb.is_path_pog(path):
        return True
    try:
        hb.make_path_pog(path, output_raster_path=path)
    except Exception as error:
        log('  could not make %s a POG: %s' % (os.path.basename(path), error))
        return False
    if hb.is_path_pog(path):
        log('  wrote %s as a POG' % os.path.basename(path))
        return True
    log('  %s is still not a POG: it is not on the pyramid, which overviews cannot fix'
        % os.path.basename(path))
    return False


def distribute_results(p, service, log=None):
    """Copy a service's registered results into the project's output directory.

    One loop for every service, so a renamed service cannot leave behind a copy that looks up the
    wrong key: `p.results[<wrong key>]` raises in a task that runs last and only on a full tree.

    Args:
        p (ProjectFlow): the project, inside gep_results_distribution.
        service (str): the service's key in p.results.
        log (callable): where to report; hazelbean's log by default.
    """
    log = log or hb.log
    log('Distributing GEP results...')
    for name, path in p.results.get(service, {}).items():
        output_path = os.path.join(p.output_dir, name)
        hb.path_copy(path, output_path)
        log('Distributed %s to %s' % (name, output_path))
        # Every raster that leaves as a result leaves as a POG, once, here -- rather than each
        # service remembering to do it.
        if str(output_path).lower().endswith(('.tif', '.tiff')):
            publish_raster_as_pog(output_path, log=log)
    log('GEP results distribution complete.')


def require_workspace_signature(workspace, signature, signature_path, prepared_inputs=()):
    """Refuse unsigned/changed cached service data; allow interrupted same-input work.

    Record the contract before writing service outputs. An existing result is
    reusable only under this same contract; this does not certify completion.
    """
    reason = outputs_reuse_reason([], signature, signature_path)
    if reason is None:
        return
    artifacts = []
    allowed = {os.path.abspath(str(p)) for p in prepared_inputs}
    if os.path.isdir(workspace):
        for folder, _, names in os.walk(workspace):
            artifacts.extend(os.path.join(folder, n) for n in names
                             if n.lower().endswith(('.tif', '.tiff', '.csv', '.parquet'))
                             and os.path.abspath(os.path.join(folder, n)) not in allowed)
    if artifacts:
        raise ValueError('Unverified or changed service workspace %s (%s). Preserve it and use a '
                         'fresh directory; example cached file: %s' % (workspace, reason, artifacts[0]))
    os.makedirs(workspace, exist_ok=True)
    write_outputs_signature(signature, signature_path)
