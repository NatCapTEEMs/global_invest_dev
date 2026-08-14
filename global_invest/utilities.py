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
