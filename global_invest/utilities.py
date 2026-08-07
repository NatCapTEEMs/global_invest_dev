
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
