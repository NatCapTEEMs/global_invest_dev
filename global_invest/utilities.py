
# The scenario-name mapping that used to live here (our label -> a service's frozen-table label) was
# NGFS-specific and has moved to the consumer: each service's static shock task now defaults to identity
# and warns loudly if a scenario is absent from its table, and ngfs_pnas sets p.<service>_scenario_map
# with its two non-identity entries. A general library should not hardcode one project's scenario names.


# example utility function

def convert_currency(value, from_currency, to_currency, exchange_rate):
    """
    Convert a value from one currency to another using the provided exchange rate.
    """
    
    pass
