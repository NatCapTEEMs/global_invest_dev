
# Our scenario label -> the label(s) that scenario goes by in a service's frozen dependency table, with
# fallbacks (stress_test reuses current_policies; net_zero appears year-suffixed in some tables). Shared
# because every service's frozen table currently uses the same labels; a service whose table diverges
# overrides it per-project via p.<service>_scenario_map rather than editing this.
ES_SCENARIO_MAP = {
    'below_2c': ['below_2c'], 'current_policies': ['current_policies'],
    'delayed_transition': ['delayed_transition'], 'fragmented_world': ['fragmented_world'],
    'low_demand': ['low_demand'], 'ndcs': ['ndcs'],
    'net_zero': ['net_zero', 'net_zero_2050'], 'stress_test': ['current_policies'],
}


# example utility function

def convert_currency(value, from_currency, to_currency, exchange_rate):
    """
    Convert a value from one currency to another using the provided exchange rate.
    """
    
    pass
