"""Water-quality (purification) science: nutrient retention valued at treatment cost.

The consortium drive's Water Quality folder carries the committed chain but no script. The
verified stages, identities exact against the committed intermediates: per country and
nutrient (nitrogen, phosphorus), EstimatedRetention(kg) = TotalRetention(kg) x the AQUASTAT
domestic-water-use fraction, and ServiceValue(USD) = EstimatedRetention x the treatment cost
per kg. The retention totals themselves are upstream science (an NDR-style estimation) taken
as given.

The final accounting output (water_quality_gep.csv) is expressed in 2019 INTERNATIONAL dollars
per the drive's method note. That conversion stage is NOT identified: it is not the plain sum
(per-country ratios up to ~11x), and the single-factor XR/PPP hypothesis was tested against
the WDI series and rejected (median 36 percent off) -- the harmonization uses per-source-year
cost inputs the folder does not carry. The committed international-dollar values are carried
as the provisional account column; the conversion script and its inputs are the open ask.
This also raises a cross-cutting question flagged on the deck: every other service values in
2019 market USD.
"""
import numpy as np
import pandas as pd

WATER_QUALITY_NUTRIENTS = ('n', 'p')   # nitrogen, phosphorus


def element_service_values(retention_df):
    """Recompute the two verified identities per nutrient and return the frame with our
    recomputed columns beside the committed ones. Raises if either identity breaks."""
    df = retention_df.copy()
    for nutrient in WATER_QUALITY_NUTRIENTS:
        estimated = df[f'{nutrient}_TotalRetention(kg)'] * df['DomesticWaterUseFraction']
        value = estimated * df[f'{nutrient}_Price(usd)']
        for ours, committed in ((estimated, f'{nutrient}_EstimatedRetention(kg)'),
                                (value, f'{nutrient}_ServiceValue(usd)')):
            both = ours.notna() & df[committed].notna()
            if both.any() and not np.allclose(ours[both], df.loc[both, committed], rtol=1e-9):
                raise ValueError(f'the committed {committed} no longer follows its identity '
                                 f'-- re-identify before trusting the chain.')
        df[f'{nutrient}_gep_usd'] = value
    return df


def water_quality_gep_by_country(retention_df, international_df, countries_df):
    """One row per country: the recomputed USD components plus the committed
    international-dollar accounting value (conversion stage unidentified; see module docstring)."""
    usd = element_service_values(retention_df)[
        ['iso3_r250_label', 'n_gep_usd', 'p_gep_usd']].rename(
        columns={'n_gep_usd': 'nitrogen_gep_usd', 'p_gep_usd': 'phosphorus_gep_usd'})
    usd['water_quality_gep_usd'] = usd['nitrogen_gep_usd'] + usd['phosphorus_gep_usd']
    df = countries_df.merge(usd, on='iso3_r250_label', how='left')
    df = df.merge(international_df[['iso3_r250_label', 'water_quality_gep']],
                  on='iso3_r250_label', how='left')
    return df
