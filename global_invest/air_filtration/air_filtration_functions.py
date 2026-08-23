"""Air-quality science: avoided mortality x VSL, two channels from one drive workbook.

The consortium drive's Air Filtration folder carries the committed per-country workbook
(air_filtration_gep.xlsx): avoided deaths from DEPOSITION (dry pollutant capture by vegetation
-- the sheet's air_filtration service) and from DUST (windblown-dust suppression -- the sheet's
sandstorm prevention service), each valued at a GDP-adjusted VSL. The upstream science behind
the deaths columns (process-based emissions models + Global InMAP + health impacts, per the
appendix) is not rebuildable here and is taken as given; the valuation layer is rebuilt and
verified against the workbook exactly. The deposition total, $17.81bn, is the manuscript's
air-filtration number.

Two identified rules, both asserted in the tests:
- The workbook's rows are the r250 geopackage in row order (FID 1..250); names differ from our
  correspondence only by which name column was used (199 of 250 identical, the rest sovereign-
  vs-territory naming of the same rows).
- Countries without a country-specific VSL take the global average of the country-source VSLs
  (identified from the workbook, exact).

VINTAGE GAP, flagged: the folder's vsl.R and raw CSVs document the VSL method (US-anchored,
GDP- and life-years-adjusted) but do NOT reproduce the workbook's VSL column -- a different
data vintage. The VSL build behind the workbook is an open ask; gdp_adjusted_vsl below ports
the documented method for transparency.
"""
import numpy as np
import pandas as pd

AIR_FILTRATION_VSL_USA = 9_900_000    # the method's US anchor VSL (vsl.R)
AIR_FILTRATION_MIN_NAME_MATCHES = 190  # positional-join sanity floor (199 observed)


def gdp_adjusted_vsl(life_expectancy_df, median_age_df, gdp_df):
    """The documented VSL method (vsl.R, ported for transparency): US VSL per life-year lost,
    scaled to GDP, applied to each country's own life-years lost (life expectancy minus median
    age). Slug-keyed like its raw inputs. NOTE the module-docstring vintage gap: with the
    committed raw data this does not reproduce the workbook's VSL column."""
    df = (life_expectancy_df[['slug', 'years']].rename(columns={'years': 'life_expectancy'})
          .merge(median_age_df[['slug', 'years']].rename(columns={'years': 'median_age'}), on='slug')
          .merge(gdp_df[['slug', 'gdp_real']], on='slug'))
    usa = df[df['slug'] == 'united-states'].iloc[0]
    vsl_per_life_year_to_gdp = (AIR_FILTRATION_VSL_USA
                                / (usa['life_expectancy'] - usa['median_age'])
                                / usa['gdp_real'])
    df['life_years_lost'] = df['life_expectancy'] - df['median_age']
    df['vsl'] = df['gdp_real'] * df['life_years_lost'] * vsl_per_life_year_to_gdp
    return df[['slug', 'vsl']]


def verify_global_average_fill(workbook_df):
    """Assert the identified fill rule: every global_avg row carries exactly the mean of the
    country-source VSLs. Returns that mean."""
    country_vsl = workbook_df.loc[workbook_df['VSL_Source'] == 'country', 'VSL']
    fill = workbook_df.loc[workbook_df['VSL_Source'] == 'global_avg', 'VSL']
    global_average = country_vsl.mean()
    if len(fill) and not np.allclose(fill, global_average, rtol=1e-9):
        raise ValueError('the workbook no longer follows the identified global-average VSL '
                         'fill rule -- re-identify before trusting the valuation.')
    return global_average


def air_quality_benefits(workbook_df):
    """Deaths x VSL per channel, recomputed from the workbook's own columns. The tests hold
    the recomputation to the workbook's benefit columns."""
    df = workbook_df.copy()
    df['air_filtration_gep'] = df['Dep_Deaths'] * df['VSL']
    df['sandstorm_prevention_gep'] = df['Dust_Deaths'] * df['VSL']
    return df


def air_quality_gep_by_country(workbook_df, r250_order_df):
    """Join the workbook onto the r250 ids by POSITION, which is not a shortcut but a requirement.

    The workbook carries no country code, only an FID that is the geopackage's feature id, so a
    name join looks like the safer option. It is not: the workbook has two rows both called
    Serbia, FID 137 and FID 232, and position 232 in the r250 order is XKX, Kosovo. Joining on the
    name would give Serbia both rows and drop Kosovo entirely. The order carries information the
    names do not.

    A name-equality floor guards it: fewer than AIR_FILTRATION_MIN_NAME_MATCHES identical names
    means the order changed and the join must not proceed. That is what stands between this and
    every country taking its neighbour's deaths."""
    if len(workbook_df) != len(r250_order_df):
        raise ValueError(f'row-count mismatch: workbook {len(workbook_df)} vs r250 {len(r250_order_df)}')
    matches = int((workbook_df['Country'].values == r250_order_df['brk_name'].values).sum())
    if matches < AIR_FILTRATION_MIN_NAME_MATCHES:
        raise ValueError(f'positional join refused: only {matches} identical names '
                         f'(floor {AIR_FILTRATION_MIN_NAME_MATCHES}) -- the row order changed.')
    df = air_quality_benefits(workbook_df).reset_index(drop=True)
    out = r250_order_df[['iso3_r250_id', 'iso3_r250_label']].reset_index(drop=True).copy()
    out[['air_filtration_gep', 'sandstorm_prevention_gep']] = (
        df[['air_filtration_gep', 'sandstorm_prevention_gep']])
    return out
