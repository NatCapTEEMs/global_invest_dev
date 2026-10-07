"""Air-quality science: avoided mortality x VSL, two channels from one drive workbook.

The consortium drive's Air Filtration folder carries the committed per-country workbook
(air_filtration_gep.xlsx): avoided deaths from DEPOSITION (dry pollutant capture by vegetation
-- the sheet's air_filtration service) and from DUST (windblown-dust suppression -- the sheet's
sandstorm prevention service), each valued at a GDP-adjusted VSL. The upstream science behind
the deaths columns (process-based emissions models + Global InMAP + health impacts, per the
appendix) is not rebuildable here and is taken as given; the valuation layer is rebuilt, prices
at the account's shared VSL panel, and reports where the workbook's own VSL column (which the
manuscript's $17.81bn deposition total carries) differs from it.

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

AIR_FILTRATION_VSL_USA = 9_900_000    # the method's US anchor VSL (vsl.R)
AIR_FILTRATION_MIN_NAME_MATCHES = 190  # positional-join sanity floor (199 observed)

# How closely the VSL rebuilt from the group's country table must match the workbook's own
# column before the difference is worth reporting. The two are the same build, so a country that
# differs is either a stale workbook row or a revision, and either way somebody has to say which.
VSL_AGREEMENT_RTOL = 1e-6

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


def vsl_from_shared_panel(workbook_df, r250_order_df, panel_df):
    """The VSL column from the account's shared panel, positionally aligned to the workbook.

    The panel is the rebuilt EPA life-years-lost table both mortality services read, keyed by
    iso3_r250_label with every account country covered, so there is no fill rule and a missing
    country is an error rather than an average. The workbook's own VSL column no longer prices
    anything; rows where the panel departs from it beyond VSL_AGREEMENT_RTOL are returned so
    the run can report the size of the repricing it applies.

    Args:
        workbook_df (pd.DataFrame): the workbook, carrying `VSL`, in r250 row order.
        r250_order_df (pd.DataFrame): the r250 order, carrying `iso3_r250_label`.
        panel_df (pd.DataFrame): the shared panel, carrying `iso3_r250_label` and `vsl_usd`.

    Returns:
        (pd.Series, pd.DataFrame): the panel VSL positionally aligned to the workbook, and the
        rows where it departs from the workbook's own column.
    """
    import pandas as pd

    order = r250_order_df.reset_index(drop=True)
    table = panel_df.drop_duplicates('iso3_r250_label').set_index('iso3_r250_label')['vsl_usd']
    vsl = order['iso3_r250_label'].map(table).rename('vsl')
    if vsl.isna().any():
        raise ValueError('the shared VSL panel is missing: %s'
                         % sorted(order.loc[vsl.isna(), 'iso3_r250_label']))
    workbook_vsl = workbook_df['VSL'].reset_index(drop=True)
    difference = (vsl - workbook_vsl).abs() / workbook_vsl.abs()
    report = pd.DataFrame({'country': order['ee_r264_description'],
                           'iso3': order['iso3_r250_label'],
                           'workbook_vsl': workbook_vsl,
                           'table_vsl': vsl,
                           'relative_difference': difference})
    return vsl, report[difference > VSL_AGREEMENT_RTOL].copy()


def air_quality_benefits(workbook_df, vsl=None):
    """Deaths x VSL per channel. The tests hold the recomputation to the workbook's benefit
    columns. `vsl` supplies the column rebuilt from the group's country table; without it the
    workbook's own column is used, which is what the benefit-column tests compare against."""
    df = workbook_df.copy()
    valuation = workbook_df['VSL'] if vsl is None else vsl
    df['air_filtration_gep'] = df['Dep_Deaths'] * valuation
    df['sandstorm_prevention_gep'] = df['Dust_Deaths'] * valuation
    return df


def air_quality_gep_by_country(workbook_df, r250_order_df, vsl=None):
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
    df = air_quality_benefits(workbook_df, vsl=vsl).reset_index(drop=True)
    out = r250_order_df[['iso3_r250_id', 'iso3_r250_label']].reset_index(drop=True).copy()
    out[['air_filtration_gep', 'sandstorm_prevention_gep']] = (
        df[['air_filtration_gep', 'sandstorm_prevention_gep']])
    return out
