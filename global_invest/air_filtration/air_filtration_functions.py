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

AIR_FILTRATION_VSL_USA = 9_900_000    # the method's US anchor VSL (vsl.R)
AIR_FILTRATION_MIN_NAME_MATCHES = 190  # positional-join sanity floor (199 observed)

# How closely the VSL rebuilt from the group's country table must match the workbook's own
# column before the difference is worth reporting. The two are the same build, so a country that
# differs is either a stale workbook row or a revision, and either way somebody has to say which.
VSL_AGREEMENT_RTOL = 1e-6

# The table keys on a slug. Slugifying our country description matches 205 of 250 outright; these
# are the ones whose slug the table spells differently, keyed by ISO3 so a name edit cannot break
# them. Macao and Palestine are deliberately absent: the workbook prices both per country and the
# table has neither, so they are the two values we cannot source and the run says so.
VSL_TABLE_SLUG_BY_ISO3 = {
    'KOR': 'korea-south', 'PRK': 'korea-north', 'FSM': 'micronesia-federated-states-of',
    'BHS': 'bahamas-the', 'GMB': 'gambia-the', 'TUR': 'turkey-turkiye', 'CIV': 'cote-divoire',
    'MMR': 'burma', 'VIR': 'virgin-islands', 'COD': 'congo-democratic-republic-of-the',
    'COG': 'congo-republic-of-the', 'HKG': 'hong-kong', 'SWZ': 'eswatini',
}


def vsl_table_slug(description, iso3):
    """The table's key for a country: its ISO3 alias if it has one, else the slugified name."""
    import re
    import unicodedata

    if iso3 in VSL_TABLE_SLUG_BY_ISO3:
        return VSL_TABLE_SLUG_BY_ISO3[iso3]
    ascii_name = (unicodedata.normalize('NFKD', str(description))
                  .encode('ascii', 'ignore').decode())
    return re.sub(r'[^a-z0-9]+', '-', ascii_name.lower().replace('&', ' and ')).strip('-')


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


def vsl_from_country_table(workbook_df, r250_order_df, vsl_df):
    """The VSL column rebuilt from the air quality group's country table.

    The workbook carries a VSL column we did not compute. This builds the same column from the
    group's published country-level table so the valuation reads a source we hold rather than a
    number handed to us. A country the table names takes its value; a country it does not takes
    the mean of the ones it does, which is the fill rule the workbook itself follows (see
    verify_global_average_fill).

    The join is on the r250 order's `ee_r264_description`, not on the workbook's `Country`, for
    the same reason air_quality_gep_by_country joins by position: the workbook calls both FID 137
    and FID 232 "Serbia" when the second is Kosovo, so a name join hands Serbia's value to Kosovo
    and never resolves Kosovo at all. The description column distinguishes them.

    Args:
        workbook_df (pd.DataFrame): the workbook, carrying `VSL`, in r250 row order.
        r250_order_df (pd.DataFrame): the r250 order, carrying `ee_r264_description`.
        vsl_df (pd.DataFrame): the group's table, carrying `country` and `vsl`.

    Returns:
        (pd.Series, pd.DataFrame): the rebuilt VSL positionally aligned to the workbook, and the
        rows where it disagrees with the workbook's own column by more than VSL_AGREEMENT_RTOL.
    """
    import pandas as pd

    order = r250_order_df.reset_index(drop=True)
    key = [vsl_table_slug(d, i) for d, i in
           zip(order['ee_r264_description'], order['iso3_r250_label'])]
    table = (vsl_df.assign(_k=vsl_df['country'].astype(str).str.strip().str.lower())
             .drop_duplicates('_k').set_index('_k')['vsl'])
    rebuilt = pd.Series(key, name='vsl').map(table)
    matched = int(rebuilt.notna().sum())

    workbook_vsl = workbook_df['VSL'].reset_index(drop=True)
    source = workbook_df['VSL_Source'].reset_index(drop=True)

    # A country the workbook prices itself but the table does not name keeps the workbook's
    # figure, and is reported: it is a value we could not source, not one to quietly average away.
    unsourced = rebuilt.isna() & (source == 'country')
    rebuilt = rebuilt.where(~unsourced, workbook_vsl)
    # The rest are the workbook's global_avg rows, which carry the mean of the priced ones.
    rebuilt = rebuilt.fillna(rebuilt[~unsourced].mean())
    difference = (rebuilt - workbook_vsl).abs() / workbook_vsl.abs()
    report = pd.DataFrame({'country': order['ee_r264_description'],
                           'iso3': order['iso3_r250_label'],
                           'workbook_vsl': workbook_vsl,
                           'table_vsl': rebuilt,
                           'relative_difference': difference,
                           'unsourced': unsourced})
    return rebuilt, matched, report[difference > VSL_AGREEMENT_RTOL].copy(), report[unsourced].copy()


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
