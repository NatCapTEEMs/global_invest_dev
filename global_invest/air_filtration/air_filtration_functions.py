"""Air-quality science: avoided mortality x VSL, two channels from one drive workbook.

The consortium drive's Air Filtration folder carries the committed per-country workbook
(air_filtration_gep.xlsx): avoided deaths from DEPOSITION (dry pollutant capture by vegetation
-- the sheet's air_filtration service) and from DUST (windblown-dust suppression -- the sheet's
sandstorm prevention service). The upstream science behind the deaths columns (process-based
emissions models + Global InMAP + health impacts, per the appendix) is not rebuildable here and
is taken as given. The valuation prices those deaths at the account's shared VSL panel -- the
same file landslide mitigation reads: the US anchor of $9.9M (2019 US$), scaled by GDP per
capita (World Bank, constant 2015 US$) and life-years lost, with countries lacking data taking
their World Bank region's median, else the global median. The workbook's own VSL column prices nothing; the run reports
how far the panel departs from it. The countries the source states it could not estimate
publish as missing (not_estimated_mask), not as the zero deaths the workbook stores.

Identified properties of the WORKBOOK, asserted in the tests:
- Its rows are the r250 geopackage in row order (FID 1..250); names differ from our
  correspondence only by which name column was used (199 of 250 identical, the rest sovereign-
  vs-territory naming of the same rows).
- Its own VSL column fills countries without a country-specific VSL with the global average of
  the country-source VSLs (exact) -- the workbook's rule, not the account's.

Provenance gap, recorded: the folder's vsl.R and raw CSVs document the workbook's VSL method
(US-anchored, GDP- and life-years-adjusted) but do not reproduce the workbook's VSL column -- a
different data vintage. Pricing does not depend on it; gdp_adjusted_vsl ports that documented
method for transparency.
"""
import numpy as np

AIR_FILTRATION_VSL_USA = 9_900_000    # the workbook method's US anchor VSL (vsl.R); the account's is the panel's US row
AIR_FILTRATION_MIN_NAME_MATCHES = 190  # positional-join sanity floor (199 observed)
AIR_FILTRATION_FID_START = 1  # the workbook's FID is the gpkg feature id, 1..250 in file order

# How closely the shared-panel VSL must match the workbook's own column before a row counts as
# repriced in the run's report.
VSL_AGREEMENT_RTOL = 1e-6

def gdp_adjusted_vsl(life_expectancy_df, median_age_df, gdp_df):
    """The documented VSL method (vsl.R, ported for transparency): US VSL per life-year lost,
    scaled to GDP, applied to each country's own life-years lost (life expectancy minus median
    age). Slug-keyed like its raw inputs. NOTE the module-docstring provenance gap: with the
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
    """Assert the WORKBOOK's own fill rule (every global_avg row carries exactly the mean of the
    country-source VSLs), so a changed workbook cannot slip by unnoticed. The account does not
    price with this column; its fills are the shared panel's regional and global medians.
    Returns the workbook's mean."""
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
    country is an error rather than an average. The workbook's own VSL column prices nothing;
    rows where the panel departs from it beyond VSL_AGREEMENT_RTOL are returned so
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


def age_at_population_percentile(age_structure_df, percentile, key):
    """The authors' elderly age per country: the last single age whose cumulative population
    share is still below the percentile (clean_pyramid.R in the authors' updated code).

    The convention is the authors' and is kept exactly: the age returned is the one just below
    the percentile, not the first age at or above it. Ages are read in ascending order and a key
    holding several source countries (Gaza Strip and West Bank in one r250 country) pools their
    populations by age before the shares are taken.

    Args:
        age_structure_df (pd.DataFrame): the single-year age structure, one row per source
            country and age, with `GROUP` (the age, '100+' for the last, empty on the total row),
            `Population` (thousands separators allowed) and the key column.
        percentile (float): the cumulative share, in percent, that marks the elderly age.
        key (str): the column that identifies a country.

    Returns:
        pd.DataFrame: the key and `elderly_age`, one row per key.

    Raises:
        ValueError: if a key has no total row or its ages do not sum to its total.
    """
    import pandas as pd

    df = age_structure_df[[key, 'GROUP', 'Population']].copy()
    df['population'] = pd.to_numeric(df['Population'].astype(str).str.replace(',', ''))
    total = df[df['GROUP'].isna()].groupby(key)['population'].sum()
    ages = df[df['GROUP'].notna()].assign(age=lambda d: d['GROUP'].str.rstrip('+').astype(int))
    ages = ages.groupby([key, 'age'], as_index=False)['population'].sum().sort_values([key, 'age'])
    if set(ages[key]) != set(total.index):
        raise ValueError('countries without a total row: %s' % sorted(set(ages[key]) ^ set(total.index)))
    summed = ages.groupby(key)['population'].sum()
    if not (summed == total.loc[summed.index]).all():
        raise ValueError('ages do not sum to the total in: %s' % sorted(summed.index[summed != total.loc[summed.index]]))
    ages['cum_share'] = (ages['population'] / ages[key].map(total)).groupby(ages[key]).cumsum() * 100
    below = ages[ages['cum_share'] < percentile]
    return below.groupby(key, as_index=False)['age'].last().rename(columns={'age': 'elderly_age'})


def vsl_with_age_statistic(panel_df, elderly_age_df):
    """The shared VSL panel with life-years lost measured from the authors' elderly age.

    The panel scales the US VSL by GDP per capita and by life-years lost, life expectancy less
    median age: vsl_c = vsl_us x (gdp_c / gdp_us) x (lll_c / lll_us). The authors' updated code
    measures life-years lost from the elderly age (`age_at_population_percentile`) instead. Only
    that term changes, in each country and in the US: the US anchor, the GDP ratio and the life
    expectancies are the panel's. A country the panel values directly keeps a direct value where
    it has an elderly age. Every other country takes the panel's own fill, the median direct VSL
    of its World Bank region, else the global median.

    Args:
        panel_df (pd.DataFrame): the shared panel (iso3_r250_id, iso3_r250_label, region_wb,
            life_expectancy, median_age, lll, gdp_pc, vsl_usd, vsl_source).
        elderly_age_df (pd.DataFrame): iso3_r250_id and elderly_age.

    Returns:
        pd.DataFrame: the panel with `elderly_age` added and `lll`, `vsl_usd` and `vsl_source`
        recomputed. Every other column is the panel's.

    Raises:
        ValueError: if the US has no elderly age or no direct panel value.
    """
    out = panel_df.copy()
    out['elderly_age'] = out['iso3_r250_id'].map(elderly_age_df.set_index('iso3_r250_id')['elderly_age'])
    out['lll'] = out['life_expectancy'] - out['elderly_age']
    direct = (panel_df['vsl_source'] == 'direct') & out['lll'].notna()
    us = out['iso3_r250_label'] == 'USA'
    if not direct[us].all() or us.sum() != 1:
        raise ValueError('the US needs a direct panel value and an elderly age')
    anchor = out.loc[us, 'vsl_usd'].iloc[0]
    gdp_us, lll_us = out.loc[us, 'gdp_pc'].iloc[0], out.loc[us, 'lll'].iloc[0]
    out['vsl_usd'] = np.where(direct, anchor * (out['gdp_pc'] / gdp_us) * (out['lll'] / lll_us), np.nan)
    region = out[direct].groupby('region_wb')['vsl_usd'].median()
    out.loc[~direct, 'vsl_usd'] = out.loc[~direct, 'region_wb'].map(region)
    out['vsl_source'] = np.where(direct, 'direct', 'region_median_fallback')
    unfilled = out['vsl_usd'].isna()
    out.loc[unfilled, 'vsl_usd'] = out.loc[direct, 'vsl_usd'].median()
    out.loc[unfilled, 'vsl_source'] = 'global_median_fallback'
    return out


def air_quality_benefits(workbook_df, vsl=None):
    """Deaths x VSL per channel. The tests hold the recomputation to the workbook's benefit
    columns. `vsl` supplies the shared-panel column; without it the workbook's own column is
    used, which is what the benefit-column tests compare against."""
    df = workbook_df.copy()
    valuation = workbook_df['VSL'] if vsl is None else vsl
    df['air_filtration_gep'] = df['Dep_Deaths'] * valuation
    df['sandstorm_prevention_gep'] = df['Dust_Deaths'] * valuation
    return df


def not_estimated_mask(workbook_df, r250_order_df, not_estimated_labels):
    """Positions of the countries the source did not estimate, from an identified list.

    The list is the source's own coverage statement for this workbook version, not a rule: a
    zero elsewhere in the workbook stays a measured zero. Every listed country must carry zero
    deaths in both channels in this workbook, so a list written for another version fails here
    instead of blanking a measured value.
    """
    labels = r250_order_df['iso3_r250_label'].reset_index(drop=True)
    listed = set(not_estimated_labels)
    unknown = sorted(listed - set(labels))
    if unknown:
        raise ValueError(f'not-estimated list names countries outside the r250 order: {unknown}')
    mask = labels.isin(listed).to_numpy()
    deaths = workbook_df[['Dep_Deaths', 'Dust_Deaths']].reset_index(drop=True)
    measured = mask & (deaths != 0).any(axis=1).to_numpy()
    if measured.any():
        raise ValueError('listed as not estimated but the workbook carries deaths: '
                         f'{sorted(labels[measured])} -- the list does not match this workbook')
    return mask


def air_quality_gep_by_country(workbook_df, r250_order_df, vsl=None, not_estimated_labels=()):
    """Join the workbook onto the r250 ids by POSITION, which is not a shortcut but a requirement.

    The workbook carries no country code, only an FID that is the geopackage's feature id, so a
    name join looks like the safer option. It is not: the workbook has two rows both called
    Serbia, FID 137 and FID 232, and position 232 in the r250 order is XKX, Kosovo. Joining on the
    name would give Serbia both rows and drop Kosovo entirely. The order carries information the
    names do not.

    A name-equality floor guards it: fewer than AIR_FILTRATION_MIN_NAME_MATCHES identical names
    means the order changed and the join must not proceed -- but a floor alone cannot see a
    two-row swap, so two per-row guards sit in front of it: the FID column must be the file
    order itself (any whole-row reorder breaks it), and no mismatched position may carry a
    name the r250 list knows (any data-column reorder among real countries breaks it). The
    known blind spot, recorded in the tests: a data-only swap between two VARIANT rows (for
    example Mayotte and Tokelau) leaves no name or FID signal and only a keyed workbook from
    the air group would close it."""
    if len(workbook_df) != len(r250_order_df):
        raise ValueError(f'row-count mismatch: workbook {len(workbook_df)} vs r250 {len(r250_order_df)}')
    fid = workbook_df['FID'].to_numpy()
    expected = np.arange(AIR_FILTRATION_FID_START, AIR_FILTRATION_FID_START + len(workbook_df))
    if not (fid == expected).all():
        moved = workbook_df.loc[fid != expected, 'Country'].tolist()
        raise ValueError(f'positional join refused: the workbook rows are not in FID order '
                         f'(first moved rows: {moved[:5]}) -- the file was reordered.')
    # Per-row guard: at a position where the names differ, a workbook name that exists anywhere
    # in the r250 list means a real country's row sits at the wrong position. A swap of whole
    # rows carries its FID and is caught above; a swap of the data columns alone is caught here.
    # The workbook's duplicated name (Serbia, whose second row is Kosovo's by FID order) is the
    # recorded exception; spelling and territory variants never appear in the r250 list (zero
    # hits on the shipped file).
    w = workbook_df['Country'].astype(str).to_numpy()
    b = r250_order_df['brk_name'].astype(str).to_numpy()
    duplicated = set(workbook_df['Country'][workbook_df['Country'].duplicated(keep=False)])
    displaced = [w[i] for i in np.where(w != b)[0] if w[i] in set(b) and w[i] not in duplicated]
    if displaced:
        raise ValueError(f'positional join refused: known country names sit at the wrong '
                         f'positions: {displaced[:5]} -- the data columns were reordered.')
    matches = int((workbook_df['Country'].values == r250_order_df['brk_name'].values).sum())
    if matches < AIR_FILTRATION_MIN_NAME_MATCHES:
        raise ValueError(f'positional join refused: only {matches} identical names '
                         f'(floor {AIR_FILTRATION_MIN_NAME_MATCHES}) -- the row order changed.')
    df = air_quality_benefits(workbook_df, vsl=vsl).reset_index(drop=True)
    out = r250_order_df[['iso3_r250_id', 'iso3_r250_label']].reset_index(drop=True).copy()
    out[['air_filtration_gep', 'sandstorm_prevention_gep']] = (
        df[['air_filtration_gep', 'sandstorm_prevention_gep']])
    unestimated = not_estimated_mask(workbook_df, r250_order_df, not_estimated_labels)
    out.loc[unestimated, ['air_filtration_gep', 'sandstorm_prevention_gep']] = np.nan
    return out
