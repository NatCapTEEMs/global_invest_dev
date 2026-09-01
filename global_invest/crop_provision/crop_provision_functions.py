# -*- coding: utf-8 -*-
"""Crop-provision science: FAOSTAT gross production value x the CWoN land rental rate.

The quantity-and-price stage is already fused in the source data: FAOSTAT's Value of Production
bulk file reports gross production value per country, crop and year, so no separate price join
happens here. Attribution is the Changing Wealth of Nations 2024 land rental rate, a per-country
share that varies by decade, applied by an as-of merge so each year takes the rate of the decade
it falls in. The result is converted from FAOSTAT's thousand USD to plain USD once, before any
grouping, and collapsed onto the r250 country list.

The subsistence component sits at the bottom of this module, the way the Lynch subsistence value
sits beside the commercial rent in fisheries: same folder, own task, own table, never summed into
the commercial figure. It narrows the same production value by two survey shares, which is what
makes the overlap between the two components exact rather than a matter of judgement.

Every function here is a pure transformation over frames, which is what the tests exercise. The
task module reads the FAOSTAT bulk file and the rental-rate table and passes the frames in.
"""
import logging

import numpy as np
import pandas as pd
import hazelbean as hb

from global_invest import utilities

# FAOSTAT's Value of Production table stacks several elements and units in one file. The
# valuation reads gross production value in current USD, which the file reports as element 57
# with unit "1000 USD".
FAOSTAT_VALUE_UNIT = '1000 USD'
FAOSTAT_GROSS_PRODUCTION_VALUE_ELEMENT = 57
# FAOSTAT ships crop values in thousand USD; every service in the library reports plain USD.
FAOSTAT_THOUSAND_USD = 1000.0
# The bulk file's year columns run Y1961 to Y2022, each shadowed by a Y<year>F data-quality flag.
# FAOSTAT area 223 is Turkiye, which recent releases spell several ways. The country join runs on
# the M49 code, so the name is normalised only to keep the crop-level table readable.
FAOSTAT_TURKIYE_AREA_CODE = 223


# The columns a crop-level row is identified by, before the year columns are melted down.
CROP_ID_COLUMNS = ['area_code', 'area_code_M49', 'country', 'crop_code', 'crop']






def attach_countries_in_usd(df_crop_value, df_countries, value_column='crop_provision_gep'):
    """Crop-level rows carrying their country's identifiers and attributes, in plain USD.

    The correspondence is collapsed to one row per country first: joining against r264 as shipped
    would repeat a split country's production once per sub-region. The join keeps every FAOSTAT
    row (a row whose M49 code the correspondence does not carry keeps missing identifiers rather
    than disappearing), and the unit conversion happens here, once, before any grouping.

    Args:
        df_crop_value (pd.DataFrame): long crop values with integer area_code_M49 and
            crop_provision_gep in thousand USD.
        df_countries (pd.DataFrame): the r264 country correspondence.

    Returns:
        pd.DataFrame: the crop rows with country identifiers attached and crop_provision_gep
        in USD.
    """
    ee_r264_to_250 = utilities.collapse_countries_to_r250(
        df_countries,
        keep_columns=['area_code_M49', 'area_code', 'country', 'crop_code', 'crop', 'year',
                      'rental_rate', 'Value'])
    df = hb.df_merge(ee_r264_to_250, df_crop_value, how='right',
                     left_on='iso3_r250_id', right_on='area_code_M49')
    df[value_column] = df[value_column] * FAOSTAT_THOUSAND_USD
    return df





# =================================================================================================
# Subsistence component: the crops a smallholder household grows and eats rather than sells.
#
# A port of the reference pipeline, step for step, so the library produces its figure rather than
# quoting it. The chain is five stages:
#
#   01  own consumption  = cropland area x smallholder area share x value per hectare x own share
#   03  interpolation    = fill a surveyed country's missing years from a per-country regression
#   04  extrapolation    = fill an unsurveyed country from a global regression on cropland area
#   05  valuation        = x the CWoN land rental rate, then deflated to the base year by CPI
#   00  delivery         = the base-year panel joined onto the account's country list
#
# Reproducing the reference is what makes a disagreement with it worth stating, which is the lesson
# pollination taught: the coffee split was only actionable because both sides ran the same numbers.
# What the reproduction cannot test is an error the reference makes that this makes too -- so where
# we found one, the account publishes our arithmetic and the reference's is published beside it.
# `subsistence_own_consumption` computes both in one pass for exactly that reason.
# =================================================================================================

# FAO RuLIS stacks four indicators in one export. The one beside this is the share of agricultural
# production SOLD at market -- close to its complement -- so the selection is on the full string.
RULIS_OWN_CONSUMPTION_INDICATOR = (
    'Value of crop used for own consumption, share of total value of crop production (%)')
RULIS_SOLD_AT_MARKET_INDICATOR = (
    'Value of agricultural production sold at the market, share of total value of agricultural '
    'production (%)')
# RuLIS reports each indicator at sixteen disaggregations. The national figure describes the
# country; the rest split it by settlement, household composition or expenditure quintile.
RULIS_NATIONAL = 'National'
RULIS_SETTLEMENT_DISAGGREGATIONS = ('Rural', 'Urban')

# Lowder et al. (2021) tabulate farm size by region four ways; the share of agricultural area is the
# row this needs, and holdings under two hectares are the two columns it sums. The second column
# name carries an en dash, which is what the published table uses.
LOWDER_AREA_SHARE_ROW = 'share of agricultural area (%)'
LOWDER_SMALLHOLDER_COLUMNS = ('< 1 ha', '1–2 ha')
LOWDER_REGION_COLUMN = 'Region'

# FAOSTAT's Land Use domain, which carries both areas and the production intensity.
LAND_USE_VALUE_PER_AREA_ELEMENT = 'Value of agricultural production (Int. $) per Area'
LAND_USE_CROPLAND_ITEM = 'Cropland'
PER_AREA_COLUMN = 'Agricultural Production per Area (USD_PPP/ha)'

# The panel's span, and the World Bank income classes the extrapolation is built over.
SUBSISTENCE_FIRST_YEAR = 2004
SUBSISTENCE_LAST_YEAR = 2021
EXTRAPOLATION_INCOME_CLASSES = ('L', 'LM')
EXTRAPOLATION_CANDIDATE_FEATURES = ('Value', 'cropland_ha_1000', 'agricultural_land_ha_1000',
                                    'ag_value_per_area_int_usd')
EXTRAPOLATION_MINIMUM_TRAINING_ROWS = 20
EXTRAPOLATION_FOLDS = 5
COVARIATE_COLUMNS = ('GDP_capita', 'Value', 'cropland_ha_1000', 'agricultural_land_ha_1000',
                     'ag_value_per_area_int_usd', 'smallholder_lt2ha_share_pct',
                     'is_low_income', 'is_lower_middle_income')

# Both survey shares are published as percentages, and FAOSTAT publishes both areas in thousands of
# hectares against an intensity per single hectare. Keeping the three constants separate is what
# makes the unit finding below legible rather than a magic factor of ten.
PERCENT = 100.0
THOUSAND_HECTARES = 1000.0

# The reference resolves countries by name and needs one alias to do it.
SUBSISTENCE_COUNTRY_ALIASES = {'United Republic of Tanzania': 'Tanzania, United Republic of'}


def fold_country_name(name):
    """A country name reduced to lowercase ASCII words, for matching one source's spelling to
    another's. Accents, punctuation and ampersands all fall out."""
    import re
    import unicodedata

    if pd.isna(name):
        return ''
    ascii_name = (unicodedata.normalize('NFKD', str(name))
                  .encode('ascii', 'ignore').decode('ascii'))
    return re.sub(r'\s+', ' ',
                  re.sub(r'[^a-z0-9]+', ' ',
                         str(ascii_name).lower().strip().replace('&', ' and '))).strip()


def attach_iso3_by_name(df, df_iso, column='Country'):
    """ISO3 codes joined on the folded country name, filling rather than replacing what is there.

    Args:
        df (pd.DataFrame): rows carrying a country name.
        df_iso (pd.DataFrame): the ISO-3166 table, with `name` and `alpha-3`.
        column (str): the name column to resolve.

    Returns:
        pd.DataFrame: with `alpha-3` filled where it was missing.
    """
    iso = df_iso[['name', 'alpha-3']].copy()
    iso['country_key'] = iso['name'].apply(fold_country_name)
    out = df.copy()
    out['country_key'] = out[column].replace(SUBSISTENCE_COUNTRY_ALIASES).apply(fold_country_name)
    out = pd.merge(out, iso[['country_key', 'alpha-3']], on='country_key', how='left',
                   suffixes=('', '_iso'))
    if 'alpha-3_iso' in out.columns:
        out['alpha-3'] = (out['alpha-3'].combine_first(out['alpha-3_iso'])
                          if 'alpha-3' in out.columns else out['alpha-3_iso'])
        out = out.drop(columns=['alpha-3_iso'])
    return out.drop(columns=['country_key'])


def national_own_consumption_shares(df_rulis):
    """RuLIS rows reduced to the national own-consumption share, one row per country and year.

    A country RuLIS surveyed but never reported nationally keeps its rural and urban figures
    averaged into a national one, which is what keeps Georgia in the panel.
    """
    df = df_rulis.drop(columns=[c for c in df_rulis.columns if 'unnamed' in c.lower()])
    df = df[df['Indicator'] != RULIS_SOLD_AT_MARKET_INDICATOR]
    if df.empty:
        raise NameError(
            'The RuLIS export carries nothing once the sold-at-market indicator is dropped; its '
            'indicators are %s. That indicator is close to this one\'s complement, so reading it '
            'would value the commercial half as subsistence.'
            % sorted(df_rulis['Indicator'].unique()))
    return df


def add_constructed_national_rows(df):
    """A national row for each country that has only rural and urban ones, as their mean."""
    national = df[df['Disaggregation'] == RULIS_NATIONAL]
    without = set(df['Country'].unique()) - set(national['Country'].unique())
    settlement = df[df['Country'].isin(without)
                    & df['Disaggregation'].isin(RULIS_SETTLEMENT_DISAGGREGATIONS)]
    constructed = settlement.groupby(['Country', 'Year', 'Indicator'], as_index=False).agg({
        'Value': 'mean', PER_AREA_COLUMN: 'mean', 'Standard Deviation': 'mean',
        'Number of observations': 'sum', 'Income Classification': 'first'})
    constructed['Disaggregation'] = RULIS_NATIONAL
    return pd.concat([national, constructed], ignore_index=True)


def income_classes_by_country_year(df_wb_hist):
    """The World Bank historical income classification, melted to one row per country and year."""
    df = df_wb_hist.melt(id_vars=['Country'], var_name='Year', value_name='Income Classification')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    return df


def production_intensity(df_area_value):
    """Value of agricultural production per hectare, one row per country and year."""
    df = df_area_value.rename(columns={'Area': 'Country'})
    df = df[df['Element'] == LAND_USE_VALUE_PER_AREA_ELEMENT]
    return df[['Country', 'Year', 'Value']].rename(columns={'Value': PER_AREA_COLUMN})


def cropland_area(df_area_value):
    """Cropland area, as FAOSTAT reports it -- in THOUSANDS of hectares, per its Unit column."""
    df = df_area_value.rename(columns={'Area': 'Country'})
    return df[df['Item'] == LAND_USE_CROPLAND_ITEM][['Item', 'Country', 'Year', 'Unit', 'Value']]


def smallholder_area_shares(df_lowder):
    """The share of agricultural area held in farms under two hectares, by region, as PERCENT.

    The percentage is left as the source publishes it, because the reference's arithmetic consumes
    it that way and the port has to reproduce the reference before it corrects it.
    """
    kind_column = 'Number or share of farms / agricultural area'
    rows = df_lowder[df_lowder[kind_column] == LOWDER_AREA_SHARE_ROW]
    if rows.empty:
        raise NameError(
            'The Lowder table carries no %r rows; it carries %s. The share of agricultural AREA is '
            'what production is multiplied by, and the share of FARMS sitting beside it in the same '
            'table is about seven times larger.'
            % (LOWDER_AREA_SHARE_ROW, sorted(df_lowder[kind_column].unique())))
    for column in LOWDER_SMALLHOLDER_COLUMNS:
        if column not in rows.columns:
            raise NameError(
                'The Lowder table has no %r column; it carries %s. Both size classes under two '
                'hectares are needed, and dropping one silently halves the component.'
                % (column, list(rows.columns)))
    return rows


def subsistence_own_consumption(df_rulis, df_wb_hist, df_area_value, df_income, df_lowder,
                                df_gross_prod, df_iso):
    """Step 01: own consumption per country and survey year, as the reference computes it.

    ⚠ The arithmetic on the last line is the reference's, and its units do not agree with the
    sources: FAOSTAT reports cropland in THOUSANDS of hectares against an intensity per SINGLE
    hectare, and the Lowder share is a PERCENTAGE that is never divided by 100. The single /100
    converts only the own-consumption share. The two errors compound to a factor of ten, and this
    reproduces it deliberately -- `subsistence_own_consumption_corrected` is the same chain with the
    units read as the sources label them.

    Args:
        df_rulis (pd.DataFrame): the FAO RuLIS export.
        df_wb_hist (pd.DataFrame): the World Bank historical income classification.
        df_area_value (pd.DataFrame): the FAOSTAT Land Use extract.
        df_income (pd.DataFrame): the World Bank region and income group table.
        df_lowder (pd.DataFrame): the Lowder et al. (2021) farm-size table.
        df_gross_prod (pd.DataFrame): FAOSTAT gross production value.
        df_iso (pd.DataFrame): the ISO-3166 country table.

    Returns:
        pd.DataFrame: Country, Year, alpha-3, Region, Income group, own_con, own_con_corrected.
    """
    df = pd.merge(national_own_consumption_shares(df_rulis),
                  income_classes_by_country_year(df_wb_hist),
                  how='left', on=['Country', 'Year'])
    df = pd.merge(df, production_intensity(df_area_value), on=['Country', 'Year'], how='left')
    df = add_constructed_national_rows(df)
    df = df.pivot_table(index=['Country', 'Year', PER_AREA_COLUMN], columns='Indicator',
                        values='Value', aggfunc='first').reset_index()
    df = pd.merge(df, cropland_area(df_area_value), on=['Country', 'Year'], how='left')
    df = pd.merge(df, df_income.rename(columns={'Code': 'alpha-3'}), on=['Country'], how='left')
    df = pd.merge(df, smallholder_area_shares(df_lowder), on='Region', how='left')
    df = pd.merge(df, df_gross_prod, on=['Country', 'Year'], how='left')
    df = attach_iso3_by_name(df, df_iso)

    smallholder_percent = sum(pd.to_numeric(df[c], errors='coerce')
                              for c in LOWDER_SMALLHOLDER_COLUMNS)
    own_share_percent = df[RULIS_OWN_CONSUMPTION_INDICATOR]

    # The reference's arithmetic, reproduced.
    df['own_con'] = (df['Value_x'] * smallholder_percent * df[PER_AREA_COLUMN]
                     * own_share_percent / PERCENT)
    # The same chain with every unit read as its source labels it: cropland out of thousands of
    # hectares, and both survey percentages out of 100. Larger by THOUSAND_HECTARES / PERCENT.
    df['own_con_corrected'] = (df['Value_x'] * THOUSAND_HECTARES
                               * (smallholder_percent / PERCENT)
                               * df[PER_AREA_COLUMN]
                               * (own_share_percent / PERCENT))
    return df[['Country', 'Year', 'alpha-3', 'Region', 'Income group',
               'own_con', 'own_con_corrected']]


def interpolate_missing_years(df_own, df_gross_prod_usd, df_gdp_per_capita):
    """Step 03: a surveyed country's missing years filled from its own observations.

    Each country is fitted separately on year, GDP per capita and commercial production; a country
    with too few observations to fit falls back to scaling commercial production by its median
    observed ratio.

    ⚠ The dropna below is the reference's, and it is why eleven countries lose observations they
    had: a country-year missing GDP per capita or commercial production leaves the panel here, and
    an OBSERVED own-consumption value goes with it. Ethiopia has three surveyed years entering this
    step and none leaving it, so every Ethiopian row downstream is a prediction.

    Args:
        df_own (pd.DataFrame): Country, Year, alpha-3, own_con.
        df_gross_prod_usd (pd.DataFrame): Country, Year, Value_gross_prof.
        df_gdp_per_capita (pd.DataFrame): Country, Year, GDP_capita.

    Returns:
        pd.DataFrame: Country, Year, alpha-3, own_con, own_con2.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    degree = 1
    features = ['Year', 'GDP_capita', 'Value_gross_prof']

    merged = pd.merge(df_own, df_gross_prod_usd, on=['Year', 'Country'], how='left')
    merged = pd.merge(merged, df_gdp_per_capita, on=['Year', 'Country'], how='left')

    years = pd.DataFrame({'Year': np.arange(SUBSISTENCE_FIRST_YEAR, SUBSISTENCE_LAST_YEAR + 1)})
    complete = df_own[['Country']].dropna().drop_duplicates().merge(years, how='cross')
    df = complete.merge(merged, on=['Country', 'Year'], how='left')
    before = int(df['own_con'].notna().sum())
    df = df.dropna(subset=['Year', 'GDP_capita', 'Value_gross_prof'])
    lost = before - int(df['own_con'].notna().sum())
    if lost:
        hb.log('Interpolation: %d observed own-consumption values left the panel with the rows '
               'that had no GDP per capita or commercial production.' % lost)

    def fit_group(group):
        known = group.dropna(subset=['own_con'])
        if len(known) == 0:
            return group
        if len(known) <= degree:
            ratio = (known['own_con'] / known['Value_gross_prof']).replace(
                [np.inf, -np.inf], np.nan).dropna()
            group['own_con2'] = group['own_con']
            if not ratio.empty:
                missing = group['own_con2'].isna() & group['Value_gross_prof'].notna()
                group.loc[missing, 'own_con2'] = (group.loc[missing, 'Value_gross_prof']
                                                  * ratio.median())
            else:
                group['own_con2'] = group['own_con2'].fillna(known['own_con'].iloc[0])
            return group
        polynomial = PolynomialFeatures(degree=degree, include_bias=False)
        model = LinearRegression().fit(polynomial.fit_transform(known[features].values),
                                       known['own_con'].values)
        group['own_con2'] = group['own_con']
        missing = group['own_con2'].isna()
        predicted = model.predict(polynomial.transform(group[features].values))
        group.loc[missing, 'own_con2'] = predicted[missing.values]
        return group

    out = df.groupby('Country', group_keys=False).apply(fit_group)
    keep = out[(out['own_con2'].notna()) & (out['own_con2'] >= 0)]['Country'].unique()
    out = out[out['Country'].isin(keep) & (out['own_con2'] >= 0)]
    return out[['Country', 'Year', 'alpha-3', 'own_con', 'own_con2']]


def fit_line(x, y):
    """Least-squares intercept and slope, or (None, None) when the fit is not defined."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.nanstd(x) == 0:
        return None, None
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(intercept), float(slope)


def country_group_folds(countries, max_folds=EXTRAPOLATION_FOLDS):
    """Cross-validation folds that split on country, so a country is never in both sides."""
    groups = sorted(pd.Series(countries).dropna().unique().tolist())
    if len(groups) < 2:
        return []
    k = min(max_folds, len(groups))
    return [set(groups[i::k]) for i in range(k)]


def rank_extrapolation_features(df, feature_columns, target_column='own_con2'):
    """Candidate regressors ranked by absolute correlation, then by cross-validated error."""
    train = df[df[target_column].notna()].dropna(subset=['Country'])
    if len(train) < EXTRAPOLATION_MINIMUM_TRAINING_ROWS:
        raise ValueError('Only %d training rows for the extrapolation, and at least %d are needed.'
                         % (len(train), EXTRAPOLATION_MINIMUM_TRAINING_ROWS))
    rows = []
    for feature in [c for c in feature_columns if c in train.columns and train[c].notna().any()]:
        sub = train[['Country', feature, target_column]].dropna()
        if len(sub) < EXTRAPOLATION_MINIMUM_TRAINING_ROWS:
            continue
        correlation = sub[feature].corr(sub[target_column])
        intercept, slope = fit_line(sub[feature].values, sub[target_column].values)
        if pd.isna(correlation) or intercept is None:
            continue
        errors = []
        for held_out in country_group_folds(sub['Country'].values) or [set()]:
            train_fold = sub[~sub['Country'].isin(held_out)]
            test_fold = sub[sub['Country'].isin(held_out)] if held_out else sub
            if len(train_fold) < 2 or len(test_fold) == 0:
                continue
            fold_intercept, fold_slope = fit_line(train_fold[feature].values,
                                                  train_fold[target_column].values)
            if fold_intercept is None:
                continue
            predicted = np.clip(fold_intercept + fold_slope * test_fold[feature].values, 0, None)
            errors.append(float(np.sqrt(np.mean((test_fold[target_column].values - predicted) ** 2))))
        if not errors:
            continue
        rows.append({'feature': feature, 'abs_pearson_r': float(abs(correlation)),
                     'intercept': intercept, 'slope': slope,
                     'cv_rmse_mean': float(np.mean(errors)), 'n_train_rows': len(sub)})
    if not rows:
        raise ValueError('No candidate regressor had enough non-missing observations.')
    return (pd.DataFrame(rows).sort_values(['abs_pearson_r', 'cv_rmse_mean'],
                                           ascending=[False, True]).reset_index(drop=True))


def extrapolate_to_unsurveyed(df_interpolated, df_wb_hist, df_covariates, df_iso):
    """Step 04: countries no survey reached, filled from a global regression on one covariate.

    ⚠ This is where the panel stops being survey evidence. The regressor chosen is cropland area,
    which is also the largest term in the formula that produced the target, so the relationship is
    close to circular; and the filled rows outnumber the observed ones by more than twenty-five to
    one. The panel records `own_con_source` per row so a reader can separate them.

    Returns:
        tuple: (panel, chosen feature name).
    """
    income = df_wb_hist.melt(id_vars=['Country'], var_name='Year', value_name='wb_income')
    income = income[income['Year'].str.isdigit()]
    income['Year'] = income['Year'].astype(int)
    income = income[income['Year'].between(SUBSISTENCE_FIRST_YEAR, SUBSISTENCE_LAST_YEAR)]
    income = income[income['wb_income'].isin(EXTRAPOLATION_INCOME_CLASSES)].copy()

    covariates = df_covariates.rename(columns={'name': 'Country'})
    present = [c for c in COVARIATE_COLUMNS if c in covariates.columns]
    covariates = (covariates[['Country', 'Year'] + present]
                  .groupby(['Country', 'Year'], as_index=False)
                  .agg({c: 'mean' for c in present}))

    df = pd.merge(income, df_interpolated, on=['Year', 'Country'], how='left')
    df = pd.merge(df, covariates, on=['Country', 'Year'], how='left')
    df = attach_iso3_by_name(df, df_iso)

    had_own_con = df['own_con'].notna()
    had_own_con2 = df['own_con2'].notna()
    features = [c for c in EXTRAPOLATION_CANDIDATE_FEATURES if c in df.columns]
    ranked = rank_extrapolation_features(df, features)
    best = ranked.iloc[0]['feature']

    train = df[[best, 'own_con2']].dropna()
    intercept, slope = fit_line(train[best].values, train['own_con2'].values)
    predict = df['own_con2'].isna() & df[best].notna()
    df.loc[predict, 'own_con2'] = np.clip(intercept + slope * df.loc[predict, best].values, 0, None)

    df['own_con_source'] = np.where(
        had_own_con, 'observed',
        np.where(had_own_con2, 'interpolated',
                 np.where(df['own_con2'].notna(), 'extrapolated_correlation', 'missing')))
    counts = df['own_con_source'].value_counts().to_dict()
    hb.log('Extrapolation on %r: %s' % (best, counts))
    return df, best


def subsistence_rental_rates(df_coefs):
    """The CWoN table melted to one row per country and decade start, keyed on ISO3.

    The commercial component reads the same workbook keyed on the FAO area code; this component
    joins on ISO3 because the panel it values is keyed that way.
    """
    df = df_coefs.rename(columns={'ISO3': 'alpha-3'})
    melted = df.melt(id_vars=['Order', 'FAO', 'alpha-3', 'Country/territory'],
                     var_name='Decade', value_name='rental_rate')
    melted['Decade_start'] = melted['Decade'].str.extract(r'(\d{4})').astype(int)
    return melted


def apply_subsistence_rental_rate(df_panel, df_coefs):
    """Step 05a: own consumption attributed to land, each year taking its decade's rate."""
    df = pd.merge(df_panel, subsistence_rental_rates(df_coefs), on='alpha-3', how='left')
    df = df[df['Year'] >= df['Decade_start']]
    df = df.sort_values(['alpha-3', 'Year', 'Decade_start']).drop_duplicates(
        ['alpha-3', 'Year'], keep='last')
    df['gep_value'] = df['own_con2'] * df['rental_rate']
    return df


def consumer_price_index(df_cpi, base_year):
    """A CPI index per country with the base year at 100, chained from annual inflation rates."""
    metadata = ['Series Name', 'Series Code', 'Country Name', 'Country Code']
    long = df_cpi.melt(id_vars=metadata, var_name='Year', value_name='Inflation_Rate')
    long['Year'] = pd.to_numeric(long['Year'].str.extract(r'(\d{4})')[0], errors='coerce')
    long = long.dropna(subset=['Year'])
    long['Year'] = long['Year'].astype(int)
    long['Inflation_Rate'] = pd.to_numeric(long['Inflation_Rate'], errors='coerce').fillna(0)

    parts = []
    for _, group in long.groupby('Country Code'):
        if len(group) < 2:
            continue
        group = group.sort_values('Year').reset_index(drop=True)
        group['CPI'] = np.nan
        if group['Inflation_Rate'].sum() == 0:
            group['CPI'] = 100
            parts.append(group)
            continue
        base_rows = group.index[group['Year'] == base_year]
        if len(base_rows) == 0:
            continue
        base = base_rows[0]
        group.loc[base, 'CPI'] = 100
        for i in range(base + 1, len(group)):
            group.loc[i, 'CPI'] = group.loc[i - 1, 'CPI'] * (
                1 + group.loc[i, 'Inflation_Rate'] / PERCENT)
        for i in range(base - 1, -1, -1):
            group.loc[i, 'CPI'] = group.loc[i + 1, 'CPI'] / (
                1 + group.loc[i + 1, 'Inflation_Rate'] / PERCENT)
        parts.append(group)
    if not parts:
        raise ValueError('No country in the CPI table has a %d row to index on.' % base_year)
    return pd.concat(parts)


def deflate_to_base_year(df_gep, df_cpi, base_year):
    """Step 05b: each year's value restated in base-year money by its own country's CPI."""
    cpi = consumer_price_index(df_cpi, base_year)[['Country Code', 'Year', 'CPI']].rename(
        columns={'Country Code': 'alpha-3'})
    base = cpi[cpi['Year'] == base_year][['alpha-3', 'CPI']].rename(columns={'CPI': 'CPI_base'})
    out = df_gep.merge(cpi, on=['alpha-3', 'Year'], how='left').merge(base, on='alpha-3', how='left')
    out['crop_subsistence_gep'] = out['gep_value'] * (out['CPI_base'] / out['CPI'])
    return out


def subsistence_on_country_list(df_deflated, df_countries, base_year):
    """Step 00: the base-year panel on the account's country list, one row per country.

    ⚠ The reference joins its panel against the r264 correspondence on a Natural Earth name column
    and delivers 16 of its own 66 valued countries. Collapsing the correspondence to one row per
    country and joining on the ISO3 label keeps every one of them, which is the house rule
    `collapse_countries_to_r250` exists for.
    """
    base = df_deflated[df_deflated['Year'] == base_year]
    countries = utilities.collapse_countries_to_r250(df_countries)
    joined = countries.merge(
        base[['alpha-3', 'own_con', 'own_con2', 'own_con_source', 'rental_rate',
              'crop_subsistence_gep']],
        how='left', left_on='iso3_r250_label', right_on='alpha-3')
    valued_in = int(base['crop_subsistence_gep'].notna().sum())
    valued_out = int(joined['crop_subsistence_gep'].notna().sum())
    if valued_out < valued_in:
        raise ValueError(
            '%d of %d valued countries did not land on the account country list. A country that '
            'does not match is dropped silently and the table still looks complete, which is how '
            'the reference delivers a fifth of its own estimate.'
            % (valued_in - valued_out, valued_in))
    joined['year'] = base_year
    return joined


# The regional median needs enough observations behind it to mean anything. Middle East and North
# Africa has one survey and East Asia and Pacific two, so those fall back to the global median and
# every row records which it used.
SUBSISTENCE_MINIMUM_SURVEYS_PER_REGION = 3


def impute_own_consumption_shares(df_observed, df_income,
                                  minimum_per_region=SUBSISTENCE_MINIMUM_SURVEYS_PER_REGION):
    """An own-consumption share for every country, observed where a survey exists and imputed where
    it does not, with the source of each recorded.

    This is the correction to the reference's extrapolation, and the reason for it is that the two
    approaches impute different things. The reference regresses the own-consumption *level* on
    cropland area across countries, which fills 1,339 of its 1,610 rows -- and cropland area is the
    largest term in the formula that produced the level being fitted, so the fit is close to
    circular and the fitted rows swamp the 49 observed ones. What the survey actually measures is a
    *share*: a ratio of two quantities that move together, bounded between nothing and everything,
    and far more portable across countries than a total is. So the share is what gets imputed here,
    and the country's own cropland, own production intensity and own regional farm-size structure
    supply everything else. An unsurveyed country's estimate is then built from four numbers of
    which three are its own.

    Args:
        df_observed (pd.DataFrame): Country and Value, the observed national shares as percentages.
        df_income (pd.DataFrame): the World Bank table carrying Country and Region.
        minimum_per_region (int): surveys a region needs before its median is used.

    Returns:
        pd.DataFrame: Country, own_consumption_share (a fraction), share_source.
    """
    observed = df_observed.merge(df_income[['Country', 'Region']], on='Country', how='left')
    counts = observed.groupby('Region')['Value'].count()
    medians = observed.groupby('Region')['Value'].median()
    usable = {region: medians[region] for region in medians.index
              if counts[region] >= minimum_per_region}
    global_median = observed['Value'].median()

    rows = []
    for _, row in df_income.iterrows():
        country, region = row['Country'], row.get('Region')
        match = observed[observed['Country'] == country]
        if len(match):
            rows.append((country, match['Value'].iloc[0] / PERCENT, 'observed'))
        elif region in usable:
            rows.append((country, usable[region] / PERCENT,
                         'regional median (%s, n=%d)' % (region, counts[region])))
        elif pd.notna(global_median):
            rows.append((country, global_median / PERCENT, 'global median'))
    out = pd.DataFrame(rows, columns=['Country', 'own_consumption_share', 'share_source'])
    hb.log('Own-consumption shares: %d observed, %d regional median, %d global median.'
           % ((out['share_source'] == 'observed').sum(),
              out['share_source'].str.startswith('regional').sum(),
              (out['share_source'] == 'global median').sum()))
    return out.drop_duplicates('Country')


def low_and_lower_middle_income_countries(df_wb_hist, year):
    """The countries the account estimates subsistence for in a given year.

    The reference restricts its panel to the World Bank's low and lower-middle-income classes and
    that scope is kept, both because subsistence cropping is negligible above it and because
    changing the scope at the same time as the method would leave the two changes confounded.
    """
    long = df_wb_hist.melt(id_vars=['Country'], var_name='Year', value_name='wb_income')
    long = long[long['Year'].str.isdigit()]
    long['Year'] = long['Year'].astype(int)
    in_year = long[(long['Year'] == year) & (long['wb_income'].isin(EXTRAPOLATION_INCOME_CLASSES))]
    return in_year[['Country', 'wb_income']].drop_duplicates('Country')


def subsistence_value_from_shares(df_area_value, df_lowder, df_income, df_wb_hist, df_shares,
                                  year):
    """Subsistence crop production, built for every eligible country from its own structural data.

    Four factors, of which three are the country's own: its cropland area, its production intensity
    and its region's farm-size structure, with only the own-consumption share imputed where no
    survey reached it. Every unit is read as its source labels it -- cropland out of thousands of
    hectares, both survey figures out of percentages.

    Args:
        df_area_value (pd.DataFrame): the FAOSTAT Land Use extract.
        df_lowder (pd.DataFrame): the Lowder et al. (2021) farm-size table.
        df_income (pd.DataFrame): the World Bank region and income group table.
        df_wb_hist (pd.DataFrame): the World Bank historical income classification.
        df_shares (pd.DataFrame): Country, own_consumption_share, share_source.
        year (int): the year to build.

    Returns:
        pd.DataFrame: Country, alpha-3, Year, and own_con with the shares and terms behind it.
    """
    cropland = cropland_area(df_area_value)
    cropland = cropland[cropland['Year'] == year][['Country', 'Value']].rename(
        columns={'Value': 'cropland_1000_ha'})
    intensity = production_intensity(df_area_value)
    intensity = intensity[intensity['Year'] == year][['Country', PER_AREA_COLUMN]]

    rows = smallholder_area_shares(df_lowder)
    smallholder = rows[[LOWDER_REGION_COLUMN]].copy()
    smallholder['smallholder_area_share'] = sum(
        pd.to_numeric(rows[c], errors='coerce') for c in LOWDER_SMALLHOLDER_COLUMNS) / PERCENT

    df = low_and_lower_middle_income_countries(df_wb_hist, year)
    df = df.merge(df_income[['Country', 'Region', 'Code']].rename(columns={'Code': 'alpha-3'}),
                  on='Country', how='left')
    df = df.merge(smallholder, on='Region', how='left')
    df = df.merge(cropland, on='Country', how='left')
    df = df.merge(intensity, on='Country', how='left')
    df = df.merge(df_shares, on='Country', how='left')

    df['Year'] = year
    df['own_con'] = (df['cropland_1000_ha'] * THOUSAND_HECTARES
                     * df['smallholder_area_share']
                     * df[PER_AREA_COLUMN]
                     * df['own_consumption_share'])
    complete = df['own_con'].notna()
    hb.log('Subsistence from shares, %d: %d eligible countries, %d with every term present.'
           % (year, len(df), int(complete.sum())))
    return df[complete].copy()
