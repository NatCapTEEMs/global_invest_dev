"""Fire-protection (wildfire) science, ported from the GEP wildfire repo's R pipeline
(gep_wildfire_2019.R, translated function by function; the committed per-country output is the
replication anchor and reproduces to the float -- see test_fire_protection).

Method in one pass: per-country AR(1) regressions of burned area on its one-year lag (run
upstream, results committed) give a fire-persistence beta per country and regression type.
Three quantities derive from them: the BASELINE persistence effect (total burned area on its
own lag), the NATURE-VS-HUMAN CAUSE difference (nature-caused fires' persistence minus
human-caused fires'), and the NATURE-VS-HUMAN AREA difference (persistence in nature-dominated
minus human-dominated districts). Each beta times the country's 2018 burned area gives the
acres avoided (or added) in 2019, valued at the country's EM-DAT wildfire damage per burned
acre (missing countries filled with the global average of positive values); GEP is the avoided
damage (the sign flip on additional acres).

Faithful-port notes, kept for anchor fidelity rather than fixed:
- The source pivots the regression table with a mean aggregator over ALL value columns
  including the CHARACTER significance stars; R's mean() on character is NA, so every wide
  significance column -- and therefore every *_signif_level in the output -- is 0 by
  construction. The pvalue columns beside them are real; the levels are not.
- Beta selection is by regression-type id, which in the source is data.table's .GRP: the
  FIRST-APPEARANCE order of (Y_var, Key_X_var, sample) in the input CSV. A reordered input
  would silently reassign the betas, so build_regression_type_ids ASSERTS the expected
  triples before any beta is read.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

HA_TO_ACRES = 2.47105

# Y_var values the source drops before building regression-type ids (days/counts/average
# variants; only the total-burned-area regressions feed the GEP).
FIRE_DROP_Y_VARS = (
    'total_burned_days_human', 'average_burned_days_human',
    'total_counts_human', 'average_counts_human',
    'total_burned_days_nature', 'average_burned_days_nature',
    'total_counts_nature', 'average_counts_nature',
    'total_burned_days_allfire', 'average_burned_days_allfire',
    'total_counts_total', 'average_counts_total',
    'average_burned_areas_ha', 'average_ba_ha_nature', 'average_ba_ha_human')

# The regression-type ids the GEP reads, with the (Y_var, Key_X_var, sample) triple each id
# must correspond to in first-appearance order (asserted at load).
FIRE_REG_TYPES = {
    1: ('total_ba_ha_human', 'L1_total_ba_ha_human', 'full'),
    2: ('total_ba_ha_nature', 'L1_total_ba_ha_nature', 'full'),
    3: ('total_burned_areas_ha', 'L1_total_burned_areas_ha', 'full'),
    6: ('total_burned_areas_ha', 'L1_total_burned_areas_ha', 'nature_subsample'),
    7: ('total_burned_areas_ha', 'L1_total_burned_areas_ha', 'human_subsample'),
}
FIRE_BASELINE_ID = 3          # fire persistence, full sample
FIRE_NN_ID, FIRE_HH_ID = 2, 1     # nature-caused minus human-caused persistence
FIRE_TTN_ID, FIRE_TTH_ID = 6, 7   # nature-dominated minus human-dominated districts


def build_regression_type_ids(regression_df):
    """Filter the dropped Y_vars and assign reg_type_id in first-appearance order of
    (Y_var, Key_X_var, sample) -- data.table's .GRP semantics. Asserts that every id the
    GEP reads carries the triple FIRE_REG_TYPES expects, so a reordered input fails loudly
    instead of silently reassigning betas."""
    df = regression_df[~regression_df['Y_var'].isin(FIRE_DROP_Y_VARS)].copy()
    keys = df[['Y_var', 'Key_X_var', 'sample']].drop_duplicates()
    id_of_triple = {tuple(row): i for i, row in enumerate(keys.itertuples(index=False), 1)}
    for reg_type_id, triple in FIRE_REG_TYPES.items():
        if id_of_triple.get(triple) != reg_type_id:
            raise ValueError(
                f'regression CSV order changed: expected {triple} at reg_type_id '
                f'{reg_type_id}, found it at {id_of_triple.get(triple)} -- the beta '
                f'selection below would be wrong. Update FIRE_REG_TYPES deliberately.')
    df['reg_type_id'] = [id_of_triple[t] for t in
                         zip(df['Y_var'], df['Key_X_var'], df['sample'])]
    return df


def _pivot_wide(df):
    """The source's dcast: GID_0 x reg_type_id, mean-aggregated. The mean of the character
    significance column is NA in R; pandas mean would raise, so the column is dropped here
    and the wide significance values are NaN by construction (see the module docstring)."""
    numeric_cols = ['coeff', 'std_err', 'n_obs', 'pvalue']
    wide = df.pivot_table(index='GID_0', columns='reg_type_id', values=numeric_cols,
                          aggfunc='mean')
    wide.columns = [f'{value}_{reg_type_id}' for value, reg_type_id in wide.columns]
    return wide.reset_index()


def difference_stats(coeff_a, se_a, coeff_b, se_b):
    """Difference of two independent estimates: diff, its standard error, and the two-sided
    normal-approximation p-value (the source's pnorm test)."""
    diff = coeff_a - coeff_b
    se_diff = np.sqrt(se_a ** 2 + se_b ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = np.where(se_diff == 0, np.nan, diff / se_diff)
    p_value = 2 * norm.cdf(-np.abs(t_stat))
    return diff, se_diff, t_stat, p_value


def significance_level(p_values):
    """Star count as an integer: 3 for p<0.01, 2 for p<0.05, 1 for p<0.10, else 0 (NaN -> 0)."""
    p = np.asarray(p_values, dtype=float)
    return np.select([np.isnan(p), p < 0.01, p < 0.05, p < 0.10], [0, 3, 2, 1], default=0)


def compute_beta_differences(regression_df):
    """Section 1 of the source: the per-country beta table the GEP reads.

    Returns one row per GID_0 with the baseline beta and the two difference betas
    (nature-vs-human cause; nature-vs-human area), their p-values, and the *_signif_level
    columns -- 0 by construction, matching the source (see module docstring)."""
    wide = _pivot_wide(build_regression_type_ids(regression_df))

    beta = pd.DataFrame({'GID_0': wide['GID_0']})
    beta['Beta_baseline_coeff'] = wide[f'coeff_{FIRE_BASELINE_ID}']
    beta['Beta_baseline_pvalue'] = wide[f'pvalue_{FIRE_BASELINE_ID}']
    beta['baseline_signif_level'] = 0    # mean-of-character NA in the source, always level 0

    for label, id_a, id_b in (('nn_hh', FIRE_NN_ID, FIRE_HH_ID),
                              ('ttn_tth', FIRE_TTN_ID, FIRE_TTH_ID)):
        diff, _, _, p_value = difference_stats(
            wide[f'coeff_{id_a}'], wide[f'std_err_{id_a}'],
            wide[f'coeff_{id_b}'], wide[f'std_err_{id_b}'])
        # The source computes the difference only where all four inputs are present.
        valid = (wide[f'coeff_{id_a}'].notna() & wide[f'std_err_{id_a}'].notna()
                 & wide[f'coeff_{id_b}'].notna() & wide[f'std_err_{id_b}'].notna())
        beta[f'Beta_diff_{label}'] = np.where(valid, diff, np.nan)
        beta[f'p_value_Beta_diff_{label}'] = np.where(valid, p_value, np.nan)
        beta[f'{label}_signif_level'] = significance_level(
            np.where(valid, p_value, np.nan))
    return beta


def compute_avoided_acres(beta_df, burned_2018_df):
    """Section 2: marginal additional acres per beta (beta x acres-per-ha) and the 2019
    additional-acre totals (marginal x the country's 2018 burned hectares).

    burned_2018_df: one row per GID_0 with total_burned_areas_ha_2018 and n_adm2_units."""
    df = beta_df.merge(burned_2018_df, on='GID_0', how='left')
    for label in ('baseline', 'nn_hh', 'ttn_tth'):
        beta_col = {'baseline': 'Beta_baseline_coeff',
                    'nn_hh': 'Beta_diff_nn_hh', 'ttn_tth': 'Beta_diff_ttn_tth'}[label]
        df[f'marginal_additional_acres_{label}'] = df[beta_col] * HA_TO_ACRES
        df[f'additional_acres_2019_{label}'] = (
            df[f'marginal_additional_acres_{label}'] * df['total_burned_areas_ha_2018'])
    df['burned_acres_2019'] = df['total_burned_areas_ha_2018'] * HA_TO_ACRES
    return df


def compute_damage_per_acre(burned_acres_all_years_df, emdat_damages_df):
    """Section 3, valuation side: EM-DAT wildfire damages per burned acre by country, with
    zero or missing filled by the global average over countries with positive values.

    burned_acres_all_years_df: GID_0, total_burned_acres_country (ALL panel years).
    emdat_damages_df: GID_0, total_damages_country_usd."""
    df = burned_acres_all_years_df.merge(emdat_damages_df, on='GID_0', how='left')
    df['total_damages_country_usd'] = df['total_damages_country_usd'].fillna(0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['damages_usd_per_acre_country'] = np.where(
            df['total_burned_acres_country'] > 0,
            df['total_damages_country_usd'] / df['total_burned_acres_country'], 0.0)
    positive = df.loc[df['damages_usd_per_acre_country'] > 0, 'damages_usd_per_acre_country']
    global_average = positive.mean()
    df.loc[df['damages_usd_per_acre_country'] <= 0, 'damages_usd_per_acre_country'] = global_average
    return df[['GID_0', 'damages_usd_per_acre_country']], global_average


def emdat_wildfire_damages(emdat_df):
    """EM-DAT extract -> total adjusted wildfire damages (USD) per ISO3. The damage column is
    thousands of USD with comma separators in the source workbook."""
    wildfire = emdat_df[emdat_df['Disaster Type'] == 'Wildfire'].copy()
    damage_col = "Total Damage, Adjusted ('000 US$)"
    wildfire['total_damages_usd'] = (
        pd.to_numeric(wildfire[damage_col].astype(str).str.replace(',', '', regex=False),
                      errors='coerce') * 1000)
    damages = (wildfire.dropna(subset=['total_damages_usd', 'ISO'])
               .groupby('ISO', as_index=False)['total_damages_usd'].sum()
               .rename(columns={'ISO': 'GID_0',
                                'total_damages_usd': 'total_damages_country_usd'}))
    return damages


def compute_gep(avoided_df, damage_per_acre_df, global_average):
    """Section 3, product: GEP = avoided damage. Additional acres carry the persistence sign,
    so the value of avoiding them is the negative product. Countries without a damage rate
    (absent from the merge) take the global average, as in the source."""
    df = avoided_df.merge(damage_per_acre_df, on='GID_0', how='left')
    df['damages_usd_per_acre_country'] = df['damages_usd_per_acre_country'].fillna(global_average)
    for label in ('baseline', 'nn_hh', 'ttn_tth'):
        df[f'GEP_wildfire_2019_{label}'] = (
            -1 * df[f'additional_acres_2019_{label}'] * df['damages_usd_per_acre_country'])
    return df


# GADM codes three of the reference's units differently from the account: Kosovo is XKO where
# the account's label is XKX, and Z01 and Z07 are Indian sub-regions GADM keeps separate. Joining
# GID_0 straight onto the r250 label silently dropped all three, stranding their value on rows
# with no country at all (34.09M USD at the 2019 nature-vs-human-cause variant, which still
# entered the reported total). The correspondence resolves all three, so the join goes through it.
GADM_LABEL_ALIASES = {'XKO': 'XKX'}
FIRE_VALUE_COLUMNS = ('GEP_wildfire_2019_baseline', 'GEP_wildfire_2019_nn_hh',
                      'GEP_wildfire_2019_ttn_tth')


def attach_countries(df_gep, df_countries):
    """The per-GADM results summed to one row per r250 country.

    GID_0 is resolved through the GADM-to-r250 correspondence rather than assumed equal to the
    r250 label. Several GADM units can share a country (India carries Z01 and Z07 beside IND),
    so the values are summed per country rather than joined row for row.

    Args:
        df_gep (pd.DataFrame): compute_gep's output, one row per GID_0.
        df_countries (pd.DataFrame): the r264 correspondence, carrying gadm_r263_label and
            iso3_r250_label.

    Returns:
        pd.DataFrame: one row per valued country, with the three variant columns summed.

    Raises:
        ValueError: if a GID_0 carrying a value resolves to no country. Dropping it would take
            its value out of the account without saying so.
    """
    import pandas as pd

    gadm_to_r250 = (df_countries[['gadm_r263_label', 'iso3_r250_label']]
                    .dropna().drop_duplicates('gadm_r263_label')
                    .set_index('gadm_r263_label')['iso3_r250_label'])
    df = df_gep.copy()
    df['iso3_r250_label'] = df['GID_0'].replace(GADM_LABEL_ALIASES).map(gadm_to_r250)

    value_columns = [c for c in FIRE_VALUE_COLUMNS if c in df.columns]
    unresolved = df[df['iso3_r250_label'].isna()]
    stranded = unresolved[value_columns].notna().any(axis=1)
    if stranded.any():
        raise ValueError(
            f"fire_protection: GID_0 {sorted(unresolved.loc[stranded, 'GID_0'])} carry a value "
            f"but resolve to no r250 country, so their value would leave the account unannounced.")

    return (df.dropna(subset=['iso3_r250_label'])
            .groupby('iso3_r250_label', as_index=False)[value_columns].sum(min_count=1))
