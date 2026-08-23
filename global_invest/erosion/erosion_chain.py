"""The erosion account's science, as functions over arrays and frames.

Nothing here opens a file. The account values the crop production that soil retention protects,
in four steps, and each one is a function below:

1. A country's severity threshold. Soil loss counts as severe above a tolerance rate, which is
   lower for small and low-lying countries because the default rate is calibrated on deep upland
   soils (`country_threshold_policy`).
2. The share of gross soil loss that is prevented, per pixel. On-farm cover prevents
   `avoided / (avoided + gross)`; the upstream layer prevents its own share; the two combine as a
   union rather than a sum, so a pixel protected twice is not counted twice
   (`onfarm_prevention_share`, `combined_prevention_share`, `restrict_to_valued_pixels`).
3. A country's production shock, as the production-weighted mean of each crop's protected share
   times that crop's yield elasticity to erosion (`country_erosion_shock`).
4. The value, as the country's crop gross production value times that shock (`country_gep`).

The threshold and prevention steps run over rasters the module reads elsewhere; the shock and
value steps run over frames. Splitting them out is what lets the arithmetic be tested on four
countries instead of on a global grid.
"""
import numpy as np
import pandas as pd

# Shocks below this are numerical residue rather than economics, but a country with a genuinely
# tiny protected share should not fall to exactly zero and drop out of the account.
MIN_SHOCK_FLOOR = 8e-10


def country_threshold_policy(df_countries, threshold_high, threshold_low,
                             small_country_area_km2, low_elevation_mean_m):
    """Each country's soil-loss tolerance rate, and why it got the one it got.

    The high rate is the default. A country takes the low rate if it is small or if its mean
    elevation is low, because the high rate assumes deep soils on upland slopes.

    Args:
        df_countries (pandas.DataFrame): one row per country, with `iso3`, `area_km2` and
            `mean_elevation_m`. Either measure may be missing, in which case it cannot trigger
            the low rate.
        threshold_high (float): the default tolerance, t/ha/yr.
        threshold_low (float): the tolerance for small and low-lying countries, t/ha/yr.
        small_country_area_km2 (float): area below which a country is small.
        low_elevation_mean_m (float): mean elevation below which a country is low-lying.

    Returns:
        pandas.DataFrame: `iso3`, `threshold_t_ha_yr`, `reason`.
    """
    df = df_countries.copy()
    area = pd.to_numeric(df.get('area_km2'), errors='coerce')
    elevation = pd.to_numeric(df.get('mean_elevation_m'), errors='coerce')

    is_small = area.notna() & (area < small_country_area_km2)
    is_low = elevation.notna() & np.isfinite(elevation) & (elevation < low_elevation_mean_m)

    reasons = []
    for small, low in zip(is_small, is_low):
        parts = (['small-area'] if small else []) + (['low-elevation'] if low else [])
        reasons.append(' & '.join(parts) if parts else 'default-high')

    return pd.DataFrame({
        'iso3': df['iso3'].to_numpy(),
        'threshold_t_ha_yr': np.where(is_small | is_low, threshold_low, threshold_high),
        'reason': reasons,
    })


def onfarm_prevention_share(avoided_erosion, gross_erosion, epsilon=1e-9):
    """The share of gross soil loss that the field's own cover prevents.

    `avoided / (avoided + gross)`, which is a rate rather than a tonnage, so it can be applied to
    production without carrying the pixel's area through.

    Args:
        avoided_erosion (numpy.ndarray): soil loss avoided by present cover, t/ha/yr.
        gross_erosion (numpy.ndarray): soil loss that still occurs, t/ha/yr.
        epsilon (float): floor on the denominator, so a pixel with neither reads as zero
            prevention rather than dividing by zero.

    Returns:
        numpy.ndarray: the share, in [0, 1].
    """
    denominator = np.maximum(np.asarray(avoided_erosion) + np.asarray(gross_erosion), epsilon)
    return np.clip(np.asarray(avoided_erosion) / denominator, 0.0, 1.0)


def combined_prevention_share(onfarm_share, upstream_share):
    """On-farm and upstream prevention combined as a union, not a sum.

    `1 - (1 - onfarm)(1 - upstream)`: the soil that gets through both. Summing the two shares
    would double-count the loss that either alone would have prevented, and could exceed 1.

    Returns:
        numpy.ndarray: the combined share, in [0, 1].
    """
    combined = 1.0 - (1.0 - np.asarray(onfarm_share)) * (1.0 - np.asarray(upstream_share))
    return np.clip(combined, 0.0, 1.0)


def restrict_to_valued_pixels(share, is_cropland, is_severe):
    """Zero a prevention share wherever the account does not value it.

    The account values prevention only on cropland, and only where soil loss exceeds the
    country's tolerance, so prevention of a loss the soil could absorb is not counted.

    Returns:
        numpy.ndarray: the share where both hold, zero elsewhere.
    """
    return np.where(np.asarray(is_cropland) & np.asarray(is_severe), np.asarray(share), 0.0)


def country_erosion_shock(df_country_crop, min_shock_floor=MIN_SHOCK_FLOOR):
    """A country's crop production shock, from its per-crop protected shares.

    The shock is the production-weighted mean over crops of `protected share x elasticity`, so a
    crop that is a large part of the country's output moves the shock more than a small one, and a
    crop whose yield barely responds to soil loss contributes little however well protected it is.
    Both factors are clipped into [0, 1]: an elasticity above 1 would claim yield falls faster
    than soil is lost.

    Args:
        df_country_crop (pandas.DataFrame): one row per country and crop, with `ISO3`,
            `protected_production_tons`, `total_production_tons`, `share_protected_production`
            and `elasticity_used`.
        min_shock_floor (float): shocks between zero and this floor to it.

    Returns:
        pandas.DataFrame: `iso3`, `protected_production_tons`, `total_production_tons`,
        `share_protected_production` and `erosion_shock_share`. A country with no production
        carries a missing share and a missing shock rather than a zero, because there is nothing
        to take a share of.
    """
    df = df_country_crop.copy()

    physical = df.groupby('ISO3', as_index=False)[
        ['protected_production_tons', 'total_production_tons']].sum()
    physical['share_protected_production'] = np.where(
        physical['total_production_tons'] > 0,
        np.clip(physical['protected_production_tons'] / physical['total_production_tons'], 0.0, 1.0),
        np.nan)
    physical = physical.rename(columns={'ISO3': 'iso3'})

    for column in ['total_production_tons', 'share_protected_production', 'elasticity_used']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df['weight'] = df['total_production_tons'].clip(lower=0.0)
    df['term'] = (df['weight']
                  * df['share_protected_production'].clip(0.0, 1.0)
                  * df['elasticity_used'].clip(0.0, 1.0))

    weighted = df.groupby('ISO3', as_index=False).agg(
        weight_sum=('weight', 'sum'), term_sum=('term', 'sum'))
    weighted['erosion_shock_share'] = np.where(
        weighted['weight_sum'] > 0,
        (weighted['term_sum'] / weighted['weight_sum']).clip(0.0, 1.0),
        np.nan)

    tiny = (weighted['erosion_shock_share'].notna()
            & (weighted['erosion_shock_share'] > 0)
            & (weighted['erosion_shock_share'] < min_shock_floor))
    weighted.loc[tiny, 'erosion_shock_share'] = min_shock_floor
    weighted = weighted.rename(columns={'ISO3': 'iso3'})

    return physical.merge(weighted[['iso3', 'erosion_shock_share']], on='iso3', how='left')


def country_gep(df_shock, df_crop_gpv, df_gdp, component):
    """The value of the protected production, and what it is as a share of GDP.

    Args:
        df_shock (pandas.DataFrame): `country_erosion_shock`'s output.
        df_crop_gpv (pandas.DataFrame): `iso3` and `crop_gpv_const2019_2019`, the country's crop
            gross production value in constant 2019 dollars.
        df_gdp (pandas.DataFrame): `iso3` and `gdp_const2019_2019`.
        component (str): which prevention channel this is, carried onto every row so the on-farm,
            upstream and combined runs can be concatenated and still told apart.

    Returns:
        pandas.DataFrame: the shock columns plus `crop_gpv_const2019_2019`,
        `gdp_const2019_2019`, `gep_const2019_usd` and `gdp_loss_pct`.
    """
    out = df_shock.merge(df_crop_gpv, on='iso3', how='left').merge(df_gdp, on='iso3', how='left')
    out['gep_const2019_usd'] = (out['crop_gpv_const2019_2019'].fillna(0.0)
                                * out['erosion_shock_share'].fillna(0.0))
    out['gdp_loss_pct'] = np.where(
        out['gdp_const2019_2019'].notna() & (out['gdp_const2019_2019'] > 0),
        100.0 * out['gep_const2019_usd'] / out['gdp_const2019_2019'],
        np.nan)
    out['component'] = component
    return out[['component', 'iso3', 'protected_production_tons', 'total_production_tons',
                'share_protected_production', 'erosion_shock_share', 'crop_gpv_const2019_2019',
                'gdp_const2019_2019', 'gep_const2019_usd', 'gdp_loss_pct']]
