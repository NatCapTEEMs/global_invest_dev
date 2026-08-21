"""Timber provision valuation chain, reimplemented from the drive's Forestry rasters.

The Forestry chain values managed forestry at 10 arcsec: annual biomass yield (m3/ha, MC2
forest types x Tian et al. timber regions, NPP-adjusted) times the national log price
(USD/m3, Siikamaki and Santiago-Avila 2014), converted from gross to net return with the
GTAP land factor share (Damania et al. 2023), minus per-pixel transport cost (Weiss et al.
travel time x trucking cost). Pixels are kept only where the Lesiv et al. (2022) forest
management map marks current timber activity (categories 2-5 on forested ESA classes);
negative net returns and nodata floor to zero.

Verified against the staged rasters (base_data/global_invest/timber_provision/input/):

  - current_forest_value.tif equals forestry_land_share_return_tcost_before_2022_07_23.tif
    bit-exactly on every one of its 265,210,379 positive pixels (0 mismatches globally);
    the remaining 802,119,863 positive net-return pixels are dropped by the management
    mask, which is external to the staged layers (it is NOT derivable from them).
  - The decomposition net_return = land_factor_share * biomass * price - transport_cost
    reproduces the net-return raster with a per-region fitted share (R2 0.78-0.9999 over
    12 continental windows, median relative error 0.5-4%): the share applies at GTAP
    region x AEZ level with separate plantation/natural estimates, so it is an input
    here, not a constant.
  - The committed per-country table (timber_provision_gep.csv, $88.74bn over 166
    countries) is the plain pixel sum of current_forest_value.tif inside each r250
    country polygon (ratio 1.0000 on all countries tested): the raster is per-pixel USD.
"""
import numpy as np

NET_RETURN_NDV = -99999.0


def net_forest_return(annual_biomass, log_price, transport_cost, land_factor_share):
    """Per-pixel net forestry return in USD: the appendix decomposition.

    Args:
        annual_biomass (np.ndarray): Annual harvestable biomass, m3 per pixel-hectare.
        log_price (np.ndarray): National producer log price, USD/m3.
        transport_cost (np.ndarray): Per-pixel transport cost, USD.
        land_factor_share (np.ndarray or float): GTAP land factor share converting gross
            to net returns (regional; broadcastable against the rasters).

    Returns:
        np.ndarray: land_factor_share * annual_biomass * log_price - transport_cost.
    """
    return land_factor_share * annual_biomass * log_price - transport_cost


def forest_value_from_net_return(net_return, managed_mask, ndv=NET_RETURN_NDV):
    """The current-forest-value raster: net return kept only where forestry is current.

    Args:
        net_return (np.ndarray): Net forestry return (may hold negatives and `ndv`).
        managed_mask (np.ndarray): Boolean, True where the Lesiv et al. (2022) management
            map marks current timber activity on forested ESA classes.
        ndv (float): The net-return raster's nodata value.

    Returns:
        np.ndarray: float32, net_return where managed and positive, else 0 (the value
        raster's own nodata convention).
    """
    kept = managed_mask & (net_return > 0) & (net_return != ndv)
    return np.where(kept, net_return, 0.0).astype(np.float32)

def gep_by_zone(value, zone_ids, n_zones):
    """Per-zone pixel sums of a value block, indexed by zone id.

    Args:
        value (np.ndarray): Value raster block (per-pixel USD).
        zone_ids (np.ndarray): Integer zone-id block, same shape; 0 = background.
        n_zones (int): Highest zone id; the output has n_zones + 1 entries.

    Returns:
        np.ndarray: float64 sums, index i = total for zone id i. Blockwise callers
        accumulate by adding successive blocks' arrays.
    """
    return np.bincount(zone_ids.ravel(), weights=value.astype(np.float64).ravel(),
                       minlength=n_zones + 1)


def timber_gep_from_zone_sums(zone_sums, countries_df):
    """One row per country: the zonal pixel sums keyed to iso3_r250_id.

    Args:
        zone_sums (np.ndarray): Per-zone sums from `gep_by_zone` (accumulated), indexed
            by iso3_r250_id.
        countries_df (pd.DataFrame): One row per country, with an iso3_r250_id column.

    Returns:
        pd.DataFrame: countries_df plus a timber_provision_gep column.
    """
    df = countries_df.copy()
    df["timber_provision_gep"] = zone_sums[df["iso3_r250_id"].to_numpy()]
    return df
