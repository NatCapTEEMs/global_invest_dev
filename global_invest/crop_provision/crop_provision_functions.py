# -*- coding: utf-8 -*-
"""Crop-provision science: FAOSTAT gross production value x the CWoN land rental rate.

The quantity-and-price stage is already fused in the source data: FAOSTAT's Value of Production
bulk file reports gross production value per country, crop and year, so no separate price join
happens here. Attribution is the Changing Wealth of Nations 2024 land rental rate, a per-country
share that varies by decade, applied by an as-of merge so each year takes the rate of the decade
it falls in. The result is converted from FAOSTAT's thousand USD to plain USD once, before any
grouping, and collapsed onto the r250 country list.

Every function here is a pure transformation over frames, which is what the tests exercise. The
task module reads the FAOSTAT bulk file and the rental-rate table and passes the frames in.
"""
import logging

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






def attach_countries_in_usd(df_crop_value, df_countries):
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
    df['crop_provision_gep'] = df['crop_provision_gep'] * FAOSTAT_THOUSAND_USD
    return df




