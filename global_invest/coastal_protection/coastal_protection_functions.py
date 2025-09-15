# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hazelbean as hb



path = '/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_protection/data_mangroves_2019.xlsx'

def read_mangrove_values(path: str):
    """
    Read FAO crop production values, filter by unit, drop unwanted columns/crops/countries,
    and reshape to long format.

    Returns DataFrame with columns: [area_code, country, crop_code, crop, year, gep].
    """

    try:
        df_mangrove_value = pd.read_excel(path, sheet_name='Sheet1', engine='openpyxl')
        logging.info(f"Loaded mangrove coastal protection values from {path} ({df_mangrove_value.shape[0]} rows).")
    except Exception as e:
        logging.error(f"Failed to read crop values file '{path}': {e}")
        raise


    # rename columns
    old_names = ["countrycode","annual_value_2019"]
    new_names = ["fao_r250_label",'Value']

    rename_dict = dict(zip(old_names, new_names))
    df_mangrove_value.rename(columns=rename_dict, inplace=True)


    logging.info(f"Finished cleaning up ({df_mangrove_value.shape[0]} rows).")

    # reshape to long format
    df_mangrove_value["year"] = pd.to_numeric(df_mangrove_value["year"], errors="coerce").astype(int)

    logging.info(f"Reshaped to long format ({df_mangrove_value.shape[0]} rows).")
    return df_mangrove_value



def group_countries(df: pd.DataFrame):
    """
    Aggregate total GEP across all countries by year.
    """
    # df = df.loc[df['year'] == 2019].copy()
    df_gep_by_year = hb.df_groupby(df, groupby_cols='year', agg_cols="Value", preserve='keep_all_valid')

    
    # START HERE: df_gep_by_year = hb.df_groupby(df, groupby_cols='iso3_r250_label', agg_dict={"Value": "sum"}). This line causes a really wrongly formatted DataFrame.
    df_gep_by_year.set_index("year", inplace=False)
    # df_gep_by_year.rename(columns={"gep": "total_gep"}, inplace=True)
    df_gep_by_year.sort_values("year", inplace=True)
    logging.info(f"Grouped total by year ({df_gep_by_year.shape[0]} rows).")
    return df_gep_by_year
