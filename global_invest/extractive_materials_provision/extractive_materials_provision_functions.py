# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hazelbean as hb

def read_mineral_values(path: str):
    """
    Read FAO crop production values, filter by unit, drop unwanted columns/crops/countries,
    and reshape to long format.

    Returns DataFrame with columns: [area_code, country, crop_code, crop, year, gep].
    """

    try:
        df_raw_mineral_values = pd.read_csv(path, encoding="ISO-8859-1")
        logging.info(f"Loaded crop values from {path} ({df_raw_mineral_values.shape[0]} rows).")
    except Exception as e:
        logging.error(f"Failed to read crop values file '{path}': {e}")
        raise

    df_raw_mineral_values.drop(columns=['ï»¿Country Name','Indicator Name','Indicator Code'], inplace=True)

    df_mineral_values = df_raw_mineral_values.melt(
    id_vars=["Country Code"],        # keep this column fixed
    var_name="year",                 # new column for year
    value_name="mineral_rent"       # new column for value
    )  
    return df_mineral_values

def read_GDP_values(path: str):
    """
    Read FAO crop production values, filter by unit, drop unwanted columns/crops/countries,
    and reshape to long format.

    Returns DataFrame with columns: [area_code, country, crop_code, crop, year, gep].
    """

    try:
        df_raw_GDP_values = pd.read_csv(path, encoding="ISO-8859-1")
        logging.info(f"Loaded crop values from {path} ({df_raw_GDP_values.shape[0]} rows).")
    except Exception as e:
        logging.error(f"Failed to read crop values file '{path}': {e}")
        raise

    df_raw_GDP_values.drop(columns=['ï»¿Country Name','Indicator Name','Indicator Code'], inplace=True)

    df_GDP_values = df_raw_GDP_values.melt(
    id_vars=["Country Code"],        # keep this column fixed
    var_name="year",                 # new column for year
    value_name="GDP_currentUSD"       # new column for value
    )  
    return df_GDP_values

def group_countries(df: pd.DataFrame):
    """
    Aggregate total GEP across all countries by year.
    """
    df_gep_by_year = hb.df_groupby(df, groupby_cols='year', agg_cols="Value", preserve='keep_all_valid')

    
    # START HERE: df_gep_by_year = hb.df_groupby(df, groupby_cols='iso3_r250_label', agg_dict={"Value": "sum"}). This line causes a really wrongly formatted DataFrame.
    df_gep_by_year.set_index("year", inplace=False)
    # df_gep_by_year.rename(columns={"gep": "total_gep"}, inplace=True)
    df_gep_by_year.sort_values("year", inplace=True)
    logging.info(f"Grouped total by year ({df_gep_by_year.shape[0]} rows).")
    return df_gep_by_year