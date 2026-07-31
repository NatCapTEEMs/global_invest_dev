# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hazelbean as hb

def read_mangrove_values(path: str):

    try:
        df_mangrove_value = pd.read_excel(path, sheet_name='Sheet1', engine='openpyxl')
        logging.info(f"Loaded mangrove coastal protection values from {path} ({df_mangrove_value.shape[0]} rows).")
    except Exception as e:
        logging.error(f"Failed to read crop values file '{path}': {e}")
        raise


    # rename columns
    old_names = ["countrycode","annual_value_2019"]
    new_names = ["ee_r264_label",'Value']

    rename_dict = dict(zip(old_names, new_names))
    df_mangrove_value.rename(columns=rename_dict, inplace=True)


    logging.info(f"Finished cleaning up ({df_mangrove_value.shape[0]} rows).")

    # reshape to long format
    df_mangrove_value["year"] = pd.to_numeric(df_mangrove_value["year"], errors="coerce").astype(int)

    logging.info(f"Reshaped to long format ({df_mangrove_value.shape[0]} rows).")
    return df_mangrove_value


def read_coral_reef_values(path: str):

    try:
        df_coral_reef_value = pd.read_excel(path, sheet_name='Sheet1', engine='openpyxl')
        logging.info(f"Loaded mangrove coastal protection values from {path} ({df_coral_reef_value.shape[0]} rows).")
    except Exception as e:
        logging.error(f"Failed to read crop values file '{path}': {e}")
        raise
        df_coral_reef_value['coral_reef_value'] = pd.to_numeric(df_coral_reef_value['coral_reef_value'], errors='coerce')*1000000  
         # original values: 2011 USD millions -> convert to USD

    logging.info(f"Finished cleaning up ({df_coral_reef_value.shape[0]} rows).")


    return df_coral_reef_value


def read_gdp_inflation_deflator(path: str):
    """
    Read the World Bank GDP inflation deflator data from the specified Excel file.
    https://data.worldbank.org/indicator/NY.GDP.DEFL.KD.ZG

    """

    try:
        df_gdp_inflation_deflator = pd.read_excel(path, engine='openpyxl')
        df_gdp_inflation_deflator = df_gdp_inflation_deflator.melt(
            id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
            var_name='year',
            value_name='value'
        )
        df_gdp_inflation_deflator['year'] = pd.to_numeric(df_gdp_inflation_deflator['year'], errors='coerce').astype('Int64')
        logging.info(f"Loaded mangrove coastal protection values from {path} ({df_gdp_inflation_deflator.shape[0]} rows).")
    except Exception as e:
        logging.error(f"Failed to read crop values file '{path}': {e}")
        raise

    logging.info(f"Finished cleaning up ({df_gdp_inflation_deflator.shape[0]} rows).")

    return df_gdp_inflation_deflator


def get_inflation_deflator_multiplier(path, start_year, end_year):

    """
    Compute cumulative GDP deflator multiplier between two years for each country.

    Parameters
    ----------
    p : object
        Parameter object containing path attributes (e.g., p.df_gdp_inflation_deflator_path)
    start_year : int
        The first year in the period (inclusive)
    end_year : int
        The last year in the period (inclusive)

    Returns
    -------
    DataFrame
        A DataFrame with columns:
        ['Country Code', 'Country Name', f'deflator_multiplier_{start_year}_{end_year}']
    """

    df_gdp_inflation_deflator = read_gdp_inflation_deflator(path)

    mask = df_gdp_inflation_deflator["year"].between(start_year, end_year)
    df_gdp_inflation_deflator = df_gdp_inflation_deflator[mask]
    df_gdp_inflation_deflator["multiplier"] = 1 + df_gdp_inflation_deflator["value"]/100
    df_gdp_inflation_deflator = (df_gdp_inflation_deflator.groupby(["Country Code", "Country Name"], as_index=False)["multiplier"]
    .prod()
    .rename(columns={"multiplier": "deflator_multiplier"})
    )

    df_gdp_inflation_deflator.rename(
        columns={
            "Country Code": "ee_r264_label",
            "Country Name": "ee_r264_name"
        },
        inplace=True
    )

    return df_gdp_inflation_deflator

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

