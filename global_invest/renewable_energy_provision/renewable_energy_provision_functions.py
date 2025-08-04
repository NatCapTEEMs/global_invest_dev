# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hazelbean as hb

# TODOO Obviously these should be built into hazelbean instead of duplicated all over.

def merge_dfs(main_df, list_of_dfs):
    """
    Merge a main DataFrame with a list of DataFrames on common columns ['ISO3 code', 'Year'].

    Parameters:
        main_df (pd.DataFrame): The main DataFrame with columns ['ISO3 code', 'Country', 'Year', 'Price'].
        list_of_dfs (list): List of renewable energy demand dfs 

    Returns:
        list: A list of merged DataFrames.
    """
    merged_dfs = []

    # Define the common columns for the merge
    merge_columns = ['ISO3 code', 'Year']

    for df in list_of_dfs:
        # Perform the merge
        merged_df = pd.merge(main_df, df, on=merge_columns, how='inner')
        # remove redundant country cols
        merged_df = merged_df.drop('Country_y', axis=1) 
        merged_df = merged_df.rename(columns={'Country_x' : 'Country'})
        # Append the merged DataFrame to the results list
        merged_dfs.append(merged_df)

    return merged_dfs



# Function to filter dataframe for 2019 and save CSVs by Technology
def filter_and_split_by_resource(df):



    # Rename columns
    df_filtered = df.rename(columns={
        # 'ISO3 code': 'Country_Code',
        'Country': 'Country_Name',
        'Group Technology': 'Resource',
        'Price (USD/GWh)': 'P_electricity_USD_per_GWh',
        'Electricity Generation (GWh)': 'energy_prod_GWh'
    })

    # Reorder columns
    # df_filtered = df_filtered[['Resource', 'Country_Name', 'Year', 'iso3_r250_label', 'P_electricity_USD_per_GWh', 'energy_prod_GWh', 'nat_contrib', 'renewable_energy_provision_gep']]



    # Create dictionary of dataframes split by Resource
    technology_dfs = {resource: df_resource.copy() for resource, df_resource in df_filtered.groupby('Resource')}


    return technology_dfs