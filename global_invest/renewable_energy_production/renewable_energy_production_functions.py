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
def export_by_resource(df, output_dir):

    # set output directory
    out_dir = output_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True) # create dir if needed

    # Filter years
    df_filtered = df.loc[df['Year'] == 2019] # 2019 only base
    # df_filtered = df[df['Year'].between(2014, 2019)].copy()

    # Rename columns
    df_filtered = df_filtered.rename(columns={
        'ISO3 code': 'Country_Code',
        'Country': 'Country_Name',
        'Group Technology': 'Resource',
        'Price (USD/GWh)': 'P_electricity_USD_per_GWh',
        'Electricity Generation (GWh)': 'energy_prod_GWh'
    })

    # Reorder columns
    df_filtered = df_filtered[['Resource', 'Country_Name', 'Year', 'Country_Code', 'P_electricity_USD_per_GWh', 'energy_prod_GWh', 'nat_contrib', 'gep']]

    # Drop rows where gep <= 0 (happens from nat_contrib calc)
    df_filtered = df_filtered.loc[df_filtered['gep'] > 0]

    # Create dictionary of dataframes split by Resource
    technology_dfs = {resource: df_resource.copy() for resource, df_resource in df_filtered.groupby('Resource')}

    # Save each df to CSV
    for tech, tech_df in technology_dfs.items():
        filename = f'{tech.replace(" ", "_").lower()}_gep_2019.csv'
        out_path = os.path.join(out_dir, filename)
        tech_df.to_csv(out_path, index=False)
        print(f'Saved: {out_path}')

    return technology_dfs