# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt


def preprocess_fao(fao_input_path, output_path):
    """
    Task: Run Commerical Agriculture
    Who to Blame: Ryan McWay, Marta Sylla, Lijing Wang
    """

    # Import the FAO commerical data 
    df_crop_value = pd.read_csv(fao_input_path, encoding='ISO-8859-1')
    # Limit to International dollar values 
    df_crop_value = df_crop_value[df_crop_value['Unit'] == "1000 Int$"]

    # Rename variables: 
    old_names = ["Area Code", "Area Code (M49)", "Area", "Item Code", 
                "Item", "Y1961", "Y1962", "Y1963", "Y1964", "Y1965",
                "Y1966", "Y1967", "Y1968", "Y1969", "Y1970", "Y1971",         
                "Y1972", "Y1973", "Y1974", "Y1975", "Y1976", "Y1977", 
                "Y1978", "Y1979", "Y1980", "Y1981", "Y1982", "Y1983", 
                "Y1984", "Y1985", "Y1986", "Y1987", "Y1988", "Y1989", 
                "Y1990", "Y1991", "Y1992", "Y1993", "Y1994", "Y1995", 
                "Y1996", "Y1997", "Y1998", "Y1999", "Y2000", "Y2001", 
                "Y2002", "Y2003", "Y2004", "Y2005", "Y2006", "Y2007", 
                "Y2008", "Y2009", "Y2010", "Y2011", "Y2012", "Y2013", 
                "Y2014", "Y2015", "Y2016", "Y2017", "Y2018", "Y2019", 
                "Y2020", "Y2021", "Y2022"]
    new_names = ["area_code", "area_code_M49", "country", "crop_code", 
                "crop", "1961", "1962", "1963", "1964", "1965",
                "1966", "1967", "1968", "1969", "1970", "1971",         
                "1972", "1973", "1974", "1975", "1976", "1977", 
                "1978", "1979", "1980", "1981", "1982", "1983", 
                "1984", "1985", "1986", "1987", "1988", "1989", 
                "1990", "1991", "1992", "1993", "1994", "1995", 
                "1996", "1997", "1998", "1999", "2000", "2001", 
                "2002", "2003", "2004", "2005", "2006", "2007", 
                "2008", "2009", "2010", "2011", "2012", "2013", 
                "2014", "2015", "2016", "2017", "2018", "2019", 
                "2020", "2021", "2022"]

    # Rename the columns
    rename_dict = dict(zip(old_names, new_names))
    df_crop_value.rename(columns=rename_dict, inplace=True)

    # Remove any regional or international values (e.g., keep only the countries)
    countries_to_drop = ['USSR', 'Yugoslav SFR', 'World' 'Africa', 'Eastern Africa', 'Middle Africa',
                        'Northern Africa', 'Southern Africa', 'Western Africa', 'Americas', 'Northern America',
                        'Central America', 'Caribbean', 'South America', 'Asia', 'Central Asia', 'Eastern Asia',
                        'Southern Asia', 'South-eastern Asia', 'Western Asia', 'Europe', 'Eastern Europe',
                        'Northern Europe', 'Southern Europe', 'Western Europe', 'Oceania', 'Australia and New Zealand',
                        'Melanesia', 'Micronesia', 'Polynesia', 'European Union (27)', 'Least Developed Countries',
                        'Land Locked Developing Countries', 'Small Island Developing States',
                        'Low Income Food Deficit Countries', 'Net Food Importing Developing Countries']
    df_crop_value = df_crop_value[~df_crop_value['country'].isin(countries_to_drop)]

    # Reshape Long: Country, Year, Crop for each row.
    df_crop_value = pd.melt(df_crop_value, 
                                id_vars = ["area_code", "country", "crop_code", "crop"], 
                                value_vars = [str(year) for year in range(1961, 2023)],  # Years 1961 to 2022
                                var_name = "year", 
                                value_name = "Value")
    

    # Save a csv file for values by crop, country, year
    df_crop_value = df_crop_value.sort_values(by = ['country', 'year'], ascending = [True, True])
    df_crop_value.to_csv(output_path, index=False)
    
    

def calculate_gep(input_path, output_path, **kwargs):
    """ Task: Calculate GEP from FAO data
    """
    
    crop_values_output_path = kwargs.get('crop_values_output_path', None)
    
    df_crop_value = pd.read_csv(input_path)

    # Rental rate adjustment... Perhaps GTAP value of somewhere in the 0.25 range
    rental_rate = 0.25
    df_crop_value['gep'] = df_crop_value['Value'] * rental_rate

    # CHECK: Summary Stats of avg year value by crop 


    # Aggregate up the crop values for country and year
    df_panel = df_crop_value.groupby(['country', 'year'], as_index = False).agg(
        gep = ('gep', 'sum')
        )

    # Save a csv file of country, year commerical agricultural values
    df_panel = df_panel.sort_values(by = ['country', 'year'], ascending = [True, True])
    df_panel.to_csv(output_path, index=False)
    
    
    # Save a csv file for values by crop, country, year
    if crop_values_output_path:
        df_crop_value = df_crop_value.sort_values(by = ['country', 'year'], ascending = [True, True])
        df_crop_value.to_csv(crop_values_output_path, index=False)




def calculate_gep_from_df_path(df_path, output_path, **kwargs):

    df_panel = pd.read_csv(df_path, output_dir)

    # What is the total value of commerical agriculture each year:
    annual_value = df_panel.groupby('year', as_index = False).agg(
        gep = ('gep', 'sum')
        )
    annual_value.set_index('year', inplace=True)
    # Plot as a time series
    plt.figure(figsize=(10, 6))
    plt.plot(annual_value.index, annual_value['gep'], marker='o', linestyle='-', color='b')
    plt.title('Times Series Commerical Agriculture')
    plt.xlabel('Year')  
    plt.ylabel('GEP')
    plt.xticks(rotation = 45)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "commerical_agriculture_time_series.png"), format='png')

    # CHECK: Top country producers values for a given year 


    # CHECK: Figure with time trend for the above countries

