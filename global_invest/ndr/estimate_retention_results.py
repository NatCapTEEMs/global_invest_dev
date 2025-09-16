import os
import sys
import yaml
import numpy as np
import pandas as pd
from itertools import product
from osgeo import gdal
gdal.UseExceptions()

def get_aqustat_fraction(aqustat_df, iso):
    """
    Retrieve the fraction of a specific aquifer status code from the aqustat dataframe.

    Parameters:
    aqustat_df (pd.DataFrame): DataFrame containing aquifer status codes and their fractions.
    aqustat_code (str): The aquifer status code to look for.

    Returns:
    float: The fraction corresponding to the specified aquifer status code.
           Returns 0.0 if the code is not found.
    """
    try:
        dom_fraction = float(aqustat_df.loc[aqustat_df['iso3'] == iso, 'municipal_water_withdrawal_percent_total'].values[0])/100
        return dom_fraction
    except IndexError:
        return 0.0

def estimate_retention_results_country(ndr_results_folder, iso, year, dom_fraction, n_price, nutrient='n'):
    """
    Estimate retention results for a specific country based on NDR results and percentage of domestic water uses.

    Parameters:
    ndr_results_folder (str): Path to the folder containing NDR results.
    iso (str): ISO code of the country.
    dom_fraction (float): Fraction of the dominant aquifer status.

    Returns:
    dict: A dictionary containing estimated retention results.
    """
    if nutrient == 'n':
        export_file = os.path.join(ndr_results_folder, iso, 'ndr', str(year), f'n_total_export_base.tif')
    else:
        export_file = os.path.join(ndr_results_folder, iso, 'ndr', str(year), f'p_surface_export_base.tif')
    load_file = os.path.join(ndr_results_folder, iso, 'ndr', str(year), 'intermediate_outputs',f'load_{nutrient}_base.tif')
    
    if not os.path.exists(export_file) or not os.path.exists(load_file):
        raise FileNotFoundError(f"NDR file for country {iso} not found")

    def _raster_open_nodata_mask(raster_path):
        ds = gdal.Open(raster_path)
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray().astype(float)
        no_data_value = band.GetNoDataValue()
        data[data == no_data_value] = float('nan')
        return data
    # Open the NDR raster files
    export_data = _raster_open_nodata_mask(export_file)
    load_data = _raster_open_nodata_mask(load_file)
    pixel_area_ha = 9  # Assuming 300m x 300m pixels, area in hectares
    #TODO: plug in pixel area raster. We need to align for this. 
    # Justin has the layer here: /projects/standard/jajohns/shared/base_data/pyramids/ha_per_cell_10sec.tif
    # Calculate retention values
    ndr_retention = np.nansum((load_data - export_data) * pixel_area_ha)  # in kg

    # Estimate retention based on mean NDR and dominant aquifer status fraction
    estimated_retention = ndr_retention * dom_fraction

    return {
        'ISO': iso,
        'Year': int(year),
        'TotalRetention(kg)': ndr_retention,
        'DomesticWaterFraction': dom_fraction,
        'EstimatedRetention(kg)': estimated_retention,
        'N_Price(usd)': n_price,
        'ServiceValue(usd)': estimated_retention * n_price
    }

if __name__ == "__main__":
    
    config  = sys.argv[1]           # config  = "/users/5/salmamun/GEP/gep/ndr/gep_sm_msi.yaml"
    with open(config) as yaml_data_file:
        args = yaml.load(yaml_data_file, Loader=yaml.FullLoader)

    aquastat_csv = args['aquastat_csv']  # '/path/to/aquastat.csv'
    prices_csv = args['price_csv'] # '/path/to/prices.csv'
    ndr_results_folder = args['result_folder']  # Base output directory for NDR results

    aqustat_df = pd.read_csv(aquastat_csv)
    prices_df = pd.read_csv(prices_csv)

    results_df = pd.DataFrame(columns=['ISO', 'Year', 'TotalRetention(kg)', 'DomesticWaterFraction',
    'EstimatedRetention(kg)', 'N_Price(usd)', 'ServiceValue(usd)'])
    iso_list = ['BGD', 'GMB']
    year_list = [2015, 2016]
    aoi_year_list = list(product(iso_list, year_list))
    for iso_year in aoi_year_list:
        iso = iso_year[0]
        year = iso_year[1]
        n_price = prices_df.loc[(prices_df['iso'] == iso) & (prices_df['year'] == year), 'price_n'].values[0]
        dom_fraction = get_aqustat_fraction(aqustat_df, iso)
        retention_results = estimate_retention_results_country(ndr_results_folder, iso, year, dom_fraction, n_price, nutrient='n')
        if len(results_df) == 0:
            results_df = pd.DataFrame([retention_results])
        else:
            results_df = pd.concat([results_df, pd.DataFrame([retention_results])], ignore_index=True)

    # Save results to CSV
    results_df.to_csv(args['estimation_output'], index=False)

