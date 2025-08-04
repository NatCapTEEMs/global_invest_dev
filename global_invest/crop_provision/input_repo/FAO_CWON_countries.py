import pandas as pd
import logging
from OLD_estimate_commerical_agriculture import read_crop_values, read_crop_coefs  # Adjust import as needed

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def check_missing_rental_rates(df_crop_value: pd.DataFrame, df_crop_coefs: pd.DataFrame):
    """
    Check which countries have FAO data but no corresponding rental rate in the CWON data.

    Args:
    - df_crop_value: FAO crop value data (including 'area_code' and 'country')
    - df_crop_coefs: CWON crop coefficient data (including 'FAO' and 'Country/territory')

    Returns:
    - DataFrame with countries missing rental rates
    """
    # Get a list of 'area_code' from FAO data
    fao_area_codes = df_crop_value["area_code"].unique()

    # Get a list of 'FAO' from CWON data
    cwon_fao_codes = df_crop_coefs["FAO"].unique()

    # Find area codes in FAO data that are not in CWON data
    missing_rental_rates = [area_code for area_code in fao_area_codes if area_code not in cwon_fao_codes]

    # Get the countries corresponding to the missing area codes
    countries_missing_rates = df_crop_value[df_crop_value["area_code"].isin(missing_rental_rates)][["country", "area_code"]]

    # Log the number of countries missing rental rates
    logging.info(f"Countries with FAO data but no rental rate data in CWON: {countries_missing_rates.shape[0]} countries")

    # Display all rows in the terminal by changing pandas display options
    pd.set_option('display.max_rows', None)  # Set to None to show all rows
    print("Countries with FAO data but no rental rate data in CWON:")
    print(countries_missing_rates)

    # Reset to default after displaying
    pd.reset_option('display.max_rows')

    # To troubleshoot, let's print the area codes that are missing in CWON
    print("Area codes missing in CWON data:")
    missing_codes = [area_code for area_code in fao_area_codes if area_code not in cwon_fao_codes]
    print(missing_codes)

    # Optionally, you can inspect what is in each dataset:
    print("\nFAO area codes:")
    print(fao_area_codes)

    print("\nCWON FAO codes:")
    print(cwon_fao_codes)

    return countries_missing_rates

def main():
    # Set the paths to your CSV files
    input_dir = "input"  # Adjust this path as needed
    fao_data_path = "input/Value_of_Production_E_All_Data2.csv"
    cwon_data_path = "input/CWON2024_crop_coef.csv"

    # Read the data
    df_crop_value = read_crop_values(fao_data_path)
    df_crop_coefs = read_crop_coefs(cwon_data_path)

    # Call the function to check for missing rental rates
    check_missing_rental_rates(df_crop_value, df_crop_coefs)

if __name__ == "__main__":
    main()
