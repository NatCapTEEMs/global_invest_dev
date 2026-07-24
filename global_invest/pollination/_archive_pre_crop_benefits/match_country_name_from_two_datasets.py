import pandas as pd

# File paths
price_file = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\Price_data\Prices_E_All_Data_ori.csv"
pollination_file = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\crop_production_by_country\pollination_dependent_apple_production_crop_production_by_country.csv"

# Output file paths for mismatches
price_not_in_pollination_file = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\mismatches\price_not_in_pollination.csv"
pollination_not_in_price_file = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\mismatches\pollination_not_in_price.csv"

# Function to load CSV with encoding error handling
def load_csv_with_encoding(file_path):
    encodings = ['utf-8', 'ISO-8859-1', 'latin1']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            print(f"Failed to decode {file_path} with {encoding} encoding.")
    raise ValueError(f"Unable to decode {file_path} with tried encodings.")

# Load the datasets with error handling
price_df = load_csv_with_encoding(price_file)
pollination_df = load_csv_with_encoding(pollination_file)

# Extract unique country names
unique_countries_price = set(price_df['Area'].unique())
unique_countries_pollination = set(pollination_df['Zone_ID'].unique())

# Find mismatches
price_not_in_pollination = unique_countries_price - unique_countries_pollination
pollination_not_in_price = unique_countries_pollination - unique_countries_price

# Convert mismatches to DataFrames
price_not_in_pollination_df = pd.DataFrame(list(price_not_in_pollination), columns=["Country"])
pollination_not_in_price_df = pd.DataFrame(list(pollination_not_in_price), columns=["Country"])

# Save mismatches as CSV files
price_not_in_pollination_df.to_csv(price_not_in_pollination_file, index=False)
pollination_not_in_price_df.to_csv(pollination_not_in_price_file, index=False)

print(f"Mismatches saved to:\n- {price_not_in_pollination_file}\n- {pollination_not_in_price_file}")
