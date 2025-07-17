import os
import sys
import pandas as pd
import hazelbean as hb

from global_invest.commercial_agriculture import commercial_agriculture_functions
from global_invest.commercial_agriculture import commercial_agriculture_defaults


def commercial_agriculture(p):
    """
    Parent task for commercial agriculture.
    """
    p.fao_input_path = p.get_path(os.path.join(p.base_data_dir, 'global_invest', 'commercial_agriculture', 'Value_of_Production_E_All_Data.csv'))


def gep_calculation(p):
    """ GEP calculation task for commercial agriculture."""
    # Ranked in order of processing, basically from least aggregated to most aggregated.
    result = {}
    p.results['commercial_agriculture'] = result   
    p.results['commercial_agriculture']['gep_by_country_year_crop'] = os.path.join(p.cur_dir, "gep_by_country_year_crop.csv")
    p.results['commercial_agriculture']['gep_by_country_year'] = os.path.join(p.cur_dir, "gep_by_country_year.csv")
    p.results['commercial_agriculture']['gep_by_country_base_year'] = os.path.join(p.cur_dir, "gep_by_country_base_year.csv")
    p.results['commercial_agriculture']['gep_by_year'] = os.path.join(p.cur_dir, "gep_by_year.csv")
        

    input_dir = os.path.join(p.base_data_dir, 'global_invest', 'commercial_agriculture')
    
    if not getattr(p, 'commercial_agriculture_subservices', None):
        p.commercial_attribute_subservices = commercial_agriculture_defaults.DEFAULT_AGRICULTURE_ITEMS

    # 1. Read and process data
    df_crop_value = commercial_agriculture_functions.read_crop_values(os.path.join(input_dir, "Value_of_Production_E_All_Data.csv"), p.commercial_attribute_subservices)
    df_crop_coefs = commercial_agriculture_functions.read_crop_coefs(os.path.join(input_dir, "CWON2024_crop_coef.csv"))

    df_gep_by_country_year_crop = commercial_agriculture_functions.merge_crop_with_coefs(df_crop_value, df_crop_coefs)
    # String mangle the FAO M49 codes to integers.
    df_gep_by_country_year_crop['area_code_M49'] = df_gep_by_country_year_crop['area_code_M49'].str.replace('\'', '')
    df_gep_by_country_year_crop['area_code_M49'] = df_gep_by_country_year_crop['area_code_M49'].astype(int)
   
    replacements = {
        159: 156,  # China
        891: 688,  # Serbia and Montenegro
        200: 203,  # Czechoslovakia
        230: 231,  # Ethiopia PDR
        736: 729,  # Sudan (former)     
    }
    
    # Replace wrong codes in the m49
    df_gep_by_country_year_crop['area_code_M49'] = df_gep_by_country_year_crop['area_code_M49'].replace(replacements)    

    # LEARNING POINT: I wasted lots of time not realizing the a how='right' operates differently than I expect. The left had IDs that were not in right under r264_id, but they thus had the a 
    # repeated ID in the r250. I had wrongly thought that the how='right' would only then return 1 row for each r250_id, but it actually a duplicate row repeated for each unique r264_id
    # even tho the r_250_id was the same. Thus, I had to drop the repeated ones.
    
    # Drop repeated ids in ee_r264_df
    ee_r264_to_250 = p.ee_r264_df.copy()
    ee_r264_to_250 = ee_r264_to_250[ee_r264_to_250['ee_r264_label'] == ee_r264_to_250['iso3_r250_label']]
    
    cols_to_keep = [
        'ee_r264_id',	
        'iso3_r250_id',
        'ee_r264_label',
        'iso3_r250_label',
        'ee_r264_name',
        'iso3_r250_name',
        'continent',
        'region_un',
        'region_wb',
        'income_grp',
        'subregion',
        'area_code_M49',
        'area_code',
        'country',
        'crop_code',
        'crop',
        'year',
        'rental_rate',
        'Value',
    ]
    ee_r264_to_250.drop([i for i in ee_r264_to_250.columns if i not in cols_to_keep], axis=1, inplace=True, errors='ignore')
    # ee_r264_to_250 = ee_r264_to_250[cols_to_keep]
    
    # Merge so it has all the good labels from the ee_r264_df 
    df_gep_by_country_year_crop = hb.df_merge(ee_r264_to_250, df_gep_by_country_year_crop, how='right', left_on='iso3_r250_id', right_on='area_code_M49')
    
    df_gep_by_country_year = commercial_agriculture_functions.group_crops(df_gep_by_country_year_crop)

    df_gep_by_year = commercial_agriculture_functions.group_countries(df_gep_by_country_year)
    
    df_gep_by_country_base_year = df_gep_by_country_year.loc[df_gep_by_country_year['year'] == 2019].copy()
    
    # Write to CSVs
    hb.df_write(df_gep_by_country_year_crop, p.results['commercial_agriculture']['gep_by_country_year_crop'])
    hb.df_write(df_gep_by_country_year, p.results['commercial_agriculture']['gep_by_country_year'])
    hb.df_write(df_gep_by_country_base_year, p.results['commercial_agriculture']['gep_by_country_base_year'])   
    hb.df_write(df_gep_by_year, p.results['commercial_agriculture']['gep_by_year'])

    # Then sum the values across all countries. 
    value_gep_base_year = df_gep_by_country_base_year['Value'].sum()
    
    hb.log(f"Total GEP value for base year 2019: {value_gep_base_year}")
    
    return value_gep_base_year


def gep_preprocess_ryan_old(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.commercial_agriculture_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    p.commercial_agriculture_input_path = os.path.join(p.cur_dir, "commercial_agriculture_value_by_crop.csv")
    commercial_agriculture_functions.preprocess_fao(p.fao_input_path, p.commercial_agriculture_input_path)


def gep_calculation_justin_try_1(p):

    # Ranked in order of processing, basically from least aggregated to most aggregated.
    result = {}
    p.results['commercial_agriculture'] = result   
    p.results['commercial_agriculture']['gep_by_country_year_crop_csv'] = os.path.join(p.cur_dir, "gep_by_country_year_crop.csv")
    p.results['commercial_agriculture']['gep_by_country_year_csv'] = os.path.join(p.cur_dir, "gep_by_country_year.csv")
    p.results['commercial_agriculture']['gep_by_country_base_year_csv'] = os.path.join(p.cur_dir, "gep_by_country_base_year.csv")
        
    if not p.validate_result(result):

        if hb.path_exists(p.results['commercial_agriculture']['gep_by_country_year_crop_csv']):
            gep_by_country_year_crop = hb.df_read(p.results['commercial_agriculture']['gep_by_country_year_crop_csv'])
        else:
            raw_fao_input = hb.df_read(p.fao_input_path)


            # keep only Int$ unit AND element code 57
            crop_value = raw_fao_input[(raw_fao_input["Element Code"] == 58)].copy()
            # crop_value = raw_fao_input[(raw_fao_input["Element Code"] == 152) | (raw_fao_input["Element Code"] == 58)].copy()

            # If rows with element 58 are empty, fill it with the value in 152
            # crop_value.loc[crop_value["Element Code"] == 58, "Value"] = crop_value.loc[crop_value["Element Code"] == 58, "Value"].fillna(crop_value.loc[crop_value["Element Code"] == 152, "Value"])
            
            
            # drop columns ending with F
            cols_to_drop = [col for col in crop_value.columns if col.endswith("F")]
            crop_value.drop(columns=cols_to_drop, inplace=True)

            # rename columns
            old_names = ["Area Code", "Area Code (M49)", "Area", "Item Code", "Item"] + [f"Y{y}" for y in range(1961, 2023)]
            new_names = ["area_code", "iso3_r250_id", "country", "crop_code", "crop"] + [str(y) for y in range(1961, 2023)]

            rename_dict = dict(zip(old_names, new_names))
            crop_value.rename(columns=rename_dict, inplace=True)

            # Mangle the stupid fao string notation into a proper int.
            crop_value['iso3_r250_id'] = crop_value['iso3_r250_id'].str.replace('\'', '')    
            crop_value['iso3_r250_id'] = crop_value['iso3_r250_id'].astype(int)
            
            # Keep only listed items
            items = commercial_agriculture_defaults.DEFAULT_AGRICULTURE_ITEMS
            crop_value = crop_value[crop_value["crop"].isin(items)].copy()

            # drop countries not in iso3_r250_id
            countries = p.ee_r264_df["iso3_r250_id"].unique().tolist()
            crop_value = crop_value[crop_value["iso3_r250_id"].isin(countries)]
            
            # write to CSV
            # crop_value.to_csv(os.path.join(p.cur_dir, 'crop_value_raw.csv'), index=False)
            hb.df_write(crop_value, os.path.join(p.cur_dir, 'crop_value_raw.csv'))
            # reshape to long format
            crop_value_melted = pd.melt(
                crop_value,
                id_vars=["area_code", "iso3_r250_id", "country", "crop_code", "crop"],
                value_vars=[str(year) for year in range(1961, 2023)],  # 1961–2022
                var_name="year", 
            )
            
            hb.df_write(crop_value_melted, os.path.join(p.cur_dir, 'crop_value_melted.csv'))
        # "G:\Shared drives\NatCapTEEMs\Files\base_data\global_invest\commercial_agriculture\CWON2024_crop_coef.csv"
            crop_coefficients_path = os.path.join(p.base_data_dir, 'global_invest', 'commercial_agriculture', "CWON2024_crop_coef.csv")
            crop_coefs = hb.df_read(crop_coefficients_path, delimiter=';')

            crop_coefs = crop_coefs.melt(
                id_vars=["Order", "FAO", "Country/territory"],
                var_name="Decade",
                value_name="rental_rate",
            )
            crop_coefs["Decade_start"] = crop_coefs["Decade"].str.extract(r"^(\d{4})").astype(float)
            crop_coefs = crop_coefs.dropna(subset=["Decade_start"])

            # build the lookup
            crop_coefs = crop_coefs[["FAO", "Decade_start", "rental_rate"]].copy()

            # drop any rows where FAO is null (so the cast can succeed)
            crop_coefs = crop_coefs.dropna(subset=["FAO"])

            # ensure ints
            crop_coefs["FAO"] = crop_coefs["FAO"].astype(int)
            crop_coefs["Decade_start"] = crop_coefs["Decade_start"].astype(int)

            crop_coefs = crop_coefs.rename(columns={"Decade_start": "year"})
            
            hb.df_write(crop_coefs, os.path.join(p.cur_dir, 'crop_coefs.csv'))
            
            # Merge the crop value with the coefficients
            value_with_coeffs = hb.df_merge(crop_value_melted, crop_coefs, how='outer', left_on=['area_code', 'year'], right_on=['FAO', 'year'])
            
            hb.df_write(value_with_coeffs, p.results['commercial_agriculture']['gep_by_country_year_crop_csv'])
            
    
def gep_result(p):
    """Display the results of the GEP calculation."""
    os.environ['QUARTO_PYTHON'] = sys.executable
    
    qmd_paths = hb.list_filtered_paths_recursively(os.path.dirname(__file__), include_extensions='.qmd')
    
    for source_qmd_path in qmd_paths:
        results_qmd_path = os.path.join(p.cur_dir, os.path.split(source_qmd_path)[-1])
    
        
        hb.path_copy(source_qmd_path, results_qmd_path)
        
        
        quarto_command = f"quarto render {results_qmd_path}"
        hb.log(f"Running quarto command: {quarto_command}")        
        os.system(quarto_command)
        
    