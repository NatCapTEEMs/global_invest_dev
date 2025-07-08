import os
import sys
import pandas as pd
import hazelbean as hb

from global_invest.commercial_agriculture import commercial_agriculture_functions
from global_invest.commercial_agriculture import commercial_agriculture_defaults

def build_standard_task_tree(p):
    """Build the default task tree for commercial agriculture."""
    p.commercial_agriculture_task = p.add_task(commercial_agriculture)
    p.commercial_agriculture_gep_calculation_task = p.add_task(gep_calculation, parent=p.commercial_agriculture_task)  
    return p

def build_gep_task_tree(p):
    """
    Build the default task tree forthe GEP application of commercial agriculture. In this case, it's very similar to the standard task tree
    but i've included it here for consistency with other models.
    """
    p.commercial_agriculture_task = p.add_task(commercial_agriculture)
    # p.commercial_agriculture_preprocess_task = p.add_task(gep_preprocess, parent=p.commercial_agriculture_task)  
    p.commercial_agriculture_gep_calculation_task = p.add_task(gep_calculation, parent=p.commercial_agriculture_task)  
    # p.commercial_agriculture_gep_result_task = p.add_task(gep_result, parent=p.commercial_agriculture_task)  
    
    return p

def commercial_agriculture(p):
    """
    Parent task for commercial agriculture.
    """
    p.fao_input_path = p.get_path(os.path.join(p.base_data_dir, 'global_invest', 'commercial_agriculture', 'Value_of_Production_E_All_Data.csv'))


def gep_preprocess_ryan_old(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.commercial_agriculture_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    p.commercial_agriculture_input_path = os.path.join(p.cur_dir, "commercial_agriculture_value_by_crop.csv")
    commercial_agriculture_functions.preprocess_fao(p.fao_input_path, p.commercial_agriculture_input_path)
    
def gep_calculation(p):

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
       
            crop_coefficients_path = os.path.join(p.base_data_dir, 'gep', "CWON2024_crop_coef.csv")
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
    """
    Display the results of the GEP calculation.
    """
    os.environ['QUARTO_PYTHON'] = sys.executable
    
    qmd_paths = hb.list_filtered_paths_recursively(os.path.dirname(__file__), include_extensions='.qmd')
    
    for source_qmd_path in qmd_paths:
        results_qmd_path = os.path.join(p.cur_dir, os.path.split(source_qmd_path)[-1])
    
        
        hb.path_copy(source_qmd_path, results_qmd_path)
        
        
        quarto_command = f"quarto render {results_qmd_path}"
        hb.log(f"Running quarto command: {quarto_command}")        
        os.system(quarto_command)
        
    