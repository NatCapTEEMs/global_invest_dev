import os
import sys
import hazelbean as hb
from global_invest.commercial_agriculture import commercial_agriculture_functions
from global_invest.commercial_agriculture import commercial_agriculture_defaults

def build_gep_task_tree(p):
    """
    Build the default task tree for commercial agriculture.
    """
    p.commercial_agriculture_task = p.add_task(commercial_agriculture)
    # p.commercial_agriculture_preprocess_task = p.add_task(gep_preprocess, parent=p.commercial_agriculture_task)  
    p.commercial_agriculture_gep_calculation_task = p.add_task(gep_calculation, parent=p.commercial_agriculture_task)  
    p.commercial_agriculture_gep_result_task = p.add_task(gep_result, parent=p.commercial_agriculture_task)  
    
    return p

def commercial_agriculture(p):
    """
    Parent task for commercial agriculture.
    """
    p.fao_input_path = p.get_path(os.path.join(p.base_data_dir, 'fao', 'Value_of_Production_E_All_Data.csv'))


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
    
    crop_result = commercial_agriculture_functions.calculate_gep(p.base_data_dir, commercial_agriculture_defaults.DEFAULT_CROP_ITEMS)
    livestock_result = commercial_agriculture_functions.calculate_gep(p.base_data_dir, commercial_agriculture_defaults.DEFAULT_LIVESTOCK_ITEMS)
    
    crop_gep_base_year = crop_result['gep_base_year']
    crop_gep_by_year = crop_result['gep_by_year']
    crop_gep_by_year_country = crop_result['gep_by_year_country']
    crop_gep_by_country_year_crop = crop_result['gep_by_country_year_crop']
    
    livestock_gep_base_year = livestock_result['gep_base_year']
    livestock_gep_by_year = livestock_result['gep_by_year']
    livestock_gep_by_year_country = livestock_result['gep_by_year_country']
    livestock_gep_by_country_year_crop = livestock_result['gep_by_country_year_crop']
    
    commercial_agriculture_gep_base_year = crop_gep_base_year + livestock_gep_base_year 
    
    p.crop_gep_base_year = crop_gep_base_year
    p.crop_gep_by_year_path = os.path.join(p.cur_dir, "crop_gep_by_year.csv")
    p.crop_gep_by_year_country_path = os.path.join(p.cur_dir, "crop_gep_by_year_country.csv")
    p.crop_gep_by_country_year_crop_path = os.path.join(p.cur_dir, "crop_gep_by_country_year_crop.csv")
    
    p.livestock_gep_base_year = livestock_gep_base_year
    p.livestock_gep_by_year_path = os.path.join(p.cur_dir, "livestock_gep_by_year.csv")
    p.livestock_gep_by_year_country_path = os.path.join(p.cur_dir, "livestock_gep_by_year_country.csv")
    p.livestock_gep_by_country_year_crop_path = os.path.join(p.cur_dir, "livestock_gep_by_country_year_crop.csv")
    
    p.commercial_agriculture_gep_base_year = commercial_agriculture_gep_base_year
    p.commercial_agriculture_gep_by_year_path = os.path.join(p.cur_dir, "commercial_agriculture_gep_by_year.csv")
    p.commercial_agriculture_gep_by_year_country_path = os.path.join(p.cur_dir, "commercial_agriculture_gep_by_year_country.csv")
    p.commercial_agriculture_gep_by_country_year_crop_path = os.path.join(p.cur_dir, "commercial_agriculture_gep_by_country_year_crop.csv")
    
    crop_gep_by_year.to_csv(p.crop_gep_by_year_path, index=False)
    crop_gep_by_year_country.to_csv(p.crop_gep_by_year_country_path, index=False)
    crop_gep_by_country_year_crop.to_csv(p.crop_gep_by_country_year_crop_path, index=False)
    
    livestock_gep_by_year.to_csv(p.livestock_gep_by_year_path, index=False)
    livestock_gep_by_year_country.to_csv(p.livestock_gep_by_year_country_path, index=False)
    livestock_gep_by_country_year_crop.to_csv(p.livestock_gep_by_country_year_crop_path, index=False)
    
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