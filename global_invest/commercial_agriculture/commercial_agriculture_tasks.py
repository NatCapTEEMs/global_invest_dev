import os
import hazelbean as hb
from global_invest.commercial_agriculture import commercial_agriculture_functions

def build_gep_task_tree(p):
    """
    Build the default task tree for commercial agriculture.
    """
    p.commercial_agriculture_task = p.add_task(commercial_agriculture)
    p.commercial_agriculture_preprocess_task = p.add_task(gep_preprocess, parent=p.commercial_agriculture_task)  
    p.commercial_agriculture_gep_calculation_task = p.add_task(gep_calculation, parent=p.commercial_agriculture_task)  
    
    return p

def commercial_agriculture(p):
    """
    Parent task for commercial agriculture.
    """
    p.fao_input_path = p.get_path(os.path.join(p.base_data_dir, 'fao', 'Value_of_Production_E_All_Data.csv'))

def gep_preprocess(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.commercial_agriculture_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    p.commercial_agriculture_input_path = os.path.join(p.cur_dir, "commercial_agriculture_value_by_crop.csv")
    commercial_agriculture_functions.preprocess_fao(p.fao_input_path, p.commercial_agriculture_input_path)

def gep_calculation(p):
    """
    Calculate GEP for commercial agriculture.
    
    Outputs saved at the canonical project attribute p.commercial_agriculture_output_path.
    Optional outputs given as kwargs to the function, such as crop_values_output_path.
    """
    p.commercial_agriculture_output_path = os.path.join(p.cur_dir, "commercial_agriculture_gep_by_country_and_year.csv")
    p.commercial_agriculture_gep_by_crop_path = os.path.join(p.cur_dir, "commercial_agriculture_gep_by_country_year_and_crop.csv")

    commercial_agriculture_functions.calculate_gep(p.commercial_agriculture_input_path, p.commercial_agriculture_output_path, crop_values_output_path=p.commercial_agriculture_gep_by_crop_path)