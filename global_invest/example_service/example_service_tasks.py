import os

from global_invest.example_service import example_service_functions

def example_parent_task(p):
    """Example parent task with no work of its own; example_task is registered as its child."""
    
def example_task(p):
    """
    Example task that calls a function.
    """
    result = example_service_functions.example_function(1, 2)
    
    # Save to a file
    p.example_task_output_path = os.path.join(p.cur_dir, 'example_output.qmd')
    with open(p.example_task_output_path, 'w') as f:
        f.write(f'Result of example function: {result}\n')
        
    
    return True