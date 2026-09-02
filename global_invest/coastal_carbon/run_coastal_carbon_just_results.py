"""Build a ProjectFlow object for rendering existing coastal-carbon results."""

import os

import hazelbean as hb
import pandas as pd

from global_invest.coastal_carbon import coastal_carbon_tasks


def build_results_project(project_dir=None):
    """Return configured EEZ and ISO-250 result paths without running tasks."""
    project_dir = project_dir or os.environ.get('COASTAL_CARBON_PROJECT_DIR')
    if not project_dir:
        project_dir = os.path.join(
            os.path.expanduser('~'), 'Files', 'global_invest', 'projects',
            'gep_coastal_carbon',
        )

    p = hb.ProjectFlow()
    p.project_name = os.path.basename(os.path.normpath(project_dir))
    p.project_dir = project_dir
    p.base_data_dir = (
        "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/"
        "Shared drives/NatCapTEEMs/Files/base_data/submissions"
    )
    p.set_project_dir(p.project_dir)

    p.df_eez_csv_path = p.get_path(
        'cartographic', 'ee', 'eemarine_r566_correspondence.csv'
    )
    p.gdf_eez_vector_path = p.get_path(
        'cartographic', 'ee', 'eemarine_r566_correspondence.gpkg'
    )
    p.mangrove_vector_path = os.path.join(
        p.base_data_dir, 'coastal_carbon', 'gmw_v3_2019_vec',
        'gmw_v3_2019_vec.shp'
    )
    p.salt_marsh_vector_path = os.path.join(
        p.base_data_dir, 'coastal_carbon', 'global_salt_marsh2019.gpkg'
    )
    p.seagrass_raster_dir = os.path.join(
        p.base_data_dir, 'coastal_carbon', 'GlobalSeagrass2019_2020'
    )
    p.seagrass_area_dir = os.path.join(
        p.intermediate_dir,
        'task_calculate_seagrass_area_within_eez',
        'seagrass_raster_eez_intersection',
    )
    p.carbon_price = 'rental scc r2%'

    result_dir = os.path.join(p.intermediate_dir, 'gep_calculation')
    p.results = {
        'coastal_carbon': {
            'gep_by_eez_base_year': os.path.join(
                result_dir, 'gep_by_eez2019.csv'
            ),
            'gep_by_country_base_year': os.path.join(
                result_dir, 'gep_by_iso2502019.csv'
            ),
        }
    }
    return p


def rebuild_iso250_results_from_existing_eez(project_dir=None):
    """Rebuild ISO-250 country results from cached EEZ GEP only.

    This does not register or execute ecosystem-area, carbon-stock, or Quarto
    tasks. It reads ``gep_by_eez2019.csv`` and rewrites only the ISO-250 CSV
    and GeoPackage in ``intermediate/gep_calculation``.
    """
    p = build_results_project(project_dir)
    result_paths = p.results['coastal_carbon']
    eez_csv = result_paths['gep_by_eez_base_year']
    iso_csv = result_paths['gep_by_country_base_year']
    required_columns = {
        'eemarine_r566_id', 'eemarine_r566_label', 'value',
    }
    if not os.path.exists(eez_csv):
        raise FileNotFoundError(f'EEZ result not found: {eez_csv}')

    eez_results = pd.read_csv(eez_csv)
    missing_columns = required_columns.difference(eez_results.columns)
    if missing_columns:
        raise ValueError(
            f'EEZ result missing required columns: {sorted(missing_columns)}'
        )
    if not eez_results['eemarine_r566_label'].astype(str).str.endswith(
            '_EEZ'
    ).all():
        raise ValueError('EEZ result contains a non-EEZ row')
    eez_results = coastal_carbon_tasks.coastal_carbon_functions \
        .normalize_integer_id_column(eez_results, 'eemarine_r566_id')

    # The aggregation helper writes its CSV/GPKG pair in p.cur_dir.
    p.cur_dir = os.path.dirname(iso_csv)
    rebuilt_iso_csv, rebuilt_iso_gpkg = (
        coastal_carbon_tasks.build_iso250_results_from_eez(p, eez_results)
    )
    iso_results = pd.read_csv(rebuilt_iso_csv)
    difference = iso_results['value'].sum() - eez_results['value'].sum()
    tolerance = max(1e-6, abs(eez_results['value'].sum()) * 1e-12)
    if abs(difference) > tolerance:
        raise ValueError(
            'ISO-250 total does not reconcile with EEZ total: '
            f'{difference:,.12f}'
        )
    return {
        'eez_csv': eez_csv,
        'iso_csv': rebuilt_iso_csv,
        'iso_gpkg': rebuilt_iso_gpkg,
        'eez_total': eez_results['value'].sum(),
        'iso_total': iso_results['value'].sum(),
        'difference': difference,
    }


if __name__ == '__main__':
    result = rebuild_iso250_results_from_existing_eez()
    print(
        'Rebuilt ISO-250 results from cached EEZ GEP: '
        f"EEZ={result['eez_total']:,.2f}; "
        f"ISO-250={result['iso_total']:,.2f}; "
        f"difference={result['difference']:,.12f}"
    )
