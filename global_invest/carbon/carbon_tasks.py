# =============================================================================
# imports
# =============================================================================

import os
import numpy as np

from global_invest.carbon import carbon_functions


# =============================================================================
# define tasks
# =============================================================================

def task_convert_carbon_density_maps_dtype(p):
    """
    Task to convert all uint TIFF carbon density rasters in a folder
    to float32 with scaled values, saving them with '_float' suffix.

    Parameters
    ----------
    p : ProjectFlow-like object
        Must contain p.base_data_dir (input folder path).
    """
    input_folder = p.base_data_dir
    output_folder = p.project_dir

    raw_carbon_density_maps = [
        f for f in os.listdir(input_folder)
        if f.endswith("biomass_carbon_2010.tif") and not f.startswith("._")
    ]

    for file in raw_carbon_density_maps:
        input_path = os.path.join(input_folder, file)

        name_root, ext = os.path.splitext(file)
        output_name = f"{name_root}_float{ext}"
        output_path = os.path.join(output_folder, output_name)

        carbon_functions.convert_uint_to_float_raster(
            input_path=input_path,
            output_path=output_path,
            scale_factor=0.1,
            compress="lzw"
        )

    print("Finished converting all carbon density rasters to float.")



def task_combine_two_carbon_density_maps(p):
    """
    Task to combine aboveground and belowground biomass carbon maps using functions.
    """

    # Input and output paths
    p.agb_path = p.get_path(os.path.join(p.project_dir,"aboveground_biomass_carbon_2010_float.tif"))
    p.bgb_path = p.get_path(os.path.join(p.project_dir,"belowground_biomass_carbon_2010_float.tif"))
    p.total_carbon_output_path = os.path.join(p.project_dir, "total_biomass_carbon_2010_float.tif")

    # Run the function
    result = carbon_functions.combine_two_float_rasters(
        raster1_path=p.agb_path,
        raster2_path=p.bgb_path,
        out_path=p.total_carbon_output_path,
        operation=lambda a, b: a + b,  # Default operation: addition
        fill_value=np.nan,
        compress="lzw")

    return True


def task_reproject_total_carbon_density(p):
    """
    Task to reproject the total carbon density raster to the project's coordinate reference system (CRS).
    """

    # Input and output paths
    p.total_carbon_density_path = p.get_path(os.path.join(p.project_dir, "total_biomass_carbon_2010_float.tif"))
    p.reprojected_total_carbon_density_path = os.path.join(p.project_dir, "total_biomass_carbon_2010_float_reprojected.tif")

    # Run the function
    result = carbon_functions.reproject_raster(
        input_path=p.total_carbon_density_path,
        reference_path=p.base_year_lulc_path,
        output_path=p.reprojected_total_carbon_density_path,
        compress="lzw",
        chunks={"x": 1024, "y": 1024},
        overwrite=False
        )

    return True


def task_reproject_carbon_zones(p):
    """
    Task to reproject the total carbon density raster to the project's coordinate reference system (CRS).
    """

    # Input and output paths
    p.reprojected_carbon_zones_path = os.path.join(p.project_dir, "carbon_zones_rasterized_reprojected.tif")

    # Run the function
    result = carbon_functions.reproject_raster(
        input_path=p.carbon_zones_path,
        reference_path=p.base_year_lulc_path,
        output_path=p.reprojected_carbon_zones_path,
        compress="lzw",
        chunks={"x": 1024, "y": 1024},
        overwrite=False
        )
    return True


def task_compute_carbon_density_table(p):

    p.reprojected_total_carbon_density_path = p.get_path(os.path.join(p.project_dir, "total_biomass_carbon_2010_float_reprojected.tif"))
    p.carbon_density_lookup_table_path = os.path.join(p.project_dir, "carbon_density_lookup_table.csv")

    result = carbon_functions.stack_layers_to_csv(
        group_layer1_path=p.base_year_lulc_path,
        group_layer2_path=p.carbon_zones_path,
        value_layer_path=p.reprojected_total_carbon_density_path,
        output_path=p.carbon_density_lookup_table_path,
        group1_name="lulc_id",
        group2_name="carbon_zone_id",
        value_name="carbon_density",
        num_slices=100)
    return True


def task_generate_carbon_density_raster_base_year(p):
    p.reprojected_total_carbon_density_path = p.get_path(os.path.join(p.project_dir, "total_biomass_carbon_2010_float_reprojected.tif"))
    p.carbon_density_lookup_table_path = p.get_path(os.path.join(p.project_dir, "carbon_density_lookup_table.csv"))
    p.carbon_density_raster_output_path = os.path.join(p.project_dir, "carbon_density_2019.tif")
    result = carbon_functions.generate_carbon_density_raster(
        lulc_path=p.base_year_lulc_path,
        cz_path=p.carbon_zones_path,
        carbon_density_lookup_table_path=p.carbon_density_lookup_table_path,
        out_path=p.carbon_density_raster_output_path)
    return True


def task_summarize_carbon_density_by_region(p):
    p.carbon_density_raster_output_path = p.get_path(os.path.join(p.project_dir, "carbon_density_2019.tif"))
    p.carbon_density_by_region_path = os.path.join(p.project_dir, "carbon_density_by_region_2019.csv")
    result = carbon_functions.summarize_raster_by_region(
        value_raster_path=p.carbon_density_raster_output_path,
        region_boundary_path=p.region_boundary_path,
        out_path=p.carbon_density_by_region_path)
    return result


def task_compute_carbon_shock(p):
    """Turn per-scenario 300 m LULC into a carbon ES-productivity shock -- region-agnostic.

    For the baseline scenario at base_year and end_year, and each scenario at end_year, build
    a carbon-density raster (generate_carbon_density_raster) and take its mean tC/ha per polygon
    of p.region_boundary_path (summarize_raster_by_region -- untouched, generic; the polygon
    geometry is identical across scenarios, so the mean is sufficient and area cancels). The
    shock is (mean_scenario - mean_baseline) / mean_base_year * 100 per zone, linearly
    interpolated base_year -> end_year. Emits an aoall table keyed by the boundary's ENDW/REG
    columns and p.carbon_shock_acts. The only region-specific knowledge is the column names,
    supplied by the caller via p; nothing GTAP-specific is hardcoded.

    Caller sets on p: scenario_lulc_paths {scenario: {year: path}}, carbon_shock_scenarios,
    region_boundary_path, carbon_zones_path, carbon_density_lookup_table_path,
    carbon_shock_output_path. Optional: carbon_shock_{base_scenario, base_year, end_year,
    endw_col, reg_col, value_col, acts}.
    """
    if not p.run_this:
        return
    import pandas as pd
    import geopandas as gpd

    base_scn  = getattr(p, 'carbon_shock_base_scenario', 'baseline_ignore_dependencies')
    base_year = getattr(p, 'carbon_shock_base_year', 2020)
    end_year  = getattr(p, 'carbon_shock_end_year', 2050)
    endw_col  = getattr(p, 'carbon_shock_endw_col', 'aez18_id')            # GTAP r50xAEZ18 defaults
    reg_col   = getattr(p, 'carbon_shock_reg_col', 'gtapv7_r50_label')
    val_col   = getattr(p, 'carbon_shock_value_col', 'mean')
    acts      = getattr(p, 'carbon_shock_acts', 'FRS')
    endw_fmt  = getattr(p, 'carbon_shock_endw_format', 'AEZ%d')            # int id -> 'AEZ1'..'AEZ18'

    # Standard GTAP-carbon inputs default here (Spawn density, r50xAEZ18 boundary, observed 2020 base
    # map); the caller overrides only when different, so project wiring stays a couple of lines.
    if not getattr(p, 'region_boundary_path', None):
        p.region_boundary_path = p.get_path('gtap_invest/region_boundaries/ee_r50_aez18_correspondence.gpkg')
    if not getattr(p, 'carbon_zones_path', None):
        p.carbon_zones_path = p.get_path('carbon_storage', 'carbon_zones_rasterized.tif')
    if not getattr(p, 'carbon_density_lookup_table_path', None):
        p.carbon_density_lookup_table_path = p.get_path('carbon_storage', 'carbon_density_lookup_seals7_spawn.csv')
    base_lulc = getattr(p, 'carbon_shock_base_lulc', None) or p.get_path('carbon_storage', 'lulc_seals7_2020_from_esa.tif')

    # Resolve the LULC map per scenario by globbing carbon_lulc_path_template ({scenario}/{year}
    # placeholders) when the caller didn't pre-build scenario_lulc_paths. Globbing lives here so a
    # project passes only a template string, not a path-building task; base_year uses base_lulc.
    scenarios = list(getattr(p, 'carbon_shock_scenarios', []))
    if not getattr(p, 'scenario_lulc_paths', None):
        import glob
        tmpl = p.carbon_lulc_path_template
        paths = {}
        for scen in [base_scn] + scenarios:
            hits = glob.glob(tmpl.format(scenario=scen, year=end_year))
            if hits:
                paths.setdefault(scen, {})[end_year] = hits[0]
        paths.setdefault(base_scn, {})[base_year] = base_lulc
        p.scenario_lulc_paths = paths
    if not scenarios:
        scenarios = [s for s in p.scenario_lulc_paths if s != base_scn]

    def zone_mean(scenario, year):
        dens = os.path.join(p.cur_dir, 'carbon_density_%s_%d.tif' % (scenario, year))
        if not os.path.exists(dens):
            carbon_functions.generate_carbon_density_raster(
                lulc_path=p.scenario_lulc_paths[scenario][year],
                cz_path=p.carbon_zones_path,
                carbon_density_lookup_table_path=p.carbon_density_lookup_table_path,
                out_path=dens)
        summ = os.path.join(p.cur_dir, 'carbon_by_zone_%s_%d.csv' % (scenario, year))
        if not os.path.exists(summ):
            carbon_functions.summarize_raster_by_region(dens, p.region_boundary_path, summ)
        return pd.read_csv(summ).set_index('region_id')[val_col]

    # summarize_raster_by_region keys each row by row.get("id", idx) and DROPS empty zones,
    # so map + align on region_id, never on row position.
    regions = gpd.read_file(p.region_boundary_path)
    def _fmt(v):
        return (endw_fmt % int(v)) if endw_fmt is not None else v
    labels = {r.get('id', i): (_fmt(r[endw_col]), r[reg_col]) for i, r in regions.iterrows()}

    base_by = zone_mean(base_scn, base_year)
    base_ey = zone_mean(base_scn, end_year)
    n_years = end_year - base_year

    rows = []
    for scenario in scenarios:
        scn = zone_mean(scenario, end_year)
        zids = base_by.index.intersection(base_ey.index).intersection(scn.index)
        shock_ey = (scn.loc[zids] - base_ey.loc[zids]) / base_by.loc[zids].replace(0, np.nan) * 100.0
        for zid, s in shock_ey.items():
            if not np.isfinite(s) or zid not in labels:
                continue
            endw, reg = labels[zid]
            for year in range(base_year, end_year + 1):
                rows.append({'ENDW': endw, 'ACTS': acts, 'REG': reg, 'scenario': scenario,
                             'year': year, 'shock_pct': s * (year - base_year) / n_years})

    out = pd.DataFrame(rows)
    out.to_csv(p.carbon_shock_output_path, index=False)
    print('  carbon shock: %d rows, %d scenarios -> %s'
          % (len(out), out['scenario'].nunique() if rows else 0, p.carbon_shock_output_path))
    return True
