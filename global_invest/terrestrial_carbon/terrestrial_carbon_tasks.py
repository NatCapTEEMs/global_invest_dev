import os
import sys
import pandas as pd
import hazelbean as hb
import subprocess
import csv
import numpy as np


from global_invest import utilities
from global_invest.terrestrial_carbon import terrestrial_carbon_functions


def terrestrial_carbon(p):
    """
    Parent task for terrestrial carbon.
    """
    return True


def task_convert_carbon_density_maps_dtype(p):
    """
    Task to convert all uint TIFF carbon density rasters in a folder
    to float32 with scaled values, saving them with '_float' suffix.

    Parameters
    ----------
    p : ProjectFlow
        Must contain p.base_data_dir (input folder path).
    """
    input_folder = p.get_path('terrestrial_carbon', 'spawn_2020')
    output_folder = p.cur_dir

    raw_carbon_density_maps = [
        f for f in os.listdir(input_folder)
        if f.endswith("biomass_carbon_2010.tif") and not f.startswith("._")
    ]

    for file in raw_carbon_density_maps:
        input_path = os.path.join(input_folder, file)

        name_root, ext = os.path.splitext(file)
        output_name = f"{name_root}_float{ext}"
        output_path = os.path.join(output_folder, output_name)

        terrestrial_carbon_functions.convert_uint_to_float_raster(
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
    p.agb_path = p.get_path(os.path.join(p.task_convert_carbon_density_maps_dtype_dir,"aboveground_biomass_carbon_2010_float.tif"))
    p.bgb_path = p.get_path(os.path.join(p.task_convert_carbon_density_maps_dtype_dir,"belowground_biomass_carbon_2010_float.tif"))
    p.total_carbon_output_path = os.path.join(p.cur_dir, "total_biomass_carbon_2010_float.tif")

    # Run the function
    result = terrestrial_carbon_functions.combine_two_float_rasters(
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

    # Input: the total biomass-carbon density raster. This is a one-off base-data product (raw Spawn
    # aboveground+belowground already converted and combined), consumed from base_data rather than
    # rebuilt per run. Overridable via p.total_carbon_density_path.
    p.total_carbon_density_path = getattr(p, 'total_carbon_density_path', None) or p.get_path('carbon_storage', 'spawn_total_biomass_carbon_2010.tif')
    p.reprojected_total_carbon_density_path = os.path.join(p.cur_dir, "total_biomass_carbon_2010_float_reprojected.tif")
    if not p.run_this:
        return True

    # Run the function
    result = terrestrial_carbon_functions.reproject_raster(
        input_path=p.total_carbon_density_path,
        reference_path=p.base_year_lulc_path,
        output_path=p.reprojected_total_carbon_density_path,
        compress="lzw",
        chunks={"x": 1024, "y": 1024},
        overwrite=False
        )

    return True


def task_compute_carbon_density_table(p):

    p.carbon_density_lookup_table_path = os.path.join(p.cur_dir, "carbon_density_lookup_table.csv")
    if not p.run_this:
        return True

    result = terrestrial_carbon_functions.stack_layers_to_csv(
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

    p.carbon_density_raster_base_year_path = os.path.join(p.cur_dir, "projected_carbon_density_2019.tif")
    if not p.run_this:
        return True
    result = terrestrial_carbon_functions.generate_carbon_density_raster(
        lulc_path=p.base_year_lulc_path,
        cz_path=p.carbon_zones_path,
        carbon_density_lookup_table_path=p.carbon_density_lookup_table_path,
        out_path=p.carbon_density_raster_base_year_path)
    return True


def task_generate_carbon_density_raster_per_cell_base_year(p):
    p.ha_per_cell_10sec_ref_path = p.get_path('pyramids', 'ha_per_cell_10sec.tif')
    p.projected_carbon_density_2019_per_cell_path = os.path.join(p.cur_dir, 'projected_carbon_density_2019_per_cell.tif')
    if not p.run_this:
        return True
    hb.multiply(p.carbon_density_raster_base_year_path, p.ha_per_cell_10sec_ref_path, p.projected_carbon_density_2019_per_cell_path)
    return True


def task_summarize_carbon_by_region(p):
    p.carbon_by_region_base_year_path = os.path.join(p.cur_dir, "gep_by_country_base_year.csv")
    if not p.run_this:
        return True
    result = terrestrial_carbon_functions.summarize_raster_by_region(
        value_raster_path=p.projected_carbon_density_2019_per_cell_path,
        region_boundary_path=p.gdf_countries_vector_path,
        out_path=p.carbon_by_region_base_year_path)
    return result


def gep_preprocess(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.terrestrial_carbon_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    For terrestrial carbon the preprocessing is the carbon-density build (Spawn + carbon zones), which is shared by
    both the GEP valuation and the ES shock.
    """
    pass # NYI


def gep_calculation(p):
    """ GEP calculation task for terrestrial carbon."""
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.
    service_results = {}
    p.results['terrestrial_carbon'] = service_results
    p.results['terrestrial_carbon']['gep_by_country_base_year'] = os.path.join(p.cur_dir, "gep_by_country_base_year.csv")
    # Only register results this task actually writes. Per-year results (gep_by_country_year, gep_by_year)
    # belong to a multi-year run and are registered there, not in this base-year valuation.

    # Check if all results exist
    if hb.path_all_exist(list(service_results.values())):
        hb.log("All results already exist. Skipping GEP calculation for terrestrial carbon.")
    else:
        hb.log("Starting GEP calculation for terrestrial carbon.")

        # Optimization here,
        # p.gdf_countries = hb.read_vector(p.gdf_countries)
        # p.gdf_countries = hb.read_vector(p.gdf_countries_simplified)

        # 1. Per-region (r264) carbon quantity -> aggregate to one row per COUNTRY (r250).
        df_carbon_q264 = pd.read_csv(p.carbon_by_region_base_year_path)
        df_carbon_q250 = (
            df_carbon_q264
            .groupby(['iso3_r250_id', 'year'], as_index=False)['total']
            .sum()
            .rename(columns={'total': 'terrestrial_carbon_quantity'})
        )

        # 2. Country-level (r250) GEP = quantity * price. ONE row per country: this is the source of truth
        #    for the CSV, the national total and every aggregation. The r264-expanded table (used for the
        #    map below) repeats a country's GEP across its sub-regions -- China spans 6 r264 rows, India 6,
        #    France/Turkey/UK/Pakistan 2 -- so summing it double-counts split countries (~23% too high).
        #    Only the map ever touches the r264 rows, and it never sums them.
        df_carbon_p = pd.read_excel(p.carbon_prices_path)[[p.carbon_price, 'year']]
        df_gep_250 = df_carbon_q250.merge(df_carbon_p, how='left', on='year')
        df_gep_250['terrestrial_carbon_gep'] = df_gep_250['terrestrial_carbon_quantity'] * df_gep_250[p.carbon_price]

        # Attach per-country attributes (one row per iso3 from the correspondence) and write the per-country CSV.
        attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                     'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
        keep_cols = attr_cols + ['year', 'terrestrial_carbon_quantity', p.carbon_price, 'terrestrial_carbon_gep']
        df_gep_by_country_base_year = df_gep_250.merge(
            df_carbon_q264[attr_cols].drop_duplicates('iso3_r250_id'), how='left', on='iso3_r250_id')[keep_cols]
        hb.df_write(df_gep_by_country_base_year, p.results['terrestrial_carbon']['gep_by_country_base_year'])

        # 3. Map only: attach each country's GEP to the r264 boundary geometry for the choropleth (each
        #    sub-region shows its country's value). These rows are for plotting and must never be summed.
        df_regions = df_carbon_q264.merge(df_gep_250[['iso3_r250_id', 'terrestrial_carbon_gep']], how='left', on='iso3_r250_id')
        gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_simplified, df_regions, how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
        gdf_gep_by_country_base_year.to_file(p.results['terrestrial_carbon']['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

        # 4. National total = sum over the one-row-per-country table.
        value_gep_base_year = df_gep_by_country_base_year['terrestrial_carbon_gep'].sum()
        hb.log(f"Total GEP value for base year 2019: {value_gep_base_year}")

        return value_gep_base_year

def gep_result(p):
    """Display the results of the GEP calculation."""
    
    # Set the quarto path to wherever the current script is running. This means that the environment used needs to have quarto, which may not be true on e.g. codespaces.
    os.environ['QUARTO_PYTHON'] = sys.executable
    
    # Get the  list of current services run
    services_run = list(p.results.keys())
    
    # Additional groupbys = []
    
    # Imply from the service name the file_path for the results_qmd
    module_root = hb.get_projectflow_module_root()
    
    for service_label in services_run:
        results_qmd_path = os.path.join(module_root, service_label, f'{service_label}_results.qmd')    
        results_qmd_project_path = os.path.join(p.cur_dir, f'{service_label}_results.qmd')
        hb.create_directories(results_qmd_project_path)  # Ensure the directory exists   
        
        # Copy it to the project dir for cmd line processing (but will be removed again later because it makes confusion when people try to edit it and then rerun the script which won't of course update the results.)
        hb.path_copy(results_qmd_path, results_qmd_project_path)
        
        
        quarto_command = f"quarto render {results_qmd_project_path}"
        hb.log(f"Running quarto command: {quarto_command}")     

        """Run quarto with debug information"""
        # Set environment for more verbose output
        env = os.environ.copy()
        env['QUARTO_LOG_LEVEL'] = 'DEBUG'
        
        cmd = ['quarto', 'render', results_qmd_project_path, '--verbose']
        
        # print(f"Running command: {' '.join(results_qmd_project_path)}")
        print(f"Working directory: {os.getcwd()}")
        print(f"File exists: {os.path.exists(results_qmd_project_path)}")
        
        
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            bufsize=1,  # Line buffering
            universal_newlines=True
        )
        
        # Read line by line as they come
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
                sys.stdout.flush()  # Force immediate display
        # remove results_qmd_project_path
        hb.path_remove(results_qmd_project_path)
        
def gep_load_results(p):
    """Load the GEP results computed by a PRIOR calculation run, so the report can render without
    recomputing. Fails loudly if they are not present -- run the calculation (run_terrestrial_carbon.py)
    or promote the results into base_data first. This is the 'results-only' entry point.
    """
    result_path = os.path.join(p.intermediate_dir, 'gep_calculation', 'gep_by_country_base_year.csv')
    if not hb.path_exists(result_path):
        raise FileNotFoundError(
            f"terrestrial_carbon GEP results not found at {result_path}. "
            f"Run the calculation first (run_terrestrial_carbon.py), then re-run results.")
    p.results.setdefault('terrestrial_carbon', {})
    p.results['terrestrial_carbon']['gep_by_country_base_year'] = result_path
        
def gep_results_distribution(p):
    """Distribute the results of the GEP calculation."""
    # This task is intended to copy the results to the output directory.
    hb.log("Distributing GEP results...")
    
    for key, value in p.results['terrestrial_carbon'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")
    
    hb.log("GEP results distribution complete.")


# =============================================================================
# ES-shock tasks. These feed the GTAP shock; the GEP valuation above is a separate
# consumer of the same carbon-density front-end. Neither depends on the other.
# =============================================================================

def task_compute_terrestrial_carbon_shock(p):
    """Turn per-scenario 300 m LULC into a carbon ES-productivity shock -- region-agnostic.

    At each SEALS anchor year in es_shock_years (5-year MAgPIE steps), build a carbon-density
    raster (generate_carbon_density_raster) for the baseline scenario and each scenario, and take
    its mean tC/ha per polygon of p.region_boundary_path (summarize_raster_by_region -- untouched,
    generic; the polygon geometry is identical across scenarios, so the mean is sufficient and area
    cancels). The shock at year Y is (mean_scenario_Y - mean_baseline_Y) / mean_baseline_Y * 100 per
    zone (contemporaneous /base_Y = the GTAP shock, column `shock_pct`), piecewise-linearly interpolated
    across the anchor years (0 at base_year). ALSO emits a fixed-base measure (column `shock_pct_fixedbase`)
    = same numerator / baseline density at base_year -- the "Value of Nature" % of base-year value, for
    comparability with pollination (denominator decision). Emits an aoall table keyed by the boundary's ENDW/REG columns and p.terrestrial_carbon_shock_acts. The only region-specific knowledge is the column names,
    supplied by the caller via p; nothing GTAP-specific is hardcoded.

    Caller sets on p: es_shock_years (SEALS anchor years, from seals_years),
    scenario_lulc_paths {scenario: {year: path}} or es_lulc_path_template, es_shock_scenarios,
    region_boundary_path, terrestrial_carbon_zones_path, terrestrial_carbon_density_lookup_table_path,
    terrestrial_carbon_shock_output_path. Optional: terrestrial_carbon_shock_{base_scenario, base_year,
    endw_col, reg_col, value_col, acts}.
    """
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'terrestrial_carbon_shock_output_path', None):
        p.terrestrial_carbon_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir, 'terrestrial_carbon_interpolated.csv')
    if not p.run_this:
        return
    import pandas as pd
    import geopandas as gpd

    base_scn     = getattr(p, 'es_shock_base_scenario', 'baseline_ignore_dependencies')
    base_year    = int(p.es_shock_base_year)          # interp 0-anchor, set by the caller from config
    anchor_years = sorted(y for y in map(int, p.es_shock_years) if y > base_year)  # SEALS anchors (seals_years)
    end_year     = anchor_years[-1]
    endw_col  = getattr(p, 'terrestrial_carbon_shock_endw_col', 'aez18_id')            # GTAP r50xAEZ18 defaults
    reg_col   = getattr(p, 'terrestrial_carbon_shock_reg_col', 'gtapv7_r50_label')
    val_col   = getattr(p, 'terrestrial_carbon_shock_value_col', 'mean')
    acts      = getattr(p, 'terrestrial_carbon_shock_acts', 'FRS')
    endw_fmt  = getattr(p, 'terrestrial_carbon_shock_endw_format', 'AEZ%d')            # int id -> 'AEZ1'..'AEZ18'

    # Standard GTAP-carbon inputs default here (Spawn density, r50xAEZ18 boundary, observed 2020 base
    # map); the caller overrides only when different, so project wiring stays a couple of lines.
    if not getattr(p, 'region_boundary_path', None):
        p.region_boundary_path = p.get_path('gtap_invest/region_boundaries/ee_r50_aez18_correspondence.gpkg')
    if not getattr(p, 'terrestrial_carbon_zones_path', None):
        p.terrestrial_carbon_zones_path = p.get_path('carbon_storage', 'carbon_zones_rasterized.tif')
    if not getattr(p, 'terrestrial_carbon_density_lookup_table_path', None):
        p.terrestrial_carbon_density_lookup_table_path = p.get_path('carbon_storage', 'carbon_density_lookup_seals7_spawn.csv')
    # Resolve the LULC map per scenario by globbing es_lulc_path_template ({scenario}/{year}
    # placeholders) when the caller didn't pre-build scenario_lulc_paths. Globbing lives here so a
    # project passes only a template string, not a path-building task.
    scenarios = list(getattr(p, 'es_shock_scenarios', []))
    if not getattr(p, 'scenario_lulc_paths', None):
        import glob
        tmpl = p.es_lulc_path_template
        p.scenario_lulc_paths = {
            scen: {y: glob.glob(tmpl.format(scenario=scen, year=y))[0]
                   for y in anchor_years if glob.glob(tmpl.format(scenario=scen, year=y))}
            for scen in [base_scn] + scenarios}
    if not scenarios:
        scenarios = [s for s in p.scenario_lulc_paths if s != base_scn]

    # generate_carbon_density_raster asserts LULC and carbon-zones share a grid. The zones raster is
    # global 300 m; a SEALS LULC map may be a sub-window (single-country or short-horizon test AOI).
    # When they differ, align the zones to the end-year LULC extent once -- same resolution and an
    # aligned grid, so nearest is a lossless clip -- so the task works at any extent, not only global.
    import hazelbean as hb
    from osgeo import gdal
    def _yx(path):
        ds = gdal.Open(path)
        return (ds.RasterYSize, ds.RasterXSize)
    _ref_lulc = p.scenario_lulc_paths[base_scn][end_year]
    if _yx(p.terrestrial_carbon_zones_path) != _yx(_ref_lulc):
        _aligned_cz = os.path.join(p.cur_dir, 'carbon_zones_aligned.tif')
        if not os.path.exists(_aligned_cz):
            hb.resample_to_match(p.terrestrial_carbon_zones_path, _ref_lulc, _aligned_cz, resample_method='near')
        p.terrestrial_carbon_zones_path = _aligned_cz

    def zone_mean(scenario, year):
        dens = os.path.join(p.cur_dir, 'carbon_density_%s_%d.tif' % (scenario, year))
        if not os.path.exists(dens):
            terrestrial_carbon_functions.generate_carbon_density_raster(
                lulc_path=p.scenario_lulc_paths[scenario][year],
                cz_path=p.terrestrial_carbon_zones_path,
                # develop's parameter name; the p attribute keeps ours (see the merge of this module)
                carbon_density_lookup_table_path=p.terrestrial_carbon_density_lookup_table_path,
                out_path=dens)
        summ = os.path.join(p.cur_dir, 'carbon_by_zone_%s_%d.csv' % (scenario, year))
        if not os.path.exists(summ):
            terrestrial_carbon_functions.summarize_raster_by_region(dens, p.region_boundary_path, summ)
        return pd.read_csv(summ).set_index('region_id')[val_col]

    # summarize_raster_by_region keys each zone by the stable ee_r50_aez18_id (see terrestrial_carbon_functions),
    # and DROPS empty zones, so map + align on that id, never on gpkg row position.
    regions = gpd.read_file(p.region_boundary_path, engine='pyogrio')
    def _fmt(v):
        return (endw_fmt % int(v)) if endw_fmt is not None else v
    labels = {(int(r['ee_r50_aez18_id']) if 'ee_r50_aez18_id' in r.index else r.get('id', i)):
              (_fmt(r[endw_col]), r[reg_col]) for i, r in regions.iterrows()}

    # per-anchor-year zone means; shock_Y = (scenario_Y - baseline_Y)/baseline_Y * 100 (contemporaneous /base_Y),
    # then piecewise-linear interp to annual values (0 at base_year) -- one computed point per SEALS map year.
    all_years = list(range(base_year, end_year + 1))
    base_by_year = {y: zone_mean(base_scn, y) for y in anchor_years}

    # FIXED-BASE denominator: baseline carbon density at the base year (the "Value of Nature"
    # reference, % of base-year value). At base_year every scenario shares the observed base map,
    # so this is just that map's zone means. Emitted ALONGSIDE the contemporaneous shock so carbon
    # and pollination are comparable on the fixed-base measure (denominator decision). Degrades to
    # NaN if the base year isn't available, so shock_pct is never affected.
    #
    # Base map comes via the ES-shared p.es_base_year_lulc_path (SEALS7-classified), injected into
    # scenario_lulc_paths[base_scn][base_year] exactly like erosion. NEVER p.base_year_lulc_path: SEALS
    # OWNS it and overwrites it at runtime with its raw-ESA source, and the density lookup is keyed on
    # SEALS7 classes, so a raw-ESA base map yields all-NaN densities. Align to the scenario grid first
    # (categorical -> 'near' is a lossless clip; regenerate every run, no stale reuse). A carbon-specific
    # terrestrial_carbon_base_year_lulc_path still overrides if a caller sets one.
    _base_map = getattr(p, 'terrestrial_carbon_base_year_lulc_path', None) or getattr(p, 'es_base_year_lulc_path', None)
    if _base_map and base_year not in p.scenario_lulc_paths.get(base_scn, {}):
        _base_map = _base_map if os.path.isabs(_base_map) else p.get_path(_base_map)
        if _yx(_base_map) != _yx(_ref_lulc):
            _aligned_base = os.path.join(p.cur_dir, 'lulc_base_year_aligned.tif')
            hb.resample_to_match(_base_map, _ref_lulc, _aligned_base, resample_method='near')
            _base_map = _aligned_base
        p.scenario_lulc_paths.setdefault(base_scn, {})[base_year] = _base_map
    # erosion-style guard: only compute the fixed-base level if the base year is actually present.
    base_at_base = zone_mean(base_scn, base_year) if base_year in p.scenario_lulc_paths.get(base_scn, {}) else None

    rows = []
    for scenario in scenarios:
        scn_by_year = {y: zone_mean(scenario, y) for y in anchor_years}
        num = {y: (scn_by_year[y] - base_by_year[y]) for y in anchor_years}  # shared numerator
        # (1) contemporaneous /base_Y -- the GTAP shock (unchanged behaviour)
        anchor_contemp = pd.DataFrame({
            y: num[y] / base_by_year[y].replace(0, np.nan) * 100.0 for y in anchor_years}).dropna()
        # (2) fixed-base /base_{base_year} -- reporting/comparability measure
        anchor_fixed = (pd.DataFrame({
            y: num[y] / base_at_base.replace(0, np.nan) * 100.0 for y in anchor_years}).dropna()
            if base_at_base is not None else None)
        for zid, s in anchor_contemp.iterrows():
            if zid not in labels:
                continue
            endw, reg = labels[zid]
            annual_c = np.interp(all_years, [base_year] + anchor_years, [0.0] + list(s.values))
            if anchor_fixed is not None and zid in anchor_fixed.index:
                annual_f = np.interp(all_years, [base_year] + anchor_years,
                                     [0.0] + list(anchor_fixed.loc[zid].values))
            else:
                annual_f = [np.nan] * len(all_years)
            for year, vc, vf in zip(all_years, annual_c, annual_f):
                # Explicit, same-named columns in both ES files (carbon + pollination) for the #14 diagnostic.
                # shock_pct = the GTAP-primary alias (carbon primary = contemporaneous, so it equals shock_pct_contemp).
                rows.append({'ENDW': endw, 'ACTS': acts, 'REG': reg, 'scenario': scenario,
                             'year': year, 'shock_pct': vc,
                             'shock_pct_fixedbase': vf, 'shock_pct_contemp': vc})

    out = pd.DataFrame(rows)
    out.to_csv(p.terrestrial_carbon_shock_output_path, index=False)
    print('  carbon shock: %d rows, %d scenarios (shock_pct=shock_pct_contemp=/base_Y, shock_pct_fixedbase=/base_%d) -> %s'
          % (len(out), out['scenario'].nunique() if rows else 0, base_year, p.terrestrial_carbon_shock_output_path))
    return True


def task_compute_terrestrial_carbon_shock_static(p):
    """Static per-scenario carbon shock -> FRS, linear ramp 0->end_year, from the frozen dependency table.

    The fallback add_terrestrial_carbon_tasks selects when <2 SEALS map years exist (the dynamic recompute needs >=2
    anchor maps to measure change). READS input_dir/raw_dependencies/carbon_storage_dependency.csv
    (override p.terrestrial_carbon_dependency_path) and subtracts the baseline_ignore_dependencies row at 2050
    (percentage_change x100), ramping that difference linearly from 0 at base_year. NEVER writes back to
    raw_dependencies -- the output goes to p.terrestrial_carbon_shock_output_path (terrestrial_carbon_interpolated.csv),
    the same file the dynamic task writes, so build_combined_afeall_cc_es is agnostic to which one ran.
    Caller sets: es_shock_base_year, es_shock_end_year, es_shock_scenarios,
    terrestrial_carbon_shock_output_path; scenario->raw name via p.terrestrial_carbon_scenario_map (default utilities.ES_SCENARIO_MAP);
    sector via p.terrestrial_carbon_shock_acts (default 'FRS', matching the dynamic task).
    """
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'terrestrial_carbon_shock_output_path', None):
        p.terrestrial_carbon_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir, 'terrestrial_carbon_interpolated.csv')
    if not p.run_this:
        return
    import pandas as pd

    base_year = int(p.es_shock_base_year)
    end_year = int(p.es_shock_end_year)
    n_years = end_year - base_year
    scenario_map = getattr(p, 'terrestrial_carbon_scenario_map', utilities.ES_SCENARIO_MAP)
    scenarios = list(p.es_shock_scenarios)
    acts = getattr(p, 'terrestrial_carbon_shock_acts', 'FRS')

    carb_path = getattr(p, 'terrestrial_carbon_dependency_path', None) or os.path.join(
        p.input_dir, 'raw_dependencies', 'carbon_storage_dependency.csv')
    if not os.path.exists(carb_path):
        print('  carbon shock: dependency csv not found (%s) -- skipping' % carb_path)
        return

    df = pd.read_csv(carb_path)
    base = df[(df['scenario'] == 'baseline_ignore_dependencies') & (df['year'] == 2050)]
    base_vals = base.set_index(['ENDW', 'REG'])['percentage_change'].astype(float) * 100

    rows = []
    for our_scn in scenarios:
        candidates = scenario_map.get(our_scn)
        raw_scn = next((c for c in candidates if c in df['scenario'].values), None) if candidates else None
        if not raw_scn:
            continue
        scn = df[(df['scenario'] == raw_scn) & (df['year'] == 2050)]
        scn_vals = scn.set_index(['ENDW', 'REG'])['percentage_change'].astype(float) * 100
        common = base_vals.index.intersection(scn_vals.index)
        shock = (scn_vals.loc[common] - base_vals.loc[common]).dropna()
        for year in range(base_year, end_year + 1):
            frac = (year - base_year) / n_years
            for (endw, reg), val in shock.items():
                rows.append({'ENDW': endw, 'ACTS': acts, 'REG': reg,
                             'scenario': our_scn, 'year': year, 'shock_pct': val * frac})

    out = pd.DataFrame(rows)
    out.to_csv(p.terrestrial_carbon_shock_output_path, index=False)
    nz = out[(out['year'] == end_year) & (out['shock_pct'] != 0)] if len(out) else out
    print('  carbon shock: %d rows, %d scenarios, %d nonzero @%d (static, uncapped) -> %s'
          % (len(out), out['scenario'].nunique() if len(out) else 0, len(nz), end_year,
             p.terrestrial_carbon_shock_output_path))
    return True
