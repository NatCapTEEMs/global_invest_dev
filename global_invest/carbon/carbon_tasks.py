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

    At each SEALS anchor year in carbon_shock_years (5-year MAgPIE steps), build a carbon-density
    raster (generate_carbon_density_raster) for the baseline scenario and each scenario, and take
    its mean tC/ha per polygon of p.region_boundary_path (summarize_raster_by_region -- untouched,
    generic; the polygon geometry is identical across scenarios, so the mean is sufficient and area
    cancels). The shock at year Y is (mean_scenario_Y - mean_baseline_Y) / mean_baseline_Y * 100 per
    zone (contemporaneous /base_Y = the GTAP shock, column `shock_pct`), piecewise-linearly interpolated
    across the anchor years (0 at base_year). ALSO emits a fixed-base measure (column `shock_pct_fixedbase`)
    = same numerator / baseline density at base_year -- the "Value of Nature" % of base-year value, for
    comparability with pollination (denominator decision). Emits an aoall table keyed by the boundary's ENDW/REG columns and p.carbon_shock_acts. The only region-specific knowledge is the column names,
    supplied by the caller via p; nothing GTAP-specific is hardcoded.

    Caller sets on p: carbon_shock_years (SEALS anchor years, from seals_years),
    scenario_lulc_paths {scenario: {year: path}} or carbon_lulc_path_template, carbon_shock_scenarios,
    region_boundary_path, carbon_zones_path, carbon_density_lookup_table_path,
    carbon_shock_output_path. Optional: carbon_shock_{base_scenario, base_year,
    endw_col, reg_col, value_col, acts}.
    """
    if not p.run_this:
        return
    import pandas as pd
    import geopandas as gpd

    base_scn     = getattr(p, 'carbon_shock_base_scenario', 'baseline_ignore_dependencies')
    base_year    = int(p.carbon_shock_base_year)          # interp 0-anchor, set by the caller from config
    anchor_years = sorted(y for y in map(int, p.carbon_shock_years) if y > base_year)  # SEALS anchors (seals_years)
    end_year     = anchor_years[-1]
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
    # Resolve the LULC map per scenario by globbing carbon_lulc_path_template ({scenario}/{year}
    # placeholders) when the caller didn't pre-build scenario_lulc_paths. Globbing lives here so a
    # project passes only a template string, not a path-building task.
    scenarios = list(getattr(p, 'carbon_shock_scenarios', []))
    if not getattr(p, 'scenario_lulc_paths', None):
        import glob
        tmpl = p.carbon_lulc_path_template
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
    if _yx(p.carbon_zones_path) != _yx(_ref_lulc):
        _aligned_cz = os.path.join(p.cur_dir, 'carbon_zones_aligned.tif')
        if not os.path.exists(_aligned_cz):
            hb.resample_to_match(p.carbon_zones_path, _ref_lulc, _aligned_cz, resample_method='near')
        p.carbon_zones_path = _aligned_cz

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

    # summarize_raster_by_region keys each zone by the stable ee_r50_aez18_id (see carbon_functions),
    # and DROPS empty zones, so map + align on that id, never on gpkg row position.
    regions = gpd.read_file(p.region_boundary_path)
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
    # NaN if the caller supplied no base map, so shock_pct is never affected.
    #
    # Prefer a carbon-specific SEALS7 base map (carbon_shock_base_year_lulc_path) over the generic
    # base_year_lulc_path: the density lookup is keyed on SEALS7 classes, so a raw-ESA base map would
    # yield all-NaN densities. Align it to the scenario grid first, exactly as the zones raster is
    # aligned above -- same res/origin makes 'near' a lossless clip (and the only correct method for
    # categorical LULC), so a global base map over an AOI sub-window works too.
    base_at_base = None
    _base_map = getattr(p, 'carbon_shock_base_year_lulc_path', None) or getattr(p, 'base_year_lulc_path', None)
    if _base_map:
        _base_map = _base_map if os.path.isabs(_base_map) else p.get_path(_base_map)
        if _yx(_base_map) != _yx(_ref_lulc):
            _aligned_base = os.path.join(p.cur_dir, 'lulc_base_year_aligned.tif')
            if not os.path.exists(_aligned_base):
                hb.resample_to_match(_base_map, _ref_lulc, _aligned_base, resample_method='near')
            _base_map = _aligned_base
        p.scenario_lulc_paths.setdefault(base_scn, {}).setdefault(base_year, _base_map)
        base_at_base = zone_mean(base_scn, base_year)

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
    out.to_csv(p.carbon_shock_output_path, index=False)
    print('  carbon shock: %d rows, %d scenarios (shock_pct=shock_pct_contemp=/base_Y, shock_pct_fixedbase=/base_%d) -> %s'
          % (len(out), out['scenario'].nunique() if rows else 0, base_year, p.carbon_shock_output_path))
    return True
