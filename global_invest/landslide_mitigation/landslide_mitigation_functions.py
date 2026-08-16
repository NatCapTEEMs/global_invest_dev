"""
landslide_mitigation_functions.py
"""
import numpy as np
from osgeo import gdal
import pygeoprocessing as pygeo

# ==================================================================== #
# Infinite-slope stability index (SI) computation
# ==================================================================== #

def compute_si_global(
    friction_angle_path,
    cohesion_soil_path,
    forest_share_path,
    c_root_max,
    unit_weight_path,
    transmissivity_path,
    static_q_path,
    slope_path,
    soil_depth_path,
    output_si_path,
    nodata=-9999.0,
    min_slope_deg=2.0,
):
    """Computes the infinite-slope stability index globally, block-wise.

    SI = tan(phi)/tan(beta) + c_total/(gamma*h*sin(beta)*cos(beta)) - q/(T*sin(beta))
      phi = friction angle (deg); beta = slope (deg)
      c_total = cohesion_soil + c_root, c_root = forest_share * c_root_max
      gamma = unit weight (kN/m^3); h = soil depth (m)
      q = specific discharge (m^2/day); T = transmissivity (m^2/day)

    c_root_max: plain scalar closed over by si_op (not passed through
    raster_calculator). 0.0 for the 'full_impacts' bound, C_ROOT_MAX_KPA
    otherwise (see preprocessing_tasks.compute_si_scenarios).

    min_slope_deg: excludes near-flat terrain entirely (infinite-slope
    theory doesn't apply there; standard in SHALSTAB/SINMAP/TRIGRS), not
    just a div/0 guard. Without it, both the friction and hydrological
    terms blow up near beta=0 and swamp the real forest-cover signal
    (median observed-vs-full_impacts diff was exactly 0 before this fix).

    Clipping: slope clipped to [0.5, 89.5] deg before trig (div/0 at flat,
    blowup near vertical); final SI clipped to [-10, 10].
    """
    paths = [
        friction_angle_path, cohesion_soil_path, forest_share_path,
        unit_weight_path, transmissivity_path, static_q_path,
        slope_path, soil_depth_path,
    ]
    nodatas = [pygeo.get_raster_info(path)['nodata'][0] for path in paths]

    def si_op(phi_deg, c_soil, forest_share, gamma, T, q, slope_deg, h):
        arrays = [phi_deg, c_soil, forest_share, gamma, T, q, slope_deg, h]
        valid = np.ones(phi_deg.shape, dtype=bool)
        for arr, nd in zip(arrays, nodatas):
            if nd is not None:
                valid &= (arr != nd)

        # Physical exclusion, not a div/0 guard
        valid &= (slope_deg >= min_slope_deg)

        slope_deg_c = np.clip(slope_deg, 0.5, 89.5)
        slope_rad = np.radians(slope_deg_c)
        phi_rad = np.radians(np.clip(phi_deg, 15, 40))  # re-clip defensively

        c_root = np.clip(forest_share, 0, 1) * c_root_max
        c_total = c_soil + c_root

        # Term 1: friction / geometry.
        term1 = np.tan(phi_rad) / np.tan(slope_rad)

        # Term 2: cohesion (stabilizing). errstate suppresses a spurious
        # div/0 warning at denom2==0 (bare rock / zero soil depth)
        denom2 = gamma * h * np.sin(slope_rad) * np.cos(slope_rad)
        with np.errstate(divide='ignore', invalid='ignore'):
            term2 = np.where(denom2 > 0, c_total / denom2, 0)

        # Term 3: hydrological (destabilizing), physically a saturation
        # ratio -> capped at 1 (standard in this model family). Uncapped,
        # ~90% of pixels clipped to the -10 SI floor even after the
        # slope filter, since upslope_area is heavily right-skewed.
        denom3 = T * np.sin(slope_rad)
        with np.errstate(divide='ignore', invalid='ignore'):
            saturation_ratio = np.where(denom3 > 0, q / denom3, 0)
        term3 = np.clip(saturation_ratio, 0, 1)

        si = np.clip(term1 + term2 - term3, -10, 10)
        return np.where(valid, si, nodata).astype(np.float32)

    pygeo.raster_calculator(
        [(path, 1) for path in paths],
        si_op, output_si_path, gdal.GDT_Float32, nodata,
        calc_raster_stats=True,
    )
    return output_si_path


# ==================================================================== #
# Thickness-weighted 0-30cm combine (SoilGrids + HiHydroSoil share this)
# ==================================================================== #

DEPTH_WEIGHTS_0_30CM = {'0-5cm': 5, '5-15cm': 10, '15-30cm': 15}  # total 30cm

def thickness_weighted_combine(depth_raster_paths, out_path, nodata=-9999.0,
                                conv_factor=None):
    """Combine 3 depth-interval rasters (0-5, 5-15, 15-30cm) into one
    thickness-weighted 0-30cm 'topsoil' raster. Inputs must share the same
    native grid (true for SoilGrids/HiHydroSoil).
    """
    keys = list(DEPTH_WEIGHTS_0_30CM.keys())
    paths = [depth_raster_paths[k] for k in keys]
    weights = [DEPTH_WEIGHTS_0_30CM[k] for k in keys]

    # Size/alignment check up front.
    infos = [pygeo.get_raster_info(p) for p in paths]
    first_size = infos[0]['raster_size']
    for path, info in zip(paths, infos):
        if info['raster_size'] != first_size:
            raise ValueError(
                f'{path} size {info["raster_size"]} does not match first '
                f'input {first_size} -- inputs must share the same native '
                f'grid before combining.'
            )
    src_nodatas = [info['nodata'][0] for info in infos]

    def combine_op(*arrays):
        weighted_sum = np.zeros(arrays[0].shape, dtype=np.float64)
        weight_present = np.zeros(arrays[0].shape, dtype=np.float64)
        for arr, w, nd in zip(arrays, weights, src_nodatas):
            arr64 = arr.astype(np.float64)
            valid = np.ones(arr.shape, dtype=bool) if nd is None else (arr != nd)
            if conv_factor is not None:
                arr64 = arr64 / conv_factor
            weighted_sum[valid] += arr64[valid] * w
            weight_present[valid] += w

        with np.errstate(invalid='ignore', divide='ignore'):
            combined = np.where(weight_present > 0, weighted_sum / weight_present, np.nan)
        return np.where(np.isnan(combined), nodata, combined).astype(np.float32)

    pygeo.raster_calculator(
        [(path, 1) for path in paths], combine_op, out_path,
        gdal.GDT_Float32, nodata,
    )
    return out_path