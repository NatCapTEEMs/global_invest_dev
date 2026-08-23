"""The landslide-mitigation science, as pure functions over arrays and frames.

Ported from m-braaksma/landslide_mitigation v0.2.0 (folded 2026-08-16); every function here
was previously a closure or an inline block inside landslide_mitigation_tasks.py. Nothing in
this module reads or writes a file or touches a ProjectFlow object -- the task module supplies
the arrays and frames and writes the results, which is what makes the arithmetic testable.

The calculation, in the order the task tree runs it:

  input data      ESA-CCI class codes -> a 0-1 root-reinforcement weight per pixel
                  (forest_weight_lut, forest_weight_from_lulc).
  preprocessing   SoilGrids texture -> friction angle, cohesion, unit weight; HiHydroSoil
                  K_sat x soil depth -> transmissivity; WorldClim rain x upslope area ->
                  specific discharge (friction_angle_from_texture, soil_cohesion_from_texture,
                  unit_weight_from_bulk_density, transmissivity_from_ksat, specific_discharge).
                  UGLC point events -> annual binary/mortality panels (uglc_year_panels).
  stability       the infinite-slope stability index, with root cohesion switched on
                  (`observed`) or off (`full_impacts`) -- stability_index, c_root_max_for_scenario.
  estimation      case-control panel construction (deduplicate_cases_to_pixels,
                  pixel_row_col, pixel_center_coords, case_control_rows) and the Prentice &
                  Pyke intercept correction (event_and_land_pixel_counts, prevalence,
                  case_control_intercept_offset, corrected_intercept), plus Duan's smearing
                  factor for the log-scale severity stage (duan_smearing_factor).
  prediction      hazard_probability x expected_deaths_given_landslide per pixel, tiled
                  (tile_offsets, tile_geotransform).
  valuation       OECD VSL per country with an income-group fallback (vsl_usd_2019_by_iso3,
                  assign_vsl); avoided deaths are the full_impacts minus observed difference,
                  and their value is that difference times the pixel's VSL.
  reporting       publication table assembly (regression_stars, publication_table,
                  coefficients_by_term, hurdle_table_rows) and figure binning
                  (fatality_bin_masks, bucket_legend_labels).
"""
import numpy as np
import pandas as pd

# Every derived raster in this chain carries the same nodata value.
NODATA = -9999.0

# METHOD CONSTANTS defining the ported v0.2.0 science (the landslide author's to bless) -- in
# code so a change costs a reviewed commit. publish_inputs applies them caller-wins, so a
# deliberate override on p survives.
DATA_PROCESSING_YEARS = list(range(2007, 2020))  # every year the input rasters are built for
MODELING_YEARS = list(range(2007, 2019))         # the years both models are fitted on
PREDICTION_YEARS = [2019]                        # held out of fitting, and the reported year
MAX_LOCATION_ACCURACY_M = 1000   # UGLC events located less precisely than this are dropped
CONTROL_RATIO = 25               # control pixel-years drawn per case pixel-year
# Scenario -> the root cohesion it gets. 'observed' keeps the mapped forest share at full root
# strength; 'full_impacts' (the value 0) strips root cohesion, and the difference in predicted
# deaths between the two is what the service values.
C_ROOT_SCENARIOS = {'observed': 'observed', 'full_impacts': 0}

# GeoTIFF creation options. A full EASE-Grid 1km raster is global and large, so every
# full-grid output is tiled, LZW-compressed and BIGTIFF-capable; per-tile outputs are small
# enough to skip BIGTIFF, and the stitched globals spell the options in the source repo's order.
GTIFF_CREATION_OPTIONS = ('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                          'BLOCKXSIZE=256', 'BLOCKYSIZE=256')
TILE_CREATION_OPTIONS = ('TILED=YES', 'COMPRESS=LZW')
STITCH_CREATION_OPTIONS = ('COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES')

# Nodata values of the raw sources, as their own documentation gives them.
DEM_SOURCE_NODATA = -9999            # SEALS alt_m.tif
LANDSCAN_SOURCE_NODATA = -2147483647  # LandScan int32 fill
SOIL_DEPTH_SOURCE_NODATA = -1.0      # Pelletier/ORNL DAAC
GRIP_SOURCE_NODATA = -9999           # GRIP4 .asc

# ESA-CCI class code -> root-reinforcement weight (0-1). Closed-canopy forest classes get the
# full weight; the two mosaic tree/shrub-herbaceous classes are 50/50 land cover and take half;
# sparse tree cover (<15%) is below meaningful root reinforcement. Every class not listed is 0.
FOREST_WEIGHT = {
    50: 1.0, 60: 1.0, 61: 1.0, 62: 1.0,
    70: 1.0, 71: 1.0, 72: 1.0, 80: 1.0, 81: 1.0, 82: 1.0,
    90: 1.0,             # tree_mixed_type
    160: 1.0, 170: 1.0,  # flooded tree cover
    100: 0.5, 110: 0.5,  # mosaic tree/shrub-herbaceous 50/50 -- medium weight (chosen)
    151: 0.0,            # sparse_tree_15 -- below meaningful root reinforcement
}
ESA_CLASS_CODE_MAX = 255  # ESA-CCI class codes are 8-bit, so a 256-entry LUT covers them all

# Thickness of each SoilGrids/HiHydroSoil depth interval (cm), summing to the 0-30cm topsoil.
DEPTH_WEIGHTS_0_30CM = {'0-5cm': 5, '5-15cm': 10, '15-30cm': 15}

# SoilGrids property -> (its file's property code, the integer scaling to divide out). SoilGrids
# ships percentages x10 and bulk density x100 as integers.
SOILGRIDS_PROPERTIES = {
    'sand_pct': ('sand', 10),
    'clay_pct': ('clay', 10),
    'org_carbon_pct': ('soc', 10),
    'bulk_density': ('bdod', 100),
}

# Slope is computed at this multiple of the 1km grid and averaged back down: computing it at
# 1km directly would flatten the terrain the model is about. 4 gives an exact ~250m subdivision.
SLOPE_FINE_FACTOR = 4

# --- Infinite-slope stability index -----------------------------------------------------
C_ROOT_MAX_KPA = 5.0          # root cohesion added by fully forested cover (kPa)
GRAVITY_M_S2 = 9.81           # converts bulk density in Mg/m3 to unit weight in kN/m3
MIN_SLOPE_DEG = 2.0           # infinite-slope theory does not apply below this: a physical
                              # exclusion, not a div/0 guard (see stability_index)
SLOPE_CLIP_DEG = (0.5, 89.5)  # tan(beta) -> 0 at flat, -> inf near vertical
FRICTION_ANGLE_CLIP_DEG = (15.0, 40.0)   # the plausible range for the texture regression
FOREST_SHARE_CLIP = (0.0, 1.0)
SATURATION_RATIO_MAX = 1.0    # q/(T sin beta) is physically a saturation ratio
SI_CLIP = (-10.0, 10.0)

# --- Soil hydraulic properties from SoilGrids/HiHydroSoil -------------------------------
FRICTION_ANGLE_INTERCEPT_DEG = 25.0     # phi = 25 + (sand% - clay%) / 20
FRICTION_ANGLE_TEXTURE_DIVISOR = 20.0
COHESION_INTERCEPT_KPA = 2.0            # c_soil = 2 + 0.03 clay% + 0.1 organic carbon%
COHESION_PER_CLAY_PCT_KPA = 0.03
COHESION_PER_ORG_CARBON_PCT_KPA = 0.1
COHESION_CLIP_KPA = (0.0, 50.0)
HIHYDROSOIL_KSAT_SCALE = 0.0001         # HiHydroSoil raw Int32 = value x 10,000, in cm/day
CM_PER_M = 100.0

# --- Static specific discharge ----------------------------------------------------------
DAYS_PER_YEAR = 365.25   # WorldClim BIO12 is an annual total, spread evenly over the year
MM_PER_M = 1000.0

# --- UGLC event cleaning ----------------------------------------------------------------
# Location accuracies the UGLC uses as fill values rather than as metres.
UGLC_ACCURACY_NDV_CODES = (-99999, -9999, -999, 0, np.nan)

# --- Case-control estimation panel ------------------------------------------------------
CONTROL_DRAW_SEED = 42                  # reproducible control draws
CONTROL_DRAW_MAX_ATTEMPTS_FACTOR = 20   # give up after 20x the needed draws land in ocean
CONTROL_DRAW_BATCH_CAP = 5000           # pixels tested per rejection-sampling batch
MIN_FATAL_ROWS_FOR_SEVERITY = 10        # below this the part-B fit is warned as unstable
SEVERITY_REGULARIZATION_ALPHA = 1.0     # L2 penalty for the part-A separation fallback

# --- Prediction tiling ------------------------------------------------------------------
DEFAULT_TILE_SIZE = 2000   # pixels per side; p.processing_resolution overrides

# --- Valuation: value of a statistical life ---------------------------------------------
# Table 6.1, OECD (2025) Mortality Risk Valuation in Policy Assessment. USD millions,
# 2022 base year (the same base as the individual-country CSV).
GROUP_BASE_VSL_2022 = {
    'Global': 2.7,
    'OECD': 7.1,
    'EU': 8.4,
    'United States': 8.5,
    'High-income': 7.9,
    'Low-and-middle-income': 1.1,
}

INCOME_GRP_TO_FALLBACK_GROUP = {
    '1. High income: OECD': 'OECD',
    '2. High income: nonOECD': 'High-income',
    '3. Upper middle income': 'Low-and-middle-income',
    '4. Lower middle income': 'Low-and-middle-income',
    '5. Low income': 'Low-and-middle-income',
}

# PMPRB CPI-based price-adjustment factors (US CPI-derived): benchmark year 2019 -> 2022
# cumulative price-adjustment factor = 1.050.
CPI_2019_TO_2022 = 1.050
DEFLATOR_2022_TO_2019 = 1 / CPI_2019_TO_2022
USD_PER_MILLION = 1e6

# --- Reporting --------------------------------------------------------------------------
SIGNIFICANCE_LEVELS = ((0.001, '***'), (0.01, '**'), (0.05, '*'))
TOP_COUNTRY_COUNT = 15                  # rows in the top-countries results table
CHOROPLETH_BUCKET_EDGES = [0, 1, 5, 15, 50, 100, float('inf')]
# UGLC event map: deaths per event, open-ended at the top.
FATALITY_BINS = (('1-5', 1, 5), ('5-25', 5, 25), ('25-100', 25, 100), ('100+', 100, None))
PLOT_RASTER_MAX_DIM = 4096              # decimate a global raster to this before plotting
PLOT_PERCENTILES = (2, 98)              # colour-scale range, robust to the long right tail


# ============================================================================ #
# Input data: forest cover
# ============================================================================ #

def forest_weight_lut(forest_weight=None):
    """ESA-CCI class code -> root-reinforcement weight, as a lookup array indexable by code.

    Args:
        forest_weight (dict): class code -> weight; defaults to FOREST_WEIGHT.

    Returns:
        np.ndarray: float32, length ESA_CLASS_CODE_MAX + 1, zero for unlisted codes.
    """
    forest_weight = FOREST_WEIGHT if forest_weight is None else forest_weight
    lut = np.zeros(ESA_CLASS_CODE_MAX + 1, dtype=np.float32)
    for class_code, weight in forest_weight.items():
        lut[class_code] = weight
    return lut


def forest_weight_from_lulc(lulc, lut, nodata=None):
    """Per-pixel forest weight for a block of ESA-CCI class codes.

    Codes are clipped into the LUT's range before lookup; nodata pixels come back as NODATA
    so the subsequent average-resampling warp treats them as absent rather than as zero forest.
    """
    weight = lut[np.clip(lulc, 0, ESA_CLASS_CODE_MAX)]
    if nodata is not None:
        weight = np.where(lulc == nodata, NODATA, weight)
    return weight


# ============================================================================ #
# Preprocessing: soil depth intervals, hydraulic properties, discharge
# ============================================================================ #

def thickness_weighted_mean(arrays, weights, nodatas, conv_factor=None):
    """Depth-interval arrays -> one thickness-weighted mean over the intervals present.

    Each interval contributes its own thickness as the weight, and only where it has data, so
    a pixel missing the 15-30cm layer is the mean of the two shallower layers rather than a
    hole. Pixels with no valid interval come back NaN for the caller to write as nodata.

    Args:
        arrays (list): one array per depth interval, same shape and grid.
        weights (list): the interval thicknesses, aligned with `arrays`.
        nodatas (list): each array's nodata value (None where it has none).
        conv_factor (float): SoilGrids' integer scaling, divided out before weighting.

    Returns:
        np.ndarray: float64 weighted mean, NaN where no interval had data.
    """
    weighted_sum = np.zeros(arrays[0].shape, dtype=np.float64)
    weight_present = np.zeros(arrays[0].shape, dtype=np.float64)
    for array, weight, nodata in zip(arrays, weights, nodatas):
        values = array.astype(np.float64)
        valid = np.ones(array.shape, dtype=bool) if nodata is None else (array != nodata)
        if conv_factor is not None:
            values = values / conv_factor
        weighted_sum[valid] += values[valid] * weight
        weight_present[valid] += weight

    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(weight_present > 0, weighted_sum / weight_present, np.nan)


def friction_angle_from_texture(sand_pct, clay_pct):
    """Internal friction angle (degrees) from sand and clay percentages.

    phi = 25 + (sand% - clay%) / 20, clipped to the plausible 15-40 degree range: sandier
    soils resist shear at a steeper angle than clay-rich ones.
    """
    phi = (FRICTION_ANGLE_INTERCEPT_DEG
           + (sand_pct - clay_pct) / FRICTION_ANGLE_TEXTURE_DIVISOR)
    return np.clip(phi, *FRICTION_ANGLE_CLIP_DEG)


def soil_cohesion_from_texture(clay_pct, org_carbon_pct):
    """Effective soil cohesion (kPa) from clay and organic carbon percentages.

    c_soil = 2 + 0.03 clay% + 0.1 organic-carbon%, clipped to 0-50 kPa. This is the cohesion
    the soil has on its own, before any root reinforcement is added.
    """
    cohesion = (COHESION_INTERCEPT_KPA
                + COHESION_PER_CLAY_PCT_KPA * clay_pct
                + COHESION_PER_ORG_CARBON_PCT_KPA * org_carbon_pct)
    return np.clip(cohesion, *COHESION_CLIP_KPA)


def unit_weight_from_bulk_density(bulk_density_mg_m3):
    """Soil unit weight (kN/m3) = bulk density (Mg/m3) x g."""
    return bulk_density_mg_m3 * GRAVITY_M_S2


def transmissivity_from_ksat(ksat_raw, soil_depth_m):
    """Soil transmissivity T (m2/day) = saturated conductivity x soil depth.

    HiHydroSoil ships K_sat as an integer scaled by 10,000 in cm/day; it is unscaled and
    converted to m/day before multiplying by the soil depth.
    """
    ksat_cm_day = ksat_raw * HIHYDROSOIL_KSAT_SCALE
    return (ksat_cm_day / CM_PER_M) * soil_depth_m


def specific_discharge(rain_mm_yr, upslope_area_m2, cell_width_m):
    """Static specific discharge q (m2/day) into a cell from its upslope contributing area.

    The annual rainfall total is spread evenly over the year, applied to the whole upslope
    area, and divided by the cell width it enters through.
    """
    rain_m_day = (rain_mm_yr / DAYS_PER_YEAR) / MM_PER_M
    return (rain_m_day * upslope_area_m2) / cell_width_m


# ============================================================================ #
# Stability: the infinite-slope index, with and without root cohesion
# ============================================================================ #

def c_root_max_for_scenario(scenario_value):
    """Root cohesion available in a scenario: 0 for the no-root counterfactual.

    The es_config scenarios are `observed` (forest share as mapped, root cohesion at
    C_ROOT_MAX_KPA) and `full_impacts` (the value 0, meaning strip root cohesion entirely so
    the difference in predicted deaths is what forest roots avert).
    """
    return 0.0 if scenario_value == 0 else C_ROOT_MAX_KPA


def stability_index(phi_deg, c_soil, forest_share, gamma, transmissivity, q, slope_deg,
                    soil_depth, c_root_max):
    """The infinite-slope stability index SI, per pixel.

        SI = tan(phi)/tan(beta) + c_total / (gamma h sin(beta) cos(beta)) - min(q/(T sin beta), 1)

    with phi the friction angle, beta the slope, c_total = c_soil + forest_share x c_root_max
    the combined soil and root cohesion, gamma the unit weight, h the soil depth, q the
    specific discharge and T the transmissivity. SI > 1 is stable; the forest signal enters
    only through c_total, so the observed-minus-counterfactual difference is exactly the root
    cohesion term.

    Slope is clipped to SLOPE_CLIP_DEG and the friction angle to FRICTION_ANGLE_CLIP_DEG
    before the trigonometry (tan blows up at both ends). The cohesion term is zero where the
    denominator is zero (bare rock, no soil). The hydrological term is physically a saturation
    ratio, so it is capped at 1 -- uncapped, the heavily right-skewed upslope area drives most
    pixels onto the SI floor. The result is clipped to SI_CLIP.

    Excluding near-flat terrain (slope below MIN_SLOPE_DEG) is the caller's job, since it
    belongs with the nodata mask rather than with the arithmetic.
    """
    slope_rad = np.radians(np.clip(slope_deg, *SLOPE_CLIP_DEG))
    phi_rad = np.radians(np.clip(phi_deg, *FRICTION_ANGLE_CLIP_DEG))

    c_total = c_soil + np.clip(forest_share, *FOREST_SHARE_CLIP) * c_root_max

    friction_term = np.tan(phi_rad) / np.tan(slope_rad)

    cohesion_denominator = gamma * soil_depth * np.sin(slope_rad) * np.cos(slope_rad)
    with np.errstate(divide='ignore', invalid='ignore'):
        cohesion_term = np.where(cohesion_denominator > 0, c_total / cohesion_denominator, 0)

    saturation_denominator = transmissivity * np.sin(slope_rad)
    with np.errstate(divide='ignore', invalid='ignore'):
        saturation_ratio = np.where(saturation_denominator > 0, q / saturation_denominator, 0)
    saturation_term = np.clip(saturation_ratio, 0, SATURATION_RATIO_MAX)

    return np.clip(friction_term + cohesion_term - saturation_term, *SI_CLIP)


# ============================================================================ #
# UGLC event panels and the case-control estimation table
# ============================================================================ #

def pixel_row_col(ease_x, ease_y, geotransform):
    """EASE-Grid metre coordinates -> the reference grid's (row, col) indices."""
    col = ((ease_x - geotransform[0]) // geotransform[1]).astype(int)
    row = ((ease_y - geotransform[3]) // geotransform[5]).astype(int)
    return row, col


def pixel_center_coords(rows, cols, geotransform):
    """(row, col) indices -> the pixel centres' EASE-Grid metre coordinates."""
    x = geotransform[0] + (cols + 0.5) * geotransform[1]
    y = geotransform[3] + (rows + 0.5) * geotransform[5]
    return x, y


def uglc_year_panels(events, geotransform, x_size, y_size):
    """One year of UGLC events stamped onto the reference grid.

    Each event is a point plus a location accuracy, so it is spread over a disc of that
    radius rather than assigned to one pixel: the binary panel marks every pixel the disc
    covers, and the mortality panel spreads the event's fatalities over the disc with a linear
    distance decay (1 - distance/accuracy), summed across events. Events with a non-positive
    accuracy, or whose disc falls entirely off the grid, contribute nothing.

    Args:
        events (pd.DataFrame): one year's events, with ease_x, ease_y, accuracy_m and
            fatality_count columns.
        geotransform (tuple): the reference grid's GDAL geotransform.
        x_size (int), y_size (int): the reference grid's dimensions.

    Returns:
        (np.ndarray, np.ndarray): the uint8 binary panel and the float32 mortality panel.
    """
    pixel_size = geotransform[1]
    binary = np.zeros((y_size, x_size), dtype=np.uint8)
    mortality = np.zeros((y_size, x_size), dtype=np.float32)

    for _, event in events.iterrows():
        center_x, center_y = event['ease_x'], event['ease_y']
        radius_m = event['accuracy_m']
        fatalities = event['fatality_count']
        if radius_m <= 0:
            continue

        center_col = int((center_x - geotransform[0]) / geotransform[1])
        center_row = int((center_y - geotransform[3]) / geotransform[5])
        pad = int(np.ceil(radius_m / pixel_size)) + 1

        row_start = max(center_row - pad, 0)
        row_stop = min(center_row + pad, y_size - 1)
        col_start = max(center_col - pad, 0)
        col_stop = min(center_col + pad, x_size - 1)
        if row_start > row_stop or col_start > col_stop:
            continue

        col_grid, row_grid = np.meshgrid(np.arange(col_start, col_stop + 1),
                                         np.arange(row_start, row_stop + 1))
        # The grid is already in metres, so pixel centres need no CRS transform.
        pixel_x, pixel_y = pixel_center_coords(row_grid, col_grid, geotransform)
        distance = np.sqrt((pixel_x - center_x) ** 2 + (pixel_y - center_y) ** 2)

        within = distance <= radius_m
        if not within.any():
            continue

        window = (slice(row_start, row_stop + 1), slice(col_start, col_stop + 1))
        binary[window][within] = 1
        if fatalities > 0:
            weight = np.where(within, 1 - (distance / radius_m), 0)
            mortality[window] += fatalities * weight

    return binary, mortality


def deduplicate_cases_to_pixels(cases, geotransform):
    """One row per grid pixel from a year's event points.

    Several UGLC events can land in the same 1km pixel; the logit is fitted on pixel-years, so
    they collapse to one row carrying the first event's coordinates and the highest fatality
    count of the group.
    """
    df = cases.copy()
    df['pixel_row'], df['pixel_col'] = pixel_row_col(df['ease_x'], df['ease_y'], geotransform)
    return df.groupby(['pixel_row', 'pixel_col'], as_index=False).agg(
        {'ease_x': 'first', 'ease_y': 'first', 'fatality_count': 'max'})


def case_control_rows(cases, control_x, control_y, year):
    """A year's estimation rows: the deduplicated events as cases, the land draws as controls.

    Controls carry a zero fatality count -- they are pixel-years where no landslide was
    recorded, not events with no deaths.
    """
    case_df = pd.DataFrame({
        'ease_x': cases['ease_x'].values,
        'ease_y': cases['ease_y'].values,
        'year': year,
        'case': 1,
        'fatality_count': cases['fatality_count'].values,
    })
    control_df = pd.DataFrame({
        'ease_x': control_x,
        'ease_y': control_y,
        'year': year,
        'case': 0,
        'fatality_count': 0.0,
    })
    return pd.concat([case_df, control_df], ignore_index=True)


# ============================================================================ #
# Calibration: case-control correction and the severity smearing factor
# ============================================================================ #

def event_and_land_pixel_counts(uglc_binary, binary_nodata, si, si_nodata):
    """(event pixel-years, SI-eligible land pixel-years) for one year's panels.

    The denominator is restricted to pixels the stability model actually scores, so the
    prevalence it feeds is the prevalence over the population the model predicts on.
    """
    valid = (np.ones(uglc_binary.shape, dtype=bool) if binary_nodata is None
             else (uglc_binary != binary_nodata))
    si_valid = (np.ones(si.shape, dtype=bool) if si_nodata is None else (si != si_nodata))
    valid &= si_valid
    return int((uglc_binary[valid] == 1).sum()), int(valid.sum())


def prevalence(event_pixels, land_pixels, fallback=float('nan')):
    """Event pixel-years per land pixel-year; `fallback` when nothing was counted."""
    return event_pixels / land_pixels if land_pixels > 0 else fallback


def case_control_intercept_offset(tau, pi):
    """The Prentice and Pyke (1979) offset, log((tau/(1-tau)) / (pi/(1-pi))).

    tau is the case fraction in the SAMPLE, pi the prevalence in the POPULATION. Ordinary
    logistic slopes are consistent on case-control data, but the intercept carries the
    oversampling; this is what has to come off it.
    """
    return np.log((tau / (1 - tau)) / (pi / (1 - pi)))


def corrected_intercept(alpha_raw, tau, pi):
    """The fitted intercept with the case-control oversampling removed."""
    return alpha_raw - case_control_intercept_offset(tau, pi)


def duan_smearing_factor(residuals):
    """Duan's (1983) smearing factor: mean(exp(residual)) from a log-scale fit.

    exp(predicted log fatalities) understates E[fatalities] by Jensen's inequality. Smearing
    corrects it from the residuals themselves rather than assuming log-normal errors.
    """
    return float(np.mean(np.exp(residuals)))


# ============================================================================ #
# Prediction: hazard probability and expected deaths
# ============================================================================ #

def logistic(log_odds):
    """The logistic link, 1 / (1 + exp(-x))."""
    return 1 / (1 + np.exp(-log_odds))


def hazard_probability(si, rain_max_daily, alpha, beta_si, beta_rain):
    """P(landslide) per pixel-year from the calibrated hazard logit.

    `alpha` is the case-control-corrected intercept, so the result is an absolute probability
    rather than a case-control one.
    """
    return logistic(alpha + beta_si * si + beta_rain * rain_max_daily)


def severity_linear_predictor(params, population_log1p, rain_max_daily, slope_degrees,
                              road_density):
    """The linear index both severity stages share, from a statsmodels params mapping."""
    return (params['Intercept']
            + params['population_log1p'] * population_log1p
            + params['rain_max_daily'] * rain_max_daily
            + params['slope_degrees'] * slope_degrees
            + params['road_density'] * road_density)


def expected_deaths_given_landslide(part_a_params, part_b_params, smearing_factor,
                                    population, rain_max_daily, slope_degrees, road_density):
    """E[deaths | a landslide occurs] per pixel, from the two-part hurdle model.

    P(fatality > 0 | landslide) from the part-A logit, times E[fatalities | fatal] from the
    part-B log-linear fit back-transformed with Duan's smearing factor. Multiply by
    `hazard_probability` to get expected deaths per pixel-year.
    """
    population_log1p = np.log1p(np.maximum(population, 0))
    p_fatal = logistic(severity_linear_predictor(
        part_a_params, population_log1p, rain_max_daily, slope_degrees, road_density))
    fatalities_if_fatal = np.exp(severity_linear_predictor(
        part_b_params, population_log1p, rain_max_daily, slope_degrees, road_density))
    return p_fatal * (fatalities_if_fatal * smearing_factor)


# ============================================================================ #
# Prediction tiling
# ============================================================================ #

def tile_offsets(n_cols, n_rows, tile_size):
    """Every [col_offset, row_offset, n_cols, n_rows] block covering the grid, row-major.

    Edge blocks are truncated rather than padded, so the blocks tile the grid exactly.
    """
    return [[col, row, min(tile_size, n_cols - col), min(tile_size, n_rows - row)]
            for row in range(0, n_rows, tile_size)
            for col in range(0, n_cols, tile_size)]


def tile_geotransform(reference_geotransform, col_offset, row_offset):
    """The geotransform of the tile whose top-left pixel is (col_offset, row_offset)."""
    gt = reference_geotransform
    return (gt[0] + col_offset * gt[1], gt[1], 0,
            gt[3] + row_offset * gt[5], 0, gt[5])


# ============================================================================ #
# Valuation: value of a statistical life
# ============================================================================ #

def vsl_usd_2019_by_iso3(oecd_df):
    """The OECD country VSL table -> {iso3: VSL in 2019 constant USD}.

    OBS_VALUE is reported with a UNIT_MULT power-of-ten exponent, and the published figures
    are 2022 USD, so they are scaled up and then deflated to the 2019 base the rest of GEP uses.
    """
    df = oecd_df
    if 'MEASURE_VSL' in df.columns:
        df = df[df['MEASURE_VSL'] == 'VSL']
    df = df.copy()
    df['vsl_usd_2022'] = df['OBS_VALUE'].astype(float) * 10 ** df['UNIT_MULT'].astype(float)
    df['vsl_usd_2019'] = df['vsl_usd_2022'] * DEFLATOR_2022_TO_2019
    return df.set_index('REF_AREA')['vsl_usd_2019'].to_dict()


def assign_vsl(regions_df, vsl_by_iso3, iso3_field):
    """A VSL for every region row: the country's own OECD estimate where there is one.

    Countries without a direct estimate take their income group's Table 6.1 figure, and a row
    whose income group is unclassified takes the global figure -- both in 2022 USD millions,
    so both are scaled and deflated to 2019 USD here.

    Returns:
        (pd.DataFrame, int, int): the regions with a vsl_usd column, the number of rows that
        got a direct OECD estimate, and the number that fell back to a group figure.
    """
    out = regions_df.copy()
    out['vsl_usd'] = out[iso3_field].map(vsl_by_iso3)
    n_direct = int(out['vsl_usd'].notna().sum())

    needs_fallback = out['vsl_usd'].isna()
    group_vsl_millions = out.loc[needs_fallback, 'income_grp'].map(
        lambda income_grp: GROUP_BASE_VSL_2022.get(
            INCOME_GRP_TO_FALLBACK_GROUP.get(income_grp), np.nan))
    group_vsl_millions = group_vsl_millions.fillna(GROUP_BASE_VSL_2022['Global'])
    out.loc[needs_fallback, 'vsl_usd'] = (group_vsl_millions * USD_PER_MILLION
                                          * DEFLATOR_2022_TO_2019)

    n_fallback = int((needs_fallback & out['vsl_usd'].notna()).sum())
    return out, n_direct, n_fallback


# ============================================================================ #
# Reporting: regression tables and figure bins
# ============================================================================ #

def regression_stars(p_value):
    """Significance stars for a p-value: *** below 0.001, ** below 0.01, * below 0.05."""
    if pd.isna(p_value):
        return ''
    for threshold, stars in SIGNIFICANCE_LEVELS:
        if p_value < threshold:
            return stars
    return ''


def publication_table(coef_rows, bottom_rows, dep_label):
    """A one-column econometrics table: coefficient with stars, standard error beneath.

    Args:
        coef_rows (list): (label, coefficient, std_error, p_value) per variable.
        bottom_rows (list): (label, value_string) appended after the coefficient block
            (observations, R-squared, fixed effects).
        dep_label (str): the column header, naming the dependent variable.

    Returns:
        pd.DataFrame: two rows per variable, then a blank row, then the bottom rows.
    """
    rows = []
    for label, coef, std_error, p_value in coef_rows:
        rows.append({' ': label, dep_label: f'{coef:.4f}{regression_stars(p_value)}'})
        rows.append({' ': '', dep_label: f'({std_error:.4f})' if pd.notna(std_error) else ''})

    rows.append({' ': '', dep_label: ''})
    for label, value_str in bottom_rows:
        rows.append({' ': label, dep_label: value_str})

    return pd.DataFrame(rows)


def coefficients_by_term(params, std_errors, p_values):
    """{term: (coefficient, std error, p value)} from a saved model's three per-term dicts."""
    return {term: (coef, std_errors.get(term, np.nan), p_values.get(term, np.nan))
            for term, coef in params.items()}


def hurdle_table_rows(part_a, part_b, term_labels, column_labels):
    """The coefficient/(SE) row pairs for the two hurdle stages shown side by side.

    Args:
        part_a, part_b (dict): {term: (coefficient, std error, p value)} per stage, as
            `coefficients_by_term` returns them.
        term_labels (dict): term -> the label to print; a term with no entry prints as itself.
        column_labels (tuple): the two column headers, part A then part B.

    Returns:
        list: row dicts, ready for the bottom rows to be appended before DataFrame().
    """
    column_a, column_b = column_labels
    term_order = list(part_a.keys()) + [term for term in part_b if term not in part_a]

    rows = []
    for term in term_order:
        a_coef, a_std_error, a_p_value = part_a.get(term, (np.nan, np.nan, np.nan))
        b_coef, b_std_error, b_p_value = part_b.get(term, (np.nan, np.nan, np.nan))
        rows.append({
            ' ': term_labels.get(term, term),
            column_a: (f'{a_coef:.4f}{regression_stars(a_p_value)}'
                       if pd.notna(a_coef) else ''),
            column_b: (f'{b_coef:.4f}{regression_stars(b_p_value)}'
                       if pd.notna(b_coef) else ''),
        })
        rows.append({
            ' ': '',
            column_a: f'({a_std_error:.4f})' if pd.notna(a_std_error) else '',
            column_b: f'({b_std_error:.4f})' if pd.notna(b_std_error) else '',
        })
    return rows


def fatality_bin_masks(fatalities):
    """{bin label: boolean mask} over FATALITY_BINS, for the UGLC event map."""
    masks = {}
    for label, low, high in FATALITY_BINS:
        in_bin = (fatalities >= low)
        if high is not None:
            in_bin &= (fatalities < high)
        masks[label] = in_bin
    return masks


def bucket_legend_labels(bucket_edges, tick_format):
    """Legend labels for a bucketed choropleth: one per bucket, open-ended at the top."""
    labels = []
    for lower, upper in zip(bucket_edges, bucket_edges[1:]):
        if upper == float('inf'):
            labels.append(f'{tick_format.format(lower)}+')
        else:
            labels.append(f'{tick_format.format(lower)} – {tick_format.format(upper)}')
    return labels
