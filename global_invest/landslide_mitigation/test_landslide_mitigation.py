"""Unit tests for the landslide_mitigation calculation.

The module has no committed anchor table to compare against (the author's v0.2.0 output is not
machine-readable here), so every test below pins the ported arithmetic on hand-built inputs:
small arrays and short frames whose expected values are worked out in the comments. Every pure
function in landslide_mitigation_functions is exercised.
"""
import math
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest import utilities
from global_invest.landslide_mitigation import landslide_mitigation_functions as chain
from global_invest.landslide_mitigation import landslide_mitigation_tasks as tasks

# A tiny reference grid: 1km pixels, origin at (0, 0), north-up.
TEST_GEOTRANSFORM = (0.0, 100.0, 0.0, 0.0, 0.0, -100.0)


# ---------------------------------------------------------------------------
# Wiring: the service's configuration rows and the files the shared render needs.
# ---------------------------------------------------------------------------

def test_es_config_and_parameters_rows_hydrate_landslide_mitigation(tmp_path):
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'landslide_mitigation', log=lambda *a: None)
    assert p.gep_quantity_input_path.endswith('landslide_mitigation/input_data_raw')
    assert p.gep_regions_id_col == 'ee_r264_id'
    utilities.hydrate_es_parameters(p, 'landslide_mitigation', log=lambda *a: None)
    assert p.processing_resolution == 2000
    assert p.force_run is False


def test_publish_inputs_applies_the_method_constants_but_a_caller_wins(tmp_path):
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    p.control_ratio = 3          # a caller-set value must survive the defaults layer

    tasks.publish_inputs(p)

    assert p.control_ratio == 3
    assert p.prediction_years == chain.PREDICTION_YEARS
    assert p.modeling_range == chain.MODELING_YEARS
    assert p.max_location_accuracy_m == chain.MAX_LOCATION_ACCURACY_M
    assert set(p.c_root_scenarios) == {'observed', 'full_impacts'}
    assert p.landslide_input_data_dir == p.gep_quantity_input_path
    assert p.results == {}


def test_the_shared_render_finds_this_services_qmds():
    # utilities.render_service_results looks for <service>/<service>_results.qmd beside itself
    # and RAISES when it is missing, so its absence is a run-time failure, not a quiet skip.
    service_dir = os.path.join(os.path.dirname(os.path.abspath(utilities.__file__)),
                               'landslide_mitigation')
    assert os.path.exists(os.path.join(service_dir, 'landslide_mitigation_results.qmd'))
    assert os.path.exists(os.path.join(service_dir, 'landslide_mitigation_method.qmd'))


# ---------------------------------------------------------------------------
# Input data: forest cover as a root-reinforcement weight
# ---------------------------------------------------------------------------

def test_forest_weight_lut_gives_closed_canopy_one_mosaics_half_and_the_rest_zero():
    lut = chain.forest_weight_lut()
    assert lut.shape == (256,)
    assert lut[50] == 1.0 and lut[170] == 1.0     # closed canopy, flooded tree cover
    assert lut[100] == 0.5 and lut[110] == 0.5    # mosaic tree/shrub-herbaceous
    assert lut[151] == 0.0                        # sparse tree cover, below reinforcement
    assert lut[10] == 0.0 and lut[190] == 0.0     # cropland, urban: unlisted -> zero


def test_forest_weight_from_lulc_maps_codes_and_passes_nodata_through():
    lut = chain.forest_weight_lut()
    lulc = np.array([[50, 100, 10],
                     [151, 190, 255]])

    weight = chain.forest_weight_from_lulc(lulc, lut, nodata=255)
    assert weight[0].tolist() == [1.0, 0.5, 0.0]
    assert weight[1, 0] == 0.0 and weight[1, 1] == 0.0
    assert weight[1, 2] == chain.NODATA

    # Without a nodata value the 255 pixel is just an unlisted class.
    assert chain.forest_weight_from_lulc(lulc, lut)[1, 2] == 0.0


# ---------------------------------------------------------------------------
# Preprocessing: depth intervals, soil hydraulics, discharge
# ---------------------------------------------------------------------------

def test_thickness_weighted_mean_weights_by_thickness_and_skips_absent_intervals():
    shallow = np.array([10.0, 10.0, -1.0])
    middle = np.array([20.0, -999.0, -999.0])
    deep = np.array([30.0, 30.0, -1.0])
    weights = [5, 10, 15]
    nodatas = [-1.0, -999.0, -1.0]

    out = chain.thickness_weighted_mean([shallow, middle, deep], weights, nodatas)
    # pixel 0: (10x5 + 20x10 + 30x15) / 30 = 700/30
    assert np.isclose(out[0], 700 / 30)
    # pixel 1: the middle interval is missing, so the mean is over 5 + 15 cm only.
    assert np.isclose(out[1], (10 * 5 + 30 * 15) / 20)
    # pixel 2: no interval has data -> NaN, for the caller to write as nodata.
    assert np.isnan(out[2])

    # conv_factor divides the integer scaling out before weighting.
    scaled = chain.thickness_weighted_mean([shallow, middle, deep], weights, nodatas,
                                           conv_factor=10)
    assert np.isclose(scaled[0], 70 / 30)


def test_friction_angle_from_texture_and_its_clip():
    sand = np.array([60.0, 10.0, 400.0, 0.0])
    clay = np.array([20.0, 90.0, 0.0, 400.0])
    phi = chain.friction_angle_from_texture(sand, clay)
    assert np.isclose(phi[0], 27.0)     # 25 + 40/20
    assert np.isclose(phi[1], 21.0)     # 25 - 80/20
    assert phi[2] == 40.0               # 25 + 20 -> clipped
    assert phi[3] == 15.0               # 25 - 20 -> clipped


def test_soil_cohesion_from_texture_and_its_clip():
    clay = np.array([20.0, 0.0, 2000.0, -200.0])
    org_carbon = np.array([5.0, 0.0, 0.0, 0.0])
    cohesion = chain.soil_cohesion_from_texture(clay, org_carbon)
    assert np.isclose(cohesion[0], 2 + 0.03 * 20 + 0.1 * 5)
    assert np.isclose(cohesion[1], 2.0)
    assert cohesion[2] == 50.0          # 2 + 60 -> clipped
    assert cohesion[3] == 0.0           # 2 - 6 -> clipped


def test_unit_weight_and_transmissivity_conversions():
    assert np.isclose(chain.unit_weight_from_bulk_density(np.array([1.4]))[0], 1.4 * 9.81)
    # HiHydroSoil ships K_sat x 10,000 in cm/day: 100000 -> 10 cm/day -> 0.1 m/day.
    assert np.isclose(chain.transmissivity_from_ksat(np.array([100000.0]),
                                                     np.array([2.0]))[0], 0.2)


def test_specific_discharge_spreads_annual_rain_over_the_upslope_area():
    # 365.25 mm/yr is exactly 1 mm/day = 0.001 m/day; over 2 km2 through a 1 km cell width
    # that is 0.001 x 2e6 / 1000 = 2 m2/day.
    q = chain.specific_discharge(np.array([365.25]), np.array([2.0e6]), cell_width_m=1000.0)
    assert np.isclose(q[0], 2.0)


# ---------------------------------------------------------------------------
# Stability index
# ---------------------------------------------------------------------------

def test_c_root_max_for_scenario_switches_root_cohesion_off_for_the_counterfactual():
    assert chain.c_root_max_for_scenario('observed') == chain.C_ROOT_MAX_KPA
    assert chain.c_root_max_for_scenario(0) == 0.0


def si_test_inputs():
    """Four pixels on a 45-degree slope: tan(beta) = 1 and sin(beta) cos(beta) = 0.5."""
    return dict(
        phi_deg=np.array([30.0, 30.0, 30.0, 30.0]),
        c_soil=np.array([10.0, 10.0, 10.0, 10000.0]),
        forest_share=np.array([0.5, 0.5, 0.5, 0.5]),
        gamma=np.array([10.0, 10.0, 10.0, 10.0]),
        transmissivity=np.array([4.0, 0.01, 4.0, 4.0]),
        q=np.array([math.sqrt(2), 1000.0, math.sqrt(2), math.sqrt(2)]),
        slope_deg=np.array([45.0, 45.0, 45.0, 45.0]),
        soil_depth=np.array([2.5, 2.5, 0.0, 2.5]),
    )


def test_stability_index_hand_computed_four_pixels():
    si = chain.stability_index(c_root_max=5.0, **si_test_inputs())
    friction_term = math.tan(math.radians(30))    # tan(30) / tan(45)

    # pixel 0: c_total = 10 + 0.5x5 = 12.5; denominator = 10 x 2.5 x 0.5 = 12.5 -> term 1.0.
    #          saturation = sqrt(2) / (4 x sin 45) = 0.5.
    assert np.isclose(si[0], friction_term + 1.0 - 0.5)
    # pixel 1: transmissivity is tiny, so the saturation ratio is capped at 1.
    assert np.isclose(si[1], friction_term + 1.0 - 1.0)
    # pixel 2: zero soil depth -> the cohesion denominator is zero, so the term is zero.
    assert np.isclose(si[2], friction_term - 0.5)
    # pixel 3: a huge cohesion pushes SI past the clip.
    assert si[3] == chain.SI_CLIP[1]


def test_stability_index_differs_between_scenarios_only_by_the_root_term():
    inputs = si_test_inputs()
    with_roots = chain.stability_index(c_root_max=5.0, **inputs)
    without_roots = chain.stability_index(c_root_max=0.0, **inputs)

    # pixel 0's cohesion denominator is 12.5, so the difference is 0.5 x 5 / 12.5.
    assert np.isclose(with_roots[0] - without_roots[0], 0.5 * 5 / 12.5)
    # pixel 2 has no soil, so root cohesion cannot help it.
    assert with_roots[2] == without_roots[2]
    # Roots never destabilise a slope.
    assert (with_roots >= without_roots).all()


def test_stability_index_clips_slope_and_friction_angle_before_the_trigonometry():
    inputs = si_test_inputs()

    steep = dict(inputs, phi_deg=np.full(4, 100.0))
    at_clip = dict(inputs, phi_deg=np.full(4, chain.FRICTION_ANGLE_CLIP_DEG[1]))
    assert np.allclose(chain.stability_index(c_root_max=5.0, **steep),
                       chain.stability_index(c_root_max=5.0, **at_clip))

    # A flat pixel would divide by tan(0); clipping to 0.5 degrees makes the friction term
    # enormous instead, and SI lands on the ceiling rather than on a NaN.
    flat = dict(inputs, slope_deg=np.zeros(4))
    assert np.isfinite(chain.stability_index(c_root_max=5.0, **flat)).all()
    assert chain.stability_index(c_root_max=5.0, **flat)[0] == chain.SI_CLIP[1]


# ---------------------------------------------------------------------------
# Grid coordinates, UGLC panels, estimation rows
# ---------------------------------------------------------------------------

def test_pixel_row_col_and_pixel_center_coords_round_trip():
    x = pd.Series([0.0, 1500.0, 2999.0])
    y = pd.Series([0.0, -1500.0, -2999.0])
    row, col = chain.pixel_row_col(x, y, TEST_GEOTRANSFORM)
    assert col.tolist() == [0, 15, 29]
    assert row.tolist() == [0, 15, 29]

    center_x, center_y = chain.pixel_center_coords(np.array([0, 1]), np.array([0, 2]),
                                                   TEST_GEOTRANSFORM)
    assert center_x.tolist() == [50.0, 250.0]
    assert center_y.tolist() == [-50.0, -150.0]

    back_row, back_col = chain.pixel_row_col(pd.Series(center_x), pd.Series(center_y),
                                             TEST_GEOTRANSFORM)
    assert back_row.tolist() == [0, 1] and back_col.tolist() == [0, 2]


def test_uglc_year_panels_stamps_a_disc_with_linear_distance_decay():
    # One event at the centre of pixel (2, 2) with a 100 m accuracy on a 100 m grid: the disc
    # covers that pixel and its four edge neighbours (at exactly 100 m), nothing diagonal.
    events = pd.DataFrame({'ease_x': [250.0], 'ease_y': [-250.0],
                           'accuracy_m': [100.0], 'fatality_count': [10.0]})
    binary, mortality = chain.uglc_year_panels(events, TEST_GEOTRANSFORM, x_size=5, y_size=5)

    assert binary.sum() == 5
    assert binary[2, 2] == 1 and binary[1, 2] == 1 and binary[2, 1] == 1
    assert binary[1, 1] == 0                       # 141 m away, outside the disc
    # Decay is 1 - d/r, so the centre takes the whole count and the edge neighbours take zero.
    assert np.isclose(mortality[2, 2], 10.0)
    assert np.isclose(mortality.sum(), 10.0)


def test_uglc_year_panels_accumulates_events_and_skips_unusable_ones():
    events = pd.DataFrame({
        'ease_x': [250.0, 250.0, 250.0, -100000.0],
        'ease_y': [-250.0, -250.0, -250.0, -250.0],
        'accuracy_m': [100.0, 100.0, 0.0, 100.0],       # the third has no usable accuracy
        'fatality_count': [10.0, 4.0, 99.0, 99.0],      # the fourth is off-grid
    })
    binary, mortality = chain.uglc_year_panels(events, TEST_GEOTRANSFORM, x_size=5, y_size=5)

    assert np.isclose(mortality[2, 2], 14.0)       # the two usable events add
    assert np.isclose(mortality.sum(), 14.0)       # neither skipped event contributed
    assert binary.sum() == 5

    # A zero-fatality event still marks the binary panel; it just adds no mortality.
    nonfatal = pd.DataFrame({'ease_x': [250.0], 'ease_y': [-250.0],
                             'accuracy_m': [100.0], 'fatality_count': [0.0]})
    binary_nonfatal, mortality_nonfatal = chain.uglc_year_panels(
        nonfatal, TEST_GEOTRANSFORM, x_size=5, y_size=5)
    assert binary_nonfatal.sum() == 5
    assert mortality_nonfatal.sum() == 0.0


def test_deduplicate_cases_to_pixels_keeps_one_row_per_pixel_with_the_worst_event():
    cases = pd.DataFrame({'ease_x': [250.0, 260.0, 1250.0],
                          'ease_y': [-250.0, -260.0, -250.0],
                          'fatality_count': [3.0, 7.0, 1.0]})
    out = chain.deduplicate_cases_to_pixels(cases, TEST_GEOTRANSFORM)

    assert len(out) == 2
    first = out[out['pixel_col'] == 2].iloc[0]
    assert first['ease_x'] == 250.0        # the first event's coordinates
    assert first['fatality_count'] == 7.0  # the highest count in the pixel
    assert out[out['pixel_col'] == 12].iloc[0]['fatality_count'] == 1.0


def test_case_control_rows_labels_cases_and_zeroes_the_controls():
    cases = pd.DataFrame({'ease_x': [1.0, 2.0], 'ease_y': [-1.0, -2.0],
                          'fatality_count': [5.0, 0.0]})
    out = chain.case_control_rows(cases, np.array([7.0, 8.0, 9.0]),
                                  np.array([-7.0, -8.0, -9.0]), year=2011)

    assert len(out) == 5
    assert out['case'].tolist() == [1, 1, 0, 0, 0]
    assert (out['year'] == 2011).all()
    assert out.loc[out['case'] == 0, 'fatality_count'].tolist() == [0.0, 0.0, 0.0]
    assert out.loc[out['case'] == 1, 'fatality_count'].tolist() == [5.0, 0.0]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_event_and_land_pixel_counts_restricts_the_denominator_to_si_eligible_pixels():
    binary = np.array([[1, 0, 255],
                       [1, 1, 0]])
    si = np.array([[1.0, chain.NODATA, 2.0],
                   [3.0, 4.0, 5.0]])

    # Valid = binary is not nodata AND the stability model scored the pixel.
    assert chain.event_and_land_pixel_counts(binary, 255, si, chain.NODATA) == (3, 4)
    # With no nodata declared, every pixel counts and 255 is simply not an event.
    assert chain.event_and_land_pixel_counts(binary, None, si, None) == (3, 6)


def test_prevalence_and_its_fallback():
    assert chain.prevalence(3, 4) == 0.75
    assert math.isnan(chain.prevalence(0, 0))
    assert chain.prevalence(0, 0, fallback=0.5) == 0.5


def test_case_control_intercept_correction_is_prentice_and_pyke():
    # tau = 0.5 in the sample, pi = 0.1 in the population: offset = log(1 / (1/9)) = log 9.
    assert np.isclose(chain.case_control_intercept_offset(0.5, 0.1), math.log(9))
    assert np.isclose(chain.corrected_intercept(1.0, 0.5, 0.1), 1.0 - math.log(9))
    # Oversampling cases always lifts the raw intercept, so the correction lowers it.
    assert chain.corrected_intercept(1.0, 0.5, 0.1) < 1.0
    # No oversampling (tau == pi) leaves the intercept alone.
    assert np.isclose(chain.corrected_intercept(1.0, 0.2, 0.2), 1.0)


def test_duan_smearing_factor_is_the_mean_of_exponentiated_residuals():
    residuals = np.array([0.0, math.log(2), math.log(4)])
    assert np.isclose(chain.duan_smearing_factor(residuals), (1 + 2 + 4) / 3)
    # Zero residuals mean no retransformation bias to correct.
    assert np.isclose(chain.duan_smearing_factor(np.zeros(5)), 1.0)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_logistic_and_hazard_probability():
    assert chain.logistic(0.0) == 0.5
    assert np.isclose(chain.logistic(math.log(3)), 0.75)

    # alpha + beta_si x si + beta_rain x rain = -1 + 0.5x2 + 0.25x4 = 1.
    prob = chain.hazard_probability(np.array([2.0]), np.array([4.0]),
                                    alpha=-1.0, beta_si=0.5, beta_rain=0.25)
    assert np.isclose(prob[0], 1 / (1 + math.exp(-1.0)))


def test_severity_linear_predictor_sums_the_intercept_and_four_covariates():
    params = {'Intercept': 1.0, 'population_log1p': 2.0, 'rain_max_daily': 3.0,
              'slope_degrees': 4.0, 'road_density': 5.0}
    value = chain.severity_linear_predictor(params, np.array([1.0]), np.array([1.0]),
                                            np.array([1.0]), np.array([1.0]))
    assert np.isclose(value[0], 15.0)


def test_expected_deaths_given_landslide_multiplies_the_hurdle_by_the_smeared_severity():
    zero = {'Intercept': 0.0, 'population_log1p': 0.0, 'rain_max_daily': 0.0,
            'slope_degrees': 0.0, 'road_density': 0.0}
    part_a = dict(zero, population_log1p=0.5)   # logit = 0.5 x log1p(population)
    part_b = dict(zero, population_log1p=1.0)   # log fatalities = log1p(population)

    population = np.array([math.expm1(2.0)])    # log1p(population) = 2
    ones = np.ones(1)
    deaths = chain.expected_deaths_given_landslide(part_a, part_b, 2.0, population,
                                                   ones, ones, ones)
    expected = (1 / (1 + math.exp(-1.0))) * (math.exp(2.0) * 2.0)
    assert np.isclose(deaths[0], expected)

    # Negative population (a nodata artifact) clamps to zero before log1p rather than warning.
    clamped = chain.expected_deaths_given_landslide(part_a, part_b, 1.0, np.array([-5.0]),
                                                    ones, ones, ones)
    assert np.isclose(clamped[0], 0.5 * 1.0)


def test_tile_offsets_cover_the_grid_exactly_with_truncated_edges():
    blocks = chain.tile_offsets(n_cols=5, n_rows=3, tile_size=2)
    assert blocks == [[0, 0, 2, 2], [2, 0, 2, 2], [4, 0, 1, 2],
                      [0, 2, 2, 1], [2, 2, 2, 1], [4, 2, 1, 1]]
    assert sum(n_cols * n_rows for _, _, n_cols, n_rows in blocks) == 5 * 3


def test_tile_geotransform_shifts_the_origin_and_keeps_the_pixel_size():
    reference = (100.0, 10.0, 0.0, 200.0, 0.0, -10.0)
    assert chain.tile_geotransform(reference, col_offset=3, row_offset=4) == (
        130.0, 10.0, 0, 160.0, 0, -10.0)
    assert chain.tile_geotransform(reference, 0, 0) == tuple(reference)


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

def test_vsl_usd_2019_by_iso3_applies_the_unit_multiplier_and_the_deflator():
    oecd = pd.DataFrame({'REF_AREA': ['AAA', 'BBB', 'CCC'],
                         'MEASURE_VSL': ['VSL', 'VSL', 'SOMETHING_ELSE'],
                         'OBS_VALUE': [3.0, 5.0, 99.0],
                         'UNIT_MULT': [6, 6, 6]})
    vsl = chain.vsl_usd_2019_by_iso3(oecd)

    assert set(vsl) == {'AAA', 'BBB'}              # non-VSL measures drop out
    assert np.isclose(vsl['AAA'], 3e6 / 1.050)     # UNIT_MULT 6 -> millions, then deflated
    assert np.isclose(vsl['BBB'], 5e6 / 1.050)

    # Without the measure column every row is a VSL row.
    assert set(chain.vsl_usd_2019_by_iso3(oecd.drop(columns=['MEASURE_VSL']))) == {
        'AAA', 'BBB', 'CCC'}


def test_assign_vsl_prefers_a_direct_estimate_then_the_income_group_then_the_global_figure():
    regions = pd.DataFrame({
        'iso3_r250_label': ['AAA', 'BBB', 'CCC', 'DDD'],
        'income_grp': ['1. High income: OECD', '5. Low income', 'not classified',
                       '1. High income: OECD'],
    })
    out, n_direct, n_fallback = chain.assign_vsl(regions, {'AAA': 1234.0}, 'iso3_r250_label')
    vsl = out.set_index('iso3_r250_label')['vsl_usd']

    assert n_direct == 1 and n_fallback == 3
    assert vsl['AAA'] == 1234.0                                 # its own OECD estimate
    assert np.isclose(vsl['BBB'], 1.1e6 / 1.050)                # low income -> Table 6.1 group
    assert np.isclose(vsl['CCC'], 2.7e6 / 1.050)                # unclassified -> global figure
    assert np.isclose(vsl['DDD'], 7.1e6 / 1.050)                # high-income OECD group
    assert out['vsl_usd'].notna().all()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_regression_stars_at_and_around_each_threshold():
    assert chain.regression_stars(0.0005) == '***'
    assert chain.regression_stars(0.001) == '**'      # the threshold itself is not below it
    assert chain.regression_stars(0.005) == '**'
    assert chain.regression_stars(0.01) == '*'
    assert chain.regression_stars(0.03) == '*'
    assert chain.regression_stars(0.05) == ''
    assert chain.regression_stars(np.nan) == ''


def test_publication_table_puts_the_standard_error_beneath_the_coefficient():
    table = chain.publication_table(
        coef_rows=[('A', 1.23456, 0.1, 0.0005), ('B', -2.0, np.nan, 0.2)],
        bottom_rows=[('Observations', '10')],
        dep_label='Y')

    assert table['Y'].tolist() == ['1.2346***', '(0.1000)', '-2.0000', '', '', '10']
    assert table[' '].tolist() == ['A', '', 'B', '', '', 'Observations']


def test_coefficients_by_term_pairs_each_coefficient_with_its_error_and_p_value():
    out = chain.coefficients_by_term({'a': 1.0, 'b': 2.0}, {'a': 0.5}, {'b': 0.01})
    assert out['a'][0] == 1.0 and out['a'][1] == 0.5 and math.isnan(out['a'][2])
    assert out['b'][0] == 2.0 and math.isnan(out['b'][1]) and out['b'][2] == 0.01


def test_hurdle_table_rows_shows_both_stages_and_leaves_absent_terms_blank():
    part_a = {'Intercept': (1.0, 0.5, 0.0005), 'x': (2.0, 0.25, 0.2)}
    part_b = {'Intercept': (3.0, 0.1, 0.02), 'z': (4.0, np.nan, np.nan)}
    rows = chain.hurdle_table_rows(part_a, part_b, {'Intercept': 'Intercept'}, ('A', 'B'))

    assert len(rows) == 6                                   # three terms, two rows each
    assert rows[0] == {' ': 'Intercept', 'A': '1.0000***', 'B': '3.0000*'}
    assert rows[1] == {' ': '', 'A': '(0.5000)', 'B': '(0.1000)'}
    assert rows[2] == {' ': 'x', 'A': '2.0000', 'B': ''}     # part B has no x term
    assert rows[4] == {' ': 'z', 'A': '', 'B': '4.0000'}     # part A has no z term
    assert rows[5] == {' ': '', 'A': '', 'B': ''}


def test_fatality_bin_masks_are_disjoint_and_start_at_one_death():
    fatalities = pd.Series([0.0, 1.0, 4.9, 5.0, 24.0, 25.0, 99.0, 100.0, 500.0])
    masks = chain.fatality_bin_masks(fatalities)

    assert list(masks) == ['1-5', '5-25', '25-100', '100+']
    assert [int(m.sum()) for m in masks.values()] == [2, 2, 2, 2]
    stacked = np.vstack([m.to_numpy() for m in masks.values()])
    assert (stacked.sum(axis=0) <= 1).all()             # no event lands in two bins
    assert stacked[:, 0].sum() == 0                     # a nonfatal event is in no bin
    assert stacked[:, 1:].sum() == 8                    # every event with a death is binned


def test_bucket_legend_labels_are_open_ended_at_the_top():
    labels = chain.bucket_legend_labels([0, 1, 5, float('inf')], '{:.2f}')
    assert labels == ['0.00 – 1.00', '1.00 – 5.00', '5.00+']
