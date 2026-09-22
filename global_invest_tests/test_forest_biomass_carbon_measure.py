"""D19: the timber resource as aboveground forest biomass carbon on scenario-specific forest cover.
Pinned: unchanged map -> zero change; conversion removes the observed stock; forest gain enters at
the zone mean (the mature-density assumption); nodata is not a conversion; the base raster marks
non-forest as NaN, not 0."""
import numpy as np
from global_invest.timber_provision import timber_provision_functions as tp

F = tp.SEALS7_FOREST_ID
NDV = 255


def _base():
    density = np.array([[10.0, 0.0, 30.0], [40.0, 50.0, np.nan]], dtype='float32')     # Mg C/ha
    ha = np.full((2, 3), 9.0, dtype='float32')
    lulc0 = np.array([[F, F, 2], [F, 3, F]], dtype='uint8')
    return density, ha, lulc0


def test_base_marks_non_forest_as_nan_and_keeps_zero_density_forest():
    density, ha, lulc0 = _base()
    base = tp.forest_biomass_carbon_base(density, ha, lulc0, lulc_ndv=NDV)
    assert base[0, 0] == 90.0 and base[0, 1] == 0.0 and base[1, 0] == 360.0
    assert np.isnan(base[0, 2]) and np.isnan(base[1, 1])
    assert base[1, 2] == 0.0                                # forest with no density data: 0, not NaN


def test_unchanged_map_gives_the_same_total():
    density, ha, lulc0 = _base()
    base = tp.forest_biomass_carbon_base(density, ha, lulc0, lulc_ndv=NDV)
    new = np.full((2, 3), 20.0 * 9.0, dtype='float32')
    on_map = tp.forest_biomass_carbon_on_map(base, new, lulc0, lulc_ndv=NDV)
    assert np.nansum(on_map) == np.nansum(base)


def test_conversion_removes_observed_stock_and_gain_enters_at_zone_mean():
    density, ha, lulc0 = _base()
    base = tp.forest_biomass_carbon_base(density, ha, lulc0, lulc_ndv=NDV)
    new = np.full((2, 3), 20.0 * 9.0, dtype='float32')
    lulc1 = lulc0.copy(); lulc1[0, 0] = 2                   # forest -> cropland: 90 leaves
    lulc1[0, 2] = F                                          # cropland -> forest: zone mean 180 enters
    on_map = tp.forest_biomass_carbon_on_map(base, new, lulc1, lulc_ndv=NDV)
    assert on_map[0, 0] == 0.0 and on_map[0, 2] == 180.0
    assert np.nansum(on_map) == np.nansum(base) - 90.0 + 180.0


def test_nodata_is_not_a_conversion():
    density, ha, lulc0 = _base()
    base = tp.forest_biomass_carbon_base(density, ha, lulc0, lulc_ndv=NDV)
    new = np.full((2, 3), 180.0, dtype='float32')
    lulc1 = lulc0.copy(); lulc1[0, 0] = NDV; lulc1[0, 2] = NDV
    on_map = tp.forest_biomass_carbon_on_map(base, new, lulc1, lulc_ndv=NDV)
    assert on_map[0, 0] == 90.0                              # kept
    assert on_map[0, 2] == 0.0                               # never forest, nodata: nothing enters


def test_carbon_to_dry_biomass_factor_is_the_ipcc_default():
    assert tp.CARBON_FRACTION_OF_DRY_MATTER == 0.47


def test_task_switches_on_the_resource_measure_and_signs_it():
    """The measure and the carbon layer are timber_provision_* attributes, so the reuse signature
    sees them; the biomass branch reads the aboveground layer attribute, never the total-carbon one."""
    import inspect
    from global_invest.timber_provision import timber_provision_tasks as t
    src = inspect.getsource(t.timber_provision_shock)
    assert "getattr(p, 'timber_provision_resource_measure'" in src
    assert "'aboveground_carbon'" in src and "_write_biomass_base_and_new_forest(" in src
    helper = inspect.getsource(t._write_biomass_base_and_new_forest)
    assert 'p.timber_provision_biomass_carbon_density_path' in helper
    assert 'total_carbon_density' not in helper
    assert src.index("utilities.reuse_reason(") < src.index("_write_biomass_base_and_new_forest(")


# --- the summed-quantity aggregation (D19 option 1, 22 Sep 2026) -------------------------------
import pandas as pd
from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf

LABELS = {1: ('AEZ1', 'VNM'), 2: ('AEZ2', 'VNM'), 3: ('AEZ3', 'KOR'), 4: ('AEZ4', 'XXX')}


def _rows(scen_by_year, base):
    return tcf.summed_shock_rows({'current_policies': scen_by_year}, base, LABELS, 2023, 'FRS', log=lambda *a: None)


def test_region_shock_is_the_ratio_of_sums_not_a_mean_of_ratios():
    base = pd.Series({1: 100.0, 2: 0.0, 3: 50.0, 4: 1e4})          # zone 2 holds nothing in the base year
    scen = {2050: pd.Series({1: 50.0, 2: 500.0, 3: 25.0, 4: 1e4})}  # zone 2 afforested: +500
    rows = pd.DataFrame(_rows(scen, base))
    vnm = rows[(rows.REG == 'VNM') & (rows.year == 2050)].shock_pct.unique()
    assert len(vnm) == 1 and abs(vnm[0] - (550 / 100 - 1) * 100) < 1e-9     # +450 %, from the sums
    kor = rows[(rows.REG == 'KOR') & (rows.year == 2050)].shock_pct.unique()
    assert abs(kor[0] - (-50.0)) < 1e-9
    assert rows[(rows.REG == 'VNM') & (rows.year == 2023)].shock_pct.abs().max() == 0.0


def test_every_zone_of_a_region_carries_the_regional_value():
    base = pd.Series({1: 100.0, 2: 100.0, 3: 50.0, 4: 1e4})
    rows = pd.DataFrame(_rows({2050: pd.Series({1: 60.0, 2: 40.0, 3: 50.0, 4: 1e4})}, base))
    vnm = rows[(rows.REG == 'VNM') & (rows.year == 2050)]
    assert set(vnm.ENDW) == {'AEZ1', 'AEZ2'} and vnm.shock_pct.nunique() == 1
    assert abs(vnm.shock_pct.iloc[0] - (-50.0)) < 1e-9


def test_region_without_base_year_quantity_is_reported_not_shocked():
    base = pd.Series({1: 100.0, 2: 0.0, 3: 0.0, 4: 1e4})            # KOR has nothing at all
    said = []
    rows = pd.DataFrame(tcf.summed_shock_rows({'current_policies': {2050: pd.Series({1: 90.0, 2: 0.0, 3: 900.0, 4: 1e4})}},
                                              base, LABELS, 2023, 'FRS', log=said.append))
    assert 'KOR' not in set(rows.REG)
    assert said and 'KOR' in said[0]


def test_negligible_region_is_excluded_by_share_of_the_world_total():
    base = pd.Series({1: 1e5, 2: 0.0, 3: 0.001, 4: 1e9})            # VNM 1e-4 of the world, KOR 1e-12
    rows = pd.DataFrame(_rows({2050: pd.Series({1: 1e5, 2: 0.0, 3: 500.0, 4: 1e9})}, base))
    assert 'KOR' not in set(rows.REG) and 'VNM' in set(rows.REG)
