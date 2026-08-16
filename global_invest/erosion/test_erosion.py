"""Unit tests for the erosion module: the GEP valuation math (folded from global_erosion_gep) and
the dependency-table normalization. Synthetic and loader-free: the FAO/WB CSV loaders are
monkeypatched so the tests pin compute_country_gep_from_country_crop's own arithmetic --
production-weighted elasticity shock, clipping, the numerical floor, GPV x share, GDP% -- which is
what the per-country GEP rests on. The SDR/biophysical chain is verified by the section-A run
against staged data, not unit-testable at this scale.
"""
from pathlib import Path

import pandas as pd
import pytest

from global_invest.erosion import erosion_functions as ef


def test_country_gep_weights_clips_and_floors(monkeypatch, tmp_path):
    # Three countries: AAA exercises the production-weighted elasticity mean; BBB the elasticity
    # clip at 1.0; CCC the tiny-positive numerical floor (MIN_SHOCK_FLOOR = 8e-10).
    dfc = pd.DataFrame({
        'ISO3': ['AAA', 'AAA', 'BBB', 'CCC'],
        'protected_production_tons': [50.0, 100.0, 25.0, 1e-10],
        'total_production_tons':     [100.0, 100.0, 100.0, 100.0],
        'share_protected_production': [0.5, 1.0, 0.25, 1e-12],
        'elasticity_used':            [0.4, 0.2, 1.5, 1.0],       # BBB's 1.5 must clip to 1.0
    })
    fao = pd.DataFrame({'iso3': ['AAA', 'BBB', 'CCC'],
                        'crop_gpv_const2019_2019': [1000.0, 400.0, 1e9]})
    gdp = pd.DataFrame({'iso3': ['AAA', 'BBB', 'CCC'],
                        'gdp_const2019_2019': [10000.0, 8000.0, 1e12]})
    monkeypatch.setattr(ef, 'load_fao_gpv_iso3_const2019_with_fallback',
                        lambda *a, **k: fao)
    monkeypatch.setattr(ef, 'load_wb_gdp_current_2019', lambda *a, **k: gdp)

    out = ef.compute_country_gep_from_country_crop(
        dfc, fao_iso3_csv=Path('unused.csv'), prices_full_csv=Path('unused.csv'),
        base_year=2019, gdp_current_2019_csv=Path('unused.csv'),
        component='combined').set_index('iso3')

    # AAA: shock = (100*0.5*0.4 + 100*1.0*0.2) / 200 = 0.2 -> GEP = 1000 * 0.2 = 200; GDP% = 2.0
    assert out.loc['AAA', 'erosion_shock_share'] == pytest.approx(0.2)
    assert out.loc['AAA', 'gep_const2019_usd'] == pytest.approx(200.0)
    assert out.loc['AAA', 'gdp_loss_pct'] == pytest.approx(2.0)
    assert out.loc['AAA', 'share_protected_production'] == pytest.approx(150.0 / 200.0)

    # BBB: elasticity 1.5 clips to 1.0 -> shock = 0.25, NOT 0.375 -> GEP = 400 * 0.25 = 100
    assert out.loc['BBB', 'erosion_shock_share'] == pytest.approx(0.25)
    assert out.loc['BBB', 'gep_const2019_usd'] == pytest.approx(100.0)

    # CCC: tiny positive shock floors at MIN_SHOCK_FLOOR (numerical, not economic)
    assert out.loc['CCC', 'erosion_shock_share'] == pytest.approx(ef.MIN_SHOCK_FLOOR)
    assert out.loc['CCC', 'gep_const2019_usd'] == pytest.approx(1e9 * ef.MIN_SHOCK_FLOOR)


def test_read_erosion_dependency_normalizes_scenario_labels(tmp_path):
    # The frozen table's labels carry a _2050 suffix and a bare 2023.0 float; the reader normalizes
    # both so the resolver sees plain scenario names (base extraction happens in the caller).
    dep = tmp_path / 'erosion_prevention_dependency.csv'
    pd.DataFrame({
        'scenario': ['below_2c_2050', 'baseline_ignore_damages_2050', '2023.0'],
        'aez18_id': [1, 1, 1], 'gtapv7_r50_label': ['usa'] * 3, 'value': [1.0, 2.0, 3.0],
    }).to_csv(dep, index=False)

    df = ef.read_erosion_dependency(dep)
    assert set(df['scenario']) == {'below_2c', 'baseline_ignore_damages', 'baseline_2023'}
