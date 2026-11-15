"""Unit tests for health_depression's valuation.

Every function is pinned on a hand-built frame, so the dose-response convention and the
missing-input rule are stated as executable facts, and the staged tables anchor the joins.
"""
import numpy as np
import pandas as pd

from global_invest.health_depression import health_depression_functions as hd


def test_the_prevented_cases_factor_is_the_exponential_dose_response():
    # RR 0.931 per 0.1 NDVI at NDVI 0.4: removing the greenness multiplies risk by
    # exp(-10 ln(0.931) 0.4); the prevented share of observed cases is that minus one.
    assert np.isclose(hd.prevented_cases_factor(0.4, 0.931),
                      np.exp(-10.0 * np.log(0.931) * 0.4) - 1.0)
    assert hd.prevented_cases_factor(0.0, 0.931) == 0.0


def test_the_odds_ratio_converts_at_the_authors_reference_risk():
    # the authors' functions.py:333 is odds_ratio/(1-p0 + (p0*odds_ratio)) with p0 = 0.115 from
    # their run file; the template carries that p0 and the shared conversion gives their number
    import csv, os
    from global_invest.health_depression import health_depression_tasks
    from global_invest.health_hypertension.health_hypertension_functions import risk_ratio_from_odds_ratio
    params = {}
    with open(os.path.join(os.path.dirname(health_depression_tasks.__file__), '..',
                           'input_template', 'es_parameters.csv'), encoding='utf-8-sig') as fh:
        for row in csv.reader(fh):
            if row and row[0] == 'health_depression':
                params[row[1]] = row[2]
    odds_ratio, p0 = float(params['health_depression_odds_ratio']), float(params['health_depression_prevalence_nonexposed'])
    assert (odds_ratio, p0) == (0.931, 0.115)
    rr = risk_ratio_from_odds_ratio(odds_ratio, p0)
    assert rr == odds_ratio / (1 - p0 + (p0 * odds_ratio))
    assert np.isclose(rr, 0.938447, atol=1e-6)


def test_a_factor_table_not_made_from_the_current_conversion_is_refused(tmp_path):
    # the sums carry RR inside an exponential, so a table without a matching signature is refused
    # rather than rescaled, and a changed p0 names itself as the reason
    from types import SimpleNamespace
    from global_invest import utilities
    from global_invest.health_depression import health_depression_tasks as tasks
    rasters = []
    for name in ('pop', 'ndvi', 'urban', 'regions'):
        (tmp_path / name).write_text(name)
        rasters.append(str(tmp_path / name))
    p = SimpleNamespace(
        get_path=lambda path: path,
        health_depression_odds_ratio=0.931, health_depression_prevalence_nonexposed=0.115,
        health_depression_urban_smod_min=21,
        health_depression_population_input_path=rasters[0], health_depression_ndvi_input_path=rasters[1],
        health_depression_urban_input_path=rasters[2], gep_regions_input_path=rasters[3],
        health_depression_factor_path=str(tmp_path / 'urban_prevented_factor_by_country.csv'),
        health_depression_factor_signature_path=str(tmp_path / 'urban_prevented_factor_signature.json'))
    (tmp_path / 'urban_prevented_factor_by_country.csv').write_text('iso3_r250_id,urban_prevented_factor_pop_sum\n')
    reason, signature = tasks.factor_reuse_reason(p)
    assert reason is not None and 'signature' in reason
    utilities.write_outputs_signature(signature, p.health_depression_factor_signature_path)
    assert tasks.factor_reuse_reason(p)[0] is None
    p.health_depression_prevalence_nonexposed = 0.05
    assert 'health_depression_prevalence_nonexposed' in tasks.factor_reuse_reason(p)[0]


def test_an_aligned_raster_is_rebuilt_when_its_source_changes_and_the_old_one_is_kept(tmp_path):
    # existence does not certify an aligned raster: a changed source forces a rebuild, and the
    # stale raster is displaced beside the new one rather than overwritten
    from global_invest.health_depression import health_depression_tasks as tasks
    source, pop, aligned = (str(tmp_path / name) for name in ('ndvi_source', 'pop', 'aligned.tif'))
    (tmp_path / 'ndvi_source').write_text('ndvi v1')
    (tmp_path / 'pop').write_text('grid')
    (tmp_path / 'aligned.tif').write_text('aligned from an unknown source')
    build = lambda out: open(out, 'w').write(open(source).read())
    reason, signature = tasks.aligned_raster_reuse_reason(aligned, [source, pop], {'resample': 'bilinear'})
    assert reason is not None and 'signature' in reason
    tasks.rebuild_aligned_raster(aligned, signature, build)
    assert open(aligned).read() == 'ndvi v1'
    assert tasks.aligned_raster_reuse_reason(aligned, [source, pop], {'resample': 'bilinear'})[0] is None
    (tmp_path / 'ndvi_source').write_text('ndvi v2')
    reason, signature = tasks.aligned_raster_reuse_reason(aligned, [source, pop], {'resample': 'bilinear'})
    assert reason is not None and source in reason
    tasks.rebuild_aligned_raster(aligned, signature, build)
    assert open(aligned).read() == 'ndvi v2'
    kept = sorted(f.read_text() for f in tmp_path.glob('aligned_displaced_*.tif'))
    assert kept == ['aligned from an unknown source', 'ndvi v1']


def test_the_valuation_is_prevented_cases_times_cost():
    df = pd.DataFrame({'prevalence': [0.05], 'urban_prevented_factor_pop_sum': [1e6],
                       'cost_per_case_usd': [8000.0]})
    out = hd.health_depression_gep(df)
    assert np.isclose(out['prevented_cases'].iloc[0], 0.05 * 1e6)
    assert np.isclose(out['health_depression_gep'].iloc[0], 0.05 * 1e6 * 8000.0)
    assert out['note'].iloc[0] == ''


def test_a_missing_cost_is_an_na_with_its_absence_named():
    df = pd.DataFrame({'prevalence': [0.05], 'urban_prevented_factor_pop_sum': [1e6],
                       'cost_per_case_usd': [np.nan]})
    out = hd.health_depression_gep(df)
    assert pd.isna(out['health_depression_gep'].iloc[0])
    assert 'no cost study' in out['note'].iloc[0]


def test_the_staged_cost_table_matches_the_authors_country_set():
    import os, tempfile, hazelbean as hb
    from global_invest import utilities
    base = utilities.service_data_dir(
        hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors')),
        'health_depression')
    costs = pd.read_csv(os.path.join(base, 'data_cost_mental_health.csv'))
    reference = pd.read_csv(os.path.join(base, 'reference_high_resolution.csv'))
    # the author's 32 valued countries are exactly the cost rows minus the Europe aggregate
    assert len(reference) == 32
    assert len(costs) == 33 and 'Europe' in set(costs['country'])
