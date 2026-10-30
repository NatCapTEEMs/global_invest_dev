"""Unit tests for the flood module.

The account's science is the counterfactual: two expected-damage integrals, their
difference, and the share that difference is of the degraded world. These run on
a handful of countries and a few return periods rather than on a global grid,
because the functions they touch take arrays and frames and return them.

No test here replaces a file reader: `flood_functions` holds the equations over
arrays and frames and opens nothing, so the science is tested directly and the
file handling stays in `flood_tasks`.

The InVEST-scale work is not covered: the amplification mosaic takes two weeks of
flow accumulation and is verified against staged data, and the depth-damage
functions are a published input rather than something this account computes.
"""

import numpy as np
import pandas as pd
import re

import tempfile
import hazelbean as hb
import pytest

from global_invest import utilities
from global_invest.flood import flood_functions as ff


# ---------------------------------------------------------------------------
# The prevention share: what the account reports, and what it refuses to report.
# ---------------------------------------------------------------------------
def test_prevention_share_is_the_fraction_of_potential_damage_avoided():
    # A world without ecosystems takes 100 of damage; with them it takes 90. The
    # service is the 10, and the share is a tenth of what could have happened --
    # not a tenth of what did.
    assert ff.prevention_share(90.0, 100.0) == pytest.approx(0.10)
    assert ff.prevention_share(50.0, 100.0) == pytest.approx(0.50)
    # No difference between the worlds is no service, which is a finding.
    assert ff.prevention_share(100.0, 100.0) == pytest.approx(0.0)


def test_a_country_with_no_potential_damage_has_no_share_rather_than_a_zero_one():
    # Zero would say ecosystems prevent nothing there, which is a claim about the
    # ecosystems. Missing says there is no damage to take a share of, which is a
    # fact about the exposure, and it keeps the country out of a median.
    assert np.isnan(ff.prevention_share(0.0, 0.0))
    assert np.isnan(ff.prevention_share(np.array([0.0]), np.array([0.0]))[0])


def test_degradation_cannot_reduce_damage():
    # A degraded world with less damage than the current one is a routing or
    # alignment fault rather than a result. The share floors at zero so it cannot
    # enter a total as a negative service.
    assert ff.prevention_share(110.0, 100.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The published table: the four columns every service shares.
# ---------------------------------------------------------------------------
def test_country_gep_is_a_difference_and_carries_the_shared_columns():
    # AAA has a real service; NOGDP has one but no denominator; TINY's difference
    # is below the floor and is numerical noise from the differencing.
    ead = pd.DataFrame({
        'iso3': ['AAA', 'NOGDP', 'TINY'],
        'ead_current_const2019_usd':  [900.0, 900.0, 1000.0],
        'ead_degraded_const2019_usd': [1000.0, 1000.0, 1000.0000004],
    })
    gdp = pd.DataFrame({'iso3': ['AAA', 'TINY'],
                        'gdp_const2019_2019': [10000.0, 1e9]})

    out = ff.country_gep(ead, gdp, component='bare').set_index('iso3')

    # AAA: 1000 - 900 = 100 of prevented damage, a tenth of the potential, and
    # one per cent of output.
    assert out.loc['AAA', 'gep_const2019_usd'] == pytest.approx(100.0)
    assert out.loc['AAA', 'flood_prevention_share'] == pytest.approx(0.10)
    assert out.loc['AAA', 'gdp_loss_pct'] == pytest.approx(1.0)

    # A missing GDP gives no percentage rather than an infinite one, and the
    # value stays visible so the gap reads as a missing denominator.
    assert out.loc['NOGDP', 'gep_const2019_usd'] == pytest.approx(100.0)
    assert pd.isna(out.loc['NOGDP', 'gdp_loss_pct'])

    # Sub-dollar differences are noise from subtracting two large integrals.
    assert out.loc['TINY', 'gep_const2019_usd'] == 0.0

    # The four shared columns, in the order every service publishes them.
    assert list(out.reset_index().columns[-4:]) == [
        'flood_prevention_share', 'gdp_const2019_2019',
        'gep_const2019_usd', 'gdp_loss_pct']
    assert (out['component'] == 'bare').all()


def test_the_two_counterfactuals_stack_rather_than_sum():
    # Bare soil and in-situ degradation are alternative answers to different
    # questions -- what ecosystems contribute in total, and what realistic
    # degradation would cost. Adding them would count the same prevented damage
    # twice under two assumptions about how much cover is lost.
    ead = pd.DataFrame({'iso3': ['AAA'],
                        'ead_current_const2019_usd': [900.0],
                        'ead_degraded_const2019_usd': [1000.0]})
    gdp = pd.DataFrame({'iso3': ['AAA'], 'gdp_const2019_2019': [10000.0]})
    bare = ff.country_gep(ead, gdp, 'bare')
    insitu = ff.country_gep(ead.assign(ead_degraded_const2019_usd=[950.0]), gdp, 'insitu')

    out = ff.combine_components(bare, insitu)
    assert len(out) == 2
    assert set(out['component']) == {'bare', 'insitu'}
    assert out.set_index('component').loc['bare', 'gep_const2019_usd'] == pytest.approx(100.0)
    assert out.set_index('component').loc['insitu', 'gep_const2019_usd'] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Expected annual damage: the integral, and both of its boundary assumptions.
# ---------------------------------------------------------------------------
def test_ead_integrates_damage_over_exceedance_probability():
    # Two return periods, 10 and 100 years, with damages of 100 and 200. The
    # trapezoid over p in [0.01, 0.1] is 0.5*(100+200)*(0.1-0.01) = 13.5, plus
    # the tail from p=0.01 to p=0 holding 200, which is 200*0.01 = 2.
    ead, pts, msgs, _ = ff.compute_ead_from_points(
        np.array([10.0, 100.0]), np.array([100.0, 200.0]), tail_mode='flat')
    assert ead == pytest.approx(13.5 + 2.0)


def test_the_rare_tail_is_held_flat_rather_than_falling_away_to_zero():
    # Both modes integrate the band beyond the rarest modelled event; they differ
    # in what they put at zero frequency. "flat" holds the 100-year damage of 200
    # there, so the band contributes a rectangle, 200 * 0.01 = 2. "zero" puts
    # nothing there, so the band contributes a triangle, half of that.
    #
    # Neither truncates. Truncating would drop the band entirely and assert that
    # floods rarer than the rarest modelled one cause no damage at all; holding
    # the damage flat asserts they cause at least as much as the rarest one, which
    # is bounded and errs in the safe direction. The reference implementation
    # makes the same choice.
    flat, _, _, _ = ff.compute_ead_from_points(
        np.array([10.0, 100.0]), np.array([100.0, 200.0]), tail_mode='flat')
    zero, _, _, _ = ff.compute_ead_from_points(
        np.array([10.0, 100.0]), np.array([100.0, 200.0]), tail_mode='zero')
    assert flat > zero
    assert flat - zero == pytest.approx(0.5 * 200.0 * 0.01)


def test_anchoring_at_probability_one_inflates_the_total():
    # Adding a zero-damage point at p=1 interpolates across the band between the
    # 10-year event and every rainfall, which is most of probability space and
    # carries no data. It is available for comparison and is not the default:
    # globally it raises expected annual damage by a factor of about five.
    off, _, _, _ = ff.compute_ead_from_points(
        np.array([10.0, 100.0]), np.array([100.0, 200.0]), add_p1_zero=False)
    on, _, _, _ = ff.compute_ead_from_points(
        np.array([10.0, 100.0]), np.array([100.0, 200.0]), add_p1_zero=True)
    assert on > off


def test_the_natural_capital_split_is_by_return_period_not_by_provider():
    # NC is damage from events rarer than the local design standard; NC+ is what
    # the defences prevent. A country protected to 20 years keeps only the tail
    # beyond p=1/20 in NC. This partitions exposure by engineered protection and
    # says nothing about how much of it ecosystems contribute.
    ead, _, _, nc = ff.compute_ead_from_points(
        np.array([10.0, 20.0, 100.0]), np.array([100.0, 150.0, 200.0]),
        protection_rp=20.0)
    assert nc < ead
    # An undefended country has no protection to exceed, so the whole of its
    # exposure is beyond the standard rather than none of it. FLOPROS records a
    # zero standard for 1,154 of its 4,650 polygons, so this is not an edge case:
    # treating it as missing would drop every unprotected country from the column
    # where the natural-capital share should be largest.
    ead0, _, _, nc0 = ff.compute_ead_from_points(
        np.array([10.0, 100.0]), np.array([100.0, 200.0]), protection_rp=0.0)
    assert nc0 == pytest.approx(ead0)


# ---------------------------------------------------------------------------
# Depth to damage.
# ---------------------------------------------------------------------------
def test_damage_interpolates_between_tabulated_depths():
    # The published functions are points on a continuous curve, and a cell at
    # 0.75 m sits between the 0.5 and 1.0 m entries.
    xs = np.array([0.0, 0.5, 1.0, 2.0])
    ys = np.array([0.0, 40.0, 60.0, 80.0])
    assert ff.interp_damage_per_m2(np.array([0.75]), xs, ys)[0] == pytest.approx(50.0)
    # Depths beyond the table clamp rather than extrapolating into a region the
    # survey did not measure.
    assert ff.interp_damage_per_m2(np.array([9.0]), xs, ys)[0] == pytest.approx(80.0)


def test_the_banded_alternative_rounds_up_and_is_not_the_default():
    # The reference implementation bands depth into nine intervals and charges
    # each cell the damage at the first boundary at or above its own depth, so a
    # cell under 0.7 m pays the 1.0 m rate. Reproduced for comparison only:
    # rounding up overstates the level of damage while suppressing the difference
    # between the two worlds, because amplification moves most cells by less than
    # a band width. Run globally it raises damage 12% and lowers GEP 7%.
    banded = ff._band_depth_inca(np.array([0.05, 0.3, 0.7, 1.2, 7.0], dtype='float32'))
    assert banded.tolist() == pytest.approx([0.25, 0.5, 1.0, 1.5, 5.5])


def test_the_flood_module_does_not_import_the_source_repo():
    """flood_tasks loads without the source repo's `flood_paths` module.

    The source repo resolves every path from a GEP_FLOOD_ROOT environment variable through
    `flood_paths`, which imports only from inside that checkout. This blocks that import and asserts
    the module still loads. Pollination carries the same guard against crop_benefits.
    """
    import importlib
    import sys

    class Blocker:
        def find_module(self, name, path=None):
            return self if name == 'flood_paths' or name.startswith('flood_paths.') else None

        def load_module(self, name):
            raise ImportError("No module named '%s'" % name)

    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    cached = {name: sys.modules.pop(name) for name in list(sys.modules)
              if name.startswith('global_invest.flood')}
    try:
        module = importlib.import_module('global_invest.flood.flood_tasks')
        assert module is not None
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(cached)


def test_every_flood_input_is_an_es_parameters_reference():
    """Inputs resolve through get_path like every other service, not from a directory convention.

    Locations derived by hand from one root make a machine reproduce a layout rather than say where
    its data is, and hide the run's inputs from anything that reads the definitions.
    """
    import csv
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, 'input_template', 'es_parameters.csv')
    keys = {row[1] for row in csv.reader(open(path, encoding='utf-8-sig')) if len(row) > 1}
    for name in ('flood_lulc_path', 'flood_sda_raster_path', 'flood_depth_aligned_path',
                 'flood_spa_path', 'flood_canonical_eur_path', 'flood_cn_table_path'):
        assert name in keys, '%s must be an es_parameters row' % name
    assert 'flood_root_dir' not in keys, 'the hand-derived root is gone; inputs are references now'


def test_the_staged_pipeline_inputs_resolve_on_a_fresh_project_and_a_machine_keeps_its_own():
    """The seven staged inputs ship as references to base_data's flood/input/pipeline_inputs, so a
    fresh project resolves every one; a project copy that already names another location keeps
    it, and a value set on p before hydration wins, so a machine with its own layout is unmoved."""
    import csv
    import os
    from global_invest.flood import flood_tasks
    staged = ('flood_lulc_path', 'flood_pop_path', 'flood_depth_aligned_path', 'flood_spa_path',
              'flood_spa_ratio_path', 'flood_canonical_eur_path', 'flood_currency_factors_path')
    root = tempfile.mkdtemp()
    p = hb.ProjectFlow(project_dir=os.path.join(root, 'gep_flood_control'))
    flood_tasks.publish_inputs(p)
    if not hb.path_exists(str(p.flood_lulc_path)):
        pytest.skip('the staged flood pipeline inputs are not on this machine')
    missing = [name for name in staged if not hb.path_exists(str(getattr(p, name)))]
    assert not missing, 'unresolved on a fresh project: %s' % missing

    local = os.path.join(p.input_dir, 'es_parameters.csv')
    rows = list(csv.reader(open(local, encoding='utf-8-sig')))
    for row in rows:
        if len(row) > 2 and row[1] == 'flood_lulc_path':
            row[2] = 'global_invest/flood/lulc/machine_layout.tif'
    with open(local, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    q = hb.ProjectFlow(project_dir=p.project_dir)
    q.flood_spa_path = '/set/by/caller.tif'
    flood_tasks.publish_inputs(q)
    assert str(q.flood_lulc_path).endswith('global_invest/flood/lulc/machine_layout.tif')
    assert q.flood_spa_path == '/set/by/caller.tif'
    assert str(q.flood_pop_path) == str(p.flood_pop_path)


def test_every_flood_module_imports():
    """Every flood module imports, including the run file no other test executes.

    The other tests exercise flood_tasks and flood_functions, never the entry point, so a stale
    import there would surface only on a real run. An import is the cheapest assertion that
    catches it.
    """
    import importlib
    for name in ('flood_functions', 'flood_initialize', 'flood_tasks', 'run_flood'):
        importlib.import_module('global_invest.flood.%s' % name)


def test_the_run_file_does_not_configure_the_service():
    """run_flood.py sets no p.flood_* attribute: everything the module needs comes from es_parameters.

    Hardcoded copies in the run file are not merely redundant: they let the valuation run on a
    machine where input_template is absent and no config hydrates, while the GEP chain, which does
    read config, returns $0 for every country and reports success. H14 in the harmonization gate
    enforces this across the library; this asserts it for the flood run file.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    source = open(os.path.join(here, 'run_flood.py'), encoding='utf-8').read()
    assignments = re.findall(r'^\s+p\.flood_[a-z_0-9]+\s*=', source, re.M)
    assert not assignments, 'run_flood.py configures the service: %s' % assignments
    assert 'def set_flood_paths' not in source


# ---------------------------------------------------------------------------
# Section D end to end, on frames small enough to check by hand.
#
# These execute the real 4A, 4C and 4D code paths on a handful of countries and
# return periods, so a fault in them shows here rather than on a cluster run.
# ---------------------------------------------------------------------------
def _flood_settings(p, directory):
    """The es_parameters values Section D reads, on a bare object."""
    import os
    p.flood_valuation_country_dir = os.path.join(directory, 'countries')
    p.flood_global_export_dir = os.path.join(directory, '_global')
    p.flood_currency_audit_dir = os.path.join(p.flood_global_export_dir, '_currency_audit')
    p.flood_ead_tail_mode = 'flat'
    p.flood_ead_add_p1_zero = False
    p.flood_ead_enforce_monotone = False
    p.flood_ead_write_points = False
    p.flood_apply_service_flow = False
    p.flood_report_protection_split = False
    p.flood_fill_missing_zero = True
    p.flood_region_col = 'region_wb'
    p.flood_set_pasture_equal_crop = True
    p.flood_protection_path = None
    p.flood_protection_evidence_path = None
    os.makedirs(p.flood_valuation_country_dir, exist_ok=True)
    os.makedirs(p.flood_global_export_dir, exist_ok=True)
    return p


def _write_country_damage(p, iso3, rows):
    """One country's damage-by-return-period table, the input 4C integrates."""
    import os
    d = os.path.join(p.flood_valuation_country_dir, iso3)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(d, 'damage_by_return_period_usd2019.csv'), index=False)
    return d


def test_expected_annual_damage_integrates_each_country_and_writes_one_file_per_folder(tmp_path):
    from types import SimpleNamespace
    from global_invest.flood import flood_tasks as ft

    p = _flood_settings(SimpleNamespace(), str(tmp_path))
    # Two return periods, 10 and 100 years, damages 100 and 200. The trapezoid over
    # p in [0.01, 0.1] is 0.5*(100+200)*0.09 = 13.5, plus the flat tail 200*0.01 = 2.
    _write_country_damage(p, 'AAA', [{'rp': 10, 'damage_total_usd2019': 100.0},
                                     {'rp': 100, 'damage_total_usd2019': 200.0}])
    _write_country_damage(p, 'BBB', [{'rp': 10, 'damage_total_usd2019': 0.0},
                                     {'rp': 100, 'damage_total_usd2019': 50.0}])

    status = ft.compute_ead_by_country(p)

    assert set(status['iso3']) == {'AAA', 'BBB'}
    assert (status['status'] == 'ok').all()
    aaa = float(status.set_index('iso3').loc['AAA', 'ead_usd2019'])
    assert aaa == pytest.approx(13.5 + 2.0)

    # One file per ISO3 folder, under the name that says what it holds.
    import os
    written = os.path.join(p.flood_valuation_country_dir, 'AAA',
                           'expected_annual_damage_usd2019.csv')
    assert os.path.exists(written)
    assert float(pd.read_csv(written)['ead_usd2019'].iloc[0]) == pytest.approx(aaa)


def test_a_country_with_no_damage_table_stops_the_run_rather_than_counting_as_zero(tmp_path):
    """A country that produced no damage table has no EAD, which is not an EAD of zero.

    A step 4B that dies for a country (an out-of-memory kill, for one) leaves its folder without a
    table. Recording zero there would cut the global total by that country's value with nothing
    raising.
    """
    import os
    from types import SimpleNamespace
    from global_invest.flood import flood_tasks as ft

    p = _flood_settings(SimpleNamespace(), str(tmp_path))
    _write_country_damage(p, 'AAA', [{'rp': 10, 'damage_total_usd2019': 100.0},
                                     {'rp': 100, 'damage_total_usd2019': 200.0}])
    os.makedirs(os.path.join(p.flood_valuation_country_dir, 'ZZZ'), exist_ok=True)  # no table

    with pytest.raises(ValueError) as raised:
        ft.compute_ead_by_country(p)
    assert 'ZZZ' in str(raised.value)
    assert 'missing_damage_by_return_period' in str(raised.value)

    # and the row it wrote for that country carries no number, rather than a zero
    written = pd.read_csv(os.path.join(p.flood_valuation_country_dir, 'ZZZ',
                                       'expected_annual_damage_usd2019.csv'))
    assert pd.isna(written['ead_usd2019'].iloc[0])


def test_the_damage_tables_convert_eur2010_to_usd2019_and_give_pasture_the_cropland_curve(tmp_path):
    """Step 4A on the wide canonical table.

    Every value takes one combined factor, the 2010 FX rate times the US inflator to 2019.
    Pasture has no curve of its own in the JRC table, so it takes cropland's.

    This is the WIDE input shape, where the depth columns are ALREADY absolute EUR/m2.
    The other shape -- a fractional curve per region times a max damage per country -- is
    build_canonical_from_components, tested below, and 4A does not call it. Which shape the
    real canonical file has decides which is right, and getting it wrong scales every damage
    by the max-damage value.
    """
    import os
    from types import SimpleNamespace
    from global_invest.flood import flood_tasks as ft

    p = _flood_settings(SimpleNamespace(), str(tmp_path))
    p.flood_canonical_eur_path = os.path.join(str(tmp_path), 'canonical_eur.csv')
    p.flood_currency_factors_path = os.path.join(str(tmp_path), 'factors.csv')
    for name in ('flood_damage_long_path', 'flood_damage_wide_table_path',
                 'flood_sda_damage_long_path', 'flood_sda_damage_wide_path'):
        setattr(p, name, os.path.join(str(tmp_path), name + '.csv'))

    pd.DataFrame([{'name': 'fx_usd_per_eur_2010_avg', 'value': 1.3260896},
                  {'name': 'inflator_us_2010_to_2019', 'value': 1.1600607}]
                 ).to_csv(p.flood_currency_factors_path, index=False)
    pd.DataFrame([
        {'ISO3': 'AAA', 'JRC_Region': 'R1', 'LandType': 'Commercial',
         'Max_Damage_Euro_per_m2': 100.0, '0m': 0.0, '1m': 40.0},
        {'ISO3': 'AAA', 'JRC_Region': 'R1', 'LandType': 'Agriculture',
         'Max_Damage_Euro_per_m2': 10.0, '0m': 0.0, '1m': 4.0},
    ]).to_csv(p.flood_canonical_eur_path, index=False)

    ft.build_damage_tables(p)

    combined = 1.3260896 * 1.1600607
    long = pd.read_csv(p.flood_damage_long_path)
    assert (long['currency'] == 'USD2019').all()
    at_1m = long[long['depth_m'] == 1.0]
    assert len(at_1m) > 0
    # 40 EUR/m2 at 1 m, converted once by the combined factor and by nothing else.
    assert at_1m['damage_per_m2'].max() == pytest.approx(40.0 * combined, rel=1e-9)

    sda_long = pd.read_csv(p.flood_sda_damage_long_path)
    type_col = 'sda_type' if 'sda_type' in sda_long.columns else sda_long.columns[1]
    types = set(sda_long[type_col])
    assert 'crop' in types and 'pasture' in types, types
    crop = sda_long[sda_long[type_col] == 'crop'].sort_values('depth_m')['damage_per_m2'].tolist()
    past = sda_long[sda_long[type_col] == 'pasture'].sort_values('depth_m')['damage_per_m2'].tolist()
    assert crop == past, 'pasture takes cropland curve when flood_set_pasture_equal_crop is on'


_AUTHOR_MULTIPLIER = 1.3260896 * 1.1600606949798336   # rebuild_damage_table.py EUR2010_TO_USD2019
_DEPTH_COLS = ['0m', '0.5m', '1m', '1.5m', '2m', '3m', '4m', '5m', '6m']


def _landtype_row(iso3, land_type, scale):
    """One source row: a curve rising with depth, plus a 2.5 m column the rebuild must ignore."""
    row = {'Country': iso3, 'ISO3': iso3, 'Continent': 'X', 'LandType': land_type,
           'Max_Damage_Euro_per_m2': 6.0 * scale}
    for i, col in enumerate(_DEPTH_COLS):
        row[col] = float(i) * scale
    row['2.5m'] = 999.0
    return row


def test_the_rebuilt_damage_table_follows_the_authors_recipe():
    """The authors' rebuild at its defaults: urban land takes the Residential curve, cropland the
    Agriculture curve converted from EUR per hectare, pasture cropland's, and a land type the
    country lacks is a zero curve. Commercial and Industrial do not enter, and no roads class is
    written."""
    src = pd.DataFrame([_landtype_row('AAA', 'Residential', 10.0),
                        _landtype_row('AAA', 'Commercial', 3.0),
                        _landtype_row('AAA', 'Agriculture', 50_000.0),
                        _landtype_row('bbb', 'Residential', 2.0)])
    out = ff.sda_curves_from_landtype_depth_table(src, _AUTHOR_MULTIPLIER,
                                                  [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6])

    assert list(out.columns) == ['iso3', 'sda_type'] + _DEPTH_COLS
    assert list(zip(out['iso3'], out['sda_type'])) == [
        ('AAA', 'artif'), ('AAA', 'crop'), ('AAA', 'pasture'),
        ('BBB', 'artif'), ('BBB', 'crop'), ('BBB', 'pasture')]
    row = out.set_index(['iso3', 'sda_type'])
    for i, col in enumerate(_DEPTH_COLS):
        assert row.loc[('AAA', 'artif'), col] == float(i) * 10.0 * _AUTHOR_MULTIPLIER
        assert row.loc[('AAA', 'crop'), col] == float(i) * 50_000.0 / 10_000.0 * _AUTHOR_MULTIPLIER
        assert row.loc[('AAA', 'pasture'), col] == row.loc[('AAA', 'crop'), col]
        assert row.loc[('BBB', 'crop'), col] == 0.0          # no Agriculture row: zero curve
        assert row.loc[('BBB', 'pasture'), col] == 0.0


def test_a_rebuilt_table_with_no_urban_damage_or_a_missing_depth_is_refused():
    zero_urban = pd.DataFrame([_landtype_row('AAA', 'Residential', 0.0),
                               _landtype_row('AAA', 'Agriculture', 1.0)])
    with pytest.raises(ValueError, match='artif'):
        ff.sda_curves_from_landtype_depth_table(zero_urban, 1.0, [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6])
    no_6m = pd.DataFrame([_landtype_row('AAA', 'Residential', 1.0)]).drop(columns=['6m'])
    with pytest.raises(ValueError, match='6m'):
        ff.sda_curves_from_landtype_depth_table(no_6m, 1.0, [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6])


def test_the_rebuild_checks_the_source_hash_and_records_its_provenance(tmp_path):
    """A workbook that is not the recorded one stops the rebuild before anything is written."""
    import json
    import os
    from global_invest.flood import flood_tasks as ft
    src_path = os.path.join(str(tmp_path), 'country_landtype_depth_damage.xlsx')
    pd.DataFrame([_landtype_row('AAA', 'Residential', 10.0),
                  _landtype_row('AAA', 'Agriculture', 10_000.0)]).to_excel(src_path, index=False)
    out_path = os.path.join(str(tmp_path), 'rebuilt.csv')

    with pytest.raises(ValueError, match='sha256'):
        ft.rebuild_sda_damage_table(src_path, '0' * 64, _AUTHOR_MULTIPLIER, out_path)
    assert not os.path.exists(out_path)

    ft.rebuild_sda_damage_table(src_path, utilities.sha256_file(src_path), _AUTHOR_MULTIPLIER,
                                out_path)
    # round_trip parsing reads back the float64 the writer wrote; the default parser can differ
    # in the last bit.
    table = pd.read_csv(out_path, float_precision='round_trip').set_index(['iso3', 'sda_type'])
    assert table.loc[('AAA', 'artif'), '6m'] == 8 * 10.0 * _AUTHOR_MULTIPLIER
    assert table.loc[('AAA', 'crop'), '6m'] == 8 * 10_000.0 / 10_000.0 * _AUTHOR_MULTIPLIER
    curves = ft.load_damage_table_wide(out_path)       # the reader Section D uses (float32)
    assert curves[('AAA', 'artif')][1][-1] == pytest.approx(80.0 * _AUTHOR_MULTIPLIER, rel=1e-6)
    record = json.load(open(os.path.splitext(out_path)[0] + '_provenance.json'))
    assert record['source_sha256'] == utilities.sha256_file(src_path)
    assert record['table_sha256'] == utilities.sha256_file(out_path)
    assert (record['rows'], record['countries'], record['artif_curves_nonzero']) == (3, 1, 1)


def test_section_d_reads_the_rebuilt_table_only_when_4a_is_skipped():
    """A path es_parameters names wins. With 4A skipped Section D reads the rebuilt table, and
    with 4A running it reads 4A's own output, so the rebuild can never be overwritten by 4A."""
    import os
    from types import SimpleNamespace
    from global_invest.flood import flood_tasks as ft
    base = dict(flood_global_export_dir='/run/_global',
                flood_rebuilt_sda_damage_wide_path='/run/rebuild/rebuilt.csv')

    p = SimpleNamespace(flood_skip_damage_tables=True, **base)
    assert ft.resolve_sda_damage_wide_path(p) == '/run/rebuild/rebuilt.csv'
    p = SimpleNamespace(flood_skip_damage_tables=False, **base)
    assert ft.resolve_sda_damage_wide_path(p) == os.path.join('/run/_global',
                                                              'damage_by_sda_usd2019_wide.csv')
    p = SimpleNamespace(flood_skip_damage_tables=True, flood_sda_damage_wide_path='/named.csv',
                        **base)
    assert ft.resolve_sda_damage_wide_path(p) == '/named.csv'


def _es_parameter(name):
    import os
    from global_invest.flood import flood_tasks as ft
    params = pd.read_csv(os.path.join(os.path.dirname(ft.__file__), '..', 'input_template',
                                      'es_parameters.csv'), encoding='utf-8-sig')
    values = params.loc[(params['service'] == 'flood') & (params['parameter'] == name), 'value']
    assert len(values) == 1, name
    return str(values.iloc[0])


def test_es_parameters_configures_the_authors_damage_table():
    """The authors' configuration (method note 11.6): 4A skipped, the land-type workbook named
    with its hash, and no roads class, which this table does not price."""
    assert _es_parameter('flood_skip_damage_tables') == 'true'
    assert _es_parameter('flood_landtype_depth_damage_path').endswith(
        'flood_damage/country_landtype_depth_damage.xlsx')
    assert re.fullmatch(r'[0-9a-f]{64}', _es_parameter('flood_landtype_depth_damage_sha256'))
    assert _es_parameter('flood_use_roads') == 'false'


def test_the_staged_workbook_reproduces_the_authors_checks(tmp_path):
    """Method note 11.7: artif non-zero at 6 m in 196 of 214 countries and Belgium's at USD2019
    246.67 per m2 (EUR2010 160.35). Every one of the 5,778 values also equals a recomputation that
    walks the workbook row by row, independently of the pivot the rebuild uses, both in memory and
    as written to disk and read back with round_trip parsing.

    Zero cropland curves come from two causes that are kept apart: 17 countries have no
    Agriculture row, and the Faroe Islands' Agriculture row is zero, so 18 cropland curves are zero.
    """
    import os
    from global_invest.flood import flood_tasks as ft
    root = hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors')).base_data_dir
    path = os.path.join(root, _es_parameter('flood_landtype_depth_damage_path'))
    if not os.path.exists(path):
        pytest.skip('the staged flood land-type workbook is not on this machine')
    assert utilities.sha256_file(path) == _es_parameter('flood_landtype_depth_damage_sha256')
    src = pd.read_excel(path, sheet_name=0)
    out = ff.sda_curves_from_landtype_depth_table(src, _AUTHOR_MULTIPLIER,
                                                  [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6])

    assert len(out) == 642 and out['iso3'].nunique() == 214
    artif = out[out['sda_type'] == 'artif'].set_index('iso3')
    assert int((artif[_DEPTH_COLS].abs().sum(axis=1) > 0).sum()) == 196
    assert artif.loc['BEL', '6m'] == pytest.approx(246.66772458807412, rel=1e-12)
    assert artif.loc['BEL', '6m'] / _AUTHOR_MULTIPLIER == pytest.approx(160.35, abs=0.005)
    assert {'SRB', 'PSE', 'XKX'} <= set(out['iso3'])
    absent = set(src['ISO3']) - set(src.loc[src['LandType'] == 'Agriculture', 'ISO3'])
    crop = out[out['sda_type'] == 'crop'].set_index('iso3')
    zero_crop = set(crop.index[crop[_DEPTH_COLS].abs().sum(axis=1) == 0])
    assert len(absent) == 17 and absent < zero_crop and zero_crop - absent == {'FRO'}

    got = out.set_index(['iso3', 'sda_type'])
    worst = 0.0
    for iso3, rows in src.groupby('ISO3'):
        for sda_type, land_type in (('artif', 'Residential'), ('crop', 'Agriculture'),
                                    ('pasture', 'Agriculture')):
            match = rows[rows['LandType'] == land_type]
            for col in _DEPTH_COLS:
                value = float(match[col].iloc[0]) if len(match) else 0.0
                value = 0.0 if np.isnan(value) else value
                if land_type == 'Agriculture':
                    value = value / 10_000.0
                worst = max(worst, abs(got.loc[(iso3.upper(), sda_type), col]
                                       - value * _AUTHOR_MULTIPLIER))
    assert worst == 0.0

    saved_path = ft.rebuild_sda_damage_table(path, _es_parameter('flood_landtype_depth_damage_sha256'),
                                             _AUTHOR_MULTIPLIER, str(tmp_path / 'rebuilt.csv'))
    saved = pd.read_csv(saved_path, float_precision='round_trip')
    pd.testing.assert_frame_equal(saved, out, check_exact=True)


def test_the_components_reader_multiplies_the_regional_fraction_by_the_country_max_damage(tmp_path):
    """The other canonical shape: a fraction per region and land type, times a max damage per
    country. Nothing in the pipeline calls this, so if the real canonical file is this shape, 4A
    reads fractions as if they were absolute euros.
    """
    import os
    from global_invest.flood import flood_tasks as ft

    frac = os.path.join(str(tmp_path), 'fractional_long.csv')
    maxd = os.path.join(str(tmp_path), 'maxdamage_long.csv')
    region = os.path.join(str(tmp_path), 'iso3_region.csv')
    pd.DataFrame([{'JRC_Region': 'R1', 'LandType': 'Commercial', 'depth_m': 1.0, 'fraction': 0.4}]
                 ).to_csv(frac, index=False)
    pd.DataFrame([{'iso3': 'AAA', 'LandType': 'Commercial', 'max_damage_eur_m2': 100.0}]
                 ).to_csv(maxd, index=False)
    pd.DataFrame([{'iso3': 'AAA', 'JRC_Region': 'R1'}]).to_csv(region, index=False)

    out = ft.build_canonical_from_components(frac, maxd, region)
    row = out[(out['iso3'] == 'AAA') & (out['depth_m'] == 1.0)]
    assert len(row) == 1
    # 100 EUR/m2 max damage at a 0.4 fraction is 40, in EUR2010.
    assert float(row['damage_per_m2'].iloc[0]) == pytest.approx(40.0)


def test_a_missing_amplification_raster_stops_the_run_rather_than_zeroing_the_gep(tmp_path):
    """The degraded world is the current depths times an amplification factor, so a missing factor
    makes the two worlds identical and that return period's GEP exactly zero. Returning None there
    would be indistinguishable from the current scenario, where None is correct.
    """
    import os
    from types import SimpleNamespace
    from global_invest.flood import flood_tasks as ft

    p = SimpleNamespace()
    p.flood_amplification_path = str(tmp_path)
    p.flood_amplification_pattern = 'global_amplification_{scenario}_rp{rp}.tif'

    # the current world legitimately has none
    assert ft._open_amplification_raster(p, 'current', 100) is None

    # a degraded world without one is a broken run, not a zero
    with pytest.raises(FileNotFoundError) as raised:
        ft._open_amplification_raster(p, 'degraded_bare', 100)
    assert 'degraded_bare' in str(raised.value)
    assert 'exactly zero' in str(raised.value)


def test_the_account_country_layer_works_as_the_flood_admin0():
    """Condition 2. The account's layer carries iso3 id, label and name but no region, so step 4D
    takes `region_wb` from the shared country table rather than from the geometry.

    Three things have to hold for the layer to serve as flood's admin0: it resolves one row per
    country, the ISO3 detector picks the right column out of it, and the region the export needs is
    available from the table rather than the geometry.
    """
    import os
    import geopandas as gpd

    layer = os.path.join(os.path.expanduser('~'), 'Files', 'base_data', 'cartographic', 'ee',
                         'ee_r250.gpkg')
    correspondence = os.path.join(os.path.expanduser('~'), 'Files', 'base_data', 'cartographic',
                                  'ee', 'ee_r264_correspondence.csv')
    if not (os.path.exists(layer) and os.path.exists(correspondence)):
        pytest.skip('the account country layer is not on this machine')

    admin0 = gpd.read_file(layer)
    assert len(admin0) == 250                                  # one row per country, not 264
    assert admin0.crs is not None

    # The detector must find the account's own label, not Natural Earth's adm0_a3. Against the r264
    # correspondence it picks adm0_a3 and would silently key the export on a different vocabulary.
    iso_col = utilities.pick_iso3_column(admin0)
    assert iso_col == 'iso3_r250_label', iso_col
    assert admin0[iso_col].nunique() == 250

    # The region step 4D needs is absent from the geometry and present in the table, which is why
    # the export reads it from the table.
    assert 'region_wb' not in admin0.columns
    countries = utilities.collapse_countries_to_r250(pd.read_csv(correspondence))
    assert 'region_wb' in countries.columns
    region = countries[['iso3_r250_label', 'region_wb']]
    joined = admin0[[iso_col]].merge(region, left_on=iso_col, right_on='iso3_r250_label', how='left')
    assert joined['region_wb'].notna().sum() > 200


def test_our_flood_gep_reproduces_the_authors_committed_table():
    """Condition 12. The account's flood GEP is $11,403,596,548 and the author publishes $11.40bn,
    his rounding; this recomputes the agreement country by country.

    `flood_gep_for_merge_v2_2024hazard.csv` is the author's table for joining against the other
    services, staged in base_data beside the account's run output.

    What it proves is narrow: the module is the author's pipeline with the account's path handling
    and ProjectFlow wiring, so this is a build test. It catches a refactor breaking the arithmetic
    and says nothing about whether the arithmetic is right."""
    import os
    base = os.path.join(os.path.expanduser('~'), 'Files', 'base_data', 'global_invest', 'flood',
                        'reference')
    his_path = os.path.join(base, 'flood_gep_for_merge_v2_2024hazard.csv')
    ours_path = os.path.join(base, 'step4e_flood_gep_USD2019_ours_20260831.csv')
    if not (os.path.exists(his_path) and os.path.exists(ours_path)):
        pytest.skip('the staged flood reference or our run output is not on this machine')

    his = pd.read_csv(his_path)
    ours = pd.read_csv(ours_path)
    joined = his.merge(ours[['iso3', 'gep_flood_bare_usd2019']],
                       left_on='iso3_r250_label', right_on='iso3', how='outer', indicator=True)
    assert (joined['_merge'] == 'both').all()                  # same 250 countries, no strays
    assert len(joined) == 250

    both = joined[joined['gep_const2019_usd'].notna()]
    difference = (both['gep_flood_bare_usd2019'] - both['gep_const2019_usd']).abs()
    assert difference.max() < 0.01, difference.max()           # agrees to the cent, country by country
    assert ours['gep_flood_bare_usd2019'].sum() == pytest.approx(
        his['gep_const2019_usd'].sum(), abs=1.0)

    # The one country that differs, named so a change in it is noticed rather than absorbed into a
    # tolerance: Iceland is NaN in his table and 2.1 cents in ours, which is the whole of the
    # 162-against-163 positive-country count.
    iceland = joined[joined['iso3_r250_label'] == 'ISL']
    assert iceland['gep_const2019_usd'].isna().all()
    assert (iceland['gep_flood_bare_usd2019'] < 1.0).all()


def test_the_water_variant_more_than_doubles_the_flood_headline():
    """The largest open question in the account, pinned so it cannot drift unnoticed. The run writes
    four variants and the delivered one removes permanent water; keeping it more than doubles the
    figure. Which the account intends is undocumented, so this asserts the gap exists rather than
    which side is right."""
    import os
    base = os.path.join(os.path.expanduser('~'), 'Files', 'base_data', 'global_invest', 'flood',
                        'reference')
    delivered = os.path.join(base, 'step4e_flood_gep_USD2019_ours_20260831.csv')
    waterkept = os.path.join(base, 'step4e_flood_gep_USD2019_waterkept_20260831.csv')
    if not (os.path.exists(delivered) and os.path.exists(waterkept)):
        pytest.skip('the staged flood variants are not on this machine')

    a = pd.read_csv(delivered)['gep_flood_bare_usd2019'].sum()
    b = pd.read_csv(waterkept)['gep_flood_bare_usd2019'].sum()
    assert 1.1e10 < a < 1.2e10, a                              # the delivered variant, ~$11.40bn
    assert 2.4e10 < b < 2.5e10, b                              # water kept, ~$24.54bn
    assert b / a > 2.0, b / a


def test_every_flood_tree_builder_assembles():
    """Every flood tree builder assembles.

    Flood publishes a family of composable partial trees and only some are reached from the run
    file: `build_flood_calculation_task_tree` and `build_flood_valuation_task_tree` are reached
    only from here. They are deliberate API, and an entry point nothing calls and nothing tests is
    indistinguishable from dead code (harmonization check H23 flags such builders). Building each
    tree also catches a builder naming a task that does not exist, which would otherwise fail only
    when somebody chose that tree.
    """
    from global_invest.flood import flood_initialize

    builders = [name for name in dir(flood_initialize)
                if name.startswith('build_') and name.endswith('_task_tree')]
    assert len(builders) >= 6, builders

    for name in builders:
        p = hb.ProjectFlow(project_dir=tempfile.mkdtemp())
        getattr(flood_initialize, name)(p)
        assert p.task_names_defined, name


def test_gep_calculation_requires_the_bare_counterfactual_in_a_computed_table(tmp_path, monkeypatch):
    """A computed table without gep_flood_bare_usd2019 is refused — the account never silently
    publishes the insitu column under the bare description — and a valid bare table publishes its
    bare column."""
    from types import SimpleNamespace
    from global_invest.flood import flood_tasks as ft
    monkeypatch.setattr(ft, 'publish_inputs', lambda p: p)
    out_path = str(tmp_path / 'gep_by_country_base_year.csv')
    monkeypatch.setattr(utilities, 'begin_gep_calculation',
                        lambda p, s: ({'gep_by_country_base_year': out_path}, False))
    monkeypatch.setattr(utilities, 'collapse_countries_to_r250', lambda df: df)
    countries = pd.DataFrame({'iso3_r250_label': ['AAA'], 'iso3_r250_id': [1],
                              'iso3_r250_name': ['A'], 'continent': ['X'], 'region_un': ['X'],
                              'region_wb': ['X'], 'income_grp': ['X'], 'subregion': ['X']})
    computed = tmp_path / 'flood_gep_usd2019.csv'
    p = SimpleNamespace(run_this=True, gep_base_year=2019, df_countries=countries,
                        flood_gep_path=str(computed), flood_gep_for_merge_path=None)
    pd.DataFrame({'iso3_r250_label': ['AAA'],
                  'gep_flood_insitu_usd2019': [5.0]}).to_csv(computed, index=False)
    with pytest.raises(NameError, match='bare'):
        ft.gep_calculation(p)
    pd.DataFrame({'iso3_r250_label': ['AAA'], 'gep_flood_bare_usd2019': [7.0],
                  'gep_flood_insitu_usd2019': [5.0]}).to_csv(computed, index=False)
    ft.gep_calculation(p)
    published = pd.read_csv(out_path)
    assert published.loc[0, 'flood_gep'] == pytest.approx(7.0)


def test_the_computed_route_publishes_the_authors_export_from_the_staged_table(tmp_path, monkeypatch):
    """The computed route reads the table compute_flood_gep writes, which keys by `iso3`. On the
    staged run output it must publish exactly what the author's export publishes, including the
    author's rule that blanks anything below $1 (Iceland's 2.1 cents is the case that rule exists
    for)."""
    import os
    import shutil
    from types import SimpleNamespace
    from global_invest.flood import flood_tasks as ft
    base = os.path.join(os.path.expanduser('~'), 'Files', 'base_data', 'global_invest', 'flood')
    ours_path = os.path.join(base, 'reference', 'step4e_flood_gep_USD2019_ours_20260831.csv')
    his_path = os.path.join(base, 'flood_gep_for_merge_v2_2024hazard.csv')
    if not (os.path.exists(ours_path) and os.path.exists(his_path)):
        pytest.skip('the staged flood run output or the author export is not on this machine')

    his = pd.read_csv(his_path)
    computed = os.path.join(str(tmp_path), 'flood_gep_usd2019.csv')
    shutil.copy(ours_path, computed)
    out_path = os.path.join(str(tmp_path), 'gep_by_country_base_year.csv')
    monkeypatch.setattr(ft, 'publish_inputs', lambda p: p)
    monkeypatch.setattr(utilities, 'begin_gep_calculation',
                        lambda p, s: ({'gep_by_country_base_year': out_path}, False))
    monkeypatch.setattr(utilities, 'collapse_countries_to_r250', lambda df: df)
    countries = his[['iso3_r250_label', 'iso3_r250_name']].copy()
    p = SimpleNamespace(run_this=True, gep_base_year=2019, df_countries=countries,
                        flood_gep_path=computed, flood_gep_for_merge_path=None)
    ft.gep_calculation(p)

    published = pd.read_csv(out_path).set_index('iso3_r250_label')['flood_gep']
    reference = his.set_index('iso3_r250_label')['gep_const2019_usd'].reindex(published.index)
    assert len(published) == 250
    assert (published.isna() == reference.isna()).all()
    assert (published - reference).abs().max() < 0.01
    assert np.isnan(published['ISL'])


def test_a_return_period_never_reads_a_longer_return_periods_depths(tmp_path):
    """The fallback patterns must not let rp50 match the rp500 file when the exact file is missing."""
    import os
    from types import SimpleNamespace
    from global_invest.flood import flood_tasks as ft
    open(os.path.join(str(tmp_path), 'JRC_flood_depth_rp500y__matchLULC.tif'), 'wb').close()
    p = SimpleNamespace(flood_depth_aligned_path=str(tmp_path))
    assert ft._find_depth_raster(p, 50) is None
    assert ft._find_depth_raster(p, 500).endswith('rp500y__matchLULC.tif')
    open(os.path.join(str(tmp_path), 'JRC_flood_depth_rp50_matchLULC.tif'), 'wb').close()
    assert ft._find_depth_raster(p, 50).endswith('rp50_matchLULC.tif')


def test_a_zip_member_in_a_subfolder_lands_beside_the_others(tmp_path):
    """A member stored under a folder inside the zip is extracted beside the others."""
    import os
    import zipfile
    from global_invest.flood import flood_tasks as ft
    zip_path = os.path.join(str(tmp_path), 'depth.zip')
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr('floodMapGL_rp10y/floodMapGL_rp10y.tif', b'tif')
    out_dir = os.path.join(str(tmp_path), 'out')
    extracted = ft._extract_zip(zip_path, out_dir)
    member = os.path.join(out_dir, 'floodMapGL_rp10y.tif')
    assert extracted == [member]
    with open(member, 'rb') as f:
        assert f.read() == b'tif'
