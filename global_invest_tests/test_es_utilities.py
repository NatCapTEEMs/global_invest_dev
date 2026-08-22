"""Shared ES-utility tests. resolve_raw_scenario is used by every service's static shock task, so it is
tested once here rather than duplicated per service."""
from global_invest import utilities


def test_resolve_scenario_identity_default():
    labels = ['scn_a', 'scn_c', 'scn_b_v2050']
    # no map entry, but the table already uses our name -> identity resolves it
    assert utilities.resolve_raw_scenario(labels, {}, 'scn_a', 'svc') == 'scn_a'


def test_resolve_scenario_explicit_map_first_present_wins():
    labels = ['scn_b_v2050', 'scn_c']
    m = {'scn_b': ['scn_b', 'scn_b_v2050'], 'scn_alias': ['scn_c']}
    # 'scn_b' is absent, 'scn_b_v2050' present -> the second candidate wins
    assert utilities.resolve_raw_scenario(labels, m, 'scn_b', 'svc') == 'scn_b_v2050'
    assert utilities.resolve_raw_scenario(labels, m, 'scn_alias', 'svc') == 'scn_c'


def test_resolve_scenario_absent_warns_loudly_and_returns_none():
    msgs = []
    got = utilities.resolve_raw_scenario(['scn_a'], {}, 'scn_b', 'terrestrial_carbon', log=msgs.append)
    assert got is None                        # never a silent match
    assert len(msgs) == 1                      # and it warned
    assert 'scn_b' in msgs[0] and 'terrestrial_carbon' in msgs[0] and 'scn_a' in msgs[0]


def test_resolve_base_scenario_tries_candidates_and_is_fatal_when_absent():
    import pytest
    # The frozen tables spell the nature-off baseline two ways; the consumer map carries both,
    # and the first candidate present in the table wins.
    m = {'baseline_ignore_dependencies': ['baseline_ignore_dependencies', 'baseline_ignore_damages']}
    assert utilities.resolve_base_scenario(
        ['baseline_ignore_damages', 'scn_a'], m, 'baseline_ignore_dependencies', 'erosion') \
        == 'baseline_ignore_damages'
    # A base that resolves to nothing is FATAL (it is the subtraction reference), never a skip.
    with pytest.raises(ValueError, match='BASE'):
        utilities.resolve_base_scenario(['scn_a'], {}, 'baseline_ignore_dependencies', 'erosion')


def test_nature_off_spellings_are_mutual_aliases_by_default():
    # The frozen tables disagree on the nature-off baseline's spelling (carbon: _dependencies;
    # pollination/erosion: _damages) -- each spelling must find the other's table without a
    # consumer-supplied map. Closes the "global_invest open" half of the two-spelling bug.
    quiet = lambda *a: None
    assert utilities.resolve_raw_scenario(
        ['baseline_ignore_damages', 'below_2c'], {}, 'baseline_ignore_dependencies', 'erosion', log=quiet) == 'baseline_ignore_damages'
    assert utilities.resolve_raw_scenario(
        ['baseline_ignore_dependencies', 'below_2c'], {}, 'baseline_ignore_damages', 'carbon', log=quiet) == 'baseline_ignore_dependencies'
    # exact spelling present: no alias needed, itself wins
    assert utilities.resolve_raw_scenario(
        ['baseline_ignore_dependencies'], {}, 'baseline_ignore_dependencies', 'carbon', log=quiet) == 'baseline_ignore_dependencies'
    # an explicit consumer map still wins over the default aliasing
    assert utilities.resolve_raw_scenario(
        ['x'], {'baseline_ignore_dependencies': ['x']}, 'baseline_ignore_dependencies', 'carbon', log=quiet) == 'x'
    # non-baseline scenarios keep pure identity: no accidental cross-matching
    assert utilities.resolve_raw_scenario(
        ['baseline_ignore_damages'], {}, 'below_2c', 'erosion', log=quiet) is None


def test_download_missing_inputs_fetches_only_what_is_absent(tmp_path, monkeypatch):
    """A missing input with a recorded source is downloaded; a present one is left alone,
    and an input with no source is reported by name."""
    import pandas as pd
    from types import SimpleNamespace
    from global_invest import utilities

    template = tmp_path / 'es_parameters.csv'
    pd.DataFrame([
        {'service': 'demo', 'parameter': 'demo_present_path', 'value': 'demo/present.csv'},
        {'service': 'demo', 'parameter': 'demo_present_source_url', 'value': 'https://example.invalid/present.csv'},
        {'service': 'demo', 'parameter': 'demo_absent_path', 'value': 'demo/absent.csv'},
        {'service': 'demo', 'parameter': 'demo_absent_source_url', 'value': 'https://example.invalid/absent.csv'},
        {'service': 'demo', 'parameter': 'demo_unsourced_path', 'value': 'demo/unsourced.csv'},
    ]).to_csv(template, index=False)

    present = tmp_path / 'present.csv'
    present.write_text('kept')
    absent = tmp_path / 'absent.csv'
    unsourced = tmp_path / 'unsourced.csv'

    p = SimpleNamespace(demo_present_path=str(present), demo_absent_path=str(absent),
                        demo_unsourced_path=str(unsourced))
    monkeypatch.setattr(utilities, 'seed_input_template', lambda *a, **k: str(template))
    fetched = []
    monkeypatch.setattr('urllib.request.urlretrieve',
                        lambda url, path: (fetched.append(url), open(path, 'w').write('new')))

    downloaded, missing = utilities.download_missing_inputs(p, 'demo', log=lambda *a: None)
    assert fetched == ['https://example.invalid/absent.csv']
    assert downloaded == [str(absent)]
    assert missing == ['demo_unsourced_path']
    assert present.read_text() == 'kept'
