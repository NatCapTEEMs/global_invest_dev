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

    downloaded, needs_a_person = utilities.download_missing_inputs(p, 'demo', log=lambda *a: None)
    assert fetched == ['https://example.invalid/absent.csv']
    assert downloaded == [str(absent)]
    assert needs_a_person == {'demo_unsourced_path': 'no recorded source'}
    assert present.read_text() == 'kept'


def test_download_missing_inputs_extracts_an_archive_member_and_reports_notes(tmp_path, monkeypatch):
    """An input whose source is an archive is extracted from it, and an input whose source is
    a note (an interactive export, a colleague's file) is reported with that note."""
    import zipfile
    import pandas as pd
    from types import SimpleNamespace
    from global_invest import utilities

    template = tmp_path / 'es_parameters.csv'
    pd.DataFrame([
        {'service': 'demo', 'parameter': 'demo_member_path', 'value': 'demo/member.csv'},
        {'service': 'demo', 'parameter': 'demo_member_source_url', 'value': 'https://example.invalid/pack.zip'},
        {'service': 'demo', 'parameter': 'demo_member_source_archive_member', 'value': 'inner/wanted.csv'},
        {'service': 'demo', 'parameter': 'demo_noted_path', 'value': 'demo/noted.csv'},
        {'service': 'demo', 'parameter': 'demo_noted_source_note', 'value': 'exported by hand from the dashboard'},
    ]).to_csv(template, index=False)

    archive = tmp_path / 'pack.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('inner/wanted.csv', 'the wanted member')
        zf.writestr('inner/other.csv', 'not this one')

    member = tmp_path / 'member.csv'
    noted = tmp_path / 'noted.csv'
    p = SimpleNamespace(demo_member_path=str(member), demo_noted_path=str(noted))
    monkeypatch.setattr(utilities, 'seed_input_template', lambda *a, **k: str(template))
    monkeypatch.setattr('urllib.request.urlretrieve',
                        lambda url, path: __import__('shutil').copyfile(archive, path))

    downloaded, needs_a_person = utilities.download_missing_inputs(p, 'demo', log=lambda *a: None)
    assert downloaded == [str(member)]
    assert member.read_text() == 'the wanted member'
    assert needs_a_person == {'demo_noted_path': 'exported by hand from the dashboard'}
    assert not noted.exists()


def test_collapse_countries_to_r250_keeps_one_row_per_country():
    """A split country (two r264 sub-regions) collapses to its single canonical row, and
    the extra columns a caller asks for come through."""
    import pandas as pd
    from global_invest import utilities

    df = pd.DataFrame([
        {'ee_r264_id': 1, 'iso3_r250_id': 156, 'ee_r264_label': 'CHN', 'iso3_r250_label': 'CHN',
         'ee_r264_name': 'China', 'iso3_r250_name': 'China', 'continent': 'Asia',
         'region_un': 'Asia', 'region_wb': 'EAP', 'income_grp': 'UM', 'subregion': 'E Asia',
         'area_code': 351, 'noise': 'drop me'},
        {'ee_r264_id': 2, 'iso3_r250_id': 156, 'ee_r264_label': 'CHN_north', 'iso3_r250_label': 'CHN',
         'ee_r264_name': 'China north', 'iso3_r250_name': 'China', 'continent': 'Asia',
         'region_un': 'Asia', 'region_wb': 'EAP', 'income_grp': 'UM', 'subregion': 'E Asia',
         'area_code': 351, 'noise': 'drop me'},
        {'ee_r264_id': 3, 'iso3_r250_id': 76, 'ee_r264_label': 'BRA', 'iso3_r250_label': 'BRA',
         'ee_r264_name': 'Brazil', 'iso3_r250_name': 'Brazil', 'continent': 'Americas',
         'region_un': 'Americas', 'region_wb': 'LAC', 'income_grp': 'UM', 'subregion': 'S America',
         'area_code': 21, 'noise': 'drop me'},
    ])
    out = utilities.collapse_countries_to_r250(df, keep_columns=['area_code'])
    assert len(out) == 2
    assert sorted(out['iso3_r250_label']) == ['BRA', 'CHN']
    assert 'area_code' in out.columns
    assert 'noise' not in out.columns


def test_assert_join_coverage_raises_when_a_country_drops_out():
    """A label the correspondence does not carry vanishes in the join, and the total still
    looks valid: the check has to catch the loss."""
    import pandas as pd
    import pytest
    from global_invest import utilities

    joined = pd.DataFrame({'iso3_r250_label': ['BRA', 'CHN', 'XXX'], 'value': [1.0, 2.0, None]})
    utilities.assert_join_coverage(joined, 'value', expected_rows=2, service='demo', log=lambda *a: None)
    with pytest.raises(ValueError, match='did not match a country'):
        utilities.assert_join_coverage(joined, 'value', expected_rows=3, service='demo', log=lambda *a: None)
