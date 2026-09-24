"""Zone ids are validated integers before anything keys on them.

The boundary gpkg stores `ee_r50_aez18_id` as TEXT while `aez18_id` beside it is int64. Rasterisation
coerced with int(i) and the label frame did not, so the two sides of a later merge carried different
dtypes and pandas refused it -- after twelve hours of upstream solving had already completed. These
tests use STRING ids throughout, because that is what the real file contains: a test built on
integers would pass against the very input that failed.
"""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from global_invest.erosion.erosion_damage_pipeline import normalise_zone_ids, zone_label_table


def boundary(ids, aez=None, labels=None):
    n = len(ids)
    return gpd.GeoDataFrame(
        {'ee_r50_aez18_id': ids,
         'aez18_id': aez if aez is not None else list(range(n)),
         'gtapv7_r50_label': labels if labels is not None else ['r%d' % i for i in range(n)],
         'geometry': [box(i, 0, i + 1, 1) for i in range(n)]},
        crs='EPSG:4326')


def test_string_ids_become_int64():
    """The real file's shape: ids as text, exactly as gpd.read_file returns them."""
    got = normalise_zone_ids(boundary(['100', '208', '4918']))
    assert got.ee_r50_aez18_id.dtype == 'int64'
    assert got.ee_r50_aez18_id.tolist() == [100, 208, 4918]


def test_the_merge_that_failed_now_works():
    """THE REGRESSION. Rasterisation yields int64 zone ids; the label table must join to them."""
    b = normalise_zone_ids(boundary(['100', '208', '4918']))
    labels = zone_label_table(b)
    from_raster = pd.DataFrame({'zone_id': [100, 208, 4918], 'damage': [1.0, 2.0, 3.0]})
    joined = from_raster.merge(labels, on='zone_id', validate='many_to_one')
    assert len(joined) == 3
    assert labels.zone_id.dtype == 'int64'


def test_missing_id_is_refused():
    with pytest.raises(ValueError, match='no zone id'):
        normalise_zone_ids(boundary(['100', None, '4918']))


def test_non_numeric_id_is_refused():
    with pytest.raises(ValueError, match='non-numeric'):
        normalise_zone_ids(boundary(['100', 'anz', '4918']))


def test_fractional_id_is_refused():
    """A fractional id means the column is not what it claims; truncating would key a wrong join."""
    with pytest.raises(ValueError, match='fractional'):
        normalise_zone_ids(boundary(['100', '208.5', '4918']))


def test_conflict_created_by_conversion_is_caught():
    """Two distinct strings can normalise to ONE integer, so the correspondence must be checked
    after conversion rather than before: '208' and '0208' look different in the file and are the
    same zone afterwards."""
    b = normalise_zone_ids(boundary(['100', '208', '0208'], aez=[1, 2, 3], labels=['a', 'b', 'c']))
    with pytest.raises(ValueError, match='Ambiguous zone correspondence after id normalisation'):
        zone_label_table(b)


def test_repeated_rows_for_one_zone_are_not_a_conflict():
    """The same zone appearing twice with the SAME correspondence is ordinary, not ambiguous."""
    b = normalise_zone_ids(boundary(['100', '100'], aez=[7, 7], labels=['anz', 'anz']))
    assert zone_label_table(b).zone_id.tolist() == [100]


def test_aez0_is_kept():
    """AEZ0 is a domain question handled downstream (kept in coverage, excluded from economic
    rows). It is not an invalid id and must survive normalisation."""
    b = normalise_zone_ids(boundary(['100', '200'], aez=[0, 1]))
    assert 0 in zone_label_table(b).aez18_id.tolist()
