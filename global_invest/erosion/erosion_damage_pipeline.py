"""Production area-damage seam for NGFS; SDR grid and source data are unchanged."""
from pathlib import Path
import importlib.metadata
import json
import numpy as np
import pandas as pd
from global_invest import utilities
from global_invest.erosion import erosion_damage as damage

ACCOUNTING = 'damage_area_8pct_v1'


def read_grid(path, expected=None):
    import rasterio
    with rasterio.open(path) as src:
        geometry = (src.shape, src.transform, src.crs)
        if expected is not None and geometry != expected:
            raise ValueError('Erosion grid mismatch: ' + str(path))
        return src.read(1, masked=True).astype('float64').filled(np.nan), geometry


def normalise_zone_ids(frame, column='ee_r50_aez18_id'):
    """The boundary's zone id as a validated int64, once, before anything keys on it.

    The gpkg stores ee_r50_aez18_id as TEXT while aez18_id beside it is int64. Rasterisation
    coerced with int(i) and the label frame did not, so the two sides of the later join carried
    different dtypes and pandas refused the merge after twelve hours of upstream work. Normalising
    at ingestion means rasterisation, labels and every join downstream use one validated set of ids
    rather than each coercing, or not coercing, on its own.

    Rejected rather than coerced: a missing id has no zone, a non-numeric one is not an id, and a
    fractional one means the column is not what it claims. Each would otherwise become a plausible
    integer and key a silently wrong join.

    Args:
        frame (gpd.GeoDataFrame): the boundary as read.
        column (str): the id column to normalise in place.

    Returns:
        gpd.GeoDataFrame: the same frame with `column` as int64.

    Raises:
        ValueError: on a missing, non-numeric or fractional id.
    """
    import numpy as _np
    import pandas as _pd
    raw = frame[column]
    if raw.isna().any():
        raise ValueError('%s: %d row(s) have no zone id' % (column, int(raw.isna().sum())))
    numeric = _pd.to_numeric(raw, errors='coerce')
    bad = numeric.isna()
    if bad.any():
        raise ValueError('%s: %d non-numeric zone id(s), e.g. %s'
                         % (column, int(bad.sum()), raw[bad].head(5).tolist()))
    fractional = numeric != _np.floor(numeric)
    if fractional.any():
        raise ValueError('%s: %d fractional zone id(s), e.g. %s'
                         % (column, int(fractional.sum()), numeric[fractional].head(5).tolist()))
    frame = frame.copy()
    frame[column] = numeric.astype('int64')
    return frame


def zone_label_table(boundary, id_column='ee_r50_aez18_id',
                     carry=('aez18_id', 'gtapv7_r50_label')):
    """One row per zone id, with the correspondence checked AFTER conversion.

    Two different strings can normalise to the same integer, so a correspondence that looked
    one-to-one in the file can become conflicting once the ids are numbers. The check therefore
    runs on the converted ids and reports what conflicts rather than only that something does.
    """
    labels = boundary[[id_column] + list(carry)].drop_duplicates().rename(columns={id_column: 'zone_id'})
    conflicting = labels[labels.zone_id.duplicated(keep=False)]
    if len(conflicting):
        raise ValueError('Ambiguous zone correspondence after id normalisation: %d row(s) over %d '
                         'zone id(s), e.g. %s' % (len(conflicting), conflicting.zone_id.nunique(),
                                                  conflicting.sort_values('zone_id').head(6).to_dict('records')))
    return labels


def crop_fraction(source, reference, destination):
    """Fraction of the SDR-cell footprint covered by full-resolution cropland pixels.

    No overview use, no nearest-neighbour cropland assignment. Source nodata is
    tracked separately; any solve cell overlapping missing source coverage is
    excluded later rather than treated as non-cropland.
    """
    import rasterio
    from osgeo import gdal
    destination = Path(destination)
    coverage = destination.with_name(destination.stem + '_source_coverage.tif')
    sigpath = str(destination) + '.json'
    signature = {'settings': {'method': 'native_fraction_with_coverage_v1'},
                 'inputs': {str(p): utilities.file_fingerprint(p) for p in (source, reference, __file__)}}
    if utilities.outputs_reuse_reason([str(destination), str(coverage)], signature, sigpath) is None:
        return str(destination), str(coverage)
    native = destination.with_name(destination.stem + '_native.tif')
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        profile.update(dtype='uint8', nodata=255, count=2, compress='deflate', BIGTIFF='YES',
                       tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(native, 'w', **profile) as dst:
            for _, window in src.block_windows(1):
                a = src.read(1, window=window, masked=True)
                valid = ~np.ma.getmaskarray(a)
                dst.write(((a.data == 2) & valid).astype('uint8'), 1, window=window)
                dst.write(valid.astype('uint8'), 2, window=window)
    with rasterio.open(reference) as ref:
        for band, path in ((1, destination), (2, coverage)):
            one = gdal.Translate('', str(native), format='MEM', bandList=[band])
            result = gdal.Warp(str(path), one, dstSRS=ref.crs.to_wkt(),
                outputBounds=tuple(ref.bounds), width=ref.width, height=ref.height,
                resampleAlg='average', overviewLevel='NONE', srcNodata=255, dstNodata=-9999.,
                outputType=gdal.GDT_Float32, creationOptions=['TILED=YES','COMPRESS=DEFLATE'])
            if result is None:
                raise ValueError('Failed to produce erosion crop fraction: ' + str(path))
            result = None
    utilities.write_outputs_signature(signature, sigpath)
    return str(destination), str(coverage)


# NAMED, AUDITABLE exceptions -- not a rule. Each was enumerated from damage_coverage.csv and
# examined before being listed; every one has 0.000 ha of severe cropland at the base year, so its
# d_2023 is zero. Excluding them asserts NOTHING about their future damage: d_t is unmeasured, and
# with d_2023 = 0 and d_t in [0, 0.08] the unscaled change could be anywhere in -8 to 0 points.
# These are not measured zeros and not a claim that the zones are immaterial.
#
# Two distinct causes, both leaving d_t unmeasurable:
#   coverage -- cropland exists but lies entirely outside the valid erosion support, so the zone
#               appears with 0 valid hectares and a non-finite level;
#   no crop  -- the zone holds no cropland footprint at all in that scenario-year, so
#               summarize_damage_areas (which returns only zones with total + excluded > 0) omits
#               its row entirely. An absent row is evidence of this, not a contradiction of it.
#
#   zone  region  AEZ  undefined   cause     base cropland   base severe
#   1001  col       1  21 of 21    coverage    447.896 ha      0.000 ha
#   1008  col       8   3 of 21    no crop    1095.930 ha      0.000 ha
#   1318  eur      18   3 of 21    no crop      20.946 ha      0.000 ha
#   1916  jpn      16  14 of 21    no crop      31.076 ha      0.000 ha
#   4118  xef      18   3 of 21    no crop      19.648 ha      0.000 ha
#
# A named zone is excluded from EVERY scenario and year even where only some are undefined. Keeping
# it where it is defined and dropping it elsewhere would put it in the economic rows for some
# pathways and not others, and the comparison between pathways is the entire point of the exercise.
#
# Any zone NOT listed here still fails, and the failure reports every offending zone at once rather
# than the first: discovering these one run at a time cost two and a half hours per zone.
FUTURE_COVERAGE_EXCLUSIONS = {
    1001: 'future cropland outside valid erosion support',
    1008: 'no future cropland footprint in low_demand',
    1318: 'no future cropland footprint in fragmented_world and net_zero',
    1916: 'no future cropland footprint in any pathway at several anchors',
    4118: 'no future cropland footprint in below_2c, low_demand and ndcs',
}


def _damage_missing(indexed_table, zone):
    """True when a zone has no usable damage level in an already-indexed table."""
    if zone not in indexed_table.index:
        return True
    return not np.isfinite(indexed_table.loc[zone, ['level', 'stressed_level']]
                           .to_numpy(dtype=float)).all()


def tables_to_seam(tables, labels, scenarios, anchors, years, base_scenario, base_year,
                    sectors, provision_loss, stress_from, baseline_domain=None,
                    excluded_path=None, coverage=None):
    """Estimate on the BASE-YEAR cropland domain; never fill a missing zone or invent a baseline.

    A zone with no baseline cropland has no d_2023, so 100(d_2023 - d_t) does not exist for it. Such
    a zone is EXCLUDED from the economic rows with its reason recorded, and stays in the coverage
    report: newly appearing cropland outside the base-year domain is reported, not shocked. That is
    not the same statement as a measured zero shock and must not be written as one.

    An exclusion is only legitimate where absence is ESTABLISHED. Baseline cropland removed by the
    coverage restriction, or coverage too poor to establish absence, leaves the baseline unknown and
    remains a failure.

    An undefined FUTURE year is likewise a failure, with one exception, and the exception is named
    zone by zone in FUTURE_COVERAGE_EXCLUSIONS rather than inferred. A named zone must be undefined
    in every scenario-year and for the stated cause, or it still fails: a zone excluded under some
    pathways and shocked under others would make the comparison between them incoherent.
    """
    indexed = {key: table.set_index('zone_id') for key, table in tables.items()}
    base = indexed[(base_scenario, base_year)]
    all_ids = sorted(set().union(*(set(t.zone_id) for t in tables.values())))
    labels = labels.set_index('zone_id')
    if labels.index.duplicated().any() or set(all_ids) - set(labels.index):
        raise ValueError('Missing or ambiguous erosion zone correspondence')
    all_ids = [i for i in all_ids if 1 <= int(labels.loc[i, 'aez18_id']) <= 18]
    rows = []
    excluded = []
    # Find EVERY zone whose future is undefined before shocking any of them. Raising on the first
    # unnamed zone hides the rest, and each rediscovery costs a full run of the seam.
    # Only zones with a DEFINED base year belong here. A zone with no base-year damage has no
    # d_2023 at all, so its future cannot be the thing that is missing; it is the baseline-domain
    # rule's business, decided below on baseline_domain evidence. Scanning it here instead reports
    # it as an unnamed future-undefined zone and stops a run the baseline rule would have handled.
    undefined_by_zone = {}
    for zone in all_ids:
        if zone not in base.index or not np.isfinite(base.loc[zone, 'level']):
            continue
        missing = [(s, y) for s in scenarios for y in anchors if _damage_missing(indexed[(s, y)], zone)]
        if missing:
            undefined_by_zone[zone] = missing
    unnamed = sorted(z for z in undefined_by_zone if int(z) not in FUTURE_COVERAGE_EXCLUSIONS)
    if unnamed:
        detail = '; '.join('zone %s (%s, AEZ%s) undefined in %d of %d scenario-year(s), first %s %s'
                           % (z, labels.loc[z, 'gtapv7_r50_label'], labels.loc[z, 'aez18_id'],
                              len(undefined_by_zone[z]), len(scenarios) * len(anchors),
                              undefined_by_zone[z][0][0], undefined_by_zone[z][0][1])
                           for z in unnamed)
        raise ValueError('Undefined damage in %d zone(s) that are not named exceptions. A new case '
                         'must be examined and named before it may be excluded, never swept in by '
                         'rule. All of them: %s' % (len(unnamed), detail))
    for zone in all_ids:
        if zone not in base.index or not np.isfinite(base.loc[zone, 'level']):
            domain = (baseline_domain or {}).get(int(zone))
            if domain is None:
                raise ValueError('Undefined base damage in economic zone %s and no baseline domain '
                                 'evidence to classify it' % zone)
            if domain['baseline_crop_ha'] > 1e-9 or domain['baseline_nodata_ha'] > 1e-9:
                raise ValueError(
                    'Economic zone %s has no base-year damage but %.4f ha of baseline cropland '
                    '(%.4f ha lost to source nodata, mean coverage %.6f): the baseline is UNKNOWN, '
                    'not zero, so no trajectory may be formed and none may be excluded'
                    % (zone, domain['baseline_crop_ha'], domain['baseline_nodata_ha'],
                       domain['mean_source_coverage']))
            excluded.append({'zone_id': int(zone),
                             'aez18_id': int(labels.loc[zone, 'aez18_id']),
                             'reg': labels.loc[zone, 'gtapv7_r50_label'],
                             'reason': 'no base-year cropland: d_2023 undefined, trajectory does '
                                       'not exist; NOT a measured zero shock',
                             'baseline_crop_ha': domain['baseline_crop_ha'],
                             'baseline_nodata_ha': domain['baseline_nodata_ha'],
                             'mean_source_coverage': domain['mean_source_coverage']})
            continue
        b = float(base.loc[zone, 'level'])
        undefined = undefined_by_zone.get(zone, [])
        if undefined:
            # The stated cause must hold for every undefined year, or the name describes nothing.
            # A zone ABSENT from a table is evidence of no cropland footprint, not a contradiction;
            # a zone PRESENT with valid cropland and no defined damage is unexplained and fails.
            contradicting = [(s, y) for s, y in undefined
                             if zone in indexed[(s, y)].index
                             and float(indexed[(s, y)].loc[zone, 'valid_crop_ha']) > 1e-9]
            if contradicting:
                s, y = contradicting[0]
                raise ValueError('Zone %s is a named exclusion, but %d of its %d undefined '
                                 'scenario-year(s) still hold valid cropland (first: %s %s); the '
                                 'named cause does not hold'
                                 % (zone, len(contradicting), len(undefined), s, y))
            excluded.append({'zone_id': int(zone),
                             'aez18_id': int(labels.loc[zone, 'aez18_id']),
                             'reg': labels.loc[zone, 'gtapv7_r50_label'],
                             'reason': FUTURE_COVERAGE_EXCLUSIONS[int(zone)],
                             'baseline_crop_ha': float(base.loc[zone, 'valid_crop_ha']),
                             'baseline_severe_crop_ha': float(base.loc[zone, 'severe_crop_ha']),
                             'baseline_damage_level': b,
                             'baseline_nodata_ha': 0.0,
                             'mean_source_coverage': float('nan'),
                             'undefined_scenario_years': len(undefined),
                             'of_scenario_years': len(scenarios) * len(anchors),
                             'max_future_excluded_crop_ha': max(
                                 [float(indexed[(s, y)].loc[zone, 'excluded_crop_ha'])
                                  for s, y in undefined if zone in indexed[(s, y)].index] or [0.0])})
            continue
        for scenario in scenarios:
            ordinary, stressed = [], []
            for year in anchors:
                t = indexed[(scenario, year)]
                ordinary.append(float(t.loc[zone, 'level']))
                stressed.append(float(t.loc[zone, 'stressed_level']))
            g = damage.annual_damage_change(b, anchors, ordinary, years, base_year=base_year)
            gs = damage.annual_damage_change(b, anchors, ordinary, years, base_year=base_year,
                                            stressed_levels=stressed, stress_from=stress_from)
            for sector in sectors:
                for year, value, value_stressed in zip(years, g, gs):
                    rows.append(dict(ENDW='AEZ%d' % labels.loc[zone,'aez18_id'], ACTS=str(sector).upper(),
                        REG=str(labels.loc[zone,'gtapv7_r50_label']).upper(), scenario=scenario,
                        year=int(year), shock_pct=float(value), damage_level_base=b,
                        damage_level=b+float(value), damage_level_stressed=b+float(value_stressed),
                        erosion_accounting=ACCOUNTING, erosion_provision_loss=provision_loss,
                        erosion_stress_from_year=stress_from))
    if excluded and excluded_path is not None:
        frame = pd.DataFrame(excluded)
        # The future cropland these zones DO acquire, reported beside the exclusion so the record
        # says what was set aside rather than only that something was.
        if coverage is not None and len(coverage):
            future = pd.concat(coverage, ignore_index=True) if isinstance(coverage, list) else coverage
            keep = future[future.zone_id.isin(frame.zone_id)]
            if len(keep):
                summary = keep.groupby('zone_id').agg(
                    scenario_years_present=('scenario', 'size'),
                    max_future_crop_ha=('valid_crop_ha', 'max'),
                    max_future_severe_crop_ha=('severe_crop_ha', 'max'),
                    max_future_excluded_crop_ha=('excluded_crop_ha', 'max')).reset_index()
                frame = frame.merge(summary, on='zone_id', how='left')
                keep.to_csv(str(excluded_path).replace('.csv', '_future_rows.csv'), index=False)
        frame.to_csv(excluded_path, index=False)
        print('  erosion: %d economic zone(s) EXCLUDED, no measured zero among them: %s -> %s'
              % (len(frame), sorted(frame.zone_id.tolist()), excluded_path))
        for reason, count in frame.reason.value_counts().items():
            print('    %d zone(s): %s' % (count, reason))
    if not rows:
        raise ValueError('No defined in-domain erosion trajectories')
    return pd.DataFrame(rows)


def erosion_damage_shock(p):
    """SDR + native crop fractions -> area damage, physical stress and annual level table.

    A single common valid soil/LULC support across all requested dates and
    scenarios prevents coverage changes masquerading as productivity changes.
    Coverage tables retain AEZ0 and excluded hectares; economic rows never do.
    """
    p.erosion_shock_output_path = str(Path(p.es_shock_dir) / 'erosion_interpolated.csv')
    if not p.run_this:
        return
    import geopandas as gpd
    from rasterio.features import rasterize
    version = importlib.metadata.version('natcap.invest')
    if tuple(map(int, version.split('.')[:2])) < (3,15):
        raise ValueError('Damage seam requires verified SDR t/ha/year outputs (InVEST >=3.15)')
    base_year, end_year = int(p.es_shock_base_year), int(p.es_shock_end_year)
    base_scenario = utilities.required_base_scenario(p, 'erosion')
    anchors = sorted({int(y) for y in p.es_shock_years if int(y) > base_year})
    if not anchors or anchors[-1] != end_year:
        raise ValueError('Damage anchors must reach the configured end year')
    stress_from = int(p.es_provision_from_year)
    if stress_from not in anchors:
        raise ValueError('Erosion stress onset requires a physical anchor')
    provision_loss = 1 - float(p.es_provision_retained)
    excluded = set(getattr(p, 'es_shock_excluded_scenarios', []) or [])
    excluded |= {base_scenario, p.es_provision_label}
    scenarios = sorted(set(p.es_shock_scenarios) - excluded)
    if not scenarios or p.es_provision_source_scenario not in scenarios:
        raise ValueError('Damage seam requires the ordinary source pathway')
    cases = [(base_scenario, base_year)] + [(s,y) for s in scenarios for y in anchors]
    work = Path(p.cur_dir); work.mkdir(parents=True, exist_ok=True)
    boundary_path = p.get_path(p.region_boundary_path)
    inputs = [boundary_path, __file__, damage.__file__]
    files = {}
    for scenario, year in cases:
        source = p.scenario_lulc_paths.get(scenario, {}).get(year)
        tag = '%s_%d' % (scenario, year)
        folder = Path(p.erosion_sdr_dir) / tag
        usle, rkls = folder / ('usle_'+tag+'.tif'), folder / ('rkls_'+tag+'.tif')
        for path in (source, usle, rkls):
            if not path or not Path(path).is_file():
                raise ValueError('Missing configured erosion input %s %s: %s' % (scenario,year,path))
            inputs.append(str(path))
        files[(scenario,year)] = (source, usle, rkls)
    signature = {'settings': {'accounting':ACCOUNTING, 'threshold_t_ha_year':11., 'alpha':.08,
        'cases': [list(x) for x in cases], 'base_year':base_year,'end_year':end_year,
        'provision_loss':provision_loss,'stress_from':stress_from,'sectors':list(p.erosion_shock_acts),
        'invest_version':version,'support':'common_all_inputs'},
        'inputs': {str(path):utilities.file_fingerprint(path) for path in inputs}}
    coverage_path = work/'damage_coverage.csv'; signature_path = work/'damage_signature.json'
    out = Path(p.erosion_shock_output_path)
    if utilities.outputs_reuse_reason([str(out),str(coverage_path)],signature,str(signature_path)) is None:
        return
    # Invalidate the old success record before a rebuild that might be interrupted.
    signature_path.unlink(missing_ok=True)
    geometry = None; support = None; fractions = {}
    for key,(source,usle,rkls) in files.items():
        loss, geo = read_grid(usle, geometry)
        if geometry is None: geometry = geo
        potential,_ = read_grid(rkls,geometry)
        tag='%s_%d'%key
        frac,covered = crop_fraction(source,usle,work/('crop_fraction_'+tag+'.tif'))
        c,_=read_grid(covered,geometry);f,_=read_grid(frac,geometry)
        valid=np.isfinite(loss)&np.isfinite(potential)&np.isfinite(f)&(c>=1-1e-7)
        support=valid if support is None else support&valid
        fractions[key]=(frac,covered)
    shape,transform,crs=geometry
    if crs.to_epsg()!=8857:
        raise ValueError('Damage area calculation requires the configured Equal Earth EPSG:8857 grid')
    hectares=abs(transform.a*transform.e-transform.b*transform.d)/10000
    boundary=normalise_zone_ids(gpd.read_file(boundary_path).to_crs(crs))
    labels=zone_label_table(boundary)
    # The SAME validated ids feed the raster and the labels, so the join keys cannot diverge.
    zones=rasterize(((g,i) for g,i in zip(boundary.geometry,boundary.ee_r50_aez18_id)
                    if g is not None and not g.is_empty),out_shape=shape,transform=transform,fill=0,dtype='int32')
    # The base-year cropland DOMAIN, classified before any trajectory is formed. A zone absent from
    # the base-year damage table has no d_2023 and no trajectory, but that is only a legitimate
    # exclusion when the zone genuinely had no baseline cropland. Cropland removed by the coverage
    # restriction means the baseline is UNKNOWN, not zero, and must stay a failure.
    base_key=(base_scenario,base_year)
    baseline_domain={}
    if base_key in fractions:
        base_frac,_=read_grid(fractions[base_key][0],geometry)
        base_cover,_=read_grid(fractions[base_key][1],geometry)
        crop_ha=np.where(np.isfinite(base_frac),base_frac,0.0)*hectares
        full=base_cover>=1-1e-7
        for z in np.unique(zones[zones>0]):
            m=zones==z
            baseline_domain[int(z)]={'baseline_crop_ha':float(crop_ha[m].sum()),
                                     'baseline_nodata_ha':float(crop_ha[m&~full].sum()),
                                     'mean_source_coverage':float(np.nanmean(base_cover[m]))}
    tables={};coverage=[]
    for key,(source,usle,rkls) in files.items():
        loss,_=read_grid(usle,geometry);potential,_=read_grid(rkls,geometry)
        frac,_=read_grid(fractions[key][0],geometry)
        table=damage.summarize_damage_areas(loss,potential,frac,hectares,zones,common_support=support,
                                          threshold=11.,provision_loss=provision_loss)
        tables[key]=table
        coverage.append(table.assign(scenario=key[0],year=key[1]).merge(labels,on='zone_id',validate='many_to_one'))
    coverage=pd.concat(coverage,ignore_index=True)
    coverage['in_economic_domain']=coverage.aez18_id.between(1,18)
    coverage.to_csv(coverage_path,index=False)
    excluded_path=Path(coverage_path).with_name('erosion_excluded_zones.csv')
    result=tables_to_seam(tables,labels,scenarios,anchors,list(range(base_year,end_year+1)),
                         base_scenario,base_year,p.erosion_shock_acts,provision_loss,stress_from,
                         baseline_domain=baseline_domain,excluded_path=excluded_path,
                         coverage=coverage)
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.with_suffix('.tmp.csv');result.to_csv(temporary,index=False);temporary.replace(out)
    utilities.write_outputs_signature(signature,str(signature_path))
