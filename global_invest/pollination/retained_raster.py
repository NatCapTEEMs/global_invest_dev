"""Area fraction of baseline cropland retained, on the crop-valuation grid.

Categorical land cover is read at native resolution. No overview or majority
resampling is used. Native pixel hectares are apportioned by grid overlap; this
is an area allocation approximation within boundary pixels, not finer crop data.
"""
from pathlib import Path
import json
import os
import numpy as np


def _overlap(n, offset, origin, step, target_origin, target_step, target_n):
    from scipy.sparse import coo_matrix
    edges = (origin + np.arange(offset, offset+n+1)*step-target_origin)/target_step
    # Snap only floating-point uncertainty at integer cell boundaries. Otherwise
    # coincident edges can invent ~1e-12 retained fractions in the adjacent cell.
    nearest = np.rint(edges)
    roundoff = 64 * np.finfo(float).eps * np.maximum(1., np.abs(edges))
    edges = np.where(np.abs(edges-nearest) <= roundoff, nearest, edges)
    if np.any(np.diff(edges) <= 0):
        raise ValueError('Aligned north-up grid axes required')
    rows, cols, weights = [], [], []
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        for i in range(max(0, int(np.floor(lo))), min(target_n, int(np.ceil(hi)))):
            overlap = max(0., min(hi, i+1)-max(lo, i))/(hi-lo)
            if overlap:
                rows.append(i); cols.append(j); weights.append(overlap)
    return coo_matrix((weights, (rows, cols)), shape=(target_n,n)).tocsr()


def retained_fraction_raster(base_path, future_path, template_path, output_path):
    import rasterio
    from rasterio.windows import Window
    import hazelbean as hb
    paths = [Path(p) for p in (base_path, future_path, template_path)]
    signature = dict(version='retained_hectares_v2', inputs=[dict(path=str(p.resolve()),
                     size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in paths])
    out = Path(output_path); sidecar = Path(str(out)+'.json')
    if out.exists():
        if sidecar.exists() and json.loads(sidecar.read_text()) == signature:
            return str(out)
        raise ValueError('Unverified retained-fraction cache; use a fresh directory: '+str(out))
    with rasterio.open(base_path) as base, rasterio.open(future_path) as future, rasterio.open(template_path) as target:
        if (base.shape, base.transform, base.crs) != (future.shape, future.transform, future.crs):
            raise ValueError('Baseline and future land grids differ')
        if base.crs != target.crs or not base.crs.is_geographic:
            raise ValueError('This fraction calculation requires the common geographic CRS')
        if any(t.b or t.d for t in (base.transform,target.transform)):
            raise ValueError('Rotated grids are unsupported')
        b, t = base.transform, target.transform
        wx = _overlap(base.width,0,b.c,b.a,t.c,t.a,target.width)
        denominator = np.zeros(target.shape,dtype=np.float64)
        numerator = np.zeros_like(denominator)
        for y in range(0,base.height,128):
            h=min(128,base.height-y)
            bv=base.read(1,window=Window(0,y,base.width,h),masked=True)
            fv=future.read(1,window=Window(0,y,base.width,h),masked=True)
            if np.any(np.ma.getmaskarray(bv) != np.ma.getmaskarray(fv)):
                raise ValueError('Land coverage differs across dates')
            crop=(~np.ma.getmaskarray(bv)) & (bv.data==2)
            retained=crop & (fv.data==2)
            lat=b.f+(np.arange(y,y+h)+.5)*b.e
            ha=np.asarray(hb.get_area_of_pixel_column_from_center_lats(abs(b.a),lat))[:,None]/1e4
            wy=_overlap(h,y,b.f,b.e,t.f,t.e,target.height)
            touched=np.unique(wy.nonzero()[0])
            if not len(touched): continue
            small=wy[touched]
            for values, accumulator in ((crop*ha,denominator),(retained*ha,numerator)):
                accumulator[touched] += np.asarray((wx @ (small @ values).T).T)
        if np.any(numerator > denominator+1e-5):
            raise ValueError('Retained area exceeds baseline cropland')
        fraction=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
        profile=target.profile.copy()
        for key in ('blockxsize','blockysize'):
            profile.pop(key,None)
        profile.update(dtype='float32',nodata=-9999.,count=1,compress='deflate',tiled=True,
                       blockxsize=256,blockysize=256)
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=str(out)+'.tmp.tif'
    with rasterio.open(temporary,'w',**profile) as dst:
        dst.write(fraction.astype(np.float32),1)
    os.replace(temporary,out)
    sidecar.write_text(json.dumps(signature,sort_keys=True,indent=2))
    return str(out)
