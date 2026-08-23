# pygeoprocessing `distance_transform_edt` is wrong past column 4096

**Version:** pygeoprocessing 2.4.9. Reached in our code as `hb.distance_transform_edt`, which
hazelbean re-exports from pygeoprocessing (hazelbean's own function of the same name is shadowed
by the re-export).

**Symptom:** for any raster **wider than 4096 pixels**, the returned Euclidean distance is too
large from about column 4079 onwards. Every wrong value is an over-estimate. Rasters 4096 wide or
narrower are exact.

**Why it matters here:** the function is the natural way to build a "within N km of a road" mask,
and InVEST models use it on global grids. A global raster at 10 arcsec is 129,600 columns wide, so
roughly 97 percent of its columns fall in the affected range.

## What the error looks like

Twelve disputed cells at 5000x5000, brute-forced against all 99,766 source pixels:

| row | col | pygeoprocessing | scipy | brute force |
|---|---|---|---|---|
| 4221 | 4374 | 17.0294 | 17.0000 | 17.0000 |
| 86 | 4930 | 8.0623 | 8.0000 | 8.0000 |
| 2492 | 4654 | 5.0990 | 5.0000 | 5.0000 |
| 3165 | 4219 | 9.8995 | 8.9443 | 8.9443 |
| 366 | 4276 | 3.1623 | 3.0000 | 3.0000 |

pygeoprocessing is wrong on 12 of 12; scipy is right on 12 of 12.

The values are the tell. 17.0294 is `sqrt(17^2 + 1)`, 8.0623 is `sqrt(8^2 + 1)`, 3.1623 is
`sqrt(3^2 + 1)`. The answer is consistently the distance to a source **one column further away in
x** than the true nearest one, which points at an off-by-one in the column index rather than at a
tolerance or a precision loss.

## Scale of it

Same seed, varying raster width:

| size | wrong cells | first affected column | all over-estimates |
|---|---|---|---|
| 4000 x 4000 | 0 | — | — |
| 4200 x 4200 | 604 | 4090 | yes |
| 5000 x 5000 | 5,894 | 4086 | yes |
| 6000 x 6000 | 14,511 | 4079 | yes |

Maximum error observed: 1.66 pixels. Cells slightly before 4096 are affected too, because their
true nearest source lies past the boundary.

What it is **not**: not raster-edge, not block-boundary (wrong cells cross a 256-row boundary
1.4 percent of the time, which is what chance gives at this distance), and not float32 precision
(at 2000x2000 the only differences are 1e-5, which is float32 noise, and they vanish at a 1e-3
threshold).

## Reproducing it

```python
import numpy as np, tempfile, os
import hazelbean as hb
from osgeo import gdal
from scipy import ndimage

n = 5000
mask = np.random.default_rng(3).random((n, n)) < 0.004
directory = tempfile.mkdtemp()
source = os.path.join(directory, 'source.tif')
dataset = gdal.GetDriverByName('GTiff').Create(source, n, n, 1, gdal.GDT_Byte)
dataset.SetGeoTransform((0, 1, 0, 0, 0, -1))
dataset.GetRasterBand(1).WriteArray(mask.astype(np.uint8))
dataset.GetRasterBand(1).SetNoDataValue(255)
dataset = None

target = os.path.join(directory, 'distance.tif')
hb.distance_transform_edt((source, 1), target)

got = hb.as_array(target).astype('float64')
truth = ndimage.distance_transform_edt(~mask)
wrong = np.abs(got - truth) > 1e-3
print(wrong.sum(), 'wrong; first affected column', np.where(wrong)[1].min())
```

The raster round-trips identically through GDAL, so the input is not the issue: reading it back
gives exactly the array the reference is computed from.

## What we do in the meantime

`global_invest/ntfp/ntfp_accessibility.py` builds its 10 km road-and-river buffer with a disk
dilation (`scipy.ndimage.binary_dilation` with a disk structuring element) rather than a distance
transform. That matches scipy's exact transform on all 36,000,000 pixels of the 6000x6000 test and
costs about the same wall clock, so there is no reason to switch until this is fixed.
