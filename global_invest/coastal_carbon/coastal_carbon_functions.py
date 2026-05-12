"""
Coastal carbon pipeline helpers.

All ecosystem-density and per-country stock primitives used by the four-file
pipeline (run / initialization / tasks / functions) live here. There are no
runtime dependencies on the per-ecosystem reference modules in `archieve/`.
"""

import contextlib
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp
import rasterio.windows
from rasterio.enums import Resampling
from tqdm import tqdm


# ============================================================================
# Per-pixel ecosystem carbon density helpers
# ============================================================================

# Mangrove (Hamilton & Friess 2018 + IPCC 2014 zone BGB + Sanderman 2018 SOC)
MANGROVE_BGB_AGB_RATIO_TROPICAL_WET = 0.49
MANGROVE_BGB_AGB_RATIO_TROPICAL_DRY = 0.29
MANGROVE_BGB_AGB_RATIO_SUBTROPICAL = 0.96
MANGROVE_TROPICAL_LAT_THRESHOLD = 23.5
MANGROVE_TROPICAL_WET_PRECIP_THRESHOLD_MM = 2000.0
MANGROVE_AGB_TO_C = 0.48   # Hamilton & Friess 2018
MANGROVE_BGB_TO_C = 0.39   # Howard et al. 2014


def calculate_mangrove_density_array(latitude_arr, precipitation_arr=None,
                                     soc_arr=None):
    """
    Vectorised mangrove total carbon density (Mg C/ha).

    AGB: Hamilton & Friess (2018) EQ5 latitude regression
        AGB (t/ha) = max(0, -6.4305 * |lat| + 271.747)
    BGB: AGB x IPCC 2014 zone-specific ratio (0.49 tropical wet, 0.29 tropical
         dry, 0.96 subtropical). If precipitation_arr is None, all tropics use
         the wet ratio (0.49). Tropical/subtropical split at |lat| = 23.5 deg.
    SOC: from soc_arr (Sanderman 2018 raster) when provided; otherwise a
         latitude step function (400 to 250 Mg C/ha by zone).
    """
    lat_abs = np.abs(latitude_arr).astype(np.float64)
    agb_t_per_ha = np.maximum(0.0, -6.4305 * lat_abs + 271.747)

    is_tropical = lat_abs < MANGROVE_TROPICAL_LAT_THRESHOLD
    if precipitation_arr is not None:
        is_wet = precipitation_arr > MANGROVE_TROPICAL_WET_PRECIP_THRESHOLD_MM
        bgb_ratio = np.where(
            is_tropical & is_wet, MANGROVE_BGB_AGB_RATIO_TROPICAL_WET,
            np.where(is_tropical & ~is_wet, MANGROVE_BGB_AGB_RATIO_TROPICAL_DRY,
                     MANGROVE_BGB_AGB_RATIO_SUBTROPICAL),
        )
    else:
        bgb_ratio = np.where(
            is_tropical, MANGROVE_BGB_AGB_RATIO_TROPICAL_WET,
            MANGROVE_BGB_AGB_RATIO_SUBTROPICAL,
        )

    bgb_t_per_ha = agb_t_per_ha * bgb_ratio
    agb_c = agb_t_per_ha * MANGROVE_AGB_TO_C
    bgb_c = bgb_t_per_ha * MANGROVE_BGB_TO_C

    if soc_arr is not None:
        # Scale Sanderman 1 m data to 20 cm assuming linear depth profile
        soc = soc_arr.astype(np.float64) * 0.2
    else:
        # Fallback step function scaled from 1 m to 20 cm (multiply by 0.2)
        soc = np.full_like(lat_abs, 50.0, dtype=np.float64)  # was 250
        soc = np.where(lat_abs < 25, 60.0, soc)   # was 300
        soc = np.where(lat_abs < 20, 70.0, soc)   # was 350
        soc = np.where(lat_abs < 15, 75.0, soc)   # was 375
        soc = np.where(lat_abs < 10, 80.0, soc)   # was 400

    return {
        'agb_c_mg_per_ha':   agb_c,
        'bgb_c_mg_per_ha':   bgb_c,
        'soil_c_mg_per_ha':  soc,
        'total_c_mg_per_ha': agb_c + bgb_c + soc,
    }


# Salt marsh (Chmura 2003 AGB + 2.5 BGB ratio + Maxwell 2024 MarSOC)
SALT_MARSH_AGB_MEDIAN_T_HA = 5.0
SALT_MARSH_BGB_AGB_RATIO = 2.5
SALT_MARSH_AGB_TO_C = 0.45
SALT_MARSH_BGB_TO_C = 0.41
SALT_MARSH_TROPICAL_BOOST_LAT = 25.0
SALT_MARSH_TROPICAL_BOOST_FACTOR = 1.2


def calculate_salt_marsh_density_array(latitude_arr, soc_arr=None):
    """
    Vectorised salt marsh total carbon density (Mg C/ha).

    AGB: Chmura et al. 2003 median (5 t/ha) with 1.2x tropical boost (|lat| < 25).
    BGB: AGB x 2.5 (extensive root systems).
    SOC: from soc_arr (Maxwell 2024 MarSOC raster) when provided; otherwise a
         latitude step function (180 to 350 Mg C/ha by zone).
    """
    lat_abs = np.abs(latitude_arr).astype(np.float64)

    agb_t_per_ha = np.where(
        lat_abs < SALT_MARSH_TROPICAL_BOOST_LAT,
        SALT_MARSH_AGB_MEDIAN_T_HA * SALT_MARSH_TROPICAL_BOOST_FACTOR,
        SALT_MARSH_AGB_MEDIAN_T_HA,
    ).astype(np.float64)
    bgb_t_per_ha = agb_t_per_ha * SALT_MARSH_BGB_AGB_RATIO
    agb_c = agb_t_per_ha * SALT_MARSH_AGB_TO_C
    bgb_c = bgb_t_per_ha * SALT_MARSH_BGB_TO_C

    if soc_arr is not None:
        # Scale MarSOC to 20 cm assuming linear depth profile
        soc = soc_arr.astype(np.float64) * 0.2
    else:
        # Fallback step function scaled from 1 m to 20 cm (multiply by 0.2)
        soc = np.full_like(lat_abs, 70.0, dtype=np.float64)   # was 350
        soc = np.where(lat_abs < 45, 50.0, soc)   # was 250
        soc = np.where(lat_abs < 35, 44.0, soc)   # was 220
        soc = np.where(lat_abs < 20, 36.0, soc)   # was 180

    return {
        'agb_c_mg_per_ha':   agb_c,
        'bgb_c_mg_per_ha':   bgb_c,
        'soil_c_mg_per_ha':  soc,
        'total_c_mg_per_ha': agb_c + bgb_c + soc,
    }


# Seagrass (Gomis et al. 2025 genus-specific biomass + Fourqurean 2012 soil scaled to 1 m)
SEAGRASS_GENUS_TOTAL_BIOMASS_C_MG_HA = {
    # Persistent strategy
    'Posidonia':       9.78,
    'Phyllospadix':    3.70,
    'Enhalus':         3.70,
    'Cymodocea':       3.70,
    'Thalassia':       3.70,
    'Amphibolis':      3.70,
    'Thalassodendron': 3.70,
    # Opportunistic
    'Zostera':         0.94,
    'Syringodium':     0.94,
    'Halodule':        0.94,
    # Colonising
    'Ruppia':          0.33,
    'Halophila':       0.19,
}
SEAGRASS_DEFAULT_BIOMASS_C_MG_HA = 1.55  # Gomis 2025 global mean
SEAGRASS_BGB_FRACTION = 0.70
SEAGRASS_AGB_FRACTION = 0.30
SEAGRASS_SOIL_C_MG_HA_20CM = 22.0  # Fourqurean 2012 global mean for top 20 cm
SEAGRASS_NON_MARINE_GENERA = frozenset({
    'Valisneria', 'Trapa', 'Myriophyllum', 'Najas', 'Heteranthera',
    'Stuckenia', 'Utricularia',
})


def _normalize_seagrass_genus(genus_str):
    if not isinstance(genus_str, str):
        return None
    primary = genus_str.strip().split('|')[0].strip()
    if not primary or primary.lower() == 'not reported':
        return None
    return primary.capitalize()


def _seagrass_biomass_for_genus(genus_str):
    """Mg C/ha total biomass for a genus, or None if non-marine."""
    if isinstance(genus_str, str) and genus_str.strip().lower() == 'not reported':
        return SEAGRASS_DEFAULT_BIOMASS_C_MG_HA
    g = _normalize_seagrass_genus(genus_str)
    if g is None:
        return SEAGRASS_DEFAULT_BIOMASS_C_MG_HA
    if g in SEAGRASS_NON_MARINE_GENERA:
        return None
    return SEAGRASS_GENUS_TOTAL_BIOMASS_C_MG_HA.get(g, SEAGRASS_DEFAULT_BIOMASS_C_MG_HA)


def calculate_seagrass_pool_densities_array(genus_array):
    """
    Vectorised per-pool seagrass densities (Mg C/ha) for an array of genus strings.
    Non-marine genera (freshwater plants) are zeroed out across all pools so the
    caller can join by row index without dropping records. Soil density is a
    constant (Fourqurean 2012 global mean for top 20 cm).
    """
    n = len(genus_array)
    agb = np.zeros(n, dtype=np.float64)
    bgb = np.zeros(n, dtype=np.float64)
    soil = np.zeros(n, dtype=np.float64)
    for i, g in enumerate(genus_array):
        biomass = _seagrass_biomass_for_genus(g)
        if biomass is None:
            continue
        agb[i] = biomass * SEAGRASS_AGB_FRACTION
        bgb[i] = biomass * SEAGRASS_BGB_FRACTION
        soil[i] = SEAGRASS_SOIL_C_MG_HA_20CM
    return {
        'agb_c_mg_per_ha':   agb,
        'bgb_c_mg_per_ha':   bgb,
        'soil_c_mg_per_ha':  soil,
        'total_c_mg_per_ha': agb + bgb + soil,
    }


# ============================================================================
# Raster utilities
# ============================================================================

def rasterize_polygons_to_template(vector_path, template_raster_path, out_path,
                                   field=None, default_value=1, dtype='uint8',
                                   nodata=0, all_touched=True):
    """
    Burn polygons from `vector_path` onto the grid of `template_raster_path`.

    Defaults produce a binary mask. Pass `field` to burn a numeric attribute
    (e.g., country id). Vectors are reprojected into the template CRS as needed.
    """
    gdf = gpd.read_file(vector_path)
    with rasterio.open(template_raster_path) as src:
        template_crs = src.crs
        if gdf.crs != template_crs:
            gdf = gdf.to_crs(template_crs)
        meta = src.meta.copy()
        meta.update({'count': 1, 'dtype': dtype, 'nodata': nodata, 'compress': 'lzw'})
        if field is None:
            shapes = ((geom, default_value) for geom in gdf.geometry if geom is not None)
        else:
            shapes = (
                (geom, val)
                for geom, val in zip(gdf.geometry, gdf[field])
                if geom is not None and pd.notna(val)
            )
        arr = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(src.height, src.width),
            fill=nodata,
            transform=src.transform,
            dtype=dtype,
            all_touched=all_touched,
        )
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(arr, 1)
    return out_path


def align_raster_to_template(src_raster_path, template_raster_path, out_path,
                             resampling='average', dtype=None):
    """
    Reproject and resample a source raster to match the grid of a template raster.

    Used to bring external products like Sanderman 2018 SOC or Maxwell 2024
    MarSOC onto the ha_per_cell template so they can be read in lockstep with
    the windowed streaming pass. If `out_path` already exists, returns it
    directly without recomputing.
    """
    if os.path.exists(out_path):
        return out_path

    resampling_method = getattr(Resampling, resampling)
    with rasterio.open(template_raster_path) as tmpl, \
         rasterio.open(src_raster_path) as src:
        out_dtype = dtype or src.dtypes[0]
        out_meta = tmpl.meta.copy()
        out_meta.update({
            'count': 1,
            'dtype': out_dtype,
            'nodata': src.nodata if src.nodata is not None else 0,
            'compress': 'lzw',
        })
        with rasterio.open(out_path, 'w', **out_meta) as dst:
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=tmpl.transform,
                dst_crs=tmpl.crs,
                resampling=resampling_method,
            )
    return out_path


# ============================================================================
# Per-country streaming carbon stock pass
# ============================================================================

def compute_carbon_stock_by_country(ecosystem_mask_path, country_id_raster_path,
                                    ha_per_cell_path, density_func, country_ids,
                                    chunk_rows=2048, extra_raster_paths=None):
    """
    Stream a global per-pixel carbon stock calculation aggregated per country.

    For each row-block, computes density(latitude, **extras) x ha_per_cell x mask
    and accumulates per-country totals via np.bincount.

    Parameters
    ----------
    ecosystem_mask_path : str
        Binary 0/1 mask aligned to ha_per_cell (any positive value is treated as 1).
    country_id_raster_path : str
        Integer country-id raster on the same grid (0 = nodata / outside).
    ha_per_cell_path : str
        Hectares-per-cell raster on the same grid; must have a geographic CRS.
    density_func : callable
        density_func(lat_array, **extras) -> dict with 'agb_c_mg_per_ha',
        'bgb_c_mg_per_ha', 'soil_c_mg_per_ha', 'total_c_mg_per_ha'.
    country_ids : iterable[int]
        Set of valid country ids (used to size accumulators and order output rows).
    chunk_rows : int
        Row-block size to read per iteration.
    extra_raster_paths : dict[str, str], optional
        Mapping kwarg name -> raster path. Each raster must be aligned to the
        ha_per_cell grid (use align_raster_to_template). The block is read per
        chunk and passed as a kwarg to density_func.

    Returns
    -------
    pandas.DataFrame
        Columns: eemarine_r566_id, agb_c_total_mg, bgb_c_total_mg,
        soil_c_total_mg, total_c_total_mg.
    """
    country_ids = np.asarray(list(country_ids), dtype=np.int64)
    max_id = int(country_ids.max()) + 1
    extra_raster_paths = extra_raster_paths or {}

    with contextlib.ExitStack() as stack:
        src_ha = stack.enter_context(rasterio.open(ha_per_cell_path))
        src_mask = stack.enter_context(rasterio.open(ecosystem_mask_path))
        src_cid = stack.enter_context(rasterio.open(country_id_raster_path))
        extra_srcs = {
            name: stack.enter_context(rasterio.open(path))
            for name, path in extra_raster_paths.items()
        }

        if not src_ha.crs.is_geographic:
            raise ValueError(
                f"ha_per_cell raster must be in a geographic CRS (degrees); got {src_ha.crs}"
            )
        height, width = src_ha.height, src_ha.width
        transform = src_ha.transform

        agb_sum = np.zeros(max_id, dtype=np.float64)
        bgb_sum = np.zeros(max_id, dtype=np.float64)
        soil_sum = np.zeros(max_id, dtype=np.float64)
        total_sum = np.zeros(max_id, dtype=np.float64)

        for row_off in tqdm(
            range(0, height, chunk_rows),
            desc=f"Stock {os.path.basename(ecosystem_mask_path)}",
        ):
            rows = min(chunk_rows, height - row_off)
            window = rasterio.windows.Window(0, row_off, width, rows)
            mask_arr = src_mask.read(1, window=window)
            if mask_arr.sum() == 0:
                continue
            ha_arr = src_ha.read(1, window=window).astype(np.float64)
            cid_arr = src_cid.read(1, window=window)

            row_indices = np.arange(row_off, row_off + rows)
            lats_col = np.array(
                [rasterio.transform.xy(transform, r, 0, offset='center')[1]
                 for r in row_indices],
                dtype=np.float64,
            )
            lat_arr = np.broadcast_to(lats_col[:, None], (rows, width))

            valid = (mask_arr > 0) & (cid_arr > 0) & (ha_arr > 0)
            if not valid.any():
                continue

            extras = {}
            for name, src in extra_srcs.items():
                arr = src.read(1, window=window).astype(np.float64)
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)
                extras[name] = arr

            d = density_func(lat_arr, **extras)
            ha_v = ha_arr[valid]
            cid_v = cid_arr[valid].astype(np.int64)

            agb_pix = ha_v * d['agb_c_mg_per_ha'][valid]
            bgb_pix = ha_v * d['bgb_c_mg_per_ha'][valid]
            soil_pix = ha_v * np.nan_to_num(d['soil_c_mg_per_ha'][valid], nan=0.0)
            total_pix = agb_pix + bgb_pix + soil_pix

            agb_sum += np.bincount(cid_v, weights=agb_pix, minlength=max_id)
            bgb_sum += np.bincount(cid_v, weights=bgb_pix, minlength=max_id)
            soil_sum += np.bincount(cid_v, weights=soil_pix, minlength=max_id)
            total_sum += np.bincount(cid_v, weights=total_pix, minlength=max_id)

    rows_out = []
    for cid in country_ids:
        cid_i = int(cid)
        rows_out.append({
            'eemarine_r566_id': cid_i,
            'agb_c_total_mg': agb_sum[cid_i],
            'bgb_c_total_mg': bgb_sum[cid_i],
            'soil_c_total_mg': soil_sum[cid_i],
            'total_c_total_mg': total_sum[cid_i],
        })
    return pd.DataFrame(rows_out)


# ============================================================================
# Per-ecosystem stock orchestrators
# ============================================================================

def compute_mangrove_carbon_stock_with_sanderman(
    project_dir,
    mangrove_mask_path,
    country_id_raster_path,
    ha_per_cell_path,
    country_ids,
    sanderman_soc_path=None,
    precipitation_path=None,
):
    """
    Single-call orchestrator for per-pixel mangrove carbon stock by country.

    Aligns Sanderman 2018 SOC and (optionally) a precipitation raster onto the
    ha_per_cell grid (cached on disk under project_dir), then streams a global
    carbon stock pass.
    """
    extra_raster_paths = {}

    if sanderman_soc_path and os.path.exists(sanderman_soc_path):
        soc_aligned = os.path.join(project_dir, "sanderman_soc_aligned_10sec.tif")
        align_raster_to_template(sanderman_soc_path, ha_per_cell_path, soc_aligned,
                                 resampling='average', dtype='float32')
        extra_raster_paths['soc_arr'] = soc_aligned
        print(f"  Mangrove SOC source: Sanderman 2018 raster ({sanderman_soc_path})")
    else:
        print("  Mangrove SOC source: latitude step function fallback")

    if precipitation_path and os.path.exists(precipitation_path):
        precip_aligned = os.path.join(project_dir, "precipitation_aligned_10sec.tif")
        align_raster_to_template(precipitation_path, ha_per_cell_path, precip_aligned,
                                 resampling='average', dtype='float32')
        extra_raster_paths['precipitation_arr'] = precip_aligned
        print(f"  Mangrove BGB ratio: IPCC zones using precipitation ({precipitation_path})")
    else:
        print("  Mangrove BGB ratio: latitude-only IPCC zones (tropics treated as wet)")

    return compute_carbon_stock_by_country(
        ecosystem_mask_path=mangrove_mask_path,
        country_id_raster_path=country_id_raster_path,
        ha_per_cell_path=ha_per_cell_path,
        density_func=calculate_mangrove_density_array,
        country_ids=country_ids,
        extra_raster_paths=extra_raster_paths,
    )


def compute_salt_marsh_carbon_stock_with_maxwell(
    project_dir,
    salt_marsh_mask_path,
    country_id_raster_path,
    ha_per_cell_path,
    country_ids,
    maxwell_soc_path=None,
):
    """
    Single-call orchestrator for per-pixel salt marsh carbon stock by country.

    Aligns the Maxwell et al. 2024 MarSOC raster onto the ha_per_cell grid
    (cached on disk under project_dir), then streams a global carbon stock pass.
    """
    extra_raster_paths = {}

    if maxwell_soc_path and os.path.exists(maxwell_soc_path):
        soc_aligned = os.path.join(project_dir, "maxwell_marsoc_aligned_10sec.tif")
        align_raster_to_template(maxwell_soc_path, ha_per_cell_path, soc_aligned,
                                 resampling='average', dtype='float32')
        extra_raster_paths['soc_arr'] = soc_aligned
        print(f"  Salt marsh SOC source: Maxwell 2024 MarSOC raster ({maxwell_soc_path})")
    else:
        print("  Salt marsh SOC source: latitude step function fallback")

    return compute_carbon_stock_by_country(
        ecosystem_mask_path=salt_marsh_mask_path,
        country_id_raster_path=country_id_raster_path,
        ha_per_cell_path=ha_per_cell_path,
        density_func=calculate_salt_marsh_density_array,
        country_ids=country_ids,
        extra_raster_paths=extra_raster_paths,
    )
