# =============================================================================
# flood_utils.py
#
# Generic, non-flood-specific helper functions pulled out of the original
# Flood GEP scripts and notebooks (download_and_prep_jrc_flood_depth,
# qa_spa_global_step1, sda_step2_build_sda_global, serviceflow_step3,
# flood_gep_step4b/4c/4d, build_sda_from_esa300m).
#
# Same rationale as erosion_utils.py: per Justin's 2026-07-07 email these
# raster-IO / masking / zonal-stat / plotting helpers are re-implemented in
# every GEP service module and are the first candidates to be pulled up into
# a shared global_invest.utils (or hazelbean).
#
# NOTE ON CONSOLIDATION: several of these were defined more than once across
# the original flood scripts, with slightly different names but identical
# bodies -- consolidated here to a single implementation each:
#   load_admin0()        was in qa_spa_global_step1, serviceflow_step3,
#                        sda_step2_build_sda_global, road_sda (4 copies)
#   pick_iso3_column()   was `find_iso3_col` in step4b, inline in the others
#   pixel_area_km2()     was in serviceflow_step3 and sda_step2
#   _norm/_find_col()    was duplicated verbatim in step4c and step4d
#   integrate_trapezoid() was in step4c and step4d
# =============================================================================
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
# The source repo pinned `gpd.options.io_engine = 'fiona'` here, to work around a GDAL-data
# sentinel check that pyogrio failed in that particular conda environment. That is removed, for
# two reasons. It is a dependency we do not need -- geopandas 1.x defaults to pyogrio, which works
# here, and fiona is not installed -- and `gpd.options` is process-global, so importing this module
# reconfigured geopandas for every other service in the same interpreter. It did: four
# terrestrial_carbon tests began failing on `to_file` the moment this module was imported first.
# An environment-specific workaround does not belong in a shared library; if a machine needs fiona
# it can install it and set the engine itself.
import rasterio
from rasterio.windows import Window

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

# Defaults for the plotting helpers below (overridden at run time by
# flood_functions.configure_maps(p) if the project sets different values).
EXCLUDE_ISO3 = {"ATA"}
ROBINSON_CRS = "+proj=robin"
USD_TO_MILLIONS = 1e6
TOP_N = 20


# -----------------------------------------------------------------------------
# Existence / assertions
# -----------------------------------------------------------------------------
def assert_exists(path: Path, hint: str = ""):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}\n{hint}")


def assert_same_grid(src_a, src_b, label_a: str = "A", label_b: str = "B", rtol: float = 1e-6):
    """
    Hard-lock the alignment principle used throughout the flood pipeline:
    depth rasters, LULC and SDA must share CRS + transform + shape. We never
    silently warp the accounting grid; if this fails, re-align the *input*
    to the LULC grid first.
    """
    problems = []
    if src_a.crs != src_b.crs:
        problems.append(f"CRS differs: {label_a}={src_a.crs} vs {label_b}={src_b.crs}")
    if (src_a.width, src_a.height) != (src_b.width, src_b.height):
        problems.append(
            f"Shape differs: {label_a}=({src_a.height},{src_a.width}) "
            f"vs {label_b}=({src_b.height},{src_b.width})"
        )
    ta, tb = src_a.transform, src_b.transform
    for name, va, vb in zip("abcdef", ta[:6], tb[:6]):
        if not np.isclose(va, vb, rtol=rtol, atol=1e-9):
            problems.append(f"Transform.{name} differs: {va} vs {vb}")
    if problems:
        raise ValueError(
            f"Grid mismatch between {label_a} and {label_b}:\n  " + "\n  ".join(problems)
        )
    return True


# -----------------------------------------------------------------------------
# Raster metadata / geometry
# -----------------------------------------------------------------------------
def pixel_area_m2(transform) -> float:
    return abs(float(transform.a) * float(transform.e))


def pixel_area_km2(transform) -> float:
    return pixel_area_m2(transform) / 1e6


def mercator_area_scale(transform, row_off: int, height: int) -> np.ndarray:
    """
    Areal scale factor for EPSG:3857, as a column vector of shape (height, 1).

        true_ground_area = nominal_pixel_area * cos^2(latitude)

    Web Mercator is conformal, not equal-area: it preserves shape and inflates
    area by 1/cos^2(lat), exactly. Reading pixel area off the affine transform
    therefore gives the value at the EQUATOR ONLY, and applying it globally
    overstates area everywhere else:

        latitude    overstatement
           0            1.00x
          45            2.00x
          60            4.00x
          65            5.60x
          70            8.55x

    Every country outside the tropics is affected, progressively worse toward
    the poles. This is analytic, not an approximation -- cos^2(lat) IS the areal
    scale factor for Web Mercator -- so correcting per pixel gives true ground
    area without reprojecting anything.

    Area depends only on latitude, hence only on raster ROW, so one value per
    row broadcasts across the whole tile.
    """
    R = 6378137.0                      # WGS84 semi-major axis, EPSG:3857 sphere
    rows = np.arange(row_off, row_off + height, dtype="float64") + 0.5
    y = float(transform.f) + rows * float(transform.e)
    lat = 2.0 * np.arctan(np.exp(y / R)) - np.pi / 2.0
    return (np.cos(lat) ** 2).astype("float32").reshape(-1, 1)


def raster_profile_string(ds) -> str:
    return (
        f"CRS: {ds.crs}\n"
        f"Transform: {ds.transform}\n"
        f"Width x Height: {ds.width} x {ds.height}\n"
        f"Res (approx): {ds.transform.a:.4f} x {abs(ds.transform.e):.4f}\n"
        f"Dtype: {ds.dtypes[0]}\n"
        f"Nodata: {ds.nodata}\n"
        f"Bounds: {ds.bounds}\n"
    )


def warn_if_geographic(ds, label: str = "raster"):
    """Pixel area from an affine transform is only m^2 in a projected CRS."""
    if ds.crs is not None and ds.crs.is_geographic:
        warnings.warn(
            f"[WARN] {label} CRS is geographic (degrees). Pixel area from the "
            f"transform is NOT m^2. Reproject to a projected CRS aligned to the "
            f"LULC grid before running valuation."
        )
        return True
    return False


def random_windows(width: int, height: int, n: int, wsize: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        col = int(rng.integers(0, max(1, width - wsize)))
        row = int(rng.integers(0, max(1, height - wsize)))
        yield Window(
            col_off=col, row_off=row,
            width=min(wsize, width - col), height=min(wsize, height - row),
        )


def atomic_write_raster(final_path: Path, profile: dict, array: np.ndarray, band: int = 1):
    """
    Write to <name>.tmp then rename, so a killed job never leaves a
    half-written GeoTIFF that a later --skip-done run mistakes for complete.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(array, band)
    tmp.replace(final_path)
    return final_path


def raster_ok(path: Path) -> bool:
    """Cheap validity probe used by the smart-skip logic."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as ds:
            _ = ds.profile
        return True
    except (OSError, rasterio.errors.RasterioIOError):
        return False


# -----------------------------------------------------------------------------
# Fingerprinting (smart-skip / provenance)
# -----------------------------------------------------------------------------
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_fingerprint(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": st.st_size,
        "mtime": st.st_mtime,
    }


# -----------------------------------------------------------------------------
# Admin0 / ISO3 handling
# -----------------------------------------------------------------------------
def pick_iso3_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    candidates = ["iso3", "ISO3", "iso_a3", "ISO_A3", "ADM0_A3", "adm0_a3", "iso3_r250_label"]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def pick_name_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    candidates = [
        "country_name", "NAME_EN", "ADMIN", "NAME_LONG", "NAME",
        "COUNTRY", "NAME_0", "ADM0_NAME", "GEOUNIT", "iso3_r250_name",
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def load_admin0(path: Path, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Load Admin0 polygons, normalize the ISO3 column to lowercase 'iso3',
    repair invalid geometries with buffer(0), drop empties.
    """
    path = Path(path)
    assert_exists(path, "Admin0 boundary file is required.")
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Admin0 has no CRS: {path}")
    iso_col = pick_iso3_column(gdf)
    if iso_col is None:
        raise ValueError(f"No ISO3-like column found. Columns: {list(gdf.columns)}")
    gdf["iso3"] = gdf[iso_col].astype(str).str.upper().str.strip()
    gdf["geometry"] = gdf["geometry"].buffer(0)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


# -----------------------------------------------------------------------------
# Column detection (tolerant of underscore/space/case differences)
# -----------------------------------------------------------------------------
def norm_label(s: str) -> str:
    s = str(s).strip().lower().replace("_", " ")
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return " ".join(s.split())


def find_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    norm_map: Dict[str, str] = {norm_label(c): c for c in df.columns}
    for cand in candidates:
        k = norm_label(cand)
        if k in norm_map:
            return norm_map[k]
    for cand in candidates:  # contains-match fallback
        k = norm_label(cand)
        for kk, orig in norm_map.items():
            if k in kk:
                return orig
    return None


def to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def write_csv(df: pd.DataFrame, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# -----------------------------------------------------------------------------
# Numerics
# -----------------------------------------------------------------------------
def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.nanmean(x)) if x.size else float("nan")


def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:

    """

    Trapezoidal integral, with the x axis sorted first.



    np.trapezoid weights each segment by diff(x). Handing it a descending or

    unordered x makes backwards segments contribute negatively, and the partial

    cancellation returns a small number of the wrong sign rather than an error.

    In this pipeline x is exceedance probability derived from return periods,

    which arrive in descending-p order naturally, so the sort is not optional.



    Both current call sites happen to sort beforehand -- one by argsort, one via

    a pandas groupby whose sorted-key behaviour is a default rather than a

    guarantee. Sorting here makes the property hold by construction. Sorting an

    already-sorted array is a no-op, so this cannot change existing results.

    """

    x = np.asarray(x, dtype="float64")

    y = np.asarray(y, dtype="float64")

    if x.size != y.size:

        raise ValueError(f"integrate_trapezoid: length mismatch {x.size} vs {y.size}")

    o = np.argsort(x)

    x, y = x[o], y[o]

    if hasattr(np, "trapezoid"):

        return float(np.trapezoid(y, x))

    return float(np.trapz(y, x))





def rp_to_p(rp: float) -> float:
    """Return period (years) -> annual exceedance probability."""
    rp = float(rp)
    return 1.0 / rp if rp > 0 else np.nan


# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------
def fmt_usd_millions(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if abs(x) >= 10:
        return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.1f}"
    if abs(x) >= 1:
        return f"{x:,.1f}"
    return f"{x:,.2f}"


def fmt_percent(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def fmt_usd(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"${x:,.0f}"


def build_interval_labels(edges: np.ndarray, label_format: str = "usd_millions") -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if label_format == "usd_millions":
            lo_txt, hi_txt = fmt_usd_millions(lo), fmt_usd_millions(hi)
        else:
            lo_txt, hi_txt = fmt_percent(lo), fmt_percent(hi)
        labels.append(f"{lo_txt} - {hi_txt}")
    return labels


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def savefig(path: Path, dpi: int = 300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def top_n(df: pd.DataFrame, col: str, n: int = None) -> pd.DataFrame:
    n = TOP_N if n is None else n
    d = df[np.isfinite(pd.to_numeric(df[col], errors="coerce"))].copy()
    return d.sort_values(col, ascending=False).head(n)


def compute_classification(values: pd.Series, scheme: str = "fisher_jenks", k: int = 5):
    s = pd.to_numeric(values, errors="coerce")
    m = np.isfinite(s)
    clean = s[m]

    if clean.empty:
        return pd.Series(index=values.index, dtype="float64"), np.array([0.0, 1.0])

    try:
        import mapclassify

        scheme = (scheme or "fisher_jenks").lower()
        k_eff = max(min(k, int(clean.nunique())), 1)

        if scheme == "equal_interval":
            classifier = mapclassify.EqualInterval(clean.to_numpy(), k=k_eff)
        elif scheme == "quantiles":
            classifier = mapclassify.Quantiles(clean.to_numpy(), k=k_eff)
        else:
            classifier = mapclassify.FisherJenks(clean.to_numpy(), k=k_eff)

        edges = np.concatenate(([clean.min()], np.asarray(classifier.bins, dtype=float)))
        class_ids = pd.Series(np.nan, index=values.index)
        class_ids.loc[m] = classifier.yb
        return class_ids, edges

    except Exception:
        warnings.warn("mapclassify unavailable or failed; falling back to qcut quantiles.")
        q = min(k, max(1, int(clean.nunique())))
        cats = pd.qcut(clean, q=q, duplicates="drop")
        codes = pd.Series(np.nan, index=values.index)
        codes.loc[m] = cats.cat.codes.astype(float)
        intervals = cats.cat.categories
        edges = [intervals[0].left] + [iv.right for iv in intervals]
        return codes, np.asarray(edges, dtype=float)


def plot_raster_global(tif_path: Path, title: str, out_png: Path,
                       downsample_factor: int = 6, cbar_label: str = "Value"):
    """Downsampled quicklook of a global raster (never loads it at full res)."""
    tif_path = Path(tif_path)
    assert_exists(tif_path)
    with rasterio.open(tif_path) as ds:
        step = max(1, int(downsample_factor))
        out_h = max(1, ds.height // step)
        out_w = max(1, ds.width // step)
        arr = ds.read(1, out_shape=(out_h, out_w)).astype("float32")
        if ds.nodata is not None:
            arr = np.where(arr == ds.nodata, np.nan, arr)
    arr = np.where(np.isfinite(arr), arr, np.nan)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(arr, interpolation="nearest")
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    savefig(out_png, dpi=300)


def plot_publication_choropleth_categorical(
    world_joined: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    out_png: Path,
    legend_title: str,
    scheme: str = "fisher_jenks",
    k: int = 5,
    value_unit: str = "raw",
    label_format: str = "usd_millions",
    legend_loc: str = "lower left",
):
    gdf = world_joined.copy()

    if "iso3" in gdf.columns:
        gdf = gdf[~gdf["iso3"].isin(EXCLUDE_ISO3)].copy()
    gdf = gdf[gdf.geometry.notna()].copy()

    if value_col not in gdf.columns:
        warnings.warn(f"Column not found for map: {value_col}")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_axis_off()
        ax.set_title(f"{title}\n[missing column: {value_col}]", fontsize=16, pad=14)
        savefig(out_png, dpi=300)
        return

    if value_unit == "usd_millions":
        gdf["_plot_value"] = pd.to_numeric(gdf[value_col], errors="coerce") / USD_TO_MILLIONS
    else:
        gdf["_plot_value"] = pd.to_numeric(gdf[value_col], errors="coerce")

    try:
        gdf = gdf.to_crs(ROBINSON_CRS)
    except Exception as e:
        warnings.warn(f"CRS transform failed ({e}). Plotting in native CRS.")

    minx, miny, maxx, maxy = gdf.total_bounds
    class_ids, edges = compute_classification(gdf["_plot_value"], scheme=scheme, k=k)

    valid_codes = pd.Series(class_ids).dropna()
    if valid_codes.empty:
        warnings.warn(f"No valid data for map: {value_col}")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_axis_off()
        ax.set_title(title, fontsize=16, pad=14)
        savefig(out_png, dpi=300)
        return

    n_classes = int(valid_codes.max()) + 1
    labels = build_interval_labels(edges[:n_classes + 1], label_format=label_format)

    gdf["_class_id"] = pd.Series(class_ids, index=gdf.index)
    gdf["_class_label"] = pd.Categorical(
        [labels[int(x)] if np.isfinite(x) and int(x) < len(labels) else np.nan
         for x in gdf["_class_id"]],
        categories=labels, ordered=True,
    )

    try:
        cmap = mpl.colormaps[mpl.rcParams["image.cmap"]].resampled(n_classes)
    except Exception:  # matplotlib < 3.6
        cmap = mpl.cm.get_cmap(mpl.rcParams["image.cmap"], n_classes)
    color_list = [mpl.colors.to_hex(cmap(i)) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_axis_off()
    gdf.plot(
        column="_class_label", ax=ax,
        cmap=mpl.colors.ListedColormap(color_list),
        legend=False, linewidth=0.35, edgecolor="white",
        missing_kwds={"color": "lightgrey", "edgecolor": "white"},
    )
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(title, fontsize=16, pad=14)

    handles = [Patch(facecolor=color_list[i], edgecolor="none", label=labels[i])
               for i in range(n_classes)]
    handles.append(Patch(facecolor="lightgrey", edgecolor="none", label="No data"))
    leg = ax.legend(
        handles=handles, title=legend_title, loc=legend_loc, frameon=True,
        fontsize=10, title_fontsize=11, borderpad=0.8, labelspacing=0.5,
        handlelength=1.6, handletextpad=0.6,
    )
    leg.get_frame().set_alpha(0.95)
    savefig(out_png, dpi=300)
