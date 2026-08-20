# =============================================================================
# erosion_utils.py
#
# Generic, non-erosion-specific helper functions pulled out of the three
# original research notebooks (step1_sdr_invest_run, Combine_PS_SES,
# combined_maps_figures). Per Justin's 2026-07-07 email, these are the
# kind of raster-IO / reprojection / zonal-stat / plotting helpers that
# other GEP service modules re-implement too, and are the first
# candidates to be pulled up into a shared global_invest.utils (or
# hazelbean) once the group compares notes.
#
# NOTE: assert_exists() below was defined 3x, identically, across the
# original notebooks (SDR script called its copy `_assert_exists`) --
# consolidated here to a single implementation.
# =============================================================================
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features
import rioxarray as rxr
import xarray as xr
import requests
from rasterio.enums import Resampling
from rasterio.crs import CRS as rioCRS
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import warnings

_HTTP_TIMEOUT = 60
_RETRY = 4

# Defaults for the plotting helpers below (overridden at run time by
# erosion_functions.configure_maps(p) if the project sets different values).
EXCLUDE_ISO3 = {"ATA"}
ROBINSON_CRS = "+proj=robin"
USD_TO_MILLIONS = 1e6
TOP_N = 20


def assert_exists(p: Path, hint: str = ""):
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}\n{hint}")



def _normcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _http_get(url, params=None, headers=None, stream=False):
    last_err = None
    for attempt in range(_RETRY):
        try:
            r = requests.get(url, params=params, headers=headers,
                             timeout=_HTTP_TIMEOUT, stream=stream)
            if r.status_code == 200:
                return r
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
        time.sleep(1 + attempt)
    raise last_err

def open_raster_1band(path: Path) -> xr.DataArray:
    """Open a single-band raster as a 2D DataArray (masked)."""
    return rxr.open_rasterio(path, masked=True).squeeze()

def _ensure_crs(da: xr.DataArray, name: str) -> xr.DataArray:
    if da.rio.crs is None:
        raise ValueError(f"{name} raster has no CRS. Fix metadata before running.")
    return da

def reproject_to_analysis_grid(da: xr.DataArray, analysis_crs: rioCRS, resampling: Resampling) -> xr.DataArray:
    """Reproject to equal-area CRS if needed (does NOT match a template grid)."""
    _ensure_crs(da, "input")
    if da.rio.crs == analysis_crs:
        return da
    return da.rio.reproject(analysis_crs, resampling=resampling)

def pixel_area_hectares(da: xr.DataArray) -> float:
    """Pixel area (ha) in a projected CRS."""
    if da.rio.crs is None or (not da.rio.crs.is_projected):
        raise ValueError("pixel_area_hectares requires a projected CRS.")
    res_x, res_y = map(abs, da.rio.resolution())
    return (res_x * res_y) / 10_000.0

def _clean_nonneg(da: xr.DataArray) -> xr.DataArray:
    """Convert negative to 0, keep NaNs as NaN."""
    out = da.copy()
    vals = out.values
    vals = np.where(np.isfinite(vals), np.maximum(vals, 0.0), np.nan)
    out.values = vals
    return out

def _clip01_arr(arr: np.ndarray) -> np.ndarray:
    out = arr.astype("float32", copy=False)
    out = np.where(np.isfinite(out), np.clip(out, 0.0, 1.0), np.nan).astype("float32")
    return out

def _write_share(path: Path, template: xr.DataArray, arr01: np.ndarray):
    """Write a float32 share raster (0–1) aligned to template."""
    da = xr.DataArray(arr01.astype("float32"), coords=template.coords, dims=template.dims)
    da = da.rio.write_crs(template.rio.crs, inplace=False)
    da = da.rio.write_transform(template.rio.transform(), inplace=False)
    da.rio.to_raster(path, compress="deflate", nodata=np.float32(-9999))

def _bincount_weighted_mean(ids: np.ndarray, x: np.ndarray, max_id: int) -> np.ndarray:
    """Compute mean(x) by integer id (1..max_id). ids and x must be 1D aligned."""
    ok = np.isfinite(x) & (ids > 0)
    if not np.any(ok):
        return np.full(max_id + 1, np.nan, dtype="float64")
    ids_ok = ids[ok].astype("int32", copy=False)
    x_ok   = x[ok].astype("float64", copy=False)
    s = np.bincount(ids_ok, weights=x_ok, minlength=max_id + 1).astype("float64")
    c = np.bincount(ids_ok, minlength=max_id + 1).astype("float64")
    return np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)


def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def savefig(path: Path, dpi: int = 300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def top_n(df: pd.DataFrame, col: str, n: int = TOP_N) -> pd.DataFrame:
    d = df[np.isfinite(df[col])].copy()
    return d.sort_values(col, ascending=False).head(n)


def pick_iso3_column(gdf: gpd.GeoDataFrame) -> str | None:
    candidates = ["iso3", "ISO3", "iso_a3", "ADM0_A3", "adm0_a3", "ISO_A3", "iso3_r250_label"]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def pick_name_column(gdf: gpd.GeoDataFrame) -> str | None:
    candidates = [
        "country_name", "NAME_EN", "ADMIN", "NAME_LONG", "NAME",
        "COUNTRY", "NAME_0", "ADM0_NAME", "GEOUNIT", "iso3_r250_name"
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def fmt_usd_millions(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:,.1f}"
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
        lo = edges[i]
        hi = edges[i + 1]
        if label_format == "usd_millions":
            lo_txt = fmt_usd_millions(lo)
            hi_txt = fmt_usd_millions(hi)
        else:
            lo_txt = fmt_percent(lo)
            hi_txt = fmt_percent(hi)
        labels.append(f"{lo_txt} – {hi_txt}")
    return labels


def compute_classification(values: pd.Series, scheme: str = "fisher_jenks", k: int = 5):
    s = pd.to_numeric(values, errors="coerce")
    m = np.isfinite(s)
    clean = s[m]

    if clean.empty:
        return pd.Series(index=values.index, dtype="float64"), np.array([0.0, 1.0])

    try:
        import mapclassify

        scheme = (scheme or "fisher_jenks").lower()
        k_eff = min(k, int(clean.nunique()))
        k_eff = max(k_eff, 1)

        if scheme == "fisher_jenks":
            classifier = mapclassify.FisherJenks(clean.to_numpy(), k=k_eff)
        elif scheme == "equal_interval":
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
        edges = [intervals[0].left]
        for iv in intervals:
            edges.append(iv.right)
        return codes, np.asarray(edges, dtype=float)



def plot_raster_global(tif_path: Path, title: str, out_png: Path, downsample_factor: int = 6):
    assert_exists(tif_path)
    da = rxr.open_rasterio(tif_path, masked=True).squeeze()

    if downsample_factor and downsample_factor > 1:
        da = da.isel(
            y=slice(None, None, downsample_factor),
            x=slice(None, None, downsample_factor),
        )

    arr = da.values.astype("float32", copy=False)
    arr = np.where(np.isfinite(arr), arr, np.nan)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(arr, interpolation="nearest")
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Share (0–1)", fontsize=12)
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
        [labels[int(x)] if np.isfinite(x) and int(x) < len(labels) else np.nan for x in gdf["_class_id"]],
        categories=labels,
        ordered=True,
    )

    cmap = mpl.cm.get_cmap(mpl.rcParams["image.cmap"], n_classes)
    color_list = [mpl.colors.to_hex(cmap(i)) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_axis_off()

    gdf.plot(
        column="_class_label",
        ax=ax,
        cmap=mpl.colors.ListedColormap(color_list),
        legend=False,
        linewidth=0.35,
        edgecolor="white",
        missing_kwds={"color": "lightgrey", "edgecolor": "white"},
    )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(title, fontsize=16, pad=14)

    handles = [Patch(facecolor=color_list[i], edgecolor="none", label=labels[i]) for i in range(n_classes)]
    handles.append(Patch(facecolor="lightgrey", edgecolor="none", label="No data"))

    leg = ax.legend(
        handles=handles,
        title=legend_title,
        loc=legend_loc,
        frameon=True,
        fontsize=10,
        title_fontsize=11,
        borderpad=0.8,
        labelspacing=0.5,
        handlelength=1.6,
        handletextpad=0.6,
    )
    leg.get_frame().set_alpha(0.95)

    savefig(out_png, dpi=300)
