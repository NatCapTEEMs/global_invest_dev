"""Erosion-control ES science: the STATIC per-scenario erosion (sediment-retention) shock.

Currently STATIC: read from a pre-computed per-scenario dependency table (raw_dependencies/
erosion_prevention_dependency.csv), NOT recomputed from our SEALS maps. The paper wants this DYNAMIC
(InVEST SDR on each SEALS map -- the erosion GEP model), which is the heavy compute upgrade (#26).
This module isolates the static read so the seam is in place and the dynamic swap is contained later.
"""
import numpy as np
import pandas as pd


def read_erosion_dependency(ero_path):
    """Load + normalize the erosion dependency table; return (df, base_vals).

    base_vals = the baseline_ignore_damages row values (per aez18_id x r50 region), i.e. the
    subtraction reference each scenario is measured against.
    """
    df = pd.read_csv(ero_path)
    df['scenario'] = df['scenario'].str.replace('_2050', '').str.replace('2023.0', 'baseline_2023')
    base = df[df['scenario'] == 'baseline_ignore_damages']
    base_vals = base.set_index(['aez18_id', 'gtapv7_r50_label'])['value'].astype(float).fillna(0)
    return df, base_vals


def find_scenario(df, candidates):
    """First candidate present in df['scenario'], else None."""
    for c in candidates:
        if c in df['scenario'].values:
            return c
    return None


# ---------------------------------------------------------------------------
# DYNAMIC valuation helpers (#26): SPAM crop -> supply-elasticity crosswalk.
# The 4-letter SPAM band codes are aliased to the elasticity table's crop names.
# ---------------------------------------------------------------------------
SPAM_ALIAS_MAP = {
    "whea": ["wheat"], "rice": ["rice"], "maiz": ["maize", "corn"], "barl": ["barley"],
    "sorg": ["sorghum"], "mill": ["millet", "small millet"], "pmil": ["pearl millet"],
    "pota": ["potato"], "cass": ["cassava"], "soyb": ["soybean", "soy"],
    "grou": ["groundnut", "peanut"], "cott": ["cotton"], "sugc": ["sugarcane"],
    "bana": ["banana"], "plnt": ["plantain"], "coco": ["cocoa"],
    "coff": ["arabica coffee", "coffee"], "rcof": ["robusta coffee"], "teas": ["tea"],
    "toba": ["tobacco"], "toma": ["tomato"], "onio": ["onion"],
    "vege": ["vegetable", "other vegetables"], "sunf": ["sunflower"], "rape": ["rapeseed", "canola"],
    "sesa": ["sesame"], "citr": ["citrus"], "lent": ["lentil"], "bean": ["bean"],
    "chic": ["chickpea"], "cowp": ["cowpea"], "pige": ["pigeon pea"], "yams": ["yams"],
    "swpo": ["sweet potato"], "sugb": ["sugarbeet"], "oilp": ["oilpalm", "oil palm"],
    "cnut": ["coconut"], "ocer": ["other cereals"], "orts": ["other roots"],
    "opul": ["other pulses"], "ooil": ["other oil crops"], "ofib": ["other fibre crops"],
    "rubb": ["rubber"], "trof": ["other tropical fruit"], "temf": ["temperate fruit"],
    "rest": ["rest of crops"],
}


def load_erosion_elasticity_map(elasticity_csv):
    """Return {crop_key (lowercased) -> supply elasticity in [0,1]} from the elasticity CSV.

    Accepts a crop-name column among crop/monfreda_crop/item/item_name plus an 'elasticity' column.
    """
    df = pd.read_csv(elasticity_csv, encoding='utf-8-sig')
    df.columns = [str(c).strip().lower() for c in df.columns]
    crop_col = next((c for c in ('crop', 'monfreda_crop', 'item', 'item_name') if c in df.columns), None)
    if crop_col is None or 'elasticity' not in df.columns:
        return {}
    df['elasticity'] = pd.to_numeric(df['elasticity'], errors='coerce').clip(0.0, 1.0)
    key = df[crop_col].astype(str).str.strip().str.lower()
    keep = key.ne('') & df['elasticity'].notna()
    df = df[keep].assign(__k=key[keep]).drop_duplicates('__k', keep='last')
    return dict(zip(df['__k'], df['elasticity']))


def get_erosion_elasticity(crop_key, elast_map, fallback=0.08):
    """crop_key -> elasticity: direct hit, else SPAM alias, else the flat fallback."""
    k = str(crop_key).strip().lower()
    v = elast_map.get(k, np.nan)
    if np.isfinite(v):
        return float(np.clip(v, 0.0, 1.0))
    for alias in SPAM_ALIAS_MAP.get(k, []):
        v2 = elast_map.get(str(alias).strip().lower(), np.nan)
        if np.isfinite(v2):
            return float(np.clip(v2, 0.0, 1.0))
    return float(np.clip(fallback, 0.0, 1.0))
