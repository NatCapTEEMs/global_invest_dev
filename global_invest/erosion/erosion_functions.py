"""Erosion-control ES science: the STATIC per-scenario erosion (sediment-retention) shock.

Currently STATIC: read from a pre-computed per-scenario dependency table (raw_dependencies/
erosion_prevention_dependency.csv), NOT recomputed from our SEALS maps. The paper wants this DYNAMIC
(InVEST SDR on each SEALS map -- Nfamara's global_erosion_gep), which is the heavy compute upgrade (#26).
This module isolates the static read so the seam is in place and the dynamic swap is contained later.
"""
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
