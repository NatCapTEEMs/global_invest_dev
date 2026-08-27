# -*- coding: utf-8 -*-
import hashlib
import io
import json
import os
import logging
from datetime import datetime, timezone
import urllib.request
import zipfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hazelbean as hb


WORLD_BANK_CSV_URL = (
    "https://api.worldbank.org/v2/en/indicator/{indicator_code}"
    "?downloadformat=csv"
)


def _validate_wdi_csv(csv_bytes, indicator_code, years_to_validate):
    """Validate World Bank CSV structure and required year columns."""
    first_line = csv_bytes.decode(
        "utf-8-sig", errors="replace"
    ).splitlines()[0].lstrip()
    skiprows = 4 if first_line.startswith(('Data Source', '"Data Source"')) else 0
    header = pd.read_csv(io.BytesIO(csv_bytes), skiprows=skiprows, nrows=1)
    if "Country Code" not in header.columns:
        raise ValueError(f"Downloaded {indicator_code} CSV lacks Country Code.")
    missing_years = [
        year for year in years_to_validate if str(year) not in header.columns
    ]
    if missing_years:
        raise ValueError(
            f"Downloaded {indicator_code} CSV lacks required years {missing_years}."
        )


def download_world_bank_indicator(
    indicator_code: str,
    target_dir: str,
    fallback_path: str,
    required_year=None,
    required_years=None,
    refresh: bool = False,
):
    """Download a World Bank indicator CSV, falling back to a local snapshot.

    The World Bank CSV endpoint returns a ZIP archive. The primary data CSV is
    extracted to a deterministic filename in the ProjectFlow input directory.
    Existing project downloads are reused unless ``refresh`` is true.

    Returns
    -------
    tuple[str, bool]
        Selected CSV path and whether a fresh API download succeeded.
    """
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"{indicator_code}.csv")
    if os.path.exists(target_path) and not refresh:
        logging.info(f"Using cached World Bank indicator: {target_path}")
        return target_path, False

    url = WORLD_BANK_CSV_URL.format(indicator_code=indicator_code)
    temp_path = f"{target_path}.download"
    years_to_validate = list(required_years or [])
    if required_year is not None:
        years_to_validate.append(required_year)
    years_to_validate = list(dict.fromkeys(years_to_validate))
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "global-invest-extractive-materials/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            archive_bytes = response.read()

        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            prefix = f"API_{indicator_code}_DS2_en_csv"
            data_members = [
                member
                for member in archive.namelist()
                if os.path.basename(member).startswith(prefix)
                and member.lower().endswith(".csv")
            ]
            if len(data_members) != 1:
                raise ValueError(
                    f"Expected one data CSV for {indicator_code}; found {data_members}."
                )
            archive_member = data_members[0]
            csv_bytes = archive.read(archive_member)

        _validate_wdi_csv(csv_bytes, indicator_code, years_to_validate)

        with open(temp_path, "wb") as target_file:
            target_file.write(csv_bytes)
        os.replace(temp_path, target_path)

        manifest = {
            "indicator_code": indicator_code,
            "source_url": url,
            "archive_member": archive_member,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        }
        manifest_path = f"{target_path}.manifest.json"
        manifest_temp_path = f"{manifest_path}.download"
        with open(manifest_temp_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
            manifest_file.write("\n")
        os.replace(manifest_temp_path, manifest_path)

        logging.info(f"Downloaded World Bank indicator {indicator_code} to {target_path}.")
        return target_path, True
    except Exception as error:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(fallback_path):
            try:
                with open(fallback_path, "rb") as fallback_file:
                    fallback_bytes = fallback_file.read()
                _validate_wdi_csv(
                    fallback_bytes,
                    indicator_code,
                    years_to_validate,
                )
            except Exception as fallback_error:
                raise RuntimeError(
                    f"World Bank API download failed for {indicator_code}, and local "
                    f"fallback is invalid: {fallback_path}"
                ) from fallback_error
            logging.warning(
                f"World Bank API download failed for {indicator_code}: {error}. "
                f"Using local fallback {fallback_path}."
            )
            return fallback_path, False
        raise RuntimeError(
            f"World Bank API download failed for {indicator_code}, and local fallback "
            f"does not exist: {fallback_path}"
        ) from error

def _read_wdi_indicator(path: str, value_name: str):
    """Read one World Bank indicator CSV and reshape annual values to long format."""
    try:
        with open(path, encoding="utf-8-sig") as source_file:
            first_line = source_file.readline().lstrip()
        skiprows = 4 if first_line.startswith(('Data Source', '"Data Source"')) else 0
        df_raw = pd.read_csv(path, skiprows=skiprows, encoding="utf-8-sig")
        logging.info(f"Loaded World Bank indicator from {path} ({df_raw.shape[0]} rows).")
    except Exception as e:
        logging.error(f"Failed to read World Bank indicator file '{path}': {e}")
        raise

    year_columns = [column for column in df_raw.columns if str(column).isdigit()]
    df = df_raw.melt(
        id_vars=["Country Code"],
        value_vars=year_columns,
        var_name="year",
        value_name=value_name,
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df


def read_mineral_values(path: str):
    """Read World Bank mineral rents as a percentage of GDP."""
    return _read_wdi_indicator(path, "mineral_rent")


def read_GDP_values(path: str):
    """Read World Bank GDP in current local currency units."""
    return _read_wdi_indicator(path, "GDP_current_LCU")


def read_GDP_deflator_values(path: str):
    """Read World Bank GDP deflator index values."""
    return _read_wdi_indicator(path, "GDP_deflator")


def read_PPP_values(path: str):
    """Read World Bank GDP PPP conversion factors in LCU per international dollar."""
    return _read_wdi_indicator(path, "PA.NUS.PPP")

def group_countries(df: pd.DataFrame):
    """
    Aggregate total GEP across all countries by year.
    """
    df_gep_by_year = hb.df_groupby(df, groupby_cols='year', agg_cols="Value", preserve='keep_all_valid')

    
    # START HERE: df_gep_by_year = hb.df_groupby(df, groupby_cols='iso3_r250_label', agg_dict={"Value": "sum"}). This line causes a really wrongly formatted DataFrame.
    df_gep_by_year.set_index("year", inplace=False)
    # df_gep_by_year.rename(columns={"gep": "total_gep"}, inplace=True)
    df_gep_by_year.sort_values("year", inplace=True)
    logging.info(f"Grouped total by year ({df_gep_by_year.shape[0]} rows).")
    return df_gep_by_year
