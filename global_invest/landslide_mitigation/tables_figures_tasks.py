"""
tables_figures_tasks.py
"""
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import pygeoprocessing as pygeo
from osgeo import gdal

import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# optional, map still renders without a basemap
try:
    import contextily as ctx
except ImportError:
    ctx = None


def tables_figures(p):
    """Creates a directory for tables and figures."""
    if p.run_this:
        return p
    return p    


def compute_zonal_statistics(p):
    if p.run_this:
        for year in p.prediction_years:
            csv_out_path = os.path.join(p.tables_figures_dir, f'zonal_statistics_{year}.csv')
            gpkg_out_path = os.path.join(p.tables_figures_dir, f'zonal_statistics_{year}.gpkg')
 
            if os.path.exists(csv_out_path) and os.path.exists(gpkg_out_path) and not p.force_run:
                p.L.info(f'{year}: zonal statistics already exist, skipping.')
                continue
 
            avoided_mortality_path = os.path.join(
                p.valuation_dir, f'avoided_mortality_{year}.tif'
            )
            avoided_mortality_value_path = os.path.join(
                p.valuation_dir, f'avoided_mortality_value_{year}.tif'
            )
 
            # Reuse the same EASE-Grid-reprojected correspondence vector
            # built for build_vsl_raster, rather than re-reprojecting
            corr_ease_path = os.path.join(p.valuation_dir, 'vsl_work', 'corr_ease_vsl.gpkg')
            if not os.path.exists(corr_ease_path):
                raise FileNotFoundError(
                    f'{corr_ease_path} not found -- run build_vsl_raster first '
                    f'(it produces this reprojected correspondence vector).'
                )
 
            corr = gpd.read_file(corr_ease_path, fid_as_index=True)
 
            id_field = 'ee_r264_id'
 
            deaths_stats = pygeo.zonal_statistics(
                (avoided_mortality_path, 1), corr_ease_path,
            )
            value_stats = pygeo.zonal_statistics(
                (avoided_mortality_value_path, 1), corr_ease_path,
            )
 
            corr['avoided_deaths_sum'] = [
                deaths_stats.get(fid, {}).get('sum', None) for fid in corr.index
            ]
            corr['avoided_value_sum_usd'] = [
                value_stats.get(fid, {}).get('sum', None) for fid in corr.index
            ]
            corr['pixel_count'] = [
                deaths_stats.get(fid, {}).get('count', None) for fid in corr.index
            ]
 
            # ---- CSV ----
            csv_cols = [id_field, 'iso3_r250_label', 'name_long', 'income_grp',
                        'region_wb', 'avoided_deaths_sum', 'avoided_value_sum_usd',
                        'pixel_count']
            csv_cols = [c for c in csv_cols if c in corr.columns]
            corr[csv_cols].to_csv(csv_out_path, index=False)
            p.L.info(f'Zonal statistics CSV: {csv_out_path}')
 
            # ---- GPKG (stats + geometry, for mapping) ----
            corr.to_file(gpkg_out_path, driver='GPKG')
            p.L.info(f'Zonal statistics GPKG: {gpkg_out_path}')
 
            total_deaths = corr['avoided_deaths_sum'].sum()
            total_value = corr['avoided_value_sum_usd'].sum()
            p.L.info(f'{year} global totals (zonal): '
                      f'{total_deaths:.4f} avoided deaths, '
                      f'${total_value:,.2f} avoided value')
    return p


def _regression_stars(pval):
    if pd.isna(pval):
        return ''
    if pval < 0.001:
        return '***'
    if pval < 0.01:
        return '**'
    if pval < 0.05:
        return '*'
    return ''
 
 
def _build_publication_table(coef_rows, bottom_rows, dep_label):
    """coef_rows: list of (label, coefficient, std_error, p_value).
    bottom_rows: list of (label, value_string) appended after the
    coefficient block (N, R^2, fixed effects, etc).
    Returns a two-row-per-variable DataFrame: coefficient+stars on one
    row, blank label + '(SE)' on the row beneath - standard econometrics
    table layout.
    """
    rows = []
    for label, coef, se, pval in coef_rows:
        stars = _regression_stars(pval)
        rows.append({' ': label, dep_label: f'{coef:.4f}{stars}'})
        se_text = f'({se:.4f})' if pd.notna(se) else ''
        rows.append({' ': '', dep_label: se_text})
 
    rows.append({' ': '', dep_label: ''})
    for label, value_str in bottom_rows:
        rows.append({' ': label, dep_label: value_str})
 
    return pd.DataFrame(rows)
 
 
def _save_table_multiple_formats(table_df, base_path, title, notes_text):
    """
    Saves {base_path}.csv, .tex, .md 
    """
    csv_path = f'{base_path}.csv'
    tex_path = f'{base_path}.tex'
    md_path = f'{base_path}.md'
 
    table_df.to_csv(csv_path, index=False)
 
    tex_lines = [
        '\\centering',
        table_df.to_latex(index=False, escape=True),
        '\\vspace{2pt}',
        '\\begin{minipage}{0.92\\linewidth}',
        f'\\footnotesize Notes: {notes_text}',
        '\\end{minipage}',
        '',
    ]
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines))
 
    md_lines = [
        table_df.to_markdown(index=False),
        '',
        f'Notes: {notes_text}',
        '',
    ]
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
 
    return csv_path, tex_path, md_path
 
 
VARIABLE_NOTES = {
    'si_observed': 'infinite-slope stability index (higher = more stable)',
    'rain_max_daily': 'annual maximum daily precipitation from ERA5-Land, mm',
    'population_log1p': 'log(1+population) from LandScan',
    'slope_degrees': 'terrain slope in degrees',
    'road_density': 'road length per square kilometer, GRIP4',
}
 
TERM_LABELS = {
    'const': 'Intercept',
    'Intercept': 'Intercept',
    'si_observed': 'Stability Index',
    'rain_max_daily': 'Extreme Precipitation',
    'population_log1p': 'Population (log scale)',
    'slope_degrees': 'Slope',
    'road_density': 'Road Density',
}
 
 
def export_regression_tables(p):
    if p.run_this:
        out_dir = p.tables_figures_dir
 
        # ---- Hazard model ----
        hazard_model_coefficients_path = os.path.join(p.modeling_dir, 'hazard_model_coefficients.json')
        with open(hazard_model_coefficients_path) as f:
            hazard = json.load(f)
 
        hazard_coef_rows = [
            ('Intercept', hazard['alpha_raw'], hazard['std_err']['const'], hazard['p_value']['const']),
            (TERM_LABELS['si_observed'], hazard['beta_si'],
             hazard['std_err']['si_observed'], hazard['p_value']['si_observed']),
            (TERM_LABELS['rain_max_daily'], hazard['beta_rain'],
             hazard['std_err']['rain_max_daily'], hazard['p_value']['rain_max_daily']),
        ]
        hazard_bottom_rows = [
            ('Observations', f"{hazard.get('n_train'):,}"),
            ('Pseudo R-squared (McFadden)', f"{hazard.get('pseudo_r_squared'):.4f}"),
            ('Case-control corrected intercept', f"{hazard.get('alpha_corrected'):.4f}"),
            ('Training AUC', f"{hazard.get('train_auc'):.4f}"),
            ('Held-out AUC', f"{hazard.get('holdout_auc'):.4f}"),
            ('Fixed effects', 'No'),
        ]
        hazard_table = _build_publication_table(hazard_coef_rows, hazard_bottom_rows, 'Pr(Landslide)')
        hazard_notes = (
            'Logit model estimated on a stratified case-control sample; absolute probabilities '
            'corrected for case-control oversampling. Standard errors in '
            'parentheses. * p < 0.05, ** p < 0.01, *** p < 0.001. Variable definitions: '
            f"Stability Index = {VARIABLE_NOTES['si_observed']}; "
            f"Extreme Precipitation = {VARIABLE_NOTES['rain_max_daily']}."
        )
        _save_table_multiple_formats(
            hazard_table, os.path.join(out_dir, 'hazard_model_table'),
            'Hazard Model of Landslide Occurrence', hazard_notes,
        )
 
        # ---- Severity model: single table, hurdle and severity stages as columns ----
        severity_model_coefficients_path = os.path.join(p.modeling_dir, 'severity_model_coefficients.json')
        with open(severity_model_coefficients_path) as f:
            severity = json.load(f)

        def _severity_coef_by_term(params_key, se_key, pval_key):
            return {
                term: (coef, severity[se_key].get(term, np.nan), severity[pval_key].get(term, np.nan))
                for term, coef in severity[params_key].items()
            }

        part_a = _severity_coef_by_term('part_a_params', 'part_a_std_err', 'part_a_p_value')
        part_b = _severity_coef_by_term('part_b_params', 'part_b_std_err', 'part_b_p_value')
        term_order = list(part_a.keys()) + [t for t in part_b if t not in part_a]

        def _severity_stars(pval):
            if pd.isna(pval):
                return ''
            return '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''

        rows = []
        for term in term_order:
            label = TERM_LABELS.get(term, term)
            a_coef, a_se, a_p = part_a.get(term, (np.nan, np.nan, np.nan))
            b_coef, b_se, b_p = part_b.get(term, (np.nan, np.nan, np.nan))
            coef_row = {
                ' ': label,
                'Pr(Mortality > 0)': f'{a_coef:.4f}{_severity_stars(a_p)}' if pd.notna(a_coef) else '',
                'log(Fatalities)': f'{b_coef:.4f}{_severity_stars(b_p)}' if pd.notna(b_coef) else '',
            }
            se_row = {
                ' ': '',
                'Pr(Mortality > 0)': f'({a_se:.4f})' if pd.notna(a_se) else '',
                'log(Fatalities)': f'({b_se:.4f})' if pd.notna(b_se) else '',
            }
            rows.extend([coef_row, se_row])

        rows.extend([
            {' ': 'Observations',
             'Pr(Mortality > 0)': f"{severity.get('n_train_landslides'):,}",
             'log(Fatalities)': f"{severity.get('n_train_fatal'):,}"},
            {' ': 'Pseudo R-squared (McFadden)',
             'Pr(Mortality > 0)': f"{severity.get('part_a_pseudo_r_squared'):.4f}",
             'log(Fatalities)': ''},
            {' ': 'R-squared',
             'Pr(Mortality > 0)': '',
             'log(Fatalities)': f"{severity.get('part_b_r_squared'):.4f}"},
            {' ': "Duan's smearing factor",
             'Pr(Mortality > 0)': '',
             'log(Fatalities)': f"{severity.get('smearing_factor'):.4f}"},
            {' ': 'Fixed effects', 'Pr(Mortality > 0)': 'No', 'log(Fatalities)': 'No'},
        ])

        severity_table = pd.DataFrame(rows)
        severity_notes = (
            'Two-part hurdle model of landslide fatalities. Column (1) is the logistic hurdle stage, '
            'estimated on realized landslide occurrences only. Column (2) is the log-linear severity '
            "stage, estimated on the fatal-event subsample and back-transformed using Duan's (1983) "
            'smearing correction. Standard errors in parentheses. * p < 0.05, ** p < 0.01, *** p < 0.001. '
            'Variable definitions: '
            f"Population (log scale) = {VARIABLE_NOTES['population_log1p']}; "
            f"Extreme Precipitation = {VARIABLE_NOTES['rain_max_daily']}; "
            f"Slope = {VARIABLE_NOTES['slope_degrees']}; "
            f"Road Density = {VARIABLE_NOTES['road_density']}."
        )
        _save_table_multiple_formats(
            severity_table, os.path.join(out_dir, 'severity_combined_table'),
            'Two-Part Hurdle Model of Landslide Fatalities', severity_notes,
        )
 
        p.L.info(f'Publication-style regression tables (csv/tex/md) exported to {out_dir}')
    return p


def _save_table_safe(df, base_path, title, notes_text=None):
    csv_path = f'{base_path}.csv'
    tex_path = f'{base_path}.tex'
    md_path = f'{base_path}.md'
 
    df.to_csv(csv_path, index=False)
 
    tex_lines = ['\\centering', df.to_latex(index=False, escape=True)]
    if notes_text:
        tex_lines += [
            '\\vspace{2pt}',
            '\\begin{minipage}{0.92\\linewidth}',
            f'\\footnotesize Notes: {notes_text}',
            '\\end{minipage}',
        ]
    tex_lines.append('')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines))
 
    md_lines = [df.to_markdown(index=False), '']
    if notes_text:
        md_lines += [f'Notes: {notes_text}', '']
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
 
    return csv_path, tex_path, md_path
 
 
def export_results_tables(p):
    if p.run_this:
        for year in p.prediction_years:
            zonal_csv_path = os.path.join(p.tables_figures_dir, f'zonal_statistics_{year}.csv')
            if not os.path.exists(zonal_csv_path):
                p.L.warning(f'{year}: {zonal_csv_path} not found, skipping.')
                continue
 
            df = pd.read_csv(zonal_csv_path)
 
            # ---- Global summary ----
            global_df = pd.DataFrame([{
                'Avoided mortality': df['avoided_deaths_sum'].sum(),
                'Value (US$ millions)': df['avoided_value_sum_usd'].sum() / 1_000_000,
            }])
            global_df['Avoided mortality'] = global_df['Avoided mortality'].map(lambda v: f'{v:,.2f}')
            global_df['Value (US$ millions)'] = global_df['Value (US$ millions)'].map(lambda v: f'${v:,.2f}')
            _save_table_safe(
                global_df, os.path.join(p.tables_figures_dir, f'global_summary_full_impacts_{year}'),
                f'Global Avoided Mortality and Value, {year}',
            )
 
            # ---- Region summary ----
            if 'region_wb' in df.columns:
                region_df = (
                    df[df['region_wb'] != 'Antarctica']
                    .groupby('region_wb', as_index=False)
                    .agg(
                        avoided_mortality_sum=('avoided_deaths_sum', 'sum'),
                        value_sum=('avoided_value_sum_usd', 'sum'),
                    )
                    .sort_values('value_sum', ascending=False)
                )
                region_df['value_sum'] = region_df['value_sum'] / 1_000_000
                region_out = region_df.rename(columns={
                    'region_wb': 'Geography',
                    'avoided_mortality_sum': 'Avoided mortality',
                    'value_sum': 'Value (US$ millions)',
                })
                region_out['Avoided mortality'] = region_out['Avoided mortality'].map(lambda v: f'{v:,.2f}')
                region_out['Value (US$ millions)'] = region_out['Value (US$ millions)'].map(lambda v: f'${v:,.2f}')
                _save_table_safe(
                    region_out, os.path.join(p.tables_figures_dir, f'region_wb_summary_full_impacts_{year}'),
                    f'Aggregate Benefits by World Bank Region, {year}',
                )
 
            # ---- Top 15 countries ----
            top = df.sort_values('avoided_deaths_sum', ascending=False).head(15).copy()
            top['avoided_value_sum_usd'] = top['avoided_value_sum_usd'] / 1_000_000
            keep_cols = [c for c in ['name_long', 'region_wb', 'avoided_deaths_sum', 'avoided_value_sum_usd']
                         if c in top.columns]
            top_out = top[keep_cols].rename(columns={
                'name_long': 'Country',
                'region_wb': 'World Bank region',
                'avoided_deaths_sum': 'Avoided mortality',
                'avoided_value_sum_usd': 'Value (US$ millions)',
            })
            top_out['Avoided mortality'] = top_out['Avoided mortality'].map(lambda v: f'{v:,.2f}')
            top_out['Value (US$ millions)'] = top_out['Value (US$ millions)'].map(lambda v: f'${v:,.2f}')
            _save_table_safe(
                top_out, os.path.join(p.tables_figures_dir, f'top_countries_mortality_full_impacts_{year}'),
                f'Top 15 Countries by Avoided Landslide Mortality, {year}',
            )
 
            p.L.info(f'{year}: results tables (csv/tex/md) exported to {p.tables_figures_dir}')
    return p



def export_si_severity_sensitivity_table(p):
    """Side-by-side comparison table for the Appendix: main (slope-based)
    vs. sensitivity-check (SI-based) severity specifications.
    """
    if p.run_this:
        severity_model_coefficients_path = os.path.join(p.modeling_dir, 'severity_model_coefficients.json')
        with open(severity_model_coefficients_path) as f:
            main_spec = json.load(f)  # slope-based, current production
 
        si_path = os.path.join(p.modeling_dir, 'severity_model_si_sensitivity.json')
        if not os.path.exists(si_path):
            p.L.warning(f'{si_path} not found -- run estimate_severity_model_si_sensitivity first.')
            return p
        with open(si_path) as f:
            si_spec = json.load(f)
 
        rows = []
        rows.append({'Model': 'Part A Pseudo R-sq.',
                      'Slope-based (main)': f"{main_spec.get('part_a_pseudo_r_squared'):.4f}",
                      'SI-based (sensitivity)': f"{si_spec.get('part_a_pseudo_r_squared'):.4f}"})
        rows.append({'Model': 'Part B R-sq.',
                      'Slope-based (main)': f"{main_spec.get('part_b_r_squared'):.4f}",
                      'SI-based (sensitivity)': f"{si_spec.get('part_b_r_squared'):.4f}"})
        rows.append({'Model': 'Terrain covariate coefficient (Part A)',
                      'Slope-based (main)': f"{main_spec['part_a_params'].get('slope_degrees', float('nan')):.4f}",
                      'SI-based (sensitivity)': f"{si_spec['part_a_params'].get('si_observed', float('nan')):.4f}"})
        rows.append({'Model': 'Terrain covariate p-value (Part A)',
                      'Slope-based (main)': f"{main_spec.get('part_a_p_value', {}).get('slope_degrees', float('nan')):.4g}",
                      'SI-based (sensitivity)': f"{si_spec.get('part_a_p_value', {}).get('si_observed', float('nan')):.4g}"})
        rows.append({'Model': 'Terrain covariate coefficient (Part B)',
                      'Slope-based (main)': f"{main_spec['part_b_params'].get('slope_degrees', float('nan')):.4f}",
                      'SI-based (sensitivity)': f"{si_spec['part_b_params'].get('si_observed', float('nan')):.4f}"})
        rows.append({'Model': 'Terrain covariate p-value (Part B)',
                      'Slope-based (main)': f"{main_spec.get('part_b_p_value', {}).get('slope_degrees', float('nan')):.4g}",
                      'SI-based (sensitivity)': f"{si_spec.get('part_b_p_value', {}).get('si_observed', float('nan')):.4g}"})
        rows.append({'Model': 'N (fatal events)',
                      'Slope-based (main)': f"{main_spec.get('n_train_fatal'):,}",
                      'SI-based (sensitivity)': f"{si_spec.get('n_train_fatal'):,}"})
 
        table_df = pd.DataFrame(rows)
        notes = (
            'Terrain covariate row reports $slope_degrees$ for the main specification and '
            '$si_observed$ for the sensitivity check, both occupy the same role (a terrain-based '
            'severity predictor) in their respective specifications, but are not the same variable.'
        )
        _save_table_safe(
            table_df, os.path.join(p.tables_figures_dir, 'si_severity_sensitivity_table'),
            'Severity Model: Slope-Based vs. Stability-Index-Based', notes,
        )
        p.L.info('SI-severity sensitivity table exported.')
    return p
 
 
def export_pi_audit_table(p):
    """Per-year event/land pixel counts underlying the population
    prevalence estimate, for Appendix transparency.
    """
    if p.run_this:
        out_path = os.path.join(p.tables_figures_dir, 'pi_audit_table')
        if not os.path.exists(f'{out_path}.csv') and not p.force_run:
            rows = []
            total_event_pixels = 0
            total_land_pixels = 0
    
            for year in p.modeling_range:
                binary_path = os.path.join(
                    p.preprocessing_dir, 'uglc_annual_panels', f'uglc_binary_{year}.tif'
                )
                si_path = p.si_paths.get('observed', {}).get(year)
                if not os.path.exists(binary_path) or not si_path or not os.path.exists(si_path):
                    continue
                
                ds = gdal.Open(binary_path)
                band = ds.GetRasterBand(1)
                arr = band.ReadAsArray()
                nodata = band.GetNoDataValue()
                ds = None
    
                si_ds = gdal.Open(si_path)
                si_band = si_ds.GetRasterBand(1)
                si_arr = si_band.ReadAsArray()
                si_nodata = si_band.GetNoDataValue()
                si_ds = None
    
                valid = (arr != nodata) if nodata is not None else np.ones_like(arr, dtype=bool)
                si_valid = (si_arr != si_nodata) if si_nodata is not None else np.ones_like(si_arr, dtype=bool)
                valid &= si_valid
    
                event_pixels = int((arr[valid] == 1).sum())
                land_pixels = int(valid.sum())
                prevalence = event_pixels / land_pixels if land_pixels > 0 else float('nan')
    
                rows.append({
                    'Year': year,
                    'Event pixels': f'{event_pixels:,}',
                    'Land pixels (SI-eligible)': f'{land_pixels:,}',
                    'Prevalence': f'{prevalence:.8f}',
                })
                total_event_pixels += event_pixels
                total_land_pixels += land_pixels
    
            overall_pi = total_event_pixels / total_land_pixels if total_land_pixels > 0 else float('nan')
            rows.append({
                'Year': 'Total',
                'Event pixels': f'{total_event_pixels:,}',
                'Land pixels (SI-eligible)': f'{total_land_pixels:,}',
                'Prevalence': f'{overall_pi:.8f}',
            })
    
            table_df = pd.DataFrame(rows)
            _save_table_safe(
                table_df, out_path,
                'Population Prevalence Audit',
            )
            p.L.info('Pi audit table exported.')
    return p


def plot_global_rasters_png(p):
    failures = []
 
    def _plot_raster(raster_path, out_png, title, cmap, q_low, q_high, cbar_format):
        if os.path.exists(out_png):
            p.L.info(f'✓ Plot already exists: {out_png}')
            return True
        if not os.path.exists(raster_path):
            msg = f'Raster not found: {raster_path}'
            p.L.info(f'WARNING: {msg}')
            failures.append(msg)
            return False
 
        try:
            with rasterio.open(raster_path) as src:
                max_dim = int(getattr(p, 'plot_raster_max_dim', 4096))
                scale = max(src.height / max_dim, src.width / max_dim, 1.0)
                out_h = max(1, int(np.ceil(src.height / scale)))
                out_w = max(1, int(np.ceil(src.width / scale)))
                arr = src.read(
                    1,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.nearest,
                ).astype(np.float32)
                ndv = src.nodata
 
            if ndv is not None:
                arr = np.where(arr == ndv, np.nan, arr)
 
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                msg = f'Raster has no finite data: {raster_path}'
                p.L.info(f'WARNING: {msg}')
                failures.append(msg)
                return False
 
            vmin, vmax = np.nanpercentile(finite, [q_low, q_high])
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                msg = f'Invalid plotting range: {raster_path}'
                p.L.info(f'WARNING: {msg}')
                failures.append(msg)
                return False
 
            if vmin == vmax:
                eps = abs(vmin) * 1e-6 if vmin != 0 else 1e-6
                vmin -= eps
                vmax += eps
 
            fig, ax = plt.subplots(figsize=(9, 4.8), dpi=220)
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_axis_off()
            cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.4)
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(cbar_format))
            plt.tight_layout()
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close(fig)
            p.L.info(f'✓ Saved figure: {out_png}')
            return True
        except Exception as e:
            msg = f'Failed plotting {os.path.basename(raster_path)}: {e}'
            p.L.info(f'WARNING: {msg}')
            failures.append(msg)
            plt.close('all')
            return False
 
    for year in p.prediction_years:
        # ---- Stability Index (SI), observed scenario ----
        si_path = p.si_paths.get('observed', {}).get(year)
        if si_path:
            p.L.info(f'\nPlotting Stability Index for {year}...')
            _plot_raster(
                si_path,
                os.path.join(p.tables_figures_dir, f'si_observed_{year}.png'),
                f'Stability Index (Observed Forest Cover), {year}',
                'RdYlGn',  # red=less stable, green=more stable -- intuitive
                2,
                98,
                '{x:.2f}',
            )
 
        # ---- Hazard probability, per scenario ----
        for scenario_name in p.si_paths.keys():
            p.L.info(f'\nPlotting hazard probability for {year} / {scenario_name}...')
            _plot_raster(
                os.path.join(p.stitch_tiles_dir, f'hazard_prob_{scenario_name}_{year}.tif'),
                os.path.join(p.tables_figures_dir, f'hazard_prob_{scenario_name}_{year}.png'),
                f'Landslide Hazard Probability ({scenario_name}), {year}',
                'Reds',
                2,
                98,
                '{x:.5f}',  # hazard probabilities are small (pixel-year scale)
                            # 3 decimals would show mostly zeros
            )
 
        # ---- Expected deaths, per scenario ----
        for scenario_name in p.si_paths.keys():
            p.L.info(f'\nPlotting expected deaths for {year} / {scenario_name}...')
            _plot_raster(
                os.path.join(p.stitch_tiles_dir, f'expected_deaths_{scenario_name}_{year}.tif'),
                os.path.join(p.tables_figures_dir, f'expected_deaths_{scenario_name}_{year}.png'),
                f'Expected Deaths ({scenario_name}), {year}',
                'Reds',
                2,
                98,
                '{x:.5f}',
            )
 
        # ---- Avoided mortality ----
        p.L.info(f'\nPlotting avoided mortality for {year}...')
        _plot_raster(
            os.path.join(p.valuation_dir, f'avoided_mortality_{year}.tif'),
            os.path.join(p.tables_figures_dir, f'avoided_mortality_{year}.png'),
            f'Avoided Landslide Mortality, {year}',
            'Purples',
            2,
            98,
            '{x:.5f}',
        )
        _plot_raster(
            os.path.join(p.valuation_dir, f'avoided_mortality_value_{year}.tif'),
            os.path.join(p.tables_figures_dir, f'avoided_mortality_value_{year}.png'),
            f'Economic Value of Avoided Mortality, {year}',
            'Greens',
            2,
            98,
            '${x:,.0f}',
        )
 
    if failures:
        p.L.info(f'plot_global_rasters_png completed with {len(failures)} warnings.')
 
    return p


def plot_country_choropleth_maps(p):
    if p.run_this: 
        for year in p.prediction_years:
            gpkg_path = os.path.join(p.tables_figures_dir, f'zonal_statistics_{year}.gpkg')
            if not os.path.exists(gpkg_path):
                p.L.warning(f'{gpkg_path} not found, skipping.')
                continue
 
            gdf = gpd.read_file(gpkg_path)
 
            specs = [
                ('avoided_deaths_sum', None, 'Purples',
                 f'Avoided Landslide Mortality, {year}', 'Avoided deaths',
                 '{:.2f}', f'avoided_mortality_choropleth_{year}.png',
                 [0, 1, 5, 15, 50, 100, float('inf')]),
                ('avoided_value_sum_usd', 1e6, 'Greens',
                 f'Economic Value of Avoided Mortality, {year}', 'Value (US$ millions)',
                 '${:,.2f}M', f'avoided_mortality_value_choropleth_{year}.png',
                 [0, 1, 5, 15, 50, 100, float('inf')]),
            ]
 
            for column, divisor, cmap, title, legend_label, tick_fmt, out_name, bucket_edges in specs:
                out_path = os.path.join(p.tables_figures_dir, out_name)
                if os.path.exists(out_path) and not p.force_run:
                    p.L.info(f'Choropleth already exists: {out_path}')
                    continue
 
                # Prepare and normalize data
                gdf_plot = gdf[gdf[column].notna()].copy()
                gdf_plot[column] = pd.to_numeric(gdf_plot[column], errors='coerce')
                values = gdf_plot[column] / divisor if divisor else gdf_plot[column]
                
                # Assign to buckets
                gdf_plot['bucket'] = pd.cut(values, bins=bucket_edges, labels=False, 
                                             include_lowest=True)
                
                fig, ax = plt.subplots(figsize=(12, 6), dpi=220)
 
                # Plot all countries with white base
                gdf.plot(ax=ax, color='white', edgecolor='#cccccc', linewidth=0.3)
                
                # Color by bucket
                cmap_obj = plt.get_cmap(cmap)
                num_buckets = len(bucket_edges) - 1
                colors = [cmap_obj(i / (num_buckets - 1)) for i in range(num_buckets)]
                
                for bucket_idx in range(num_buckets):
                    bucket_data = gdf_plot[gdf_plot['bucket'] == bucket_idx]
                    if not bucket_data.empty:
                        bucket_data.plot(ax=ax, color=colors[bucket_idx], 
                                        edgecolor='#666666', linewidth=0.3)
                
                # Create legend
                legend_elements = []
                for i in range(num_buckets):
                    lower = bucket_edges[i]
                    upper = bucket_edges[i + 1]
                    if upper == float('inf'):
                        label = f'{tick_fmt.format(lower)}+'
                    else:
                        label = f'{tick_fmt.format(lower)} – {tick_fmt.format(upper)}'
                    legend_elements.append(mpatches.Patch(color=colors[i], label=label))
                
                ax.legend(handles=legend_elements, loc='lower left', frameon=False, 
                         fontsize=8, title=legend_label, title_fontsize=9)
                
                ax.set_axis_off()
                plt.tight_layout()
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                p.L.info(f'Saved choropleth: {out_path}')
    return p


def plot_uglc_from_vector(p):
    """
    Plot UGLC event geometry as a point map with fatality bins.
    """
    if not p.run_this:
        return p
 
    out_png = os.path.join(p.tables_figures_dir, 'uglc_events_fatality_bins.png')
    if os.path.exists(out_png) and not p.force_run:
        p.L.info(f'✓ UGLC events plot already exists: {out_png}')
        return p
 
    p.L.info(f'Plotting UGLC events from: {p.uglc_path}')
    gdf = gpd.read_file(p.uglc_path)
    if gdf.empty:
        raise ValueError('WARNING: UGLC vector is empty, skipping.')
 
    # Points already, not buffered polygons
    points = gdf.to_crs('EPSG:3857').copy()
 
    fatalities = points['fatality_count'].fillna(0).clip(lower=0)
    nonfatal = fatalities <= 0
    bins = {
        '1-5': (fatalities >= 1) & (fatalities < 5),
        '5-25': (fatalities >= 5) & (fatalities < 25),
        '25-100': (fatalities >= 25) & (fatalities < 100),
        '100+': fatalities >= 100,
    }
 
    size_map = {'1-5': 12, '5-25': 24, '25-100': 44, '100+': 80}
    color_map = {
        '1-5': '#f28e2b',
        '5-25': '#e15759',
        '25-100': '#b51d39',
        '100+': '#5b0f1f',
    }
 
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=220)
    if nonfatal.any():
        points.loc[nonfatal].plot(
            ax=ax, color='#6c757d', markersize=6, alpha=0.32, linewidth=0, zorder=2,
        )
 
    for label, mask in bins.items():
        if mask.any():
            points.loc[mask].plot(
                ax=ax, color=color_map[label], markersize=size_map[label],
                alpha=0.82, linewidth=0, zorder=3,
            )
 
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', label='Nonfatal landslide',
               markerfacecolor='#6c757d', markeredgecolor='none', markersize=6, alpha=0.4),
        Line2D([0], [0], marker='o', color='none', label='1-5 deaths',
               markerfacecolor=color_map['1-5'], markeredgecolor='none', markersize=5, alpha=0.9),
        Line2D([0], [0], marker='o', color='none', label='5-25 deaths',
               markerfacecolor=color_map['5-25'], markeredgecolor='none', markersize=6, alpha=0.9),
        Line2D([0], [0], marker='o', color='none', label='25-100 deaths',
               markerfacecolor=color_map['25-100'], markeredgecolor='none', markersize=7, alpha=0.9),
        Line2D([0], [0], marker='o', color='none', label='100+ deaths',
               markerfacecolor=color_map['100+'], markeredgecolor='none', markersize=9, alpha=0.9),
    ]
 
    ax.legend(
        handles=legend_handles, title='Fatality count', loc='lower left',
        frameon=True, framealpha=0.9, facecolor='white', edgecolor='none',
        title_fontproperties={'family': 'serif', 'size': 9},
        prop={'family': 'serif', 'size': 8},
    )
 
    if ctx is not None:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels,
                             crs=points.crs, attribution=False)
        except Exception as e:
            p.L.info(f'WARNING: basemap fetch failed ({e}), continuing without it.')
 
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    p.L.info(f'✓ Saved figure: {out_png}')
    return p

