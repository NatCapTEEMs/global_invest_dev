"""
model_tasks.py
"""
import os
import json
import numpy as np
import pandas as pd
from osgeo import gdal
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score


def modeling(p):
    """
    Task to create a modeling_dir for the downstream tasks.
    """
    if p.run_this:
        os.makedirs(p.modeling_dir, exist_ok=True)
        return p
    return p

def calibrate_si_to_probability(p):
    if p.run_this:
        out_coef_path = os.path.join(p.modeling_dir, 'hazard_model_coefficients.json')
        if os.path.exists(out_coef_path) and not p.force_run:
            with open(out_coef_path) as f:
                p.hazard_model_coefficients = json.load(f)
            return p
 
        panel = pd.read_csv(p.estimation_table_path)
 
        train = panel[panel['year'].isin(set(p.modeling_range))].copy()
        holdout = panel[panel['year'].isin(set(p.prediction_years))].copy()
 
        # ---- Fit raw logistic on the case-control sample ----
        X_train = sm.add_constant(train[['si_observed', 'rain_max_daily']])
        y_train = train['case']
        raw_model = sm.Logit(y_train, X_train).fit(disp=False)
        p.L.info(f'Raw case-control logistic fit:\n{raw_model.summary()}')
 
        # ---- Case-control intercept correction (Prentice & Pyke, 1979) ----
        # Slope coefficients from ordinary logistic on case-control data
        # are consistent estimates of the true population slopes -- but
        # the intercept is biased because the sample over-represents
        # cases relative to the real population. Correct via the offset:
        #   alpha_corrected = alpha_raw - log((tau/(1-tau)) / (pi/(1-pi)))
        # where tau = fraction of cases in the SAMPLE, pi = fraction of
        # cases in the true POPULATION (estimated below from the UGLC
        # binary panels' actual hit-rate across all land pixel-years).
        tau = y_train.mean()
 
        # Estimate population prevalence pi from the UGLC annual binary
        # panels: (total event pixel-years) / (total land pixel-years)
        # across the SAME years used for training.
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
            valid &= si_valid  # restrict to the SI-eligible (slope-passing) population
 
            total_event_pixels += int((arr[valid] == 1).sum())
            total_land_pixels += int(valid.sum())
 
        pi = total_event_pixels / total_land_pixels if total_land_pixels > 0 else tau
        p.L.info(f'Sample case fraction (tau) = {tau:.6f}, '
                  f'estimated population prevalence (pi) = {pi:.8f}')
 
        offset = np.log((tau / (1 - tau)) / (pi / (1 - pi)))
        alpha_corrected = raw_model.params['const'] - offset
 
        coefficients = {
            'alpha_raw': float(raw_model.params['const']),
            'alpha_corrected': float(alpha_corrected),
            'beta_si': float(raw_model.params['si_observed']),
            'beta_rain': float(raw_model.params['rain_max_daily']),
            'tau_sample_case_fraction': float(tau),
            'pi_population_prevalence': float(pi),
            'n_train': int(len(train)),
            'pseudo_r_squared': float(raw_model.prsquared),
            'std_err': {k: float(v) for k, v in raw_model.bse.items()},
            'z_stat': {k: float(v) for k, v in raw_model.tvalues.items()},
            'p_value': {k: float(v) for k, v in raw_model.pvalues.items()},
        }
 
        # ---- Training-set AUC, for comparison against the held-out AUC.
        train_pred_raw = raw_model.predict(X_train)
        train_auc = roc_auc_score(y_train, train_pred_raw)
        coefficients['train_auc'] = float(train_auc)
        p.L.info(f'Training AUC: {train_auc:.4f}')

        # ---- Validate on held-out prediction_years 
        if len(holdout) > 0:
            X_holdout = sm.add_constant(
                holdout[['si_observed', 'rain_max_daily']], has_constant='add'
            )
            holdout_pred_raw = raw_model.predict(X_holdout)
            auc = roc_auc_score(holdout['case'], holdout_pred_raw)
            coefficients['holdout_auc'] = float(auc)
            p.L.info(f'Held-out ({sorted(set(p.prediction_years))}) AUC: {auc:.4f}')

        with open(out_coef_path, 'w') as f:
            json.dump(coefficients, f, indent=2)
        p.L.info(f'Hazard model coefficients saved: {out_coef_path}')
        p.hazard_model_coefficients = coefficients
    return p
 
 

def estimate_severity_model(p):
    if p.run_this:
        out_coef_path = os.path.join(p.modeling_dir, 'severity_model_coefficients.json')
        if os.path.exists(out_coef_path) and not p.force_run:
            with open(out_coef_path) as f:
                p.severity_model_coefficients = json.load(f)
            return p
 
        panel = pd.read_csv(p.estimation_table_path)
        train = panel[
            (panel['year'].isin(set(p.modeling_range))) & (panel['case'] == 1)
        ].copy()
 
        train['population_log1p'] = np.log1p(train['population'].clip(lower=0))
        train['fatality_occurred'] = (train['fatality_count'] > 0).astype(int)
 
        base_formula = 'population_log1p + rain_max_daily + slope_degrees + road_density'
 
        # ---- Part A: P(fatality > 0 | landslide occurred) ----
        logit_formula = f'fatality_occurred ~ {base_formula}'
        try:
            part_a_model = smf.logit(logit_formula, data=train).fit(disp=False)
        except np.linalg.LinAlgError:
            p.L.warning(
                'Standard MLE logit fit failed (likely separation even '
                'without GAEZ FE) -- falling back to L2-regularized fit. '
                'Coefficients will be shrunk toward 0 and standard errors '
                'from the regularized fit are not directly comparable to '
                'an unpenalized fit -- treat this as a point-estimate '
                'fallback, not full inference.'
            )
            part_a_model = smf.logit(logit_formula, data=train).fit_regularized(
                alpha=1.0, disp=False
            )
        p.L.info(f'Severity part A (fatality occurrence | landslide):\n{part_a_model.summary()}')
 
        # ---- Part B: E[log(fatalities) | fatality > 0] ----
        positive = train[train['fatality_count'] > 0].copy()
        if len(positive) < 10:
            p.L.warning(
                f'Only {len(positive)} fatal-landslide rows available for '
                f'part B -- coefficients will be unstable. Consider a '
                f'coarser covariate set or pooling more years.'
            )
        positive['log_fatalities'] = np.log(positive['fatality_count'])
        ols_formula = f'log_fatalities ~ {base_formula}'
        part_b_model = smf.ols(ols_formula, data=positive).fit()
        p.L.info(f'Severity part B (log fatalities | fatal):\n{part_b_model.summary()}')
 
        # ---- Duan's smearing correction ----
        # exp(predicted log fatalities) is a biased (too-low) estimate of
        # E[fatalities] due to Jensen's inequality on the log transform.
        # Duan (1983) smearing: multiply by the mean of exp(residuals)
        # from the log-scale fit, rather than assuming log-normal errors.
        residuals = part_b_model.resid
        smearing_factor = float(np.mean(np.exp(residuals)))
        p.L.info(f"Duan's smearing factor: {smearing_factor:.4f}")
 
        coefficients = {
            'part_a_params': {k: float(v) for k, v in part_a_model.params.items()},
            'part_a_std_err': {k: float(v) for k, v in part_a_model.bse.items()},
            'part_a_z_stat': {k: float(v) for k, v in part_a_model.tvalues.items()},
            'part_a_p_value': {k: float(v) for k, v in part_a_model.pvalues.items()},
            'part_a_pseudo_r_squared': float(part_a_model.prsquared),
            'part_b_params': {k: float(v) for k, v in part_b_model.params.items()},
            'part_b_std_err': {k: float(v) for k, v in part_b_model.bse.items()},
            'part_b_t_stat': {k: float(v) for k, v in part_b_model.tvalues.items()},
            'part_b_p_value': {k: float(v) for k, v in part_b_model.pvalues.items()},
            'part_b_r_squared': float(part_b_model.rsquared),
            'smearing_factor': smearing_factor,
            'n_train_landslides': int(len(train)),
            'n_train_fatal': int(len(positive)),
        }
 
        with open(out_coef_path, 'w') as f:
            json.dump(coefficients, f, indent=2)
        p.L.info(f'Severity model coefficients saved: {out_coef_path}')
        p.severity_model_coefficients = coefficients
    return p


def estimate_severity_model_si_sensitivity(p):
    """
    Sensitivity check: does SI subsume slope's information in severity?
 
    Diagnostic only -- writes to a separate output, does NOT overwrite
    the main severity_model_coefficients.json used by prediction.
    """
    if p.run_this:
        out_coef_path = os.path.join(p.modeling_dir, 'severity_model_si_sensitivity.json')
        if os.path.exists(out_coef_path) and not p.force_run:
            with open(out_coef_path) as f:
                p.severity_model_si_sensitivity = json.load(f)
            return p
 
        panel = pd.read_csv(p.estimation_table_path)
        train = panel[
            (panel['year'].isin(set(p.modeling_range))) & (panel['case'] == 1)
        ].copy()
 
        train['population_log1p'] = np.log1p(train['population'].clip(lower=0))
        train['fatality_occurred'] = (train['fatality_count'] > 0).astype(int)
 
        # slope_degrees -> si_observed, road_density kept
        si_formula = 'population_log1p + rain_max_daily + road_density + si_observed'
 
        logit_formula = f'fatality_occurred ~ {si_formula}'
        part_a_model = smf.logit(logit_formula, data=train).fit(disp=False)
        p.L.info(f'[SI sensitivity] Severity part A:\n{part_a_model.summary()}')
 
        positive = train[train['fatality_count'] > 0].copy()
        positive['log_fatalities'] = np.log(positive['fatality_count'])
        ols_formula = f'log_fatalities ~ {si_formula}'
        part_b_model = smf.ols(ols_formula, data=positive).fit()
        p.L.info(f'[SI sensitivity] Severity part B:\n{part_b_model.summary()}')
 
        residuals = part_b_model.resid
        smearing_factor = float(np.mean(np.exp(residuals)))
 
        result = {
            'formula': si_formula,
            'part_a_pseudo_r_squared': float(part_a_model.prsquared),
            'part_a_params': {k: float(v) for k, v in part_a_model.params.items()},
            'part_a_p_value': {k: float(v) for k, v in part_a_model.pvalues.items()},
            'part_b_r_squared': float(part_b_model.rsquared),
            'part_b_params': {k: float(v) for k, v in part_b_model.params.items()},
            'part_b_p_value': {k: float(v) for k, v in part_b_model.pvalues.items()},
            'smearing_factor': smearing_factor,
            'n_train_landslides': int(len(train)),
            'n_train_fatal': int(len(positive)),
        }
 
        # Quick side-by-side vs. the baseline (slope, no SI)
        baseline_path = os.path.join(p.modeling_dir, 'severity_model_coefficients.json')
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                baseline = json.load(f)
            p.L.info(
                f'\nCOMPARISON (baseline slope-based vs. SI-based):\n'
                f'  Part A pseudo R2: {baseline.get("part_a_pseudo_r_squared"):.4f} '
                f'(baseline) vs. {result["part_a_pseudo_r_squared"]:.4f} (SI)\n'
                f'  Part B R2:        {baseline.get("part_b_r_squared"):.4f} '
                f'(baseline) vs. {result["part_b_r_squared"]:.4f} (SI)'
            )
 
        with open(out_coef_path, 'w') as f:
            json.dump(result, f, indent=2)
        p.L.info(f'SI sensitivity results saved: {out_coef_path}')
        p.severity_model_si_sensitivity = result
    return p

