"""Fire-protection (wildfire) GEP tasks: persistence betas -> avoided acres -> avoided damage.

Ported from the GEP wildfire repo's R pipeline in the house shape. Two input modes, decided
per task by what is on disk: the FULL calculation reads the ADM2 burned-area panel and the EM-DAT
extract (panel not yet staged -- its own README says it lives on the drive; the open data
ask); until then the burned-area and damage-rate columns come from the frozen reference
output in reference/ (the committed anchor this port reproduces to the float), so the beta
science and all downstream arithmetic run and verify on any machine.

Which of the three GEP variants is the account's fire_protection value is an OPEN question
(baseline and nature-vs-human total NEGATIVE globally); fire_protection_gep defaults to the
nature-vs-human cause difference PROVISIONALLY.
"""
import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities
from global_invest.fire_protection import fire_protection_functions as ffn

FIRE_GEP_PROVISIONAL_VARIANT = 'nn_hh'   # the account's variant: OPEN question, see docstring
EMDAT_SHEET_NAME = 'EM-DAT Data'         # the extract's only data sheet


def read_emdat_wildfire_damages(path):
    """The EM-DAT extract read and totalled per ISO3. See emdat_wildfire_damages."""
    return ffn.emdat_wildfire_damages(pd.read_excel(path, sheet_name=EMDAT_SHEET_NAME))


def publish_inputs(p):
    """Every GEP task's first line: the fire_protection es_config row and the data references
    from es_parameters (defaults layer -- a caller-set value wins), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'fire_protection', log=hb.log)
    utilities.hydrate_es_parameters(p, 'fire_protection', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def beta_differences(p):
    """Per-country persistence betas from the committed regression results (Section 1)."""
    publish_inputs(p)
    p.fire_beta_path = os.path.join(p.cur_dir, 'beta_differences.csv')
    if not p.run_this:
        return
    if not hb.path_exists(p.fire_beta_path):
        regression_df = hb.df_read(p.fire_regression_results_path)
        beta = ffn.compute_beta_differences(regression_df)
        beta.to_csv(p.fire_beta_path, index=False)
    return True


def avoided_burned_area(p):
    """Betas x the country's 2018 burned area -> marginal and total additional acres 2019.
    Burned-area aggregates come from the ADM2 panel when staged, else from the frozen
    reference (loud log; the panel is the open data ask)."""
    publish_inputs(p)
    p.fire_avoided_acres_path = os.path.join(p.cur_dir, 'avoided_acres.csv')
    if not p.run_this:
        return
    if not hb.path_exists(p.fire_avoided_acres_path):
        beta = hb.df_read(p.fire_beta_path)
        if hb.path_exists(getattr(p, 'fire_panel_path', None)):
            panel = hb.df_read(p.fire_panel_path)
            year_2018 = panel[panel['year'] == 2018]
            burned_2018 = (year_2018.groupby('GID_0')
                           .agg(total_burned_areas_ha_2018=('total_burned_areas_ha', 'sum'),
                                n_adm2_units=('total_burned_areas_ha', 'size'))
                           .reset_index())
        else:
            hb.log('fire_protection: ADM2 panel not staged -- 2018 burned areas read from '
                   'the frozen reference output (the open data ask).')
            burned_2018 = hb.df_read(os.path.join(
                utilities.service_data_dir(p, 'fire_protection'), 'GEP_wildfire_2019_results.csv'))[
                ['GID_0', 'total_burned_areas_ha_2018', 'n_adm2_units']]
        ffn.compute_avoided_acres(beta, burned_2018).to_csv(p.fire_avoided_acres_path, index=False)
    return True


def damage_per_acre(p):
    """EM-DAT wildfire damages over all-years burned acres, global-average filled. Needs the
    panel for the denominator; without it the reference's damage-rate column is used (loud
    log). Emits one row per GID_0."""
    publish_inputs(p)
    p.fire_damage_per_acre_path = os.path.join(p.cur_dir, 'damage_per_acre.csv')
    if not p.run_this:
        return
    if not hb.path_exists(p.fire_damage_per_acre_path):
        if hb.path_exists(getattr(p, 'fire_panel_path', None)):
            panel = pd.read_csv(p.fire_panel_path, usecols=['GID_0', 'total_burned_areas_ha'])
            panel = panel.dropna(subset=['GID_0', 'total_burned_areas_ha'])
            burned_acres = (panel.groupby('GID_0', as_index=False)['total_burned_areas_ha'].sum()
                            .rename(columns={'total_burned_areas_ha': 'total_burned_acres_country'}))
            burned_acres['total_burned_acres_country'] *= ffn.HA_TO_ACRES
            damages = read_emdat_wildfire_damages(p.fire_emdat_path)
            rates, _ = ffn.compute_damage_per_acre(burned_acres, damages)
        else:
            hb.log('fire_protection: ADM2 panel not staged -- damage rates read from the '
                   'frozen reference output (the open data ask).')
            rates = hb.df_read(os.path.join(
                utilities.service_data_dir(p, 'fire_protection'), 'GEP_wildfire_2019_results.csv'))[
                ['GID_0', 'damages_usd_per_acre_country']]
        rates.to_csv(p.fire_damage_per_acre_path, index=False)
    return True


def gep_calculation(p):
    """GEP = avoided damage, three variants, one row per country. Writes the reference-shaped
    CSV (diffable against the source repo's committed output) and the house per-country GEP
    table; fire_protection_gep carries the PROVISIONAL variant until the account blesses one."""
    publish_inputs(p)
    # The reference-shaped CSV sits beside the house table, named for what it is: the same
    # columns the source repo's committed output carries, so the two diff directly.
    p.fire_results_reference_shape_path = os.path.join(
        p.cur_dir, 'GEP_wildfire_2019_results.csv')
    service_results, already_done = utilities.begin_gep_calculation(p, 'fire_protection')
    if already_done:
        return

    avoided = hb.df_read(p.fire_avoided_acres_path)
    rates = hb.df_read(p.fire_damage_per_acre_path)
    global_average = float(rates['damages_usd_per_acre_country'].mode().iloc[0])
    out = ffn.compute_gep(avoided, rates, global_average)
    out.to_csv(p.fire_results_reference_shape_path, index=False)

    # House per-country table. GID_0 is a GADM label, not an r250 label: three of the
    # reference's units (Kosovo as XKO, and India's Z01 and Z07) only resolve through the
    # correspondence, so the values are summed per country there rather than joined here.
    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    attrs = utilities.collapse_countries_to_r250(p.df_countries)[attr_cols]
    by_country = ffn.attach_countries(out, p.df_countries)
    df_gep = by_country.merge(attrs, how='left', on='iso3_r250_label')
    df_gep['year'] = int(p.gep_base_year)
    df_gep['fire_protection_gep'] = df_gep[f'GEP_wildfire_2019_{FIRE_GEP_PROVISIONAL_VARIANT}']
    keep_cols = attr_cols + ['year', 'GEP_wildfire_2019_baseline', 'GEP_wildfire_2019_nn_hh',
                             'GEP_wildfire_2019_ttn_tth', 'fire_protection_gep']
    hb.df_write(df_gep[keep_cols], service_results['gep_by_country_base_year'])

    for variant in ('baseline', 'nn_hh', 'ttn_tth'):
        hb.log(f'fire_protection GEP {variant}: '
               f'{out[f"GEP_wildfire_2019_{variant}"].sum():,.2f} USD '
               f'{"(PROVISIONAL account variant)" if variant == FIRE_GEP_PROVISIONAL_VARIANT else ""}')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)
