"""Water-supply GEP tasks. First component: hydropower (CWoN resource-rent method).

The hydropower rent derives from CWoN 2024's capitalized wealth (see the functions module for
the identified method and its anchor); the agriculture and household components join here when
their science surfaces.
"""
import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities
from global_invest.water_supply import water_supply_functions as wf


def publish_inputs(p):
    """Every GEP task's first line: the water_supply es_config row and the CWoN data reference
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'water_supply', log=hb.log)
    utilities.hydrate_es_parameters(p, 'water_supply', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def hydropower_rent(p):
    """CWoN capitalized hydropower wealth -> the implied constant annual rent per country."""
    publish_inputs(p)
    p.hydropower_rent_path = os.path.join(p.cur_dir, 'hydropower_rent.csv')
    if not p.run_this:
        return
    if not hb.path_exists(p.hydropower_rent_path):
        wealth = pd.read_stata(p.water_supply_cwon_hydro_wealth_path)
        # The rate is configuration rather than a module constant, so the one number the
        # hydropower figure turns on is visible beside every other service's parameters.
        wf.hydropower_rent_from_wealth(
            wealth,
            capitalization_rate=float(p.water_supply_hydropower_capitalization_rate),
        ).to_csv(p.hydropower_rent_path, index=False)
    return True


def gep_calculation(p):
    """GEP valuation for water_supply: the hydropower component on the r250 country list,
    one row per country. water_supply_gep currently equals the hydropower component; the
    agriculture and household components add columns here when they arrive."""
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'water_supply')
    if already_done:
        return

    hydropower = hb.df_read(p.hydropower_rent_path)
    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = utilities.collapse_countries_to_r250(p.df_countries)[attr_cols]
    df_gep = wf.water_supply_gep_by_country(hydropower, countries)
    if hb.path_exists(getattr(p, 'water_use_components_path', None)):
        components = hb.df_read(p.water_use_components_path)
        df_gep = df_gep.merge(components.drop(columns=['iso3_r250_label']),
                              on='iso3_r250_id', how='left')
    else:
        df_gep['water_use_agriculture_value_added'] = float('nan')
        df_gep['water_use_all_sector_value_added'] = float('nan')
    df_gep['year'] = int(p.gep_base_year)
    # water_supply_gep stays the hydropower component alone: how (and whether) the water-use
    # components combine with it is the account's subgroup question, flagged on the deck.
    df_gep['water_supply_gep'] = df_gep['hydropower_gep']
    # ⚠ The irrigation and domestic columns are VALUE ADDED unless a water share is set, in which
    # case the `_gep` pair appears beside them. Listing only what is present is what keeps the two
    # apart: a fixed list would either drop the denominator or invent an answer.
    water_use_cols = [c for c in ('water_use_irrigation_value_added',
                                  'water_use_domestic_value_added',
                                  'water_use_irrigation_gep', 'water_use_domestic_gep',
                                  'water_use_agriculture_value_added', 'water_use_all_sector_value_added')
                      if c in df_gep.columns]
    utilities.write_gep_by_country(
        p, df_gep[attr_cols + ['year', 'hydropower_gep', 'hydropower_gep_reference_variant']
                  + water_use_cols + ['water_supply_gep']],
        service_results['gep_by_country_base_year'])

    ours = df_gep['hydropower_gep'].sum()
    reference = df_gep['hydropower_gep_reference_variant'].sum()
    n_extra = int(df_gep['hydropower_gep'].notna().sum()
                  - df_gep['hydropower_gep_reference_variant'].notna().sum())
    hb.log(f'Total water_supply GEP (hydropower component) for base year {p.gep_base_year}: '
           f'{ours:,.2f}')
    hb.log(f'  reference-matching variant {reference:,.2f} over {n_extra} fewer countries; '
           f'the gap is the reference exclusions, which the reported value does not apply')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)


def water_use_components(p):
    """The water-use calculation computed from the raw inputs (script-01 cleaning, then script-02
    efficiency x withdrawal products at the survey years), reported as OUR run's components.
    The drive's committed per-country tables are NOT this calculation's outputs -- a newer AQUASTAT
    vintage plus the appendix's deflate-to-2015 step for the all-sector total, a separate
    crop-water calculation for agriculture (see the functions module) -- so they are the comparison
    anchors, logged and pinned in the test suite, never the reported values."""
    publish_inputs(p)
    p.water_use_efficiency_path = os.path.join(p.cur_dir, 'aquastat_water_efficiency_cleaned.csv')
    p.water_use_gep_path = os.path.join(p.cur_dir, 'water_use_gep_by_country_year.csv')
    p.water_use_components_path = os.path.join(p.cur_dir, 'water_use_components.csv')
    if not p.run_this:
        return
    if not hb.path_exists(p.water_use_efficiency_path):
        raw = pd.read_csv(p.water_use_efficiency_input_path, encoding='utf-8-sig')
        wf.clean_aquastat_water_efficiency(raw).to_csv(
            p.water_use_efficiency_path, index=False, encoding='utf-8-sig')
    if not hb.path_exists(p.water_use_gep_path):
        efficiency = pd.read_csv(p.water_use_efficiency_path, encoding='utf-8-sig')
        withdrawal = pd.read_csv(p.water_use_withdrawal_path, encoding='utf-8-sig')
        df_gep = wf.water_use_gep_by_country_year(efficiency, withdrawal)
        df_gep.to_csv(p.water_use_gep_path, index=False, encoding='utf-8-sig')
        sector_cols = ['gep_water_agricultural', 'gep_water_industrial', 'gep_water_municipal']
        hb.log('water_use chain: %d country-year rows, %.4g USD summed over survey years '
               'and sectors' % (len(df_gep), df_gep[sector_cols].sum().sum()))
    if not hb.path_exists(p.water_use_components_path):
        df_gep = pd.read_csv(p.water_use_gep_path, encoding='utf-8-sig')
        name_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name', 'name_long']
        countries = p.df_countries[[c for c in name_cols if c in p.df_countries.columns]].drop_duplicates('iso3_r250_id')
        out = wf.water_use_components_from_chain(df_gep, countries)
        # The chain returns VALUE ADDED, which is what SDG 6.4.1 inverts back to. The account's
        # figure is a share of it, and the share has no default: blank leaves the GEP columns
        # absent, so nothing downstream can read the denominator as the answer.
        # The irrigation premium: what an irrigated hectare earns above the same land rainfed.
        # Published beside the value added whenever both inputs are staged, so the account can see
        # the quantity the share would be applied to.
        premium = None
        premium_path = getattr(p, 'water_use_irrigation_premium_input_path', None)
        cropland_path = getattr(p, 'water_use_cropland_area_input_path', None)
        if hb.path_exists(premium_path) and hb.path_exists(cropland_path):
            aquastat = pd.read_csv(premium_path)
            # The FAOSTAT bulk zip carries the data plus four code tables, so the data member is
            # named rather than letting pandas guess (it refuses a multi-file zip).
            if str(cropland_path).endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(cropland_path) as archive:
                    member = next(n for n in archive.namelist()
                                  if n.endswith('All_Data_(Normalized).csv'))
                    with archive.open(member) as data:
                        land_use = pd.read_csv(data, encoding='utf-8-sig', low_memory=False)
            else:
                land_use = pd.read_csv(cropland_path, encoding='utf-8-sig', low_memory=False)
            premium = wf.irrigation_premium_by_country(
                aquastat, wf.cropland_area_from_faostat(land_use), int(p.gep_base_year))
            hb.log('water_use: irrigation premium %.6g USD over %d countries at %d'
                   % (premium['irrigation_premium_usd'].sum(), len(premium), int(p.gep_base_year)))
            premium.to_csv(os.path.join(p.cur_dir, 'irrigation_premium.csv'), index=False)
        else:
            hb.log('water_use: the premium inputs are not staged, so no premium is computed.')

        # Irrigation GEP: the premium times the rent share. ⚠ The share applies to the PREMIUM,
        # never to the whole irrigated value added -- that is the error the premium exists to fix.
        share = getattr(p, 'water_use_water_share_of_value_added', None)
        irrigation_gep = None
        if share is not None and premium is not None:
            irrigation_gep = wf.irrigation_gep_from_premium(premium, float(share))
            hb.log('water_use: irrigation GEP %.6g USD (premium x rent share %.4g, negatives '
                   'clipped)' % (irrigation_gep['water_use_irrigation_gep'].sum(), float(share)))
            out = out.merge(irrigation_gep[['m49', 'water_use_irrigation_gep']].rename(
                columns={'m49': 'iso3_r250_id'}), on='iso3_r250_id', how='left')
        elif share is None:
            hb.log('water_use: no rent share set, so the account publishes the premium and no '
                   'irrigation GEP.')

        # Domestic GEP: the ecosystem-provided cubic metres times a raw-water price. An explicit
        # price in es_parameters wins; absent one, the price is IMPLIED by irrigation -- what a
        # cubic metre earns in a field bounds what a city would have paid for it.
        withdrawal_path = getattr(p, 'water_use_withdrawal_by_sector_input_path', None)
        if hb.path_exists(withdrawal_path):
            withdrawal = pd.read_csv(withdrawal_path)
            volumes = wf.domestic_withdrawal_by_country(withdrawal, int(p.gep_base_year))
            price = getattr(p, 'water_use_raw_water_price_usd_per_m3', None)
            if price is None and irrigation_gep is not None:
                agricultural_m3 = withdrawal[
                    (withdrawal['Year'] == int(p.gep_base_year))
                    & (withdrawal['VariableCode'] == wf.AQUASTAT_AGRICULTURAL_WITHDRAWAL_CODE)
                ]['Value'].sum() * 1e9
                price = wf.implied_raw_water_price(
                    irrigation_gep['water_use_irrigation_gep'].sum(), agricultural_m3)
                hb.log('water_use: raw-water price implied by irrigation: %.4g USD/m3' % price)
            volumes = wf.apply_raw_water_price(volumes, price)
            if price is None:
                hb.log('water_use: domestic withdrawal %.4g m3 over %d countries at %d; no '
                       'raw-water price available, so the account publishes the VOLUME and no '
                       'GEP.' % (volumes['domestic_withdrawal_m3'].sum(), len(volumes),
                                 int(p.gep_base_year)))
            else:
                hb.log('water_use: domestic GEP %.6g USD at %.4g USD/m3'
                       % (volumes['water_use_domestic_gep'].sum(), float(price)))
                out = out.merge(volumes[['m49', 'water_use_domestic_gep']].rename(
                    columns={'m49': 'iso3_r250_id'}), on='iso3_r250_id', how='left')
            volumes.to_csv(os.path.join(p.cur_dir, 'domestic_withdrawal.csv'), index=False)
        else:
            hb.log('water_use: the sector withdrawal pull is not staged, so no domestic volume '
                   'is computed.')
        out.to_csv(p.water_use_components_path, index=False, encoding='utf-8-sig')
        hb.log('water_use components (OUR chain): agriculture %.4g USD (%d countries), all-sector %.4g USD '
               '(%d countries); %d chain countries without an r250 match' % (
                   out['water_use_agriculture_value_added'].sum(), out['water_use_agriculture_value_added'].notna().sum(),
                   out['water_use_all_sector_value_added'].sum(), out['water_use_all_sector_value_added'].notna().sum(),
                   out['iso3_r250_label'].isna().sum()))
        committed_ag = hb.df_read(p.water_use_agriculture_path)
        committed_all = hb.df_read(p.water_use_all_sector_path)
        hb.log('committed anchors: agriculture %.4g USD, all-sector %.4g USD -- different source '
               'chains (see the functions module), compared in the test suite' % (
                   committed_ag['wateruse_ag_gep'].sum(), committed_all['wateruse_gep'].sum()))
    return True
