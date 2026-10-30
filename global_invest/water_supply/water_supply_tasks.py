"""Country valuations for hydropower, agricultural water and domestic water."""
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


HYDROPOWER_RENT_SIGNATURE = 'hydropower_rent_signature.json'


def hydropower_rent(p):
    """The hydropower table: the CWoN-wealth annual value the account reports, and beside it the
    authors' 2019 resource rent, revenues minus total costs from the World Bank's hydropower
    valuation workbook, with its sign and a status per country.

    The workbook is not public. Where it is staged it is checked against its recorded sha256;
    where it is not, the run goes on with the revenue-minus-cost columns missing and says so.
    """
    publish_inputs(p)
    p.hydropower_rent_path = os.path.join(p.cur_dir, 'hydropower_rent.csv')
    if not p.run_this:
        return
    wealth_path = str(p.water_supply_cwon_hydro_wealth_path)
    workbook_path = str(p.water_supply_cwon_hydro_workbook_path)
    staged = hb.path_exists(workbook_path)
    if staged:
        found = utilities.sha256_file(workbook_path)
        if found != str(p.water_supply_cwon_hydro_workbook_sha256).strip().lower():
            raise ValueError('%s has sha256 %s, not the recorded water_supply_cwon_hydro_workbook_sha256 %s'
                             % (workbook_path, found, p.water_supply_cwon_hydro_workbook_sha256))
    # The rate is configuration rather than a module constant, so the one number the
    # wealth-based figure turns on is visible beside every other service's parameters.
    rate = float(p.water_supply_hydropower_capitalization_rate)
    inputs = {'wealth': utilities.file_fingerprint(wealth_path)}
    if staged:
        inputs['workbook'] = utilities.file_fingerprint(workbook_path)
    signature = utilities._signature({'capitalization_rate': rate, 'workbook_staged': staged}, inputs)
    signature_path = os.path.join(p.cur_dir, HYDROPOWER_RENT_SIGNATURE)
    reason = utilities.outputs_reuse_reason([p.hydropower_rent_path], signature, signature_path)
    if reason is None:
        hb.log('  hydropower: reusing %s (same inputs, same settings)' % p.hydropower_rent_path)
        return True
    hb.log('  hydropower: computing because %s' % reason)
    database = None
    if staged:
        database = pd.read_excel(workbook_path, sheet_name=wf.HYDROPOWER_WORKBOOK_SHEET)
    else:
        hb.log('  hydropower: the valuation workbook is not staged at %s, so the revenue-minus-cost '
               'columns are missing' % workbook_path)
    table = wf.hydropower_rent_table(pd.read_stata(wealth_path), rate, database)
    table.to_csv(p.hydropower_rent_path, index=False)
    utilities.write_outputs_signature(signature, signature_path)
    counts = table['hydropower_rent_status'].value_counts().to_dict()
    hb.log('  hydropower: wealth-based value %.2f over %d countries; revenue minus cost by status %s'
           % (table['hydropower_gep'].sum(), int(table['hydropower_gep'].notna().sum()), counts))
    return True


def gep_calculation(p):
    """Separate hydropower, agriculture and domestic values on the r250 country list."""
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'water_supply')
    dependencies = [p.hydropower_rent_path, p.water_use_components_path,
                    p.water_supply_agriculture_country_path, __file__, wf.__file__]
    outputs = [service_results['gep_by_country_base_year']]
    if already_done and utilities.reuse_reason(
            p, 'water_supply', outputs, light_inputs=dependencies) is None:
        return

    if not p.run_this:
        return p
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
    # The domestic value is the authors' final municipal table (their least-cost replacement
    # model). The water-use chain's implied-price construction stays beside it as a variant.
    df_gep = df_gep.rename(columns={'water_use_domestic_gep': 'water_use_domestic_gep_implied_price_variant'})
    municipal = wf.municipal_gep_by_country(pd.read_csv(p.water_supply_municipal_results_path), countries)
    df_gep = df_gep.merge(municipal, on='iso3_r250_label', how='left', validate='one_to_one')
    agriculture = pd.read_csv(p.water_supply_agriculture_country_path)
    # The premium remains an input to the implied-price variant only.
    df_gep = df_gep.drop(columns=['water_use_irrigation_gep'], errors='ignore')
    df_gep = df_gep.merge(agriculture.drop(columns=['year']),
                          on='iso3_r250_label', how='left', validate='one_to_one')
    df_gep['year'] = int(p.gep_base_year)
    # water_supply_gep is the hydropower component alone: whether and how the water-use
    # components combine with it is an open question for the account's subgroups.
    df_gep['water_supply_gep'] = df_gep['hydropower_gep']
    # The value-added columns are denominators. water_use_irrigation_gep is the agriculture
    # component from agriculture_water, not the premium x share product dropped above. Listing
    # only what is present keeps denominators and answers apart.
    water_use_cols = [c for c in ('water_use_irrigation_value_added',
                                  'water_use_domestic_value_added',
                                  'water_use_irrigation_gep', 'water_use_domestic_gep',
                                  'water_use_domestic_gep_implied_price_variant',
                                  'water_use_agriculture_value_added', 'water_use_all_sector_value_added')
                      if c in df_gep.columns]
    water_use_cols += [c for c in agriculture.columns
                       if c.startswith('water_use_') and c not in water_use_cols]
    utilities.write_gep_by_country(
        p, df_gep[attr_cols + ['year', 'hydropower_gep', 'hydropower_gep_reference_variant']
                  + water_use_cols + ['water_supply_gep']],
        service_results['gep_by_country_base_year'])

    utilities.write_reuse_signature(p, 'water_supply', outputs, light_inputs=dependencies)
    ours = df_gep['hydropower_gep'].sum()
    reference = df_gep['hydropower_gep_reference_variant'].sum()
    n_extra = int(df_gep['hydropower_gep'].notna().sum()
                  - df_gep['hydropower_gep_reference_variant'].notna().sum())
    hb.log(f'Total water_supply GEP (hydropower component) for base year {p.gep_base_year}: '
           f'{ours:,.2f}')
    hb.log(f'  reference-matching variant {reference:,.2f} over {n_extra} fewer countries; '
           f'the gap is the reference exclusions, which the reported value does not apply')
    hb.log(f'  municipal (the authors\' final table): {df_gep["water_use_domestic_gep"].sum():,.2f} '
           f'over {int(df_gep["water_use_domestic_gep"].notna().sum())} countries')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
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

        # Irrigation GEP: the premium times the rent share. The share applies to the PREMIUM,
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

        # The implied-price domestic value, published by gep_calculation as the comparison
        # variant beside the authors' municipal table: the ecosystem-provided cubic metres times a
        # raw-water price. An explicit price in es_parameters wins; absent one, the price is
        # IMPLIED by irrigation -- what a cubic metre earns in a field bounds what a city would
        # have paid for it.
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


def agriculture_water(p):
    """Apply the GEP-WaterUse-Agriculture processed coefficients to the account's crop-rent
    values in USD."""
    publish_inputs(p)
    p.water_supply_agriculture_country_path = os.path.join(p.cur_dir, 'agriculture_by_country.csv')
    p.water_supply_agriculture_crop_path = os.path.join(p.cur_dir, 'agriculture_by_crop_country.csv')
    if not p.run_this:
        return
    crop_path = getattr(p, 'water_supply_crop_values_path', None)
    if crop_path is None:
        crop_path = p.results.get('crop_provision', {}).get('gep_by_country_year_crop')
    if crop_path is None:
        crop_path = os.path.normpath(os.path.join(
            p.project_dir, p.water_supply_crop_values_relative_file))
    if not os.path.isfile(crop_path):
        raise FileNotFoundError('Agricultural water requires the commercial crop country-item '
                                f'table in USD: {crop_path}. Run crop_provision first or set '
                                'water_supply_crop_values_path.')
    p.water_supply_crop_values_path = crop_path
    outputs = [p.water_supply_agriculture_country_path, p.water_supply_agriculture_crop_path,
               os.path.join(p.cur_dir, 'agriculture_basis.csv'),
               os.path.join(p.cur_dir, 'agriculture_coverage.csv')]
    outputs += [os.path.join(p.cur_dir, 'agriculture_' + name + '.csv') for name in
                ['missing_coefficients', 'unmapped_items', 'shared_value_unallocated',
                 'unsplit_value_by_status', 'correspondence_items_absent']]
    dependencies = [__file__, wf.__file__]
    if utilities.reuse_reason(p, 'water_supply', outputs, light_inputs=dependencies) is None:
        return
    values = pd.read_csv(crop_path)
    correspondence = pd.read_csv(p.water_supply_agriculture_correspondence_path)
    spam = pd.read_csv(p.water_supply_agriculture_spam_path).rename(
        columns={'iso3': 'iso3_r250_label'})
    coefficients = pd.read_csv(p.water_supply_agriculture_lambda_path).rename(
        columns={'iso3': 'iso3_r250_label'})
    basis, diagnostics = wf.agriculture_basis_from_crop_values(
        values, correspondence, spam, int(p.gep_base_year), 'crop_provision_gep')
    crop = wf.agriculture_gep_from_lambda(coefficients, basis)
    country = wf.aggregate_agriculture_gep_to_country(crop).rename(columns={
        'GEP_gn_rf': 'water_use_rainfed_green_gep',
        'GEP_gn_ir': 'water_use_irrigated_green_gep',
        'GEP_bl_irr': 'water_use_irrigated_blue_gep',
        'GEP_total_irr': 'water_use_irrigated_total_gep',
        'GEP_total': 'water_use_agriculture_gep',
        'GEP_gn_rf_vmr': 'water_use_rainfed_vmr_gep',
        'GEP_gn_ir_vmr': 'water_use_irrigated_vmr_gep',
        'n_crop': 'water_use_agriculture_crop_count'})
    # Compatibility with the existing comparison sheet. This is all agricultural water.
    country['water_use_irrigation_gep'] = country['water_use_agriculture_gep']
    country['year'] = int(p.gep_base_year)
    country.to_csv(outputs[0], index=False)
    crop.to_csv(outputs[1], index=False)
    basis.to_csv(outputs[2], index=False)
    covered = basis.merge(coefficients, on=['CropGBWater_crop', 'iso3_r250_label'],
                          how='left', validate='one_to_one')
    missing = covered[list(wf.AGRICULTURE_LAMBDA_COLUMNS)].isna().all(axis=1)
    covered.loc[missing].to_csv(os.path.join(p.cur_dir, 'agriculture_missing_coefficients.csv'),
                               index=False)
    summary = {'mapped_value_usd': diagnostics['mapped_value_total'],
               'split_value_usd': diagnostics['split_value_total'],
               'allocated_basis_usd': diagnostics['basis_value_total'],
               'split_basis_without_coefficients_usd':
                   covered.loc[missing, ['basis_rain', 'basis_irr']].sum().sum(),
               'agriculture_gep_usd': country['water_use_agriculture_gep'].sum(),
               'valued_countries': len(country), 'valued_crop_country_rows': len(crop)}
    pd.DataFrame([summary]).to_csv(outputs[3], index=False)
    for name in ['unmapped_items', 'shared_value_unallocated', 'unsplit_value_by_status']:
        diagnostics[name].to_csv(os.path.join(p.cur_dir, f'agriculture_{name}.csv'),
                                 index=name == 'unsplit_value_by_status')
    pd.DataFrame({'crop': diagnostics['correspondence_items_absent']}).to_csv(
        os.path.join(p.cur_dir, 'agriculture_correspondence_items_absent.csv'), index=False)
    utilities.write_reuse_signature(p, 'water_supply', outputs, light_inputs=dependencies)
    hb.log(f'Agricultural water: {summary}')
    return True
