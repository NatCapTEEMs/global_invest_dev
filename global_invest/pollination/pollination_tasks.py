import os
import hazelbean as hb
from global_invest.commercial_agriculture import commercial_agriculture_functions

def build_gep_task_tree(p):
    """
    Build the default task tree for commercial agriculture.
    """
    p.pollination_task = p.add_task(pollination)
    # p.commercial_agriculture_preprocess_task = p.add_task(gep_preprocess, parent=p.commercial_agriculture_task)  
    p.add_task(gep_calculation, parent=p.pollination_task)  
    
    return p

def pollination(p):
    """
    Parent task for commercial agriculture.
    """
    # p.fao_input_path = p.get_path(os.path.join(p.base_data_dir, 'fao', 'Value_of_Production_E_All_Data.csv'))
    pass 

def pollination_biophysical(p):

    # TODOO CURRENTLY SAVES IN WEIRD WORISPACE PLACE
    # C:\Files\Research\cge\gtap_invest\gtap_invest_dev\gtap_invest\workspace_poll_suff\lulc_esa_gtap1_rcp45_ssp2_2030_SR_RnD_20p

    baseline_scenario_label = 'lulc_esa_gtap1_baseline_' + str(p.base_year)
    p.baseline_clipped_lulc_path = os.path.join(p.stitched_lulc_esa_scenarios_dir, baseline_scenario_label + '.tif')
    if p.run_this:

        luc_scenario_path = p.base_year_lulc_path
        # base_year_lulc_label = 'lulc_esa_gtap1_baseline_' + str(p.base_year)


        final_raster_path = os.path.join(p.cur_dir, 'poll_suff_ag_coverage_prop_' + baseline_scenario_label + '.tif')
        if not hb.path_exists(final_raster_path):
            L.info('Running global_invest_main.make_poll_suff on LULC: ' + str(luc_scenario_path) + ' and saving results to ' + str(final_raster_path))
            current_landuse_path = os.path.join(p.stitched_lulc_esa_scenarios, baseline_scenario_label + '.tif')


            # base_year_lulc_label = 'lulc_seals7_gtap1_baseline_' + str(p.base_year)
            # esa_include_string = 'lulc_esa_gtap1_' + luh_scenario_label + '_' + str(year) + '_' + policy_scenario_label
            # p.lulc_projected_stitched_path = os.path.join(p.cur_dir, esa_include_string + '.tif')
            #

            pollination_sufficiency = 'fix' # TODOO
            pollination_sufficiency.make_poll_suff.execute(current_landuse_path, p.cur_dir)

            created_raster_path = os.path.join(p.cur_dir, 'churn\poll_suff_hab_ag_coverage_rasters', "poll_suff_ag_coverage_prop_10s_" + baseline_scenario_label + ".tif")

            hb.copy_shutil_flex(created_raster_path, final_raster_path)

        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    current_scenario_label = luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                    final_raster_path = os.path.join(p.cur_dir, 'poll_suff_ag_coverage_prop_gtap1_' + current_scenario_label + '.tif')
                    current_landuse_path = os.path.join(p.stitched_lulc_esa_scenarios, 'lulc_esa_gtap1_' + current_scenario_label + '.tif')

                    if not hb.path_exists(final_raster_path):


                        L.info('Running pollination model for ' + current_scenario_label + ' from ' + current_landuse_path + ' to ' + final_raster_path)

                        pollination_sufficiency.make_poll_suff.execute(current_landuse_path, p.cur_dir)

                        # After it finishes, move the file to the root dir and get rid of the cruft.
                        created_raster_path = os.path.join(p.cur_dir, 'churn\poll_suff_hab_ag_coverage_rasters',  'poll_suff_ag_coverage_prop_10s_lulc_esa_gtap1_' + current_scenario_label + '.tif')
                        hb.copy_shutil_flex(created_raster_path, final_raster_path)
                    else:
                        L.info('Skipping running pollination model for ' + current_scenario_label + ' from ' + current_landuse_path + ' to ' + final_raster_path)

def pollination_shock(p):
    """Convert pollination into shockfile."""
    # # OLD SHORTCUT
    # p.policy_scenario_labels = [p.policy_scenario_labels[0], p.policy_scenario_labels[8]]

    ### Thought: think again about adding a penalty to only crediting pollination losses in bau
    p.pollination_shock_change_per_region_path = os.path.join(p.cur_dir, 'pollination_shock_change_per_region.gpkg')
    p.pollination_shock_csv_path = os.path.join(p.cur_dir, 'pollination_shock.csv')
    p.crop_data_dir = r"C:\Users\jajohns\Files\Research\base_data\crops\earthstat\crop_production"
    p.crop_prices_dir = r"C:\Users\jajohns\Files\Research\base_data\pyramids\crops\price"
    # p.pollination_biophysical_dir = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\projects\feedback_with_policies\intermediate\pollination_biophysical"
    p.pollination_dependence_spreadsheet_input_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\pollination\rspb20141799supp3.xls" # Note had to fix pol.dep for cofee and greenbroadbean as it was 25 not .25

    p.crop_value_baseline_path = os.path.join(p.cur_dir, 'crop_value_baseline.tif')
    p.crop_value_no_pollination_path = os.path.join(p.cur_dir, 'crop_value_no_pollination.tif')
    p.crop_value_max_lost_path = os.path.join(p.cur_dir, 'crop_value_max_lost.tif')
    p.crop_value_max_lost_10s_path = os.path.join(p.cur_dir, 'crop_value_max_lost_10s.tif')
    p.crop_value_baseline_10s_path = os.path.join(p.cur_dir, 'crop_value_baseline_10s.tif')

    if p.run_this:
        df = None

        # # TODO HACK: scenario subset
        # p.policy_scenario_labels = p.gtap_bau_and_combined_labels

        ###########################################
        ###### Calculate base-data necessary to do conversion of biophysical to shockfile
        ###########################################

        if not all([hb.path_exists(i) for i in [p.crop_value_baseline_path,
                                                p.crop_value_no_pollination_path,
                                                p.crop_value_max_lost_path,]]):
            df = pd.read_excel(p.pollination_dependence_spreadsheet_input_path, sheet_name='Crop nutrient content')

            crop_names = list(df['Crop map file name'])[:-3] # Drop last three which were custom addons in manuscript and don't seem to have earthstat data for.
            pollination_dependence = list(df['poll.dep'])
            crop_value_baseline = np.zeros(hb.get_shape_from_dataset_path(p.ha_per_cell_300sec_path))
            crop_value_no_pollination = np.zeros(hb.get_shape_from_dataset_path(p.ha_per_cell_300sec_path))
            for c, crop_name in enumerate(crop_names):
                L.info('Calculating value yield effect from pollination for ' + str(crop_name) + ' with pollination dependence ' + str(pollination_dependence[c]))
                crop_price_path = os.path.join(p.crop_prices_dir, crop_name + '_prices_per_ton.tif')
                crop_price = hb.as_array(crop_price_path)
                crop_price = np.where(crop_price > 0, crop_price, 0.0)
                crop_yield = hb.as_array(os.path.join(p.crop_data_dir, crop_name + '_HarvAreaYield_Geotiff', crop_name + '_Production.tif'))
                crop_yield = np.where(crop_yield > 0, crop_yield, 0.0)

                crop_value_baseline += (crop_yield * crop_price)
                crop_value_no_pollination += (crop_yield * crop_price) * (1 - float(pollination_dependence[c]))

            crop_value_max_lost = crop_value_baseline - crop_value_no_pollination
            #
            # crop_value_baseline_path = os.path.join(p.cur_dir, 'crop_value_baseline.tif')
            # crop_value_no_pollination_path = os.path.join(p.cur_dir, 'crop_value_no_pollination.tif')
            # crop_value_max_lost_path = os.path.join(p.cur_dir, 'crop_value_max_lost.tif')

            hb.save_array_as_geotiff(crop_value_baseline, p.crop_value_baseline_path, p.match_300sec_path, ndv=-9999, data_type=6)
            hb.save_array_as_geotiff(crop_value_no_pollination, p.crop_value_no_pollination_path, p.match_300sec_path, ndv=-9999, data_type=6)
            hb.save_array_as_geotiff(crop_value_max_lost, p.crop_value_max_lost_path, p.match_300sec_path, ndv=-9999, data_type=6)


        ### Resample the base data to match LULC
        global_bb = hb.get_bounding_box(p.base_year_lulc_path)
        stitched_bb = hb.get_bounding_box(p.baseline_clipped_lulc_path)
        if stitched_bb != global_bb:
            current_path = os.path.join(p.cur_dir, 'crop_value_max_lost_clipped.tif')
            hb.clip_raster_by_bb(p.crop_value_max_lost_path, stitched_bb, current_path)
            p.crop_value_max_lost_path = current_path

        if not hb.path_exists(p.crop_value_baseline_10s_path):
            hb.resample_to_match(p.crop_value_baseline_path, p.baseline_clipped_lulc_path, p.crop_value_baseline_10s_path, ndv=-9999., output_data_type=6)


        if not hb.path_exists(p.crop_value_max_lost_10s_path):
            hb.resample_to_match(p.crop_value_max_lost_path, p.baseline_clipped_lulc_path, p.crop_value_max_lost_10s_path, ndv=-9999., output_data_type=6)


        ###########################################
        ###### Calculate crop_value_pollinator_adjusted.
        ###########################################

        # Incorporate the "sufficient pollination threshold" of 30%
        # TODOO Go through and systematically pull into config files to initialize model and write output summary of what were used.
        sufficient_pollination_threshold = 0.3

        ### BASELINE crop_value_pollinator_adjusted:
        policy_scenario_label = 'gtap1_baseline_' + str(p.base_year)
        current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_' + policy_scenario_label + '_zonal_stats.xlsx')
        suff_path = os.path.join(p.pollination_biophysical_dir, 'poll_suff_ag_coverage_prop_lulc_esa_' + policy_scenario_label + '.tif')
        crop_value_pollinator_adjusted_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_' + policy_scenario_label + '.tif')

        if not hb.path_exists(crop_value_pollinator_adjusted_path):
            hb.raster_calculator_af_flex([p.crop_value_baseline_10s_path, p.crop_value_max_lost_10s_path, suff_path, p.base_year_simplified_lulc_path], lambda baseline_value, max_loss, suff, lulc:
                    np.where((max_loss > 0) & (suff < sufficient_pollination_threshold) & (lulc == 2), baseline_value - max_loss * (1 - (1/sufficient_pollination_threshold) * suff),
                        np.where((max_loss > 0) & (suff >= sufficient_pollination_threshold) & (lulc == 2), baseline_value, -9999.)), output_path=crop_value_pollinator_adjusted_path)

        # Do zonal statistics on outputed raster by AEZ-REG. Note that we need sum and count for when/if we calculate mean ON GRIDCELLS WITH AG.
        if not hb.path_exists(current_output_excel_path):
            df = hb.zonal_statistics_flex(crop_value_pollinator_adjusted_path,
                                          p.gtap37_aez18_path,
                                          zone_ids_raster_path=p.zone_ids_raster_path,
                                          id_column_label='pyramid_id',
                                          zones_raster_data_type=5,
                                          values_raster_data_type=6,
                                          zones_ndv=-9999,
                                          values_ndv=-9999,
                                          all_touched=None,
                                          stats_to_retrieve='sums_counts',
                                          assert_projections_same=False, )
            generated_scenario_label = 'gtap1_baseline_2014'
            df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
            df.to_excel(current_output_excel_path)
        else:
            generated_scenario_label = 'gtap1_baseline_2014'
            df = pd.read_excel(current_output_excel_path, index_col=0)
            df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
        merged_df = df

        ### SCENARIO crop_value_pollinator_adjusted
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_' + policy_scenario_label + '_zonal_stats.xlsx')
                    suff_path = os.path.join(p.pollination_biophysical_dir, 'poll_suff_ag_coverage_prop_gtap1_' + luh_scenario_label
                                                + '_' + str(scenario_year) + '_' + policy_scenario_label + '.tif')
                    lulc_path = os.path.join(p.stitched_lulc_esa_scenarios_dir, 'lulc_seals7_gtap1_' + luh_scenario_label
                                                + '_' + str(scenario_year) + '_' + policy_scenario_label + '.tif')

                    crop_value_pollinator_adjusted_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_' + policy_scenario_label + '.tif')

                    if not hb.path_exists(crop_value_pollinator_adjusted_path):
                        hb.raster_calculator_af_flex([p.crop_value_baseline_10s_path, p.crop_value_max_lost_10s_path, suff_path, lulc_path], lambda baseline_value, max_loss, suff, lulc:
                                np.where((max_loss > 0) & (suff < sufficient_pollination_threshold) & (lulc == 2), baseline_value - max_loss * (1 - (1/sufficient_pollination_threshold) * suff),
                                        np.where((max_loss > 0) & (suff >= sufficient_pollination_threshold) & (lulc == 2), baseline_value, -9999.)), output_path=crop_value_pollinator_adjusted_path)


                        # TODOOO: Continue thinking about what the right shock is overall. Is it the average on NEW land? Or the aggregate value
                        # To isolate the effect, maybe calculate the average value of crop loss on cells that are cultivated in both scenarios? Start on a dask function that does that?

                    if not hb.path_exists(current_output_excel_path):
                        df = hb.zonal_statistics_flex(crop_value_pollinator_adjusted_path,
                                                      p.gtap37_aez18_path,
                                                      zone_ids_raster_path=p.zone_ids_raster_path,
                                                      id_column_label='pyramid_id',
                                                      zones_raster_data_type=5,
                                                      values_raster_data_type=6,
                                                      zones_ndv=-9999,
                                                      values_ndv=-9999,
                                                      all_touched=None,
                                                      stats_to_retrieve='sums_counts',
                                                      assert_projections_same=False, )

                        generated_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
                        df.to_excel(current_output_excel_path)
                    else:
                        generated_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        df = pd.read_excel(current_output_excel_path, index_col=0)
                        df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count', generated_scenario_label + '_total': generated_scenario_label + '_sum',}, inplace=True)
                    merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)

        ###########################################
        ###### Calculate change from scenario to baseline, on and not on existing ag
        ###########################################

        baseline_policy_scenario_label = 'gtap1_baseline_' + str(p.base_year)
        baseline_crop_value_pollinator_adjusted_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_' + baseline_policy_scenario_label + '.tif')
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:

                    # Calculate difference between scenario and BASELINE for crop value adjusted
                    bau_crop_value_pollinator_adjusted_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_BAU.tif')
                    current_crop_value_pollinator_adjusted_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_' + policy_scenario_label + '.tif')

                    crop_value_difference_from_baseline_path = os.path.join(p.cur_dir, 'crop_value_difference_from_baseline_' + policy_scenario_label + '.tif')
                    if not hb.path_exists(crop_value_difference_from_baseline_path):
                        hb.dask_compute([baseline_crop_value_pollinator_adjusted_path, current_crop_value_pollinator_adjusted_path], lambda x, y: y - x, crop_value_difference_from_baseline_path)

                    # Zonal stats for difference from Baseline
                    current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_difference_from_baseline_' + policy_scenario_label + '_zonal_stats.xlsx')
                    if not hb.path_exists(current_output_excel_path):
                        df = hb.zonal_statistics_flex(crop_value_difference_from_baseline_path,
                                                      p.gtap37_aez18_path,
                                                      zone_ids_raster_path=p.zone_ids_raster_path,
                                                      id_column_label='pyramid_id',
                                                      zones_raster_data_type=5,
                                                      values_raster_data_type=6,
                                                      zones_ndv=-9999,
                                                      values_ndv=-9999,
                                                      all_touched=None,
                                                      stats_to_retrieve='sums_counts',
                                                      assert_projections_same=False, )
                        generated_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
                        df.to_excel(current_output_excel_path)

                    # Calc difference between scenario and BASELINE for crop_value on grid-cells that were agri in both lulc maps.
                    lulc_path = os.path.join(p.stitched_lulc_esa_scenarios_dir, 'lulc_seals7_gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '.tif')
                    crop_value_difference_from_baseline_existing_ag_path = os.path.join(p.cur_dir, 'crop_value_difference_from_baseline_existing_ag_' + policy_scenario_label + '.tif')

                    def op(x, y, w, z):
                        r = dask.array.where((w == 2) & (z == 2), y - x, 0.)
                        rr = (z * 0.0) + r # HACK. Dask.array.where was returning a standard xarray rather than a rioxarray. This dumb hack makes it inherit the rioxarray parameters of z
                        return rr

                    if not hb.path_exists(crop_value_difference_from_baseline_existing_ag_path):
                        op_paths = [
                            baseline_crop_value_pollinator_adjusted_path,
                            crop_value_pollinator_adjusted_path,
                            lulc_path,
                            p.base_year_simplified_lulc_path,
                        ]

                        hb.dask_compute(op_paths, op, crop_value_difference_from_baseline_existing_ag_path)

                    # Zonal stats for difference from Baseline
                    current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_difference_from_baseline_existing_ag_' + policy_scenario_label + '_zonal_stats.xlsx')
                    if not hb.path_exists(current_output_excel_path):
                        df = hb.zonal_statistics_flex(crop_value_difference_from_baseline_existing_ag_path,
                                                      p.gtap37_aez18_path,
                                                      zone_ids_raster_path=p.zone_ids_raster_path,
                                                      id_column_label='pyramid_id',
                                                      zones_raster_data_type=5,
                                                      values_raster_data_type=6,
                                                      zones_ndv=-9999,
                                                      values_ndv=-9999,
                                                      all_touched=None,
                                                      stats_to_retrieve='sums_counts',
                                                      assert_projections_same=False, )
                        generated_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_existing_ag'
                        df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
                        df.to_excel(current_output_excel_path)
                    else:
                        df = pd.read_excel(current_output_excel_path, index_col=0)
                    merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)

                    # Also need to compute the value on that cropland that was cropland in both
                    # crop_value_baseline_existing_ag_path = os.path.join(p.cur_dir, 'crop_value_baseline_existing_ag_' + policy_scenario_label + '.tif')

                    def op(y, w, z):
                        r = dask.array.where((w == 2) & (z == 2), y, 0.)
                        rr = (z * 0.0) + r  # HACK. Dask.array.where was returning a standard xarray rather than a rioxarray. This dumb hack makes it inherit the rioxarray parameters of z
                        return rr
                    
                    crop_value_baseline_existing_ag_path  = 'fix' # TODOO

                    if not hb.path_exists(crop_value_baseline_existing_ag_path):
                        op_paths = [
                            baseline_crop_value_pollinator_adjusted_path,
                            lulc_path,
                            p.base_year_simplified_lulc_path,
                        ]

                        hb.dask_compute(op_paths, op, crop_value_baseline_existing_ag_path)

                    # Zonal stats for difference from Baseline
                    current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_baseline_existing_ag_' + policy_scenario_label + '_zonal_stats.xlsx')
                    if not hb.path_exists(current_output_excel_path):
                        df = hb.zonal_statistics_flex(crop_value_baseline_existing_ag_path,
                                                      p.gtap37_aez18_path,
                                                      zone_ids_raster_path=p.zone_ids_raster_path,
                                                      id_column_label='pyramid_id',
                                                      zones_raster_data_type=5,
                                                      values_raster_data_type=6,
                                                      zones_ndv=-9999,
                                                      values_ndv=-9999,
                                                      all_touched=None,
                                                      stats_to_retrieve='sums_counts',
                                                      assert_projections_same=False, )
                        generated_scenario_label = 'crop_value_baseline_existing_ag_compared_to_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
                        df.to_excel(current_output_excel_path)
                    else:
                        df = pd.read_excel(current_output_excel_path, index_col=0)
                    merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)


                    if policy_scenario_label != 'BAU':
                        pass

                        # Calc difference between scenario and BAU for crop_value adjusted
                        # IMPORTANT NOTE: This is really just for plotting and visualization. The shockfiles themselves are all defined relative to the baseline, not relative to bau.
                        crop_value_difference_from_bau_path = os.path.join(p.cur_dir, 'crop_value_difference_from_bau_' + policy_scenario_label + '.tif')
                        bau_lulc_path = os.path.join(p.stitched_lulc_esa_scenarios_dir, 'lulc_seals7_gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_BAU.tif')
                        if not hb.path_exists(crop_value_difference_from_bau_path):
                            hb.dask_compute([bau_crop_value_pollinator_adjusted_path, current_crop_value_pollinator_adjusted_path], lambda x, y: y - x, crop_value_difference_from_bau_path)

                        # Zonal stats for difference from BAU
                        current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_difference_from_bau_' + policy_scenario_label + '_zonal_stats.xlsx')
                        if not hb.path_exists(current_output_excel_path):
                            df = hb.zonal_statistics_flex(crop_value_difference_from_bau_path,
                                                          p.gtap37_aez18_path,
                                                          zone_ids_raster_path=p.zone_ids_raster_path,
                                                          id_column_label='pyramid_id',
                                                          zones_raster_data_type=5,
                                                          values_raster_data_type=6,
                                                          zones_ndv=-9999,
                                                          values_ndv=-9999,
                                                          all_touched=None,
                                                          stats_to_retrieve='sums_counts',
                                                          assert_projections_same=False, )
                            generated_scenario_label = 'gtap2_crop_value_difference_from_bau_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                            df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
                            df.to_excel(current_output_excel_path)
                        else:
                            df = pd.read_excel(current_output_excel_path, index_col=0)
                        merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)

                        # Calc difference between scenario and BAU for crop_value adjusted ON EXISTING AG
                        # ie for crop_value on grid-cells that were agri in both lulc maps.
                        lulc_path = os.path.join(p.stitched_lulc_esa_scenarios_dir, 'lulc_seals7_gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '.tif')
                        crop_value_difference_from_bau_existing_ag_path = os.path.join(p.cur_dir, 'crop_value_difference_from_bau_existing_ag_' + policy_scenario_label + '.tif')

                        def op(x, y, w, z):
                            r = dask.array.where((w == 2) & (z == 2), y - x, 0.)
                            rr = (z * 0.0) + r  # HACK. Dask.array.where was returning a standard xarray rather than a rioxarray. This dumb hack makes it inherit the rioxarray parameters of z
                            return rr

                        if not hb.path_exists(crop_value_difference_from_bau_existing_ag_path):
                            op_paths = [
                                bau_crop_value_pollinator_adjusted_path,
                                crop_value_pollinator_adjusted_path,
                                lulc_path,
                                bau_lulc_path,
                            ]

                            hb.dask_compute(op_paths, op, crop_value_difference_from_bau_existing_ag_path)

                        current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_difference_from_bau_existing_ag_' + policy_scenario_label + '_zonal_stats.xlsx')
                        if not hb.path_exists(current_output_excel_path):
                            df = hb.zonal_statistics_flex(crop_value_difference_from_bau_existing_ag_path,
                                                          p.gtap37_aez18_path,
                                                          zone_ids_raster_path=p.zone_ids_raster_path,
                                                          id_column_label='pyramid_id',
                                                          zones_raster_data_type=5,
                                                          values_raster_data_type=6,
                                                          zones_ndv=-9999,
                                                          values_ndv=-9999,
                                                          all_touched=None,
                                                          stats_to_retrieve='sums_counts',
                                                          assert_projections_same=False, )


                            generated_scenario_label = 'gtap2_crop_value_difference_from_bau_existing_ag_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                            df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
                            df.to_excel(current_output_excel_path)
                        else:
                            df = pd.read_excel(current_output_excel_path, index_col=0)
                        merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)

        ###########################################
        ###### Calculate the actual shock as the mean change.
        ###########################################

        scenario_shock_column_names_to_keep = []
        # scenario_shock_column_names_to_keep = ['pyramid_id', 'pyramid_ids_concatenated', 'pyramid_ids_multiplied', 'gtap37v10_pyramid_id', 'aez_pyramid_id', 'gtap37v10_pyramid_name', 'ISO3', 'AZREG', 'AEZ_COMM']
        if not hb.path_exists(p.pollination_shock_csv_path):
            baseline_generated_scenario_label = 'gtap1_baseline_2014'
            baseline_generated_scenario_label_existing_ag = 'gtap1_baseline_2014_existing_ag'

            scenario_shock_column_names_to_keep.append(baseline_generated_scenario_label + '_sum')
            # scenario_shock_column_names_to_keep.append(baseline_generated_scenario_label_existing_ag + '_sum')

            for luh_scenario_label in p.luh_scenario_labels:
                for scenario_year in p.scenario_years:
                    for policy_scenario_label in p.policy_scenario_labels:

                        generated_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_pollination_shock'
                        merged_df[generated_label] = merged_df[generated_scenario_label + '_sum'] / merged_df[baseline_generated_scenario_label + '_sum']
                        scenario_shock_column_names_to_keep.append(generated_label)

                        # # NOTE: When calculating the value only on existing cells, cannot use the sum / sum method above. Need to use the new rasters created ad calculate their mean.
                        # generated_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_existing_ag'
                        # merged_df[generated_scenario_label + '_mean'] = merged_df[generated_scenario_label_existing_ag + '_sum'] / merged_df[baseline_generated_scenario_label_existing_ag + '_count']

                        # TODOOO: ALMOST got the full sim ready to run on the new pollination method but didn't finish getting the averages here calculated.

                        # generated_scenario_label_existing_ag = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_existing_ag'
                        # generated_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_pollination_shock'
                        # merged_df[generated_label] = merged_df[generated_scenario_label_existing_ag + '_sum'] / merged_df[baseline_generated_scenario_label_existing_ag + '_sum']
                        # scenario_shock_column_names_to_keep.append(generated_label)

                        # generated_scenario_existing_ag_label = 'gtap2_crop_value_difference_from_baseline_existing_ag_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        # generated_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_pollination_shock_existing_ag'
                        #
                        # merged_df[generated_label] = merged_df[generated_scenario_existing_ag_label + '_sum'] / merged_df[baseline_generated_scenario_label + '_sum']

                        # # Also subtract the difference with BAU for each other policy
                        # if policy_scenario_label != 'BAU':
                        #     bau_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_BAU_pollination_shock'
                        #     scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_pollination_shock'
                        #     new_label = luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_shock_minus_bau'
                        #     merged_df[new_label] = merged_df[scenario_label] - merged_df[bau_label]
                        #
                        #     # generated_scenario_existing_ag_label = 'gtap2_crop_value_difference_from_bau_existing_ag_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        #     # merged_df[generated_scenario_existing_ag_label + '_mean'] = merged_df[generated_scenario_existing_ag_label + '_sum'] / merged_df[generated_scenario_existing_ag_label + '_count']
                        #     # merged_df[generated_scenario_existing_ag_label + '_mean_minus_baseline'] = merged_df[generated_scenario_existing_ag_label + '_mean'] - merged_df[baseline_generated_scenario_label + '_mean']
                        #
                        #
                        #     generated_bau_label  = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_BAU'
                        #     generated_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        #
                        #     generated_bau_existing_ag_label  = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_BAU'
                        #     merged_df[generated_scenario_label + '_sum_div_bau'] = merged_df[generated_scenario_label + '_sum'] / merged_df[generated_bau_label + '_sum']
                        #
                        #     generated_scenario_existing_ag_label = 'gtap2_crop_value_difference_from_bau_existing_ag_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        #     # merged_df[generated_scenario_existing_ag_label + '_mean_minus_bau'] = merged_df[generated_scenario_existing_ag_label + '_mean'] - merged_df[generated_scenario_existing_ag_label + '_mean']
                        #
                        #     # merged_df[generated_scenario_existing_ag_label + '_sum_div_bau'] = merged_df[generated_scenario_existing_ag_label + '_sum'] / merged_df[generated_bau_existing_ag_label + '_sum']
                        #
                        #     generated_scenario_label= 'gtap2_crop_value_difference_from_bau_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label


            # write to csv and gpkg
            merged_df.to_csv(hb.suri(p.pollination_shock_csv_path, 'comprehensive'))
            gdf = gpd.read_file(p.gtap37_aez18_path)
            gdf = gdf.merge(merged_df, left_on='pyramid_id', right_index=True, how='outer')
            gdf.to_file(hb.suri(p.pollination_shock_change_per_region_path, 'comprehensive'), driver='GPKG')

            merged_df = merged_df[scenario_shock_column_names_to_keep]
            merged_df.to_csv(p.pollination_shock_csv_path)
            gdf = gpd.read_file(p.gtap37_aez18_path)
            gdf = gdf.merge(merged_df, left_on='pyramid_id', right_index=True, how='outer')
            gdf.to_file(p.pollination_shock_change_per_region_path, driver='GPKG')

def gep_calculation(p):
    r = commercial_agriculture_functions.calculate_gep(p.base_data_dir)

    (df_gep_by_country_year_crop, df_gep_by_year_country, df_gep_by_year) = r