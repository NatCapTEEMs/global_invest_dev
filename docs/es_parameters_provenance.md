# Where the es_parameters values come from

Provenance for the rows in `global_invest/input_template/es_parameters.csv`. It lives here
rather than in the CSV because that file holds values a run reads, and a note is not a value:
every one of these was being hydrated onto the ProjectFlow object as an attribute nothing
read. The `*_source_url` and `*_source_archive_member` rows stay in the CSV, because a URL and
an archive member name are values you can act on.

## erosion

**`erosion_fao_gpv_iso3`** — the author's processed FAOSTAT table, from the drive's erosion folder

**`erosion_yield_bridge`** — alpha is the flat erosion-to-yield coefficient Method A applies to every crop; yield_coefficient_fallback is what Method B uses for a crop with no coefficient of its own. They are the same 0.08 today and the code comment says they are the same bridge, but they answer different questions, so they are two rows rather than one. 19 of 46 crops take the fallback and 5 have no entry at all.

**`erosion_sdr_params`** — one calibration for both the static GEP and the dynamic shock path. ⚠ They used to differ, with sdr_max and ic_0_param transposed between them; these are the dynamic values, the conventional Borselli pair.

**`erosion_threshold`** — threshold_high 11.0 t/ha/yr is the SES-11 severe-erosion threshold the account is named for, and threshold_low 2.0 the tolerable one. They decide which pixels count as severe, so they are the two rows most worth checking against the methodology before a run.

**`erosion_yield_reduction`** — 0.08 is the fallback yield response used when a crop has no row in the elasticity table. 19 of 46 crops take it, and 5 crops have no entry at all, so it is doing more work than a fallback should.

**`erosion_gdp_csv`** — World Bank NY.GDP.MKTP.CD, GDP in current US dollars, filtered to 2019. Staged rather than fetched per run because the World Bank revises it, which would move the erosion valuation without recording why.

## fire_protection

**`fire_panel`** — the district panel, from the drive Wildfires folder

## air_filtration

**`air_filtration_workbook`** — the committed workbook, from the drive air filtration folder

**`air_filtration_vsl`** — value of a statistical life per country, the price half of the air-quality valuation. Rebuilt from the workbook the group circulated; sandstorm prevention reads the same column.

## flood

**`flood_return_periods`** — the six on disk under floodplain_depth_v2_nowater/aligned_to_lulc, which is the reported 2024 hazard set with permanent water removed.

**`flood_input_reference`** — machine-specific, so these ship blank: base_data carries no global_invest/flood tree. On MSI they are under /projects/standard/jajohns/shared/flood_gep/inputs/. ⚠ Three are not where a tidy layout would put them: the reported depths are floodplain_depth_v2_nowater/, the SPA ratio is in counterfactual_mosaic/, and the CN table is in counterfactual/ while the amplification rasters are in counterfactual_mosaic/.

**`flood_amplification_pattern`** — the staged files are global_amplification_<scenario>_rp<rp>.tif. The f0p3 and f0p5 variants beside them are alternative depth exponents; the unsuffixed file is the default.

**`flood_country_dir`** — blank means each task writes its per-country tree into its own cur_dir. A machine holding an earlier run points these at it to reuse Section B rather than rebuild it.

**`flood_optional_tables`** — protection_path and protection_evidence_path are optional companions: blank means the table is not supplied, and the readers branch on None. When an evidence table IS given and has no ISO3 column the run stops, because which countries count as documented decides how much damage is truncated.

**`flood_skip_damage_tables`** — ⚠ flood_tasks reads this with TWO different defaults -- False at the run_valuation_chain call site and True at the GEP one -- so the behaviour depended on which ran. The damage tables are prebuilt inputs and the inputs directory is not writable, so building them raises PermissionError; the deleted set_flood_paths set True. Setting it here makes both call sites agree

**`flood_divergent_defaults`** — ⚠ apply_service_flow and report_protection_split are TRUE here and FALSE as getattr defaults in configure_valuation. The deleted set_flood_paths set both True, so the run behaviour was True while the code default said otherwise. Deleting it without moving these would have silently turned off the ecosystem attribution that produces ead_attributed_to_spa_usd2019

**`flood_valuation_settings`** — the Section D switches, moved out of configure_valuation on 2026-08-29 where they were getattr defaults in code. Code set before hydration WINS over the CSV -- hydrate_es_parameters skips an attribute that is already set -- so defaults living in code silently shadowed anything put here

**`flood_settings`** — the settings the module reads. iso3_list all means every country in the boundary layer; iso3_n 0 means all remaining; an MSI array runner overwrites start and n from SLURM_ARRAY_TASK_ID.

**`flood_gep_for_merge`** — the author export gep_calculation falls back to when the counterfactual chain has not run here: iso3_r250_label and gep_const2019_usd, $11.40bn over 162 non-zero countries, v2 2024 hazard. Produced by his scripts/export_gep_for_merge.py from step4e_flood_gep_USD2019.csv; that script is not carried here because the output is a base_data input like any other author artifact

## local_climate_regulation

**`local_climate_regulation_city_savings`** — the chain's own per-country city-month valuations, one file per country: avoided kilowatt-hours, national price and their product

## ntfp

**`ntfp_roads_vector`** — the global roads layer the source module buffers, staged from its submission folder. The accessibility polygon is built from these geometries rather than from a rasterised road length, which is what the source does.

**`ntfp_countries_vector`** — the boundary polygons the source module takes its zonal statistics over. Burned onto the analysis grid rather than reprojected from an id raster, so a cell is assigned to the country whose polygon covers its centre.

**`ntfp_ndvi_mean`** — five-year mean NDVI from MODIS MOD13Q1 Collection 6.1 at 250 m over 2015 to 2019, quality masked and exported from Google Earth Engine. int16 storing NDVI times 10000, nodata -9999. Screens bare and sparse cells out of the land-cover forest mask.

**`ntfp_value_per_ha`** — derived from the public CWoN 2024 reproducibility package: the non-wood forest product annual value (forest_nontimber_annual_nwfp.dta) over the matching forest area (forest_nontimber_total_area.dta), per country and year. Note the areas differ in definition: CWoN prices over ITS forest area while the quantity we multiply is accessible forest from the ESA land cover, which is the source method's own construction

## crop_provision

**`cwon_crop_coefficients`** — the CWoN land rental-rate coefficients, from the drive

## livestock_provision

**`cwon_crop_coefficients`** — the CWoN land rental-rate coefficients, from the drive

**`gleam_dmi`** — GLEAM 3 dry-matter intake per country, harvested from FAO's public dashboard by the recipe in howto_harvest_gleam_intake.md. Carries all eight feed categories: the four the ruminant tables omit are served on the chicken and pig tables instead.

## pollination

**`pollination_crop_benefits_repo`** — the source author's pipeline. The GEP value raster is his output, not a construction of ours: CropGrids harvested area times a Monfreda within-country yield pattern times FAO calibration. The raster is staged into base_data by hand when a base year he has not published is needed; docs/runbook_pollination_value_raster.md has the six commands. Deliberately not automated: running his pipeline from ours would make this the only task in the library that executes another repository code

**`pollination_fx`** — World Bank PA.NUS.FCRF, official exchange rate in local currency per US dollar, period average, fetched 2026-08-25. 7,149 rows over 214 countries and 1990 to 2024. Pinned rather than fetched per run because the World Bank revises the series, which would move the pollination total by whatever had changed since.

**`pollination_value_raster_dir`** — the precomputed baseline pollination-value raster the shock path resamples, poll_value_global_<year>usd.tif. Output of the source pipeline run once over Monfreda yields, CropGrids area, FAO producer prices and pollination-dependence ratios. Reused rather than rebuilt, so it has no URL and a person stages it.

**`pollination_crosswalk_m49_iso3`** — FAO M49 area codes to ISO3.

**`pollination_fao_classification`** — the FAO item classification.

**`pollination_crosswalk_fao_cropgrids`** — FAO items to CropGrids crop names.

## coastal_protection

**`coastal_protection_coral_reef`** — annual expected coastal-protection benefit of coral reefs, per country. Supplied by the author rather than computed here, which is why coastal protection is only half ours.

**`coastal_protection_gdp_deflator`** — GDP inflation deflator, for putting the mangrove and coral figures in one price year. The drive folder spells it gdp_inflation_delator, sic; staged under the corrected name.

## landslide_mitigation

**`landslide_elevation`** — elevation in metres, the slope input to the stability model. Read from the shared SEALS static regressors rather than a landslide copy, so the two models see one terrain.

**`landslide_lulc_path_template`** — the ESA land cover per year, formatted with the year. A template rather than a path because the task iterates data_processing_range.
