# GEP: Crop Pollination 

This project aims to estimate global GEP for crop pollination. 

# Formula: 

$$
GEP_{poll} = Crop Yield × 
            Pollination Sufficiency × 
            Dependence Ratio × 
            Producer Price ×  
            λ (Agrue this equals to one -- no labor or capital inputs into pollination)
$$

# Requirements: 
- Pollination dependency ratios (poll_dep_greater_0.xlsx)
- Pollinator sufficiency raster map: 
  - We create. 
  - Requires a LULC map
- Crop yields: 
  - EarthSata (HarvestedAreaYield175Crops_Geotiff\GeoTiff)
  - MapSPAM (mapspam-2XXX)
- Crop prices (FAO): Prices_E_All_Data_ori.csv
  - Exchange rates (World Bank)


# Current Steps: 
1. Clean price data 
   1. crop-price.py
2. Estimate pollination sufficiency (range 0-1)
   1. poll-suff.py
3. Do raster math on crop yields and pollinator sufficency with modifier for dependency ratios, prices, and nature's contribution 
   1. To Modify: Have this script occur for each set of crop yield maps: EarthStat 2000, and MapSpam 2000, 2005, 2010, 2020
   2. calculate-gep-pollination.py


# Need to Update and Questions for Lingling: 
1. For Prices: Want to know how much improvement is made
   1. Want to know how many values are USD, SLC/LCU, or missing. 
   2. Want to know the % for each crop for each year.
   3. add to crop-price.py file (output file)
   4. Table: % (#) of each crop that have USD, LCU, SLC, and missing for pre and post processing. Start just with year 2000 and then move to all years. 
2. Update price imputation:
   1. limit for crops within country
   2. Table: % of prices that are imputed, average difference from actual price to forecasted price. 
3. We want two final results
   1. Maps of GEP values for each crop globally (raster maps by crop)
   2. A CSV aggregated by crop-country-year (this is for big GEP project)
4. Small update: Try to use LULC for year 2000 (LULC used in pollination sufficency calculation)
   1. This used 2017 and 2022 ESA.
   2. But EarthStat is for 2000. And if we move if Mapspam we will have several years
   3. So we will need to have different LULC for each year. 
5. Switch from EarthStat to MapSpam
   1. https://mapspam.info/
   2. Should run for both and compare on the year 2000.



------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------


# Answered Questions: 
1. Why are the pollination suffiencey ratios only aligned with apple?
   1. pollination_suff_path = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\aligned_poll_suff_to_apple.tif"
   2. Preprocess spatial reference for later analysis. So just bringing the two rasters to the same resolution prior to calculation.
   3. Resample the Pollination Raster: Resample the pollination sufficiency raster to match the spatial resolution and extent of the apple production raster using the average resampling method. This ensures that the pollination data aligns correctly with the apple production data


# Where are these things calculated:
1. Crop Yield: as a raster data set - Ryan (DONE)
   1. raster file
   2. EarthStat
   3. MapSpam
2. Pollination Sufficiency - Subin
   1. raster file
   2. Bring in and should be more than just 0/1. should be some fraction.
   3. Raster math: crop yield * pollinator map = yields from pollinators
3. Dependence Ratio (by crop) - Subin (DONE)
   1. csv
   2. Multiply onto the yield_polly_map given crop type for pixel
4. Crop Price - Ryan - DONE (ish)
   1. csv
   2. Mutlipy onto the yield_polly_map given crop type for pixel
5. nature's contribution - Ryan - DONE
   1. = 1 (for sake of argument)
6. Aggregation at country level
   1. Aggreate pixels from map at end


# Past Steps: 
1. match_country_name_from_two_datasets.py
   1. Checks if crops not in either pollination list or crop price list
2. resample_pollination_suffciency_yield.py
   1. Ensure alignment of raster data
2. pollination_suffcienciy.py
   1. Given the habitat, estimate the pollination sufficiency
3. Calculate_Pollination-Dependent_Production_65_crops_step_1.py
   1. Estimates pollination dependency for main set of crops
4. crop_production_zonal_statistics_by_country_65_crops_step_1_2.py
   1. TODO: THESE TWO ABOVE MAY BE THE SAME
5. calculate_GEP_pollination_65_crops_final.py
   1. Estimates the final GEP calculation

# Don't Know what this is useful for: 
1. make_poll_suff_seals.py
   1. Same file: make_poll_suff.py
