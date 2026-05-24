README

Yingjie Li adapted from README.docx, and reformatted input data list by model. 

	•	Some of the scripts (can be re-used for explorer) https://github.com/natcap/urban-online-workflow/tree/main/backend-worker 


Urban Cooling model: 
	•	data folder: LULC and Parameters August 2024 
	•	
	•	land use/land cover: lulc_overlay_3857.tif
	•	reference evapotranspiration: et0_annual_cgiar_3857.tif
	•	area of interest: acs_block_group.gpkg
	•	biophysical table: ucm__nlcd_nlud_tree.csv
	•	reference air temperature: 35
	•	UHI effect: 11
	•	air blending distance: 600
	•	maximum cooling distance: 450
	•	cooling capacity calculation method: factors
	•	run energy savings valuation: False
	•	energy consumption table: False
	•	buildings: False
	•	run work productivity valuation: False
	•	average relative humidity: False
	•	shade weight: 0.6
	•	albedo weight: 0.2
	•	evapotranspiration weight: 0.2


Carbon model
	•	data folder: LULC and Parameters August 2024 
	•	
	•	baseline LULC: lulc_overlay_3857.tif
	•	carbon pools (csv) : carbon__nlcd_nlud_tree.csv



Urban Nature Access
	•	data folder: LULC and Parameters August 2024 
	•	
	•	land use/land cover (raster): lulc_overlay_3857.tif
	•	LULC attribute table (csv): una__nlcd_nlud_tree.csv
	•	population raster (raster) (count): population_per_pixel_2020_3857.tif
	•	administrative boundaries (vector): acs_block_group.gpkg
	•	population group radii table (csv): (leaving blank)
	•	urban nature demand per capita (number) (m²): 16.7
	•	Aggregate by population groups (optional): False
	•	search radius mode: "uniform radius"
	•	decay function: dichotomy
	•	uniform search radius (number) (m): 800



Flood mitigation: 
	•	I have compiled data to this folder San Antonio, with the data source listed below 
	•	~~~~~~~~~~~~~~~
	•	area of interest: 
	•	Redistricted_Council_Districts_2022 [RedistrictedCouncilDistricts2022.shp]
	•	land use/land cover:  (folder) 
	•	sa_lc_w_20ac_foodfor_10m.tif (converted 20 acreage available natural ‘underutilized’ land for food forests)
	•	sa_lc_w_40ac_foodfor_10m.tif (converted 40 acreage available natural ‘underutilized’ land for food forests)
	•	biophysical table:  
	•	biophys_floodmitig_sa.csv 
	•	n_workers: 
	•	-1
	•	rainfall depth: 
	•	157
	•	soil hydrologic group (raster): 
	•	sa_env_hsg_int_10m.tif
	•	built infrastructure (vector, optional): 
	•	(leaving blank)
	•	damage loss table (csv): 
	•	(leaving blank)
	•	more info: Ben NDR and Flood Mar_2023.pptx 


Nutrient retention (Nutrient Delivery Ratio)
	•	I have compiled data to this folder San Antonio, with the data source listed below 
	•	~~~~~~~~~~~~~~~
	•	biophysical_table_path
	•	ndr_biophysical_parameters_vNLCDTree_SA.csv
	•	calc_n: True
	•	calc_p: True
	•	dem_path   
	•	sa_dem_3m_proj.tif
	•	k_param: 2
	•	lulc_path (folder) 
	•	sa_lc_w_20ac_foodfor_10m.tif (converted 20 acreage available natural ‘underutilized’ land for food forests)
	•	sa_lc_w_40ac_foodfor_10m.tif (converted 40 acreage available natural ‘underutilized’ land for food forests)
	•	n_workers: -1
	•	results_suffix: (leaving blank)
	•	runoff_proxy_path:
	•	E:/GIS/_natcap/san_antonio/Precip/sa_precip_32in.tif
	•	subsurface_critical_length_n: 0
	•	subsurface_critical_length_p: 0
	•	subsurface_eff_n: 0
	•	subsurface_eff_p: 0
	•	threshold_flow_accumulation: 500
	•	watersheds_path
	•	E:/GIS/_natcap/san_antonio/LULC_Vacant_Misc/San_Antonio_TX_buffer_mod.shp
	•	User guide: https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/ndr.html#interpreting-results 




References: 

	•	Doug Denu
	•	LULC and Parameters August 2024 
	•	https://github.com/natcap/urban-online-workflow#data-requirements
	•	By Chris Nootenboom
	•	in the Livable Cities/San Antonio/Results/Data folder, although it is pretty poorly documented and without a README. I believe the final outputs are in the Livable Cities/San Antonio/Results/Results folder. 
