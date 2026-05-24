README
TL;DR:
	•	please update all input data to what is contained here! The parameter tables may not play nicely with prior versions of the LULC overlay
	•	for filtering land cover values for user scenario generations: classification_structure_qaqc.xlsx
	•	for bivariate plotting: acs_block_group_equity_data.csv
	•	for final parameter tables: ucm__nlcd_nlud_tree.csv una__nlcd_nlud_tree.csv carbon__nlcd_nlud_tree.csv (and see notes Notes on NASA Urban parameterization QA.docx)

CONTENTS
Folders
	•	InVEST Results (folder): Results of InVEST models run using the data and parameter tables in the main folder

Spreadsheets
	•	acs_block_group_equity_data.csv: CSV containing processed ACS data (percent BIPOC, per capita income) and zonal stat results from the InVEST model runs provided.
	•	Joins to acs_block_group.gpkg with the “GEO_ID” field
	•	Contains columns for creating bivariate colorscheme maps comparing Percent BIPOC and Air Temperature
	•	percent_bipoc_stdev_bin: bins representing
	•	0: more than 1 standard deviation below the mean
	•	1: within 1 standard deviation of the mean
	•	2: more than 1 standard deviation above the mean
	•	average_temp_stdev_bin: (see above, but for temperature)
	•	bivariate_category: a combination of the two bins, used for linking to colormap csv in code (not provided)
	•	first digit is percent_bipoc
	•	second digit is average_temp
	•	bivariate_colors: hex colors for each bivariate category
	•	ucm__nlcd_nlud_tree.csv: final parameter table for the Urban Cooling model
	•	una__nlcd_nlud_tree.csv: final parameter table for the Urban Nature Access model
	•	carbon__nlcd_nlud_tree.csv: final parameter table for the Carbon model
	•	classification_structure_qaqc.xlsx: documentation of the lucodes in the lulc_overlay_3857.tif
	•	Contains column is_realistic_to_create that should be used to filter the user’s choice of which land cover classes and/or tree canopy levels can be selected within a zoning type
	•	lulc_crosswalk.csv
	•	a CSV export of classification_structure_qaqc.xlsx with some changes:
	•	renamed “lulc_desc” to “nlcd_lulc” to match previous version of this table
	•	“is_realistic_to_create” contains “yes” or “no” instead of “no” or null.

Word Docs
	•	Notes on NASA Urban parameterization QA.docx: detailed notes on the QA/QC process for land cover categories and parameterization

Rasters
	•	et0_annual_cgiar_3857.tif: et0 raster for use in Urban Cooling model
	•	lulc_overlay_3857.tif: the NLCD + NLUD + Tree canopy overlay dataset, for use in all LULC inputs to InVEST. Harmonized with parameter tables. 
	•	nlcd_3857.tif: NLCD landcover data, retained for reference purposes
	•	nlud_3857.tif: NLUD land use data, retained for reference purposes
	•	population_per_pixel_2020_3857.tif: population density raster for use in Urban Nature Access model
	•	tree_3857.tif: tree canopy data, retained for reference purposes

Vectors
	•	acs_block_group.gpkg: census block groups for the study area
	•	Joins to acs_block_group_equity_data.csv with the “GEO_ID” field


