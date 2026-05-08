
Sources for Urban Flood Risk example data
-----------------------------------------

Land use/land cover (LULC_NLCD_2021_MN): United States National Land Cover Dataset (NLCD) 2021
https://www.usgs.gov/centers/eros/science/national-land-cover-database

Soil hydrologic group (soil_group_MN.tif): SSURGO Portal (https://www.nrcs.usda.gov/resources/data-and-reports/ssurgo-portal) was used to generate an initial map, but most developed areas had no data. To fill in these values, pixels that are "Developed, high intensity" or "Developed, medium intensity" in the land use/land cover map were set to a value of 4 (highest runoff, assuming mostly impervious surface). The remaining missing values were set to 3, assuming more of a mix of pervious and impervious surfaces. 

The general area of interest shapefile (area_of_interest_MN.shp) was created manually, a single polygon based on the area covered by the InVEST Urban Cooling model.

Area of interest may also be defined by admin_boundaries_census_tracts.shp, which are census tract polygons from the US Census Tiger data.

Curve number values in the biophysical table (UFR_biophysical_table_MN.csv) were taken from previous sample data, whose sources are undocumented. So please don't rely on them for your study.

The layer buildings.shp was taken from the previous sample data, whose sources are undocumented, but might be from Open Street Map.

The values in Damage_loss_table_MN.csv are completely made up.
