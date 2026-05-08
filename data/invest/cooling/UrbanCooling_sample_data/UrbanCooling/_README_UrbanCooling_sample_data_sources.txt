--------------------------------------
Sources for Urban Cooling sample data
--------------------------------------

Land Use/Land Cover (land_use_2021.tif): United States National Land Cover Dataset (NLCD) 2021
https://www.usgs.gov/centers/eros/science/national-land-cover-database

Reference Evapotranspiration (reference_evapotranspiration_annual.tif): Global Aridity Index and Potential Evapotranspiration Database v3: https://www.global-ai-pet.org/global-aridity-index-pet-database, which is based on WorldClim data 1970-2000.

Area Of Interest (AOI_admin_boundaries_census_tracts.shp): Census tract polygons from the US Census Tiger data.

Area of interest may also be defined by AOI_polygon.shp, which was created manually, to cover a reasonably small area that’s useful for testing.

Biophysical table (biophysical_table_urban_cooling.csv): Values are from undocumented sources. So do not rely on them for your study.

Buildings (buildings.shp): Data source is undocumented, but might be from Open Street Map.

Energy Consumption Table (energy_consumption.csv): Values are from undocumented sources. So do not rely on them for your study.

The other non-spatial parameter values are based on either User Guide defaults, or a quick internet search for values relevant to the Minneapolis area (like air temperature). For your own study, be sure to use values that are relevant to your location, which these may not be. 

Reference air temperature (deg C):		23.2
UHI effect (deg C):		2.05
	https://yceo.users.earthengine.app/view/uhimap
Air Blending Distance (m):	600
Maximum Cooling Distance (m):	450

Work Productivity Valuation:
Average relative humidity (0-100%):		30

Manually Adjust Cooling Capacity Index Weights (optional):
Shade Weight:						0.6
Albedo Weight:						0.2
Evapotranspiration Weight:			0.2

--------------------
For more information
--------------------

See the InVEST User Guide Data Needs section for more information on the inputs and outputs of this model: https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html#data-needs

And the InVEST User Guide Data Sources section for pointers to global data that may be used as input: https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html#appendix-data-sources-and-guidance-for-parameter-selection