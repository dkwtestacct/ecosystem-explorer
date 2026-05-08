# San Antonio, TX — Data Setup

Scaffold directory for the second city. The app's `CITIES['San Antonio, TX']` entry
has `available: False` until the inputs below are in place; once they are, fill in
the placeholder constants and flip the flag to `True`.

## Status

- [ ] LULC raster (NLCD 2021, clipped to San Antonio boundary)
- [ ] Soil hydrologic group raster (SSURGO)
- [ ] Population raster (Census 2020 for Bexar County, rasterized to LULC grid)
- [ ] Curve Number biophysical table (per-NLCD-code × soil group)
- [ ] Urban Cooling biophysical table (per-NLCD-code shade / Kc / albedo)
- [ ] Reference evapotranspiration raster
- [ ] Buildings + roads + tracts shapefiles for Bexar County
- [ ] Damage-loss table + crop-yield table specific to San Antonio
- [ ] Baseline constants in `app.py`'s `CITIES` dict

## Project context

**Source:** NatCap San Antonio Urban Agriculture Project 2023.

Modeled outcomes (per project scope): Urban Cooling, Urban Flood Risk (UFR),
Urban Nature Access (UNA), Carbon Storage, Crop Yield, Nutrient Delivery (NDR).

**Land eligibility for conversion:** publicly owned, > 1 acre, excludes wetlands.
This is more restrictive than the Minneapolis "all developed land minus building
footprints" rule and will likely require an additional eligibility mask raster.

**Food forest crops (per project report):** pecan, fig, mulberry, nopal —
yield benchmarks should be averaged from these rather than reusing the
Minneapolis 11,500 lbs/acre figure.

**Urban farm crops (per project report):** 8 vegetables (TODO: enumerate from
report). May warrant a separate "urban farm" land use category alongside the
existing food forest type.

## Where to get each input

### LULC (`flood/LULC_NLCD_2021_SA.tif` and `cooling/land_use_2021.tif`)
- Download NLCD 2021 from <https://www.mrlc.gov/data>
- Clip to San Antonio city boundary (or Bexar County for full coverage)
- Reproject to **EPSG:3857** (Web Mercator) per project hint, **30 m** pixel
  size, integer Byte type
- Confirm bounds and pixel count, then update `pixel_area_acres` in `CITIES`

### Soil group (`flood/soil_group_SA.tif`)
- USDA SSURGO via <https://websoilsurvey.nrcs.usda.gov> (Bexar County) or
  <https://www.nrcs.usda.gov/resources/data-and-reports/soil-survey-geographic-database-ssurgo>
- Rasterize hydrologic group (A=1 / B=2 / C=3 / D=4) to match the LULC grid

### Population (`population/sa_pop_2020.tif`)
- Easiest: Census 2020 block totals via the decennial PL API for Bexar County
  (FIPS 48029), TIGER 2020 tabulation-block polygons, rasterized to the LULC
  grid. Pattern matches `download_census_pop.py`.
- Alternative: WorldPop if the model area extends outside Bexar County.

### CN biophysical table (`flood/UFR_biophysical_table_SA.csv`)
- Same schema as Minneapolis (`lucode, NLCD_Land, CN_A, CN_B, CN_C, CN_D`)
- Either copy values from the Minneapolis table (CN values are USDA-standard
  per land cover, not city-specific) or use SA-specific values from the
  NatCap project deliverables if they tuned them.

### Urban Cooling biophysical table (`cooling/biophysical_table_urban_cooling.csv`)
- Same schema (`lucode, lulc_desc, shade, kc, albedo, green_area, building_intensity`)
- SA's hotter / drier climate may justify lower kc and shade values for some
  classes — confirm with project report.

### Reference ET (`cooling/reference_evapotranspiration_annual.tif`)
- Coarse-resolution ET from NatCap or USGS climate normals
- Same coverage area as LULC, lower resolution acceptable (the app already
  resamples 1 km ET to 30 m via bilinear)

### Buildings, roads, tracts (`flood/buildings.shp`, etc.)
- City of San Antonio Open Data portal: <https://data.sanantonio.gov>
- TIGER 2020 tracts for Bexar County:
  `https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_48_tabblock20.zip`
- Reproject to EPSG:3857 to match LULC

### Damage-loss + yield tables (`flood/Damage_loss_table_SA.csv`, etc.)
- Damage rates may differ from Minneapolis ($/m² by building type) — request
  from Yingjie or use national defaults
- Crop-yield benchmarks: per-crop lbs/acre/year from the SA project report
  (pecan + fig + mulberry + nopal averaged for food forest; 8-veg average
  for urban farm)

## Constants to fill in once data is in place

In `app.py`'s `CITIES['San Antonio, TX']` block:

```python
'baseline_cn':          XX.X,    # mean CN of unmodified SA LULC × soil grid
'baseline_hm':          0.XXXX,  # mean HM index of unmodified SA LULC
'pixel_area_acres':     0.XXX,   # depends on EPSG:3857 + 30 m pixel size
'food_forest_lbs_acre': XXXXX,   # average of pecan + fig + mulberry + nopal
'available':            True,    # flip last
```

After flipping `available: True`, restart the app and select San Antonio in
the city picker — the existing scenario engine, surrogate, and UI all read
from the dict, so no further code changes should be needed for the basic
flow. (The Crop Yield and NDR models are *new* outputs not yet wired into
the metric cards; those need their own `evaluate_scenario` extensions.)

## Contact

For early access to baseline constants or the SA-specific biophysical
tables, contact the project team.
