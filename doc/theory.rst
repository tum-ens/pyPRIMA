******
Theory
******
Load time series
================

Purpose
-------

Time series for load, demand, and supply are widely available, however the published datasets are usually restricted to predefined spatial regions such as 
countries, and their usual subdivisions. The generate_load_timeseries() function takes the more general datasets available for these spatial resolutions 
and disaggregate them according to a set of parameters, before aggregating them at a different spatial level. The obtained time series is then fitted to 
any region, and over a wider range of temporal resolutions.

.. figure:: img/generate_load_ts.png
   :width: 100%
   :align: center
   :alt: Generate_load_timeseries description
   
   Generate_load_timeseries - Process description

Inputs
------

The main inputs of this script are:

* Load time series of the countries or regions to be disaggregated (hourly). 
* Shapefiles of the countries or regions to be disaggregated 
* Shapefiles of the regions of interest (subregions) 
* Assumptions (land use and sector correspondence) 
* Load profiles of the different sectors 
* Raster of the population and land use correspondent to the country or region 

Sectorial disaggregation
------------------------

The load is assumed to be perfectly divided into four distinct sectors (load sources):
 
* Commercial
* Industrial
* Residential
* Agricultural

.. figure:: img/sector_load_profiles.png
   :width: 60%
   :align: center
   :alt: Sectorial load profiles
   
   Sectorial load profiles

Sector Load Shares:

.. tabularcolumns:: |l|c|c|c|c|
	
+---------+----------+----------+-------------+-------------+
| Region  | Industry | Commerce | Residential | Agriculture |
+=========+==========+==========+=============+=============+
|    A    |   0.41%  |   0.28%  |     0.29%   |     0.02%   |
+---------+----------+----------+-------------+-------------+
|    B    |   0.31%  |   0.30%  |     0.38%   |     0.01%   |
+---------+----------+----------+-------------+-------------+
|    C    |   0.44%  |   0.30%  |     0.25%   |     0.01%   |
+---------+----------+----------+-------------+-------------+

An hourly load profile for each sector has been predefined for one week, the same load profile is assumed to repeat over the year. These load profiles are 
scaled and normalized based on sectorial load shares for each region(assumed to be constant throughout the spatial scope), by multiplying the load profiles by their 
corresponding share and normalizing their hourly sum to be equal to 1.

.. figure:: img/scaled_sectorial_load_profiles.png
   :width: 80%
   :align: center
   :alt: Scaled sectorial load profiles
   
   Scaled sectorial load profiles
  
.. figure:: img/normalized_sectorial_load_profiles.png
   :width: 80%
   :align: center
   :alt: Normalized Sectorial load profiles
   
   Normalized sectorial load profiles
   
   
 
Once the load profiles are normalized, we can multiply them with the actual load time series to obtain the load timeseries for each sector. 

.. figure:: img/load_time_series.png
   :width: 80%
   :align: center
   :alt: Load time series
   
   Load time series
	
.. figure:: img/load_per_sector.png
   :width: 80%
   :align: center
   :alt: Load per Sector
   
   Sectorial Load time series
   

Spatial disaggregation
----------------------

The next step is the spatial disaggregation, based on the land use and population concentration rasters. First, each land-use type is assigned a sectorial 
load percentage corresponding to the load components of the land use category. Then, the population concentration raster is used to calculate the population 
of each pixel.

.. figure:: img/sector_commercial_disaggregation.png
   :width: 80%
   :align: center
   :alt: Example - Commerce sector disaggregation 
   
   Example - Commerce sector spatial disaggregation

Counting the pixels, and land use occurrences inside of region for which the sectorial load timeseries has been calculated, the sectorial load 
for the Industry, commercial, and agricultural can be retrieved for each pixel of that region. Similarly, the residential load timeseries can 
be assigned to each pixel based on the population contained in the said pixel. The spatial disaggregation results in the assignment to every pixel inside a 
given region to be assigned a specific sectorial load timeseries.

	
.. figure:: img/sector_residence_disaggregation.png
   :width: 80%
   :align: center
   :alt: Example - Residential sector disaggregation
   
   Example - Residential sector spatial disaggregation

Re-aggregation
--------------

The result of the sectorial and spatial disaggregation, performed in the first two sections can be used to retrieve the sectorial load timeseries and, 
therefore, the general load time series of any desired region by summing up the loads of every pixel contained within the region. If a subregion spans 
more than one region or country, it is divided into subregions restrained to each of those countries. 
