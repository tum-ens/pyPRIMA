import os
# from data_functions import *
# from model_functions import *
import numpy as np
# from scipy.ndimage import convolve
# import datetime
import geopandas as gpd
import pandas as pd
# from rasterio import windows
# from shapely.geometry import mapping, Point
# import fiona
import hdf5storage
# from multiprocessing import Pool
# from itertools import product
# import h5netcdf
# import cProfile
# import pstats
# import shapely
# from osgeo import gdal, ogr
# import osr
from helping_functions import *


def initialization():
    # timecheck('Start')
    # import param and paths
    from config import paths, param

    return paths, param


def generate_sites_from_shapefile(paths):
    '''
    description
    '''
    # Read the shapefile containing the map data
    regions = gpd.read_file(paths["SHP"])
    for i in regions.index:
        regions.loc[i, 'Longitude'] = regions.geometry.centroid.iloc[i].x
        regions.loc[i, 'Latitude'] = regions.geometry.centroid.iloc[i].y

    # Remove duplicates
    df = regions.set_index('NAME_SHORT')
    df = df.loc[((df.index.duplicated(keep=False)) & (df['Population'] > 0)) | (
                (~df.index.duplicated(keep=False)) & (df['Population'] == 0))]
    df.reset_index(inplace=True)
    regions = df.copy()

    # Extract series of site names, rename it and convert it into a dataframe
    zones = pd.DataFrame(
        regions[['NAME_SHORT', 'Area', 'Population', 'Longitude', 'Latitude']].rename(columns={'NAME_SHORT': 'Site'}))
    zones = zones[['Site', 'Area', 'Population', 'Longitude', 'Latitude']]
    zones.sort_values(['Site'], inplace=True)
    zones.reset_index(inplace=True, drop=True)

    zones.to_csv(paths["model_regions"] + 'Sites.csv', index=False, sep=';', decimal=',')

    ### Eventually move this part to special modules that create urbs / evrys models!
    # Preparing output for evrys
    zones_evrys = zones[['Site', 'Latitude', 'Longitude']].rename(columns={'Latitude': 'lat', 'Longitude': 'long'})
    zones_evrys['slacknode'] = 0
    zones_evrys.loc[0, 'slacknode'] = 1
    zones_evrys['syncharea'] = 1
    zones_evrys['ctrarea'] = 1
    zones_evrys['primpos'] = 0
    zones_evrys['primneg'] = 0
    zones_evrys['secpos'] = 0
    zones_evrys['secneg'] = 0
    zones_evrys['terpos'] = 0
    zones_evrys['terneg'] = 0
    zones_evrys = zones_evrys[
        ['Site', 'slacknode', 'syncharea', 'lat', 'long', 'ctrarea', 'primpos', 'primneg', 'secpos', 'secneg', 'terpos',
         'terneg']]

    # Preparing output for urbs
    zones_urbs = zones[['Site', 'Area']].rename(columns={'Site': 'Name', 'Area': 'area'})
    zones_urbs['area'] = zones_urbs['area'] * 1000000  # in m²

    zones_evrys.to_csv(paths["evrys"] + 'Sites_evrys.csv', index=False, sep=';', decimal=',')
    zones_urbs.to_csv(paths["urbs"] + 'Sites_urbs.csv', index=False, sep=';', decimal=',')


def generate_intermittent_supply_timeseries(paths, param):
    '''
	description
	'''


def generate_load_timeseries(paths, param):
    '''
	The goal of this script is to extract and preprocess data to be used as load timeseries for every site modeled in evrys and urbs. The data is extracted from excel spreadsheets and raster maps, and output into two .csv files. The data needs to be disaggregated sectorially and spatially before being aggregated again into regions, formatted, and exported. 
# 
# 
# ## Requirements:
# 
# The folder “Files Europe” contains the input and output folders, named “00 Input” and “NUTS0_wo_Balkans” respectively. The output folder’s name corresponds to the regionalization desired, and is referred to specific shape and raster files in the input folder.
# 
# In order to run this code, the user will need to install python 3.6 and all the libraries as well as their dependencies. 
# Anaconda3 is recommended. Create a new environment and install the required packages (see libraries). 
# 
# “Zonal_stats.py” is also needed to run the script
# 
# ### Libraries:
# 
# - Pandas (Python Data Analysis Library) - Used for manipulating and analyzing large data structures
# - GeoPandas – In tandem with Pandas, allows to process geospatial data
# - Gdal and Rasterio – Makes working with rasters easier
# 
# ### Definitions
# 
# Raster:
# 
# A bitmap image, consisting of a grid of pixel corresponding to a specific data set. Use QGIS (open source) to visualize the data.
# 
# ## Parameter Glossary
# <a id='SectionA'></a>
#  
# ### A - Data Acquisition
# 
# 1. countries :
# Table of all countries present in the raster with Population and GDP information
# 2. df_load_countries : 
# Hourly load over the year for all countries
# <a id= 'SectionB'></a> 
# 
# ### B - Sectorial Disaggregation
# 
# 1. sec_share : 
# Share of sectors in electricity demand
# 2. Profiles :
# Generic shape of load profile for each sector
# 3. df_sectors :
# Hourly load for each sector in each country over the year
# 4. load_sector :
# Annual load for each sector in each country
# <a id= 'SectionC'></a>
# 
# ### C - Spatial Disaggregation
#  
# 1. sector_lu : 
# Normalized land use coefficient by sectors
# 2. stat :
# Land use and corresponding sectorial spatial distribution per country
# 3. load_landuse :
# Hourly load associated to each land use type in each country over the year
# <a id= 'SectionD'></a>
# 
# ### D - Regional Aggregation
# 
# 1. border_region :
# Geodataframes of each region, corresponding to location, shape, name and population within the region
# 2. border_country :
# Geodataframes of each country, corresponding to location, shape, name and population within the country
# 3. stat_sub :
# Table of the land use and population in each sub region, denominated by region and country
# 4. load_subregion :
# Hourly sectorial load for each sub region over the year
# 5. load_region :
# Houlry sectorial load for each region over the year
# 
# #### Legend of land use map
# * 0	Water
# * 1	Evergreen Needle leaf Forest
# * 2 	Evergreen Broadleaf Forest
# * 3	Deciduous Needle leaf Forest
# * 4 	Deciduous Broadleaf Forest
# * 5 	Mixed Forests
# * 6 	Closed Shrublands
# * 7	Open Shrublands
# * 8 	Woody Savannas
# * 9 	Savannas
# * 10 	Grasslands
# * 11 	Permanent Wetland
# * 12 	Croplands
# * 13 	Urban and Built-Up
# * 14	Cropland/Natural Vegetation Mosaic
# * 15 	Snow and Ice
# * 16 	Barren or Sparsely Vegetated
# 
# ## Input:
# 
# - Hourly load values for all country of the ENTSOE each month of the year 2015. 
#     - name: Statistics_entsoe_YYYYMM_all.xls  
# - Raster of the country's name, shape, population, and GDP
#     - name: Europe_ NUTS0_wo_Balkans.shp
# - Raster of the regions of interest
#     - name: Europe_ NUTS0_wo_Balkans.shp
# - Correspondence between land use and sector as well as other user defined assumptions
#     - name: assumptions_const.xlsx
# - Load profiles of typical for Households, industry, street lighting, commercial and agriculture.
#     - folder: Load profiles
# - Rasters of population and land use of Europe
#     - names: Europe_Population.tif & Europe_Landuse.tif
#     
# ## Outputs:
# 
# - CSV file with time series for every site modeled in evrys.
#     - name: demand_evrys2015.csv
#     - information: t, sit, co, value
# - CSV file with time series for urbs site modeled in urbs.
#     - name: demand_urbs2015.csv
#     - information: t, XX.Elec
    '''
    # Sector land use allocation
    # The land use coefficients found in the assumptions table are normalized over each sector, and the results are stored in the table 'sector_lu'
    sector_lu = pd.read_excel(paths["assumptions"], sheet_name='Landuse', index_col=0, usecols=[0, 2, 3, 4])
    landuse_types = [str(i) for i in sector_lu.index]
    sector_lu = sector_lu.transpose().div(np.repeat(sector_lu.sum(axis=0)[:, None], len(sector_lu), axis=1))
    sec = [str(i) for i in sector_lu.index]
    sec_dict = dict(zip(sec, param["load"]["sectors"]))

    # Share of sectors in electricity demand
    sec_share = pd.read_csv(paths["sector_shares"], sep=';', decimal=',', index_col=0)

    if not (os.path.isfile(paths["load"] + 'df_sectors.csv') and os.path.isfile(
            paths["load"] + 'load_sector.csv') and os.path.isfile(paths["load"] + 'load_landuse.csv')):
        countries = gpd.read_file(paths["Countries"])
        countries = countries.drop(countries[countries["Population"] == 0].index)
        countries = countries[['NAME_SHORT', 'Population']].rename(
            columns={'NAME_SHORT': 'Country'})  # Eventually add GDP

        # Get dataframe with cleaned timeseries for countries
        df_load_countries = clean_load_data(paths, param, countries)

        # ADD IF statement to jump to urbs/evrys, if countries = desired resolution

        # Get sectoral profiles
        profiles = get_sectoral_profiles(paths, param)

        # Plot figures?
        # plt.figure(figsize=(10,7))
        # plt.plot(range(169), profiles.loc[0:168,'RES'], 'r',
        #         range(169), profiles.loc[0:168,'IND'], 'k',
        #         range(169), profiles.loc[0:168,'COM'], 'b',
        #         range(169), profiles.loc[0:168,'AGR'], 'g--',
        #         range(169), profiles.loc[0:168,'STR'], 'y--',
        #         linewidth=2)
        # plt.legend(['RES', 'IND', 'COM', 'AGR'])
        # plt.xlabel('Hours',fontsize=15)
        # plt.ylabel('Load p.u. yearly sectoral load', fontsize=15)
        # plt.title('First week', fontsize=15)

        # Prepare an empty table of the hourly load for the five sectors in each countries.
        df_sectors = pd.DataFrame(0, index=df_load_countries.index, columns=pd.MultiIndex.from_product(
            [df_load_countries.columns.tolist(), sec + ['RES']], names=['Country', 'Sector']))
        # Copy the load profiles for each sector in the columns of each coutry, and multiply each sector by the share defined in 'sec_share'. 
        # Note that at the moment the values are the same for all countries
        for c in df_load_countries.columns:
            for s in sec + ['RES']:
                df_sectors.loc[:, (c, s)] = profiles[s] * sec_share.loc[c, s]

        # Normalize the loads profiles over all sectors by the hour ei. The Sum of the loads of all sectors = 1 for each hour
        df_scaling = df_sectors.groupby(level=0, axis=1).sum()
        for c in df_load_countries.columns:
            for s in sec + ['RES']:
                df_sectors.loc[:, (c, s)] = df_sectors.loc[:, (c, s)] / df_scaling[c]

        # Multiply the normalized hourly loads profiles by the actual hourly loads for each country 
        # e.i. the share of the actual load for each sector is captured and stored in the table 'df_sectors' 
        for c in df_load_countries.columns:
            for s in sec + ['RES']:
                df_sectors.loc[:, (c, s)] = df_sectors.loc[:, (c, s)] * df_load_countries[c]

        # Calculate the yearly load per sector and country
        load_sector = df_sectors.sum(axis=0).rename('Load in MWh')

        # Count pixels of each land use type and create weighting factors for each country:
        # Population
        stat_pop = pd.DataFrame.from_dict(zonal_stats(paths["Countries"], paths["POP"], 'population'))
        stat_pop.rename(columns={'NAME_SHORT': 'Country', 'sum': 'RES'}, inplace=True)
        stat_pop.set_index('Country', inplace=True)

        # Land use
        stat = pd.DataFrame.from_dict(zonal_stats(paths["Countries"], paths["LU"], 'landuse'))
        stat.rename(columns={'NAME_SHORT': 'Country'}, inplace=True)
        stat.set_index('Country', inplace=True)
        ### TODO: move list of landuse types to config?
        stat = stat.loc[:,
               ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']].fillna(0)
        stat = stat[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']]

        # Join the two dataframes
        stat = stat.join(stat_pop[['RES']])

        # Weighting by sector
        for s in sec:
            for i in stat.index:
                stat.loc[i, s] = np.dot(stat.loc[i, landuse_types], sector_lu.loc[s])

        rows = landuse_types.copy()
        rows.append('RES')
        m_index = pd.MultiIndex.from_product([stat.index.tolist(), rows], names=['Country', 'Land use'])
        load_landuse = pd.DataFrame(0, index=m_index, columns=df_sectors.index)
        for c in stat.index.tolist():  # Countries
            load_landuse.loc[c, 'RES'] = load_landuse.loc[c, 'RES'] + df_sectors[(c, 'RES')] / stat.loc[c, 'RES']
            for lu in landuse_types:  # Land use types
                for s in sec:  # other sectors
                    load_landuse.loc[c, lu] = load_landuse.loc[c, lu] + sector_lu.loc[s, int(lu)] * df_sectors[(c, s)] / \
                                              stat.loc[c, s]

        # Save the data into HDF5 files for faster execution
        df_sectors.to_csv(paths["load"] + 'df_sectors.csv', sep=';', decimal=',', index=True, header=True)
        print("files saved: " + paths["load"] + 'df_sectors.csv')
        load_sector.to_csv(paths["load"] + 'load_sector.csv', sep=';', decimal=',', index=True, header=True)
        print("files saved: " + paths["load"] + 'load_sector.csv')
        load_landuse.to_csv(paths["load"] + 'load_landuse.csv', sep=';', decimal=',', index=True)
        print("files saved: " + paths["load"] + 'load_landuse.csv')

    # Read CSV files		
    df_sectors = pd.read_csv(paths["load"] + 'df_sectors.csv', sep=';', decimal=',')
    load_sector = pd.read_csv(paths["load"] + 'load_sector.csv', sep=';', decimal=',')
    load_landuse = pd.read_csv(paths["load"] + 'load_landuse.csv', sep=';', decimal=',')

    # Split regions into subregions (a region can overlap with many countries, but a subregion belongs to only one country)
    intersection_regions_countries(paths)

    # Count number of pixels for each subregion
    # Population
    stat_pop_sub = pd.DataFrame.from_dict(
        zonal_stats(paths["model_regions"] + 'intersection.shp', paths["POP"], 'population'))
    stat_pop_sub.rename(columns={'NAME_SHORT': 'Subregion', 'sum': 'RES'}, inplace=True)
    stat_pop_sub.set_index('Subregion', inplace=True)

    # Land use
    stat_sub = pd.DataFrame.from_dict(zonal_stats(paths["model_regions"] + 'intersection.shp', paths["LU"], 'landuse'))
    stat_sub.rename(columns={'NAME_SHORT': 'Subregion'}, inplace=True)
    stat_sub.set_index('Subregion', inplace=True)
    stat_sub.fillna(0, inplace=True)
    stat_sub = stat_sub[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']]

    # Join the two dataframes
    stat_sub = stat_sub.join(stat_pop_sub[['RES']])

    # Add attributes for country/region
    stat_sub['Region'] = 0
    stat_sub['Country'] = 0
    for i in stat_sub.index:
        stat_sub.loc[i, ['Region', 'Country']] = i.split('_')

    # Calculate the hourly load for each subregion
    load_subregions = pd.DataFrame(0, index=stat_sub.index, columns=df_sectors.index.tolist() + ['Region', 'Country'])
    load_subregions[['Region', 'Country']] = stat_sub[['Region', 'Country']]
    for sr in load_subregions.index:
        c = load_subregions.loc[sr, 'Country']
        # For residential:
        load_subregions.loc[sr, df_sectors.index.tolist()] = load_subregions.loc[sr, df_sectors.index.tolist()] + \
                                                             stat_sub.loc[sr, 'RES'] * load_landuse.loc[c, 'RES']
        for lu in landuse_types:
            load_subregions.loc[sr, df_sectors.index.tolist()] = load_subregions.loc[sr, df_sectors.index.tolist()] + \
                                                                 stat_sub.loc[sr, lu] * load_landuse.loc[c, lu]
            ### TO BE CONTINUED


if __name__ == '__main__':
    paths, param = initialization()
    # generate_sites_from_shapefile(paths)
    generate_intermittent_supply_timeseries(paths, param)
    generate_load_timeseries(paths, param)
    # generate_commodities(paths, param) # corresponds to 04
    # distribute_renewable_capacities(paths, param) # corresponds to 05a
    # clean_processes_and_storage_data(paths, param) # corresponds to 05b I think
    # generate_processes(paths, param) # corresponds to 05c
    # generate_storage(paths, param) # corresponds to 05d
    # clean_grid_data(paths, param) # corresponds to 06a
    # generate_aggregated_grid(paths, param) # corresponds to 06b
    # generate_urbs_model(paths, param)
    # generate_evrys_model(paths, param)
