import os
from helping_functions import *
import shutil
import openpyxl
from scipy.ndimage import convolve
import datetime


def initialization():
    timecheck('Start')
    from config import paths, param
    res_weather = param["res_weather"]
    res_desired = param["res_desired"]

    # read shapefile of regions
    regions_shp = gpd.read_file(paths["SHP"])
    # Extract onshore and offshore areas separately
    param["regions_land"] = regions_shp.drop(regions_shp[regions_shp["Population"] == 0].index)
    param["regions_eez"] = regions_shp.drop(regions_shp[regions_shp["Population"] != 0].index)
    # Recombine the maps in this order: onshore then offshore

    regions_all = gpd.GeoDataFrame(pd.concat([param["regions_land"], param["regions_eez"]],
                                             ignore_index=True), crs=param["regions_land"].crs)

    param["nRegions_land"] = len(param["regions_land"])
    param["nRegions_eez"] = len(param["regions_eez"])

    nRegions = param["nRegions_land"] + param["nRegions_eez"]
    Crd_regions = np.zeros((nRegions, 4))
    for reg in range(0, nRegions):
        # Box coordinates for MERRA2 data
        r = regions_all.bounds.iloc[reg]
        box = np.array([r["maxy"], r["maxx"], r["miny"], r["minx"]])[np.newaxis]
        Crd_regions[reg, :] = crd_merra(box, res_weather)
    Crd_all = np.array([max(Crd_regions[:, 0]), max(Crd_regions[:, 1]), min(Crd_regions[:, 2]), min(Crd_regions[:, 3])])
    param["Crd_regions"] = Crd_regions
    param["Crd_all"] = Crd_all
    timecheck('End')
    return paths, param


def generate_sites_from_shapefile(paths):
    '''
    description
    '''
    timecheck('Start')

    # Read the shapefile containing the map data
    regions = gpd.read_file(paths["SHP"])
    for i in regions.index:
        regions.loc[i, 'Longitude'] = regions.geometry.centroid.iloc[i].x
        regions.loc[i, 'Latitude'] = regions.geometry.centroid.iloc[i].y

    # Remove duplicates
    df = regions.set_index('NAME_SHORT')
    df = df.loc[((~df.index.duplicated(keep=False)) & (df['Population'] > 0)) |
                ((df.index.duplicated(keep=False)) & (~df['Population'] == 0))]
    df.reset_index(inplace=True)
    regions = df.copy()

    # Extract series of site names, rename it and convert it into a dataframe
    zones = pd.DataFrame(
        regions[['NAME_SHORT', 'Area', 'Population', 'Longitude', 'Latitude']].rename(columns={'NAME_SHORT': 'Site'}))
    zones = zones[['Site', 'Area', 'Population', 'Longitude', 'Latitude']]
    zones.sort_values(['Site'], inplace=True)
    zones.reset_index(inplace=True, drop=True)

    zones.to_csv(paths["model_regions"] + 'Sites.csv', index=False, sep=';', decimal=',')

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

    timecheck('End')


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

    timecheck('Start')

    # Sector land use allocation
    # The land use coefficients found in the assumptions table are normalized over each sector, and the results are
    #  stored in the table 'sector_lu'
    sector_lu = pd.read_excel(paths["assumptions"], sheet_name='Landuse', index_col=0, usecols=[0, 2, 3, 4])
    landuse_types = [str(i) for i in sector_lu.index]
    sector_lu = sector_lu.transpose().div(np.repeat(sector_lu.sum(axis=0)[:, None], len(sector_lu), axis=1))
    sec = [str(i) for i in sector_lu.index]
    sec_dict = dict(zip(sec, param["load"]["sectors"]))

    # Share of sectors in electricity demand
    sec_share = pd.read_csv(paths["sector_shares"], sep=';', decimal=',', index_col=0)
    stat = None

    # Count pixels of each land use type and create weighting factors for each country:
    # Population
    stat_pop = pd.DataFrame.from_dict(zonal_stats(paths["Countries"], paths["POP"], 'population'))
    stat_pop.rename(columns={'NAME_SHORT': 'Country', 'sum': 'RES'}, inplace=True)
    stat_pop.set_index('Country', inplace=True)

    # Land use
    stat = pd.DataFrame.from_dict(zonal_stats(paths["Countries"], paths["LU"], 'landuse'))
    stat.rename(columns={'NAME_SHORT': 'Country'}, inplace=True)
    stat.set_index('Country', inplace=True)
    stat = stat.loc[:, landuse_types].fillna(0)

    # Join the two dataframes
    stat = stat.join(stat_pop[['RES']])

    # Weighting by sector
    for s in sec:
        for i in stat.index:
            stat.loc[i, s] = np.dot(stat.loc[i, landuse_types], sector_lu.loc[s])

    if not (os.path.isfile(paths["load"] + 'df_sectors.csv') and
            os.path.isfile(paths["load"] + 'load_sector.csv') and
            os.path.isfile(paths["load"] + 'load_landuse.csv')):

        countries = gpd.read_file(paths["Countries"])
        countries = countries.drop(countries[countries["Population"] == 0].index)
        countries = countries[['NAME_SHORT', 'Population']].rename(
            columns={'NAME_SHORT': 'Country'})  # Eventually add GDP

        # Get dataframe with cleaned timeseries for countries
        df_load_countries = clean_load_data(paths, param, countries)

        # ADD IF statement to jump to urbs/evrys, if countries = desired resolution
        # I didn't get it *

        # Get sectoral profiles
        profiles = get_sectoral_profiles(paths, param)

        # Prepare an empty table of the hourly load for the five sectors in each countries.
        df_sectors = pd.DataFrame(0, index=df_load_countries.index, columns=pd.MultiIndex.from_product(
            [df_load_countries.columns.tolist(), sec + ['RES']], names=['Country', 'Sector']))

        # Copy the load profiles for each sector in the columns of each country, and multiply each sector by the share
        # defined in 'sec_share'.
        # Note that at the moment the values are the same for all countries

        for c in df_load_countries.columns:
            for s in sec + ['RES']:
                df_sectors.loc[:, (c, s)] = profiles[s] * sec_share.loc[c, s]

        # Normalize the loads profiles over all sectors by the hour ei. The Sum of the loads of all sectors = 1
        # for each hour

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

        rows = landuse_types.copy()
        rows.append('RES')
        m_index = pd.MultiIndex.from_product([stat.index.tolist(), rows], names=['Country', 'Land use'])
        load_landuse = pd.DataFrame(0, index=m_index, columns=df_sectors.index)
        status = 0
        length = len(stat.index.tolist()) * len(landuse_types) * len(sec)
        display_progress("Computing regions load", (length, status))
        for c in stat.index.tolist():  # Countries
            load_landuse.loc[c, 'RES'] = load_landuse.loc[c, 'RES'] \
                                         + df_sectors[(c, 'RES')] \
                                         / stat.loc[c, 'RES']
            for lu in landuse_types:  # Land use types
                for s in sec:  # other sectors
                    load_landuse.loc[c, lu] = load_landuse.loc[c, lu] \
                                              + sector_lu.loc[s, int(lu)] \
                                              * df_sectors[(c, s)] \
                                              / stat.loc[c, s]
                    status = status + 1
                    display_progress("Computing regions load", (length, status))

        # Save the data into HDF5 files for faster execution
        df_sectors.to_csv(paths["load"] + 'df_sectors.csv', sep=';', decimal=',', index=False, header=True)
        print("Dataframe df_sector saved: " + paths["load"] + 'df_sector.csv')
        load_sector.to_csv(paths["load"] + 'load_sector.csv', sep=';', decimal=',', index=True, header=True)
        print("Dataframe load_sector saved: " + paths["load"] + 'load_sector.csv')
        load_landuse.to_csv(paths["load"] + 'load_landuse.csv', sep=';', decimal=',', index=True)
        print("Dataframe load_landuse saved: " + paths["load"] + 'load_landuse.csv')

    # Read CSV files
    df_sectors = pd.read_csv(paths["load"] + 'df_sectors.csv', sep=';', decimal=',', header=[0, 1])
    load_sector = pd.read_csv(paths["load"] + 'load_sector.csv', sep=';', decimal=',', index_col=[0, 1])["Load in MWh"]
    load_landuse = pd.read_csv(paths["load"] + 'load_landuse.csv', sep=';', decimal=',', index_col=[0, 1])

    # Split regions into subregions
    # (a region can overlap with many countries, but a subregion belongs to only one country)
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

    stat_sub = stat_sub.loc[:, landuse_types].fillna(0)

    # Join the two dataframes
    stat_sub = stat_sub.join(stat_pop_sub[['RES']])

    # Add attributes for country/region
    stat_sub['Region'] = 0
    stat_sub['Country'] = 0
    for i in stat_sub.index:
        stat_sub.loc[i, ['Region', 'Country']] = i.split('_')

    # Calculate the hourly load for each subregion

    load_subregions = pd.DataFrame(0, index=stat_sub.index,
                                   columns=df_sectors.index.tolist() + ['Region', 'Country'])
    load_subregions[['Region', 'Country']] = stat_sub[['Region', 'Country']]
    status = 0
    length = len(load_subregions.index) * len(landuse_types)

    display_progress("Computing sub regions load:", (length, status))

    for sr in load_subregions.index:
        c = load_subregions.loc[sr, 'Country']
        # For residential:
        load_subregions.loc[sr, df_sectors.index.tolist()] = load_subregions.loc[sr, df_sectors.index.tolist()] \
                                                             + stat_sub.loc[sr, 'RES'] \
                                                             * load_landuse.loc[c, 'RES'].to_numpy()
        for lu in landuse_types:
            load_subregions.loc[sr, df_sectors.index.tolist()] = load_subregions.loc[sr, df_sectors.index.tolist()] \
                                                                 + stat_sub.loc[sr, lu] \
                                                                 * load_landuse.loc[c, lu].to_numpy()
            # show_progress
            status = status + 1
            display_progress("Computing sub regions load", (length, status))

    load_regions = load_subregions.groupby(['Region', 'Country']).sum()
    load_regions.reset_index(inplace=True)
    load_regions.set_index(['Region'], inplace=True)
    load_regions.to_hdf('load_subregion.hdf', 'df')

    for s in sec + ['RES']:
        zonal_weighting(paths, load_sector, stat, s)

    # Export to evrys/urbs

    # Aggregate subregions
    load_regions.reset_index(inplace=True)
    load_regions = load_regions.groupby(['Region']).sum()

    # Calculate the sum (yearly consumption) in a separate vector
    yearly_load = load_regions.sum(axis=1)

    # Calculte the ratio of the hourly load to the yearly load
    df_normed = load_regions / np.tile(yearly_load.to_numpy(), (8760, 1)).transpose()

    # Prepare the output in the desired format
    df_output = pd.DataFrame(list(df_normed.index) * 8760, columns=['sit'])
    df_output['value'] = np.reshape(df_normed.to_numpy(), -1, order='F')
    df_output['t'] = df_output.index // len(df_normed) + 1
    df_output = pd.concat([df_output, pd.DataFrame({'co': 'Elec'}, index=df_output.index)], axis=1)

    df_evrys = df_output[['t', 'sit', 'co', 'value']]  # .rename(columns={'Region': 'sit'})

    # Transform the yearly load into a dataframe
    df_load = pd.DataFrame()

    df_load['annual'] = yearly_load

    # Preparation of dataframe
    df_load = df_load.reset_index()
    df_load = df_load.rename(columns={'Region': 'sit'})

    # Merging load dataframes and calculation of total demand
    df_load['total'] = param["load"]["degree_of_eff"] * df_load['annual']

    if param["load"]["distribution_type"] == 'population_GDP':
        try:
            # Calculate regional information for load destribution
            demand_des_sum = demand_des.groupby(['Countries']).sum()
            demand_des_sum = demand_des_sum.reset_index().rename(
                columns={'GDP': 'GDP_Sum', 'Population': 'Population_Sum'})

            # Merging of regional information, annual load and normalized load
            df_help = pd.merge(df_load, demand_des_sum, how='inner', on=['Countries'])
            df_merged = pd.merge(df_output, demand_des, how='inner', on=['Site'])
            df_merged = df_merged.rename(columns={'Countries_x': 'Countries'})
            df_merged = pd.merge(df_merged, df_help, how='inner', on=['Countries'])

            # Calculation of the distributed load per region
            df_merged['value_normal'] = df_merged['value'] * df_merged['total'] * (
                    factor_GDP * df_merged['GDP'] / df_merged['GDP_Sum'] + factor_Pop * df_merged['Population'] /
                    df_merged['Population_Sum'])
        except:
            raise Exception('Demand_des not implemented')
    else:
        df_merged = pd.merge(df_output, df_load, how='outer', on=['sit'])
        df_merged['value_normal'] = df_merged['value'] * df_merged['total']

    # Calculation of the absolute load per country
    df_absolute = df_merged  # .reset_index()[['t','Countries','value_normal']]

    # Rename the countries
    df_absolute['sitco'] = df_absolute['sit'] + '.Elec'

    df_urbs = df_absolute.pivot(index='t', columns='sitco', values='value_normal')
    df_urbs = df_urbs.reset_index()

    # Yearly consumption for each zone
    Load_EU = pd.DataFrame(df_absolute.groupby('sit').sum()['value_normal'].rename('Load'))

    # Output
    Load_EU.to_csv(paths["load_EU"], sep=',', index=True)
    df_evrys.to_csv(paths["evrys"] + 'Demand_evrys' + '%04d' % (param["year"]) + '.csv', sep=',', index=False)
    df_urbs.to_csv(paths["urbs"] + 'Demand_urbs' + '%04d' % (param["year"]) + '.csv', sep=';', decimal=',', index=False)
    timecheck('End')


def generate_commodities(paths, param):
    timecheck('Start')

    assumptions = pd.read_excel(paths["assumptions"], sheet_name='Commodity')
    commodities = list(assumptions['Commodity'].unique())

    dict_price_instate = dict(zip(assumptions['Commodity'], assumptions['price mid']))
    dict_price_outofstate = dict(zip(assumptions['Commodity'], assumptions['price out-of-state']))
    dict_type_evrys = dict(zip(assumptions['Commodity'], assumptions['Type_evrys']))
    dict_type_urbs = dict(zip(assumptions['Commodity'], assumptions['Type_urbs']))
    dict_annual = dict(zip(assumptions['Commodity'], assumptions['annual']))
    dict_co_max = dict(zip(assumptions['Commodity'], assumptions['max']))
    dict_maxperstep = dict(zip(assumptions['Commodity'], assumptions['maxperstep']))

    # Read the CSV containing the list of sites
    sites = pd.read_csv(paths["model_regions"] + 'Sites.csv', sep=';', decimal=',')

    # Read the CSV containing the annual load
    load = pd.read_csv(paths["load_EU"], index_col=['sit'])

    # Prepare output tables for evrys and urbs

    output_evrys = pd.DataFrame(columns=['Site', 'Co', 'price', 'annual', 'losses', 'type'], dtype=np.float64)
    output_urbs = pd.DataFrame(columns=['Site', 'Commodity', 'Type', 'price', 'max', 'maxperstep'])

    # Fill tables
    for s in sites["Site"]:
        for c in commodities:
            if c == 'Elec':
                if s in load.index:
                    annual = load.loc[s][0]
                else:
                    annual = 0
            else:
                annual = dict_annual[c]
            if len(s) >= 2:
                output_evrys = output_evrys.append(
                    {'Site': s, 'Co': c, 'price': dict_price_instate[c], 'annual': annual, 'losses': 0,
                     'type': dict_type_evrys[c]}, ignore_index=True)
                output_urbs = output_urbs.append(
                    {'Site': s, 'Commodity': c, 'Type': dict_type_urbs[c], 'price': dict_price_instate[c],
                     'max': dict_co_max[c], 'maxperstep': dict_maxperstep[c]}, ignore_index=True)
            else:
                output_evrys = output_evrys.append(
                    {'Site': s, 'Co': c, 'price': dict_price_outofstate[c], 'annual': annual, 'losses': 0,
                     'type': dict_type_evrys[c]}, ignore_index=True)
                output_urbs = output_urbs.append(
                    {'Site': s, 'Commodity': c, 'Type': dict_type_urbs[c], 'price': dict_price_outofstate[c],
                     'max': dict_co_max[c], 'maxperstep': dict_maxperstep[c]}, ignore_index=True)

    output_evrys.to_csv(paths["evrys"] + 'Commodities_evrys.csv', index=False, sep=';', decimal=',')
    output_urbs.to_csv(paths["urbs"] + 'Commodities_urbs.csv', index=False, sep=';', decimal=',')

    timecheck('End')


def distribute_renewable_capacities(paths, param):

    # Shapefile with countries
    countries = gpd.read_file(paths["Countries"])

    # Countries to be considered
    sites = pd.DataFrame(countries[['NAME_SHORT']].rename(columns={'NAME_SHORT': 'Site'}))
    sites = sites.sort_values(by=['Site'], axis=0)['Site'].unique()

    # Read input file, extracted from IRENA
    data_raw = pd.read_excel(paths["IRENA"], skiprows=[0, 1, 2, 3, 4, 5, 6])

    # Add missing country names
    for i in np.arange(1, len(data_raw.index)):
        if data_raw.isnull().loc[i, 'Country/area'] is True:
            data_raw.loc[i, 'Country/area'] = data_raw.loc[i - 1, 'Country/area']

    # Select technologies needed in urbs and rename them
    data_raw = data_raw.loc[data_raw["Technology"].isin(param["dist_ren"]["renewables"])].reset_index(drop=True)
    data_raw["Technology"] = data_raw["Technology"].replace(param["dist_ren"]["renewables"])
    data_raw = data_raw.rename(columns={'Country/area': 'Site', 'Technology': 'Process', 2015: 'inst-cap'})

    # Create new dataframe with needed information, rename sites and extract chosen sites
    data = data_raw[["Site", "Process", "inst-cap"]]
    data = data.replace({"site": param["dist_ren"]["country_names"]}).fillna(value=0)
    data = data.loc[data["site"].isin(sites)].reset_index(drop=True)

    # Group by and sum
    data = data.groupby(["Site", "Process"]).sum().reset_index()

    # Estimate number of units
    units = param["dist_ren"]["units"]
    for p in data["process"].unique():
        data.loc[data["Process"] == p, "Unit"] = data.loc[data["Process"] == p, "inst-cap"] // units[p] \
                                                 + (data.loc[data["Process"] == p, "inst_cap"] % units[p] > 0)
    for p in data["Process"].unique():
        x = y = c = []
        for counter in range(0, len(countries) - 1):
            print(counter)
            if float(data.loc[(data["site"] == countries.loc[counter, "NAME_SHORT"]) & (
                    data["Process"] == p), 'inst-cap']) == 0:
                continue
            if (countries.loc[counter, "Population"]) & (p == 'WindOff'):
                continue
            if (countries.loc[counter, "Population"] == 0) & (p != 'WindOff'):
                continue
            name, x_off, y_off, potential = rasclip(paths["rasters"][p], paths["Countries"], counter)
            raster_shape = potential.shape
            potential = potential.flatten()

            # Calculate the part of the probability that is based on the potential
            potential_nan = np.isnan(potential) | (potential == 0)
            potential = (potential - np.nanmin(potential)) / (np.nanmax(potential) - np.nanmin(potential))
            potential[potential_nan] = 0

            # Calculate the random part of the probability
            potential_random = np.random.random_sample(potential.shape)
            potential_random[potential_nan] = 0

            # Combine the two parts
            potential_new = (1 - param["dist_ren"]["randomness"]) * potential \
                            + param["dist_ren"]["randomness"] * potential_random

            # Sort elements based on their probability and keep the indices
            ind_sort = np.argsort(potential_new, axis=None)  # Ascending
            ind_needed = ind_sort[-int(data.loc[(data["Site"] == name) & (data["Process"] == p), "units"].values):]

            # Free memory
            del ind_sort, potential, potential_nan, potential_random

            # Get the coordinates of the power plants and their respective capacities
            power_plants = [units[p]] * len(ind_needed)
            if data.loc[(data["Site"] == name) & (data["Process"] == p), "Inst-cap"].values % units[p] > 0:
                power_plants[-1] = data.loc[(data["Site"] == name) & (data["Process"] == p), "Inst-cap"].values % units[
                    p]
            y_pp, x_pp = np.unravel_index(ind_needed, raster_shape)
            x = x + ((x_pp + x_off + 0.5) * param["res_desired"][1] + param["Crd_all"][3]).tolist()
            y = y + (param["Crd_all"][0] - (y_pp + y_off + 0.5) * param["res_desired"][0]).tolist()
            c = c + potential_new[ind_needed].tolist()  # Power_plants

            del potential_new

        # Create map
        map_power_plants(p, x, y, c, paths)


def clean_processes_and_storage_data(paths, param):

    timecheck("Start")
    assumptions = pd.read_excel(paths["assumptions"], sheetname='Process')

    depreciation = dict(zip(assumptions['Process'], assumptions['depreciation'].astype(float)))
    year_mu = dict(zip(assumptions['Process'], assumptions['year_mu'].astype(float)))
    year_stdev = dict(zip(assumptions['Process'], assumptions['year_stdev'].astype(float)))

    # Get data from fresna database
    Process = pd.read_csv(paths["database"], header=0, skipinitialspace=True,
                          usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Process.rename(columns={'Capacity': 'inst-cap', 'YearCommissioned': 'year', 'lat': 'Latitude', 'lon': 'Longitude'},
                   inplace=True)
    print('Number of power plants: ', len(Process))

    # ### Process name

    # Use the name of the processes in OPSD as a standard name
    Process['Pro'] = Process['OPSD']

    # If the name is missing in OPSD, use the name in other databases
    Process.loc[Process['Pro'].isnull(), 'Pro'] = Process.loc[Process['Pro'].isnull(), 'CARMA']
    Process.loc[Process['Pro'].isnull(), 'Pro'] = Process.loc[Process['Pro'].isnull(), 'ENTSOE']
    Process.loc[Process['Pro'].isnull(), 'Pro'] = Process.loc[Process['Pro'].isnull(), 'GEO']
    Process.loc[Process['Pro'].isnull(), 'Pro'] = Process.loc[Process['Pro'].isnull(), 'WRI']

    # Add suffix to duplicate names
    Process['Pro'] = Process['Pro'] + Process.groupby(['Pro']).cumcount().astype(str).replace('0', '')

    # Remove spaces from the name and replace them with underscores
    Process['Pro'] = [Process.loc[i, 'Pro'].replace(' ', '_') for i in Process.index]

    # Remove useless columns
    Process.drop(['CARMA', 'ENTSOE', 'GEO', 'OPSD', 'WRI'], axis=1, inplace=True)

    print('Number of power plants with distinct names: ', len(Process['Pro'].unique()))

    # ### Type

    Process['CoIn'] = np.nan
    for i in Process.index:
        # Get the pumped storage facilities
        if Process.loc[i, 'Technology'] in ['Pumped Storage', 'Pumped Storage With Natural Inflow',
                                            'Pumped Storage, Pumped Storage With Natural Inflow, Reservoir',
                                            'Pumped Storage, Reservoir', 'Pumped Storage, Run-Of-River']:
            Process.loc[i, 'CoIn'] = 'PumSt'

        # Get the tidal power plants
        if Process.loc[i, 'Technology'] == 'Tidal':
            Process.loc[i, 'CoIn'] = 'Tidal'

        # Assign an input commodity
        if pd.isnull(Process.loc[i, 'CoIn']):
            Process.loc[i, 'CoIn'] = param["clean_pro_sto"]["proc_dict"][Process.loc[i, 'Fueltype']]

        # Distinguish between small and large hydro
        if (Process.loc[i, 'CoIn'] == 'Hydro_Small') and (Process.loc[i, 'inst-cap'] > 30):
            Process.loc[i, 'CoIn'] = 'Hydro_Large'

    # Remove useless columns
    Process.drop(['Fueltype', 'Technology', 'Set'], axis=1, inplace=True)

    # Remove renewable power plants (except Tidal and Geothermal)
    Process.set_index('CoIn', inplace=True)
    Process.drop(list(set(Process.index.unique()) & set(param["clean_pro_sto"]["renewable_powerplants"])),
                 axis=0, inplace=True)

    Process.reset_index(inplace=True)

    print('Possible processes: ', Process['CoIn'].unique())

    # ### Include renewable power plants

    for pp in param["clean_pro_sto"]["renewable_powerplants"]:
        # Shapefile with power plants
        pp_shapefile = gpd.read_file(paths["PPs_"] + pp + '.shp')
        pp_df = pd.DataFrame(pp_shapefile.rename(columns={'CapacityMW': 'inst-cap'}))
        pp_df['Longitude'] = [pp_df.loc[i, 'geometry'].x for i in pp_df.index]
        pp_df['Latitude'] = [pp_df.loc[i, 'geometry'].y for i in pp_df.index]
        pp_df['CoIn'] = pp
        pp_df['Pro'] = [pp + '_' + str(i) for i in pp_df.index]
        pp_df.drop(['geometry'], axis=1, inplace=True)
        Process = Process.append(pp_df, ignore_index=True)

    print('Possible processes: ', Process['CoIn'].unique())

    # ### Year

    # Assign a dummy year for missing entries (will be changed later)
    for c in Process['CoIn'].unique():
        if c in param["clean_pro_sto"]["storage"]:
            Process.loc[(Process['CoIn'] == c) & (Process['year'].isnull()), 'year_mu'] = 1980
            Process.loc[(Process['CoIn'] == c) & (Process['year'].isnull()), 'year_stdev'] = 5
        else:
            Process.loc[(Process['CoIn'] == c) & (Process['year'].isnull()), 'year_mu'] = year_mu[c]
            Process.loc[(Process['CoIn'] == c) & (Process['year'].isnull()), 'year_stdev'] = year_stdev[c]

    Process.loc[Process['year'].isnull(), 'year'] = np.floor(
        np.random.normal(Process.loc[Process['year'].isnull(), 'year_mu'],
                         Process.loc[Process['year'].isnull(), 'year_stdev']))

    # Process['year'].fillna(year_ref, inplace=True)

    # Drop recently built plants (after the reference year)
    Process = Process[(Process['year'] <= param["year"])]

    # ### Coordinates

    P_missing = Process[Process['Longitude'].isnull()].copy()
    P_located = Process[~Process['Longitude'].isnull()].copy()

    # Assign dummy coordinates within the same country (will be changed later)
    for country in P_missing['Country'].unique():
        P_missing.loc[P_missing['Country'] == country, 'Latitude'] = P_located[P_located['Country'] == country].iloc[
            0, 2]
        P_missing.loc[P_missing['Country'] == country, 'Longitude'] = P_located[P_located['Country'] == country].iloc[
            0, 3]

    Process = P_located.append(P_missing)
    Process = Process[Process['Longitude'] > -11]

    # ### Consider lifetime of power plants

    if param["year"] > param["clean_pro_sto"]["year_ref"]:
        for c in Process['CoIn'].unique():
            if c in param["clean_pro_sto"]["storage"]:
                Process.loc[Process['CoIn'] == c, 'lifetime'] = 60
            else:
                Process.loc[Process['CoIn'] == c, 'lifetime'] = depreciation[c]
        print(len(Process.loc[(Process['lifetime'] + Process['year']) < param["year"]]), ' processes will be deleted')
        Process.drop(Process.loc[(Process['lifetime'] + Process['year']) < param["year"]].index, inplace=True)

    # ### Site

    # Create point geometries (shapely)
    Process['geometry'] = list(zip(Process.Longitude, Process.Latitude))
    Process['geometry'] = Process['geometry'].apply(Point)

    P_located = gpd.GeoDataFrame(Process, geometry='geometry', crs='')
    P_located.crs = {'init': 'epsg:4326'}

    # Define the output commodity
    P_located['CoOut'] = 'Elec'
    P_located.drop(['Latitude', 'Longitude', 'year_mu', 'year_stdev'], axis=1, inplace=True)
    P_located = P_located[['Pro', 'CoIn', 'CoOut', 'inst-cap', 'Country', 'year', 'geometry']]

    # Save the GeoDataFrame
    P_located.to_file(driver='ESRI Shapefile', filename=paths["pro_sto"])

    timecheck("End")


def generate_processes(paths, param):

    return 1


def generate_urbs_model(paths, param):
    """
    Read model's .csv files, and create relevant dataframes.
    Writes dataframes to urbs and evrys input excel files.
    """
    timecheck('Start')

    # ## To be done later when all modules are implemented ## #

    # # Prepare blank excel file
    # output = pd.ExcelWriter(paths["urbs_model"], mode='w')
    #
    # # Read and Format zones
    # site_urbs = pd.read_csv(paths["urbs"] + 'Sites_urbs.csv', sep=';', decimal=',')
    #
    # # Write Sites
    #
    # site_urbs.to_excel(output, "Site")
    # # Read load
    # load_urbs = pd.read_csv(paths["urbs"] + 'Demand_urbs' + '%04d' % (param["year"]) + '.csv', sep=';', decimal=',')
    # load_urbs.to_excel(output, "Demand")
    #
    # # TO IMPLEMENT FOR EVRYS TOO
    timecheck('End')


if __name__ == '__main__':
    paths, param = initialization()
    # generate_sites_from_shapefile(paths) - done
    # generate_intermittent_supply_timeseries(paths, param) - separate module
    # generate_load_timeseries(paths, param) - done
    # generate_commodities(paths, param) # corresponds to 04 - done
    # distribute_renewable_capacities(paths, param)  # corresponds to 05a - done (still need testing)
    # clean_processes_and_storage_data(paths, param)  # corresponds to 05b I think - done (still needs testing)
    generate_processes(paths, param)  # corresponds to 05c
    # generate_storage(paths, param) # corresponds to 05d
    # clean_grid_data(paths, param) # corresponds to 06a
    # generate_aggregated_grid(paths, param) # corresponds to 06b
    generate_urbs_model(paths, param)
    # generate_evrys_model(paths, param)
