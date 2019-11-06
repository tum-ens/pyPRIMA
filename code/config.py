import os
from pathlib import Path
import numpy as np


def configuration():
    """
    This function is the main configuration function that calls all the other modules in the code.

    :return: The dictionary param containing all the user preferences, and the dictionary path containing all the paths to inputs and outputs.
    :rtype: tuple of dict
    """
    paths, param = general_settings()
    paths, param = scope_paths_and_parameters(paths, param)

    param = resolution_parameters(param)
    param = load_parameters(param)
    param = grid_parameters(param)
    param = processes_parameters(param)

    paths = global_maps_input_paths(paths)
    paths = assumption_paths(paths)
    paths = load_input_paths(paths)
    paths = grid_input_paths(paths)
    paths = processes_input_paths(paths, param)
    paths = output_folders(paths, param)
    paths = output_paths(paths, param)
    paths = local_maps_paths(paths, param)

    return paths, param


def general_settings():
    """
    This function creates and initializes the dictionaries param and paths. It also creates global variables for the root folder ``root``
    and the system-dependent file separator ``fs``.

    :return: The empty dictionary paths, and the dictionary param including some general information.
    :rtype: tuple of dict
    """
    # These variables will be initialized here, then read in other modules without modifying them.
    global fs
    global root

    param = {}
    param["author"] = "Kais Siala"  # the name of the person running the script
    param["comment"] = "Testing"

    paths = {}
    fs = os.path.sep
    current_folder = os.path.dirname(os.path.abspath(__file__))
    # For personal Computer:
    # root = str(Path(current_folder).parent.parent.parent) + fs + "Database_KS" + fs
    # For Server Computer:
    root = str(Path(current_folder).parent.parent.parent) + "Database_KS" + fs

    return paths, param


def scope_paths_and_parameters(paths, param):
    """
    This function defines the path of the geographic scope of the output *spatial_scope* and of the subregions of interest *subregions*.
    It also associates two name tags for them, respectively *region_name* and *subregions_name*, which define the names of output folders.
    Both paths should point to shapefiles of polygons or multipolygons.
    
    For *spatial_scope*, only the bounding box around all the features matters.
    Example: In case of Europe, whether a shapefile of Europe as one multipolygon, or as a set of multiple features (countries, states, etc.) is used, does not make a difference.
    Potential maps (theoretical and technical) will be later generated for the whole scope of the bounding box.
    
    For *subregions*, the shapes of the individual features matter, but not their scope.
    For each individual feature that lies within the scope, you can later generate a summary report and time series.
    The shapefile of *subregions* does not have to have the same bounding box as *spatial_scope*.
    In case it is larger, features that lie completely outside the scope will be ignored, whereas those that lie partly inside it will be cropped using the bounding box
    of *spatial_scope*. In case it is smaller, all features are used with no modification.
    
    *year* defines the year of the weather data, and *technology* the list of technologies that you are interested in.
    Currently, four technologies are defined: onshore wind ``'WindOn'``, offshore wind ``'WindOff'``, photovoltaics ``'PV'``, concentrated solar power ``'CSP'``.
    
    *frameworks* is a list of model frameworks, for which models will be generated. Currently only ``'urbs'`` and ``'evrys'`` are supported.

    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the user preferences.
    :type param: dict
    :return: The updated dictionaries paths and param.
    :rtype: tuple of dict
    """

    # Paths to the shapefiles
    PathTemp = root + "02 Shapefiles for regions" + fs + "User-defined" + fs
    paths["spatial_scope"] = PathTemp + "Europe_NUTS0_wo_Balkans_with_EEZ.shp"
    paths["subregions"] = PathTemp + "Bavaria_WGC.shp"

    # Name tags for the scope and the subregions
    param["region_name"] = "Europe"  # Name tag of the spatial scope
    param["subregions_name"] = "Geothermal_WGC"  # Name tag of the subregions

    # Year
    param["year"] = 2015

    # Technologies
    param["technology"] = ["WindOn", "PV", "WindOff", "CSP"]  # ['PV', 'CSP', 'WindOn', 'WindOff']

    # Frameworks
    # param["frameworks"] = ['urbs', 'evrys'] # ['urbs', 'evrys']

    return paths, param


def resolution_parameters(param):
    """
    This function defines the resolution of weather data (low resolution), and the desired resolution of output rasters (high resolution).
    Both are numpy array with two numbers. The first number is the resolution in the vertical dimension (in degrees of latitude),
    the second is for the horizontal dimension (in degrees of longitude).

    :param param: Dictionary including the user preferences.
    :type param: dict
    :return: The updated dictionary param.
    :rtype: dict
    """

    param["res_weather"] = np.array([1 / 2, 5 / 8])
    param["res_desired"] = np.array([1 / 240, 1 / 240])
    return param


def load_parameters(param):
    """
    """

    param["load"] = {
        "sectors": ["RES", "COM", "IND", "AGR"],
        "sectors_eurostat": {
            "Residential": "RES",
            "Agriculture/Forestry": "AGR",
            "Services": "COM",
            "Non-specified (Other)": "COM",
            "Final Energy Consumption - Industry": "IND",
        },
        "default_sec_shares": "DEU",
    }
    return param


def grid_parameters(param):
    """
    """

    param["grid"] = {
        "quality": {"voltage": 1, "wires": 0, "cables": 0.5, "frequency": 0},
        "default": {"voltage": 220000, "wires": 1, "cables": 3, "frequency": 50},  # in Volt
        # from literature (see CITAVI files: "References for Reactances and SILVersion6")
        "specific_reactance": {
            110: 0.39,  # in Ohm/km
            220: 0.3,
            345: 0.3058,
            # 380: 0.25, inconsistent
            500: 0.2708,
            # 765: 0.2741, inconsistent
            1000: 0.2433,  # upper bound
        },
        # probably same sources? Or a St. Clair's curve found somewhere...
        "loadability": {
            80: 3,  # dimensionless
            100: 2.75,
            150: 2.5,
            200: 2,
            250: 1.75,
            300: 1.5,
            350: 1.37,
            400: 1.25,
            450: 1.12,
            500: 1,
            550: 0.9,
            600: 0.85,
            650: 0.8,
            700: 0.77,
            1000: 0.6,  # upper bound
        },
        # dummy values??
        "SIL": {
            10: 0.3, # in MW
            30: 2.7,
            69: 15,
            110: 32,
            138: 59.4,
            220: 175,
            345: 504,
            380: 602,
            500: 1200,
            765: 2736,
            1000: 6312, # upper bound
        }, 
        "efficiency": {
            "AC_OHL": 0.92, # 8% losses / 1000 km
            "AC_CAB": 0.90,
            "DC_OHL": 0.95,
            "DC_CAB": 0.95,
        },  
        "wacc": 0.07,
        "depreciation": 50,
    }

    return param


def processes_parameters(param):
    """
    """
# dist_ren = {"p_landuse": {0: 0, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2, 6: 0.2, 7: 0.2, 8: 0.1, 9: 0.1, 10: 0.5, 11: 0,
# 12: 1, 13: 0, 14: 1, 15: 0, 16: 0},
# "cap_lo": 0,
# "cap_up": np.inf,
# "drop_params": ['Site', 'inst-cap', 'cap_lo', 'cap-up', 'year']
# }

    param["dist_ren"] = {
        "units": {
            "Solar": 2,
            "WindOn": 5,
            "WindOff": 10,
            "Bioenergy": 5,
            "Hydro": 50,
        },
        "randomness": 0.99,
        "default_pa_type": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "default_pa_availability": np.array([1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 1.00, 1.00, 1.00, 1.00]),
    }
    return param


def global_maps_input_paths(paths):
    """
    This function defines the paths where the global maps are saved:
    
      * *LU_global* for the land use raster
      * *Topo_tiles* for the topography tiles (rasters)
      * *Pop_tiles* for the population tiles (rasters)
      * *Bathym_global* for the bathymetry raster
      * *Protected* for the shapefile of protected areas
      * *GWA* for the country data retrieved from the Global Wind Atlas (missing the country code, which will be filled in a for-loop in :mod:data_functions.calc_gwa_correction)
      * *Countries* for the shapefiles of countries
      * *EEZ_global* for the shapefile of exclusive economic zones of countries
    
    :param paths: Dictionary including the paths.
    :type paths: dict
    :return: The updated dictionary paths.
    :rtype: dict
    """
    global root
    global fs

    # Global maps
    PathTemp = root + "01 Raw inputs" + fs + "Maps" + fs
    paths["LU_global"] = PathTemp + "Landuse" + fs + "LCType.tif"
    # paths["Topo_tiles"] = PathTemp + "Topography" + fs
    paths["Pop_tiles"] = PathTemp + "Population" + fs
    # paths["Bathym_global"] = PathTemp + "Bathymetry" + fs + "ETOPO1_Ice_c_geotiff.tif"
    paths["Protected"] = PathTemp + "Protected Areas" + fs + "WDPA_Nov2018-shapefile-polygons.shp"
    # paths["GWA"] = PathTemp + "Global Wind Atlas" + fs + fs + "windSpeed.csv"
    paths["Countries"] = PathTemp + "Countries" + fs + "gadm36_0.shp"
    paths["EEZ_global"] = PathTemp + "EEZ" + fs + "eez_v10.shp"

    return paths


def assumption_paths(paths):
    """
    """
    global root
    global fs

    paths["assumptions_landuse"] = root + "00 Assumptions" + fs + "assumptions_landuse.csv"
    paths["assumptions_processes"] = root + "00 Assumptions" + fs + "assumptions_processes.csv"
    paths["dict_season"] = root + "00 Assumptions" + fs + "dict_season_north.csv"
    paths["dict_daytype"] = root + "00 Assumptions" + fs + "dict_day_type.csv"
    paths["dict_countries"] = root + "00 Assumptions" + fs + "dict_countries.csv"
    paths["dict_lines_costs"] = root + "00 Assumptions" + fs + "dict_lines_costs.csv"
    paths["dict_technologies"] = root + "00 Assumptions" + fs + "dict_technologies.csv"

    return paths


def load_input_paths(paths):
    """
    """
    global root
    global fs

    PathTemp = root + "01 Raw inputs" + fs + "Load" + fs
    paths["sector_shares"] = PathTemp + "Eurostat" + fs + "nrg_105a_1_Data.csv"
    paths["load_ts"] = PathTemp + "ENTSOE" + fs + "Monthly-hourly-load-values_2006-2015.xlsx"  # CA: "CA_Load Profiles_11 Regions_correct names.csv"
    paths["profiles"] = {
        "RES": PathTemp + "Load profiles" + fs + "Lastprofil_Haushalt_H0.xlsx",  # CA: "Residential Load Profile_2017_SCE.csv"
        "IND": PathTemp + "Load profiles" + fs + "Lastprofil_Industrie_Tag.xlsx",  # CA: "Medium Commercial and Industrial_Load Profile_2017 SCE.xlsx"
        "COM": PathTemp
        + "Load profiles"
        + fs
        + "VDEW-Lastprofile-Gewerbe-Landwirtschaft_G0.csv",  # CA: "Small Commercial_Load Profile_2017 SCE.xlsx"
        "AGR": PathTemp + "Load profiles" + fs + "VDEW-Lastprofile-Landwirtschaft_L0.csv",  # CA: "VDEW-Lastprofile-Landwirtschaft_L0.csv"
        "STR": PathTemp + "Load profiles" + fs + "Lastprofil_Strassenbeleuchtung_S0.xlsx",  # CA: "Lastprofil_Strassenbeleuchtung_S0.xlsx"
    }

    return paths


def grid_input_paths(paths):
    """
    """
    global root
    global fs

    PathTemp = root + "01 Raw inputs" + fs + "Grid" + fs
    paths["transmission_lines"] = PathTemp + "gridkit_europe" + fs + "gridkit_europe-highvoltage-links.csv"

    return paths
    
    
def processes_input_paths(paths, param):
    """
    """
    global root
    global fs
    
    year = str(param["year"])
    
    PathTemp = root + "01 Raw inputs" + fs + "Renewable energy" + fs
    paths["IRENA"] = PathTemp + "IRENA" + fs + "IRENA_RE_electricity_statistics_allcountries_alltech_" + year + ".csv"
    
    PathTemp = root + "03 Intermediate files" + fs + "Files Europe" + fs + "Renewable energy" + fs + "Potential" + fs
    paths["dist_ren"] = {
        "rasters": {
            "Solar": PathTemp + "Europe_PV_0_FLH_mask_2015.tif",
            "WindOn": PathTemp + "Europe_WindOn_100_FLH_mask_2015.tif",
            "WindOff": PathTemp + "Europe_WindOff_80_FLH_mask_2015.tif",
            "Bioenergy": PathTemp + "Europe_Bioenergy_potential_distribution.tif",
            "Hydro": PathTemp + "Europe_Hydro_potential_distribution.tif",
        },
    }
    
    PathTemp = root + '01 Raw inputs' + fs + 'Power plants and storage' + fs
    paths["FRESNA"] = PathTemp + 'EU_Powerplants' + fs + 'FRESNA2' + fs + 'Matched_CARMA_ENTSOE_ESE_GEO_GPD_OPSD_reduced.csv'
    
    return paths
    




def output_folders(paths, param):
    """
    This function defines the paths to multiple output folders:
    
      * *region* is the main output folder.
      * *weather_data* is the output folder for the weather data of the spatial scope.
      * *local_maps* is the output folder for the local maps of the spatial scope.
      * *potential* is the output folder for the ressource and technical potential maps.
      * *regional_analysis* is the output folder for the time series and the report of the subregions.
      * *regression_out* is the output folder for the regression results.
      
    All the folders are created at the beginning of the calculation, if they do not already exist,
    
    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the user preferences.
    :type param: dict
    :return: The updated dictionary paths.
    :rtype: dict
    """
    global root
    global fs

    region = param["region_name"]
    subregions = param["subregions_name"]

    # Main output folder
    paths["region"] = root + "03 Intermediate files" + fs + "Files " + region + fs

    # Output folder for local maps of the scope
    paths["local_maps"] = paths["region"] + "Maps" + fs
    if not os.path.isdir(paths["local_maps"]):
        os.makedirs(paths["local_maps"])

    # Output folder for sites
    paths["sites"] = paths["region"] + "Sites" + fs + subregions + fs
    if not os.path.isdir(paths["sites"]):
        os.makedirs(paths["sites"])

    # Output folder for load data
    paths["load_sub"] = paths["region"] + "Load" + fs + subregions + fs
    if not os.path.isdir(paths["load_sub"]):
        os.makedirs(paths["load_sub"])
    paths["load"] = paths["region"] + "Load" + fs

    # Output folder for the grid data
    paths["grid_sub"] = paths["region"] + "Grid" + fs + subregions + fs
    if not os.path.isdir(paths["grid_sub"]):
        os.makedirs(paths["grid_sub"])
    paths["grid"] = paths["region"] + "Grid" + fs

    # Output folder for the regional analysis of renewable energy
    paths["regional_analysis"] = paths["region"] + "Renewable energy" + fs + "Regional analysis" + fs + subregions + fs
    if not os.path.isdir(paths["regional_analysis"]):
        os.makedirs(paths["regional_analysis"])
        
    # Output folder for processes and storage
    paths["proc_sub"] = paths["region"] + "Power plants and storage" + fs + subregions + fs
    if not os.path.isdir(paths["proc_sub"]):
        os.makedirs(paths["proc_sub"])
    paths["proc"] = paths["region"] + "Power plants and storage" + fs

    # Output folder for urbs models
    paths["urbs"] = root + "04 Model files" + fs + "Files " + region + fs + subregions + fs + "urbs" + fs
    if not os.path.isdir(paths["urbs"]):
        os.makedirs(paths["urbs"])

    # Output folder for evrys models
    paths["evrys"] = root + "04 Model files" + fs + "Files " + region + fs + subregions + fs + "evrys" + fs
    if not os.path.isdir(paths["evrys"]):
        os.makedirs(paths["evrys"])

    return paths


def output_paths(paths, param):
    """
    """

    region = param["region_name"]
    subregions = param["subregions_name"]
    year = str(param["year"])

    # Sites
    paths["sites_sub"] = paths["sites"] + "Sites.csv"

    # Load
    paths["stats_countries"] = paths["load"] + "Statistics_countries.csv"
    paths["sector_shares_clean"] = paths["load"] + "Sector_shares_" + year + ".csv"
    paths["load_ts_clean"] = paths["load"] + "TS_countries_clean_" + year + ".csv"
    paths["df_sector"] = paths["load"] + "TS_countries_sectors_" + year + ".csv"
    paths["load_sector"] = paths["load"] + "Yearly_demand_countries_sectors_" + year + ".csv"
    paths["load_landuse"] = paths["load"] + "TS_countries_land_use_" + year + ".csv"
    paths["intersection_subregions_countries"] = paths["load_sub"] + "Intersection_with_" + param["subregions_name"] + ".shp"
    paths["stats_country_parts"] = paths["load_sub"] + "Statistics_country_parts.csv"
    paths["load_regions"] = paths["load_sub"] + "TS_subregions_" + param["subregions_name"] + "_" + year + ".csv"

    # Grid
    paths["grid_expanded"] = paths["grid"] + "grid_expanded.csv"
    paths["grid_filtered"] = paths["grid"] + "grid_filtered.csv"
    paths["grid_corrected"] = paths["grid"] + "grid_corrected.csv"
    paths["grid_filled"] = paths["grid"] + "grid_filled.csv"
    paths["grid_cleaned"] = paths["grid"] + "grid_cleaned.csv"
    paths["grid_shp"] = paths["grid"] + "grid_cleaned.shp"
    paths["grid_completed"] = paths["grid_sub"] + "transmission.csv"
    
    # Renewable processes
    paths["IRENA_summary"] = paths["region"] + "Renewable energy" + fs + "IRENA_summary_" + year + ".csv"
    paths["locations_ren"] = {
        "Solar": paths["proc"] + "Solar.shp",
        "WindOn": paths["proc"] + "WindOn.shp",
        "WindOff": paths["proc"] + "WindOff.shp",
        "Bioenergy": paths["proc"] + "Bioenergy.shp",
        "Hydro": paths["proc"] + "Hydro.shp",
    }
    
    # Other processes and storage
    paths["process_raw"] = paths["proc"] + "processes_raw.csv"

    # Framework models
    paths["urbs_model"] = paths["urbs"] + region + "_" + subregions + "_" + year + ".xlsx"
    paths["evrys_model"] = paths["evrys"] + region + "_" + subregions + "_" + year + ".xlsx"

    return paths


def local_maps_paths(paths, param):
    """
    This function defines the paths where the local maps will be saved:
    
      * *LAND* for the raster of land areas within the scope
      * *EEZ* for the raster of sea areas within the scope
      * *SUB* for the raster of areas covered by subregions (both land and sea) within the scope
      * *LU* for the land use raster within the scope
      * *BATH* for the bathymetry raster within the scope
      * *TOPO* for the topography raster within the scope
      * *SLOPE* for the slope raster within the scope
      * *PA* for the raster of protected areas within the scope
      * *POP* for the population raster within the scope
      * *BUFFER* for the raster of population buffer areas within the scope
      * *CORR_GWA* for correction factors based on the Global Wind Atlas (mat file)
      * *CORR_ON* for the onshore wind correction factors (raster)
      * *CORR_OFF* for the offshore wind correction factors (raster)
      * *AREA* for the area per pixel in m² (mat file)
    
    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the user preferences.
    :type param: dict
    :return: The updated dictionary paths.
    :rtype: dict
    """

    # Local maps
    PathTemp = paths["local_maps"] + param["region_name"]
    paths["LAND"] = PathTemp + "_Land.tif"  # Land pixels
    paths["EEZ"] = PathTemp + "_EEZ.tif"  # Sea pixels
    # paths["SUB"] = PathTemp + "_Subregions.tif"  # Subregions pixels
    paths["LU"] = PathTemp + "_Landuse.tif"  # Land use types
    # paths["TOPO"] = PathTemp + "_Topography.tif"  # Topography
    paths["PA"] = PathTemp + "_Protected_areas.tif"  # Protected areas
    # paths["SLOPE"] = PathTemp + "_Slope.tif"  # Slope
    # paths["BATH"] = PathTemp + "_Bathymetry.tif"  # Bathymetry
    paths["POP"] = PathTemp + "_Population.tif"  # Population
    # paths["BUFFER"] = PathTemp + "_Population_Buffered.tif"  # Buffered population
    # paths["AREA"] = PathTemp + "_Area.mat"  # Area per pixel in m²

    # # Correction factors for wind speeds
    # turbine_height_on = str(param["WindOn"]["technical"]["hub_height"])
    # turbine_height_off = str(param["WindOff"]["technical"]["hub_height"])
    # paths["CORR_ON"] = PathTemp + "_WindOn_Correction_" + turbine_height_on + '.tif'
    # paths["CORR_OFF"] = PathTemp + "_WindOff_Correction_" + turbine_height_off + '.tif'

    return paths


# ##############################
# #### Move to assumptions #####
# ##############################

# # Processes and Storages in California

# Cal_urbs = {'cap_lo': {'Biomass': 0, 'Coal': 0, 'Gas': 0, 'Geothermal': 0, 'Hydro_Large': 0, 'Hydro_Small': 0,
# 'Nuclear': 0, 'Oil': 0, 'Slack': 999999, 'Solar': 0, 'Waste': 0, 'WindOn': 0},
# 'cap_up': {'Biomass': 100, 'Coal': 0, 'Gas': 10000, 'Geothermal': 0, 'Hydro_Large': 0, 'Hydro_Small': 0,
# 'Nuclear': 0, 'Oil': 0, 'Slack': 999999, 'Solar': np.inf, 'Waste': 0, 'WindOn': np.inf},
# 'max_grad': {'Biomass': 1.2, 'Coal': 0.6, 'Gas': 4.8, 'Geothermal': 0.6, 'Hydro_Large': np.inf,
# 'Hydro_Small': np.inf,
# 'Nuclear': 0.3, 'Oil': 3, 'Slack': np.inf, 'Solar': np.inf, 'Waste': 1.2, 'WindOn': np.inf},
# 'min_fraction': {'Biomass': 0, 'Coal': 0.5, 'Gas': 0.25, 'Geothermal': 0, 'Hydro_Large': 0,
# 'Hydro_Small': 0,
# 'Nuclear': 0.7, 'Oil': 0.4, 'Slack': 0, 'Solar': 0, 'Waste': 0, 'WindOn': 0},
# 'inv_cost': {'Biomass': 875000, 'Coal': 600000, 'Gas': 450000, 'Geothermal': 1000000,
# 'Hydro_Large': 1600000, 'Hydro_Small': 320000,
# 'Nuclear': 1600000, 'Oil': 600000, 'Slack': 0, 'Solar': 600000, 'Waste': 800000,
# 'WindOn': 900000},
# 'fix_cost': {'Biomass': 28000, 'Coal': 18000, 'Gas': 6000, 'Geothermal': 10000, 'Hydro_Large': 20000,
# 'Hydro_Small': 1000,
# 'Nuclear': 50000, 'Oil': 9000, 'Slack': 0, 'Solar': 25000, 'Waste': 4000, 'WindOn': 30000},
# 'var_cost': {'Biomass': 5, 'Coal': 14, 'Gas': 25, 'Geothermal': 5, 'Hydro_Large': 3, 'Hydro_Small': 3,
# 'Nuclear': 10, 'Oil': 35, 'Slack': 200, 'Solar': 0, 'Waste': 2, 'WindOn': 0},
# 'startup_cost': {'Biomass': 0, 'Coal': 90, 'Gas': 40, 'Geothermal': 0, 'Hydro_Large': 0, 'Hydro_Small': 0,
# 'Nuclear': 150, 'Oil': 40, 'Slack': 0, 'Solar': 0, 'Waste': 0, 'WindOn': 0},
# 'wacc': 0.07,
# 'depreciation': {'Biomass': 25, 'Coal': 40, 'Gas': 30, 'Geothermal': 100, 'Hydro_Large': 100,
# 'Hydro_Small': 30,
# 'Nuclear': 60, 'Oil': 30, 'Slack': 1, 'Solar': 25, 'Waste': 25, 'WindOn': 25},
# 'area_per_cap': {'Biomass': np.nan, 'Coal': np.nan, 'Gas': np.nan, 'Geothermal': np.nan,
# 'Hydro_Large': np.nan, 'Hydro_Small': np.nan,
# 'Nuclear': np.nan, 'Oil': np.nan, 'Slack': np.nan, 'Solar': 14000, 'Waste': np.nan,
# 'WindOn': np.nan}
# }

# pro_sto_Cal = {'proc_dict': {'AB': 'Biomass', 'BLQ': 'Biomass', 'OBG': 'Biomass', 'WDS': 'Biomass',
# 'BIT': 'Coal', 'RC': 'Coal',
# 'DFO': 'Oil', 'JF': 'Oil', 'PG': 'Oil', 'PC': 'Oil',
# 'LFG': 'Gas', 'NG': 'Gas', 'OG': 'Gas', 'PUR': 'Gas', 'WH': 'Gas',
# 'GEO': 'Geothermal',
# 'WAT': 'Hydro_Small',
# # Later, we will define Hydro_Large as power plants with capacity > 30MW
# 'MWH': 'Battery',
# 'NUC': 'Nuclear',
# 'SUN': 'Solar',
# 'MSW': 'Waste',
# 'WND': 'WindOn'},
# 'storage': ['PumSt', 'Battery'],
# 'status': ['(OP) Operating', '(SB) Standby/Backup: available for service but not normally used'],
# 'states': ['CA'],
# 'Cal_urbs': Cal_urbs
# }

# ###########################
# #### User preferences #####
# ###########################

# # Models input file Sheets
# param["model_sheets"] = {'urbs': ['Global', 'Site', 'Commodity', 'Process', 'Process-Commodity', 'Transmission', 'Storage',
# 'DSM', 'Demand', 'Suplm', 'Buy-Sell-Price'],
# 'evrys': ['Flags', 'Sites', 'Commodity', 'Process', 'Transmission', 'Storage', 'DSM', 'Demand']}

# # urbs Global paramters
# urbs_global = {"Support timeframe": param["year"],
# "Discount rate": 0.03,
# "CO2 limit": 'inf',
# "Cost budget": 6.5e11,
# "CO2 budget": 'inf'
# }
# param["urbs_global"] = urbs_global

# # Distribution_renewables


# dist_ren = {"p_landuse": {0: 0, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2, 6: 0.2, 7: 0.2, 8: 0.1, 9: 0.1, 10: 0.5, 11: 0,
# 12: 1, 13: 0, 14: 1, 15: 0, 16: 0},
# "units": {'Solar': 1, 'WindOn': 2.5, 'WindOff': 2.5, 'Biomass': 5, 'Biogas': 5, 'Liquid biofuels': 5,
# 'Hydro_Small': 0.2, 'Hydro_Large': 200},  # MW
# "randomness": 0.4,
# "cap_lo": 0,
# "cap_up": np.inf,
# "drop_params": ['Site', 'inst-cap', 'cap_lo', 'cap-up', 'year']
# }

# param["dist_ren"] = dist_ren

# # Clean Process and Storage data, Process, and Storage
# pro_sto = {"year_ref": 2015,
# "proc_dict": {'Hard Coal': 'Coal',
# 'Hydro': 'Hydro_Small',
# # Later, we will define Hydro_Large as power plants with capacity > 30MW
# 'Nuclear': 'Nuclear',
# 'Natural Gas': 'Gas',
# 'Lignite': 'Lignite',
# 'Oil': 'Oil',
# 'Bioenergy': 'Biomass',
# 'Other': 'Waste',
# 'Waste': 'Waste',
# 'Wind': 'WindOn',
# 'Geothermal': 'Geothermal',
# 'Solar': 'Solar'},
# "storage": ['PumSt', 'Battery'],
# "renewable_powerplants": ['Hydro_Large', 'Hydro_Small', 'WindOn', 'WindOff',
# 'Solar', 'Biomass', 'Biogas', 'Liquid biofuels'],
# "agg_thres": 20,
# "wacc": 0.07
# }

# param["pro_sto"] = pro_sto


# ###########################
# ##### Define Paths ########
# ###########################

# region = param["region_name"]
# model_regions = param["subregions_name"]
# year = str(param["year"])


# ##################################
# #           Input files          #
# ##################################

# # Assumptions
# paths["assumptions"] = root + "00 Assumptions" + fs + "assumptions_const.xlsx"
# # paths["assumptions"] = root + "00 Assumptions" + fs + "assumptions_const_v04_4NEMO.xlsx"


# # Process and storage data
# PathTemp = root + '01 Raw inputs' + fs + 'Power plants and storage' + fs
# paths["database"] = PathTemp + 'EU_Powerplants' + fs + 'Matched_CARMA_ENTSOE_GEO_OPSD_WRI_reduced.csv'
# paths["database_FRESNA"] = PathTemp + 'EU_Powerplants' + fs + 'FRESNA2' + fs + \
# 'Matched_CARMA_ENTSOE_ESE_GEO_GPD_OPSD_reduced.csv'
# paths["database_Cal"] = PathTemp + 'CA_Powerplants' + fs + 'april_generator2017 (original data).xlsx'


# # ## Renewable Capacities
# # Rasters for wind and solar

# timestamp = '4NEMO'
# pathtemp = root + "03 Intermediate files" + fs + "Files " + region + fs + "Renewable energy" + fs + timestamp + fs

# paths["Renewable energy"] = pathtemp
# rasters = {'WindOn': pathtemp + 'Europe_WindOn_FLH_mask_2015.tif',
# 'WindOff': pathtemp + 'Europe_WindOff_FLH_mask_2015.tif',
# 'Solar': pathtemp + 'Europe_PV_FLH_mask_2015.tif',
# 'Biomass': pathtemp + 'Europe_Biomass.tif',
# 'Liquid biofuels': pathtemp + 'Europe_Biomass.tif',
# 'Biogas': pathtemp + 'Europe_Biomass.tif'}

# paths["rasters"] = rasters

# # IRENA Data
# paths["IRENA"] = root + '01 Raw inputs' + fs + 'Renewable energy' + fs + 'Renewables.xlsx'

# # Intermittent Supply Timeseries
# paths["raw_TS"] = {}
# paths["reg_coef"] = {}
# paths["regression_out"] = pathtemp + "Regression_Outputs" + fs
# for tech in param["technology"]:
# st = ''
# settings = np.sort(np.array(param["settings"][tech]))
# for set in settings:
# st += '_' + str(set)
# paths["reg_coef"][tech] = \
# paths["regression_out"] + region + '_' + tech[:-1] + '_reg_coefficients' + st + '.csv'


# ##################################
# #     General Ouputs Folders     #
# ##################################


# # Intermediate files
# pathtemp = root + '03 Intermediate files' + fs + 'Files ' + region + fs
# paths["pro_sto"] = pathtemp + 'Power plants and storage' + fs + 'Processes_and_Storage_' + year + '.shp'
# paths["process_raw"] = pathtemp + 'Power plants and storage' + fs + 'Processes_raw.csv'
# paths["map_power_plants"] = pathtemp + fs + 'Renewable energy' + fs

# # Model files
# pathtemp = root + '04 Model files' + fs + 'Files ' + region + fs + model_regions + fs
# paths["model_regions"] = pathtemp
# paths["annual_load"] = pathtemp + 'Load_' + region + '_' + year + '.csv'
# paths["suplm_TS"] = pathtemp + 'intermittent_supply_timeseries_' + year + '.csv'
# paths["strat_TS"] = paths["model_regions"] + 'Stratified_intermittent_TS' + str(year) + '_'
# paths["Process_agg"] = pathtemp + 'Processes_agg_2.csv'
# paths["Process_agg_bis"] = pathtemp + 'Processes_agg_3.csv'


# # urbs
# paths["urbs"] = pathtemp + 'urbs' + fs
# if not os.path.isdir(paths["urbs"]):
# os.makedirs(paths["urbs"])
# paths["urbs_suplm"] = paths["urbs"] + 'Suplm_urbs_' + year + '.csv'
# paths["urbs_commodity"] = paths["urbs"] + 'Commodity_urbs_' + year + '.csv'
# paths["urbs_process"] = paths["urbs"] + 'Process_urbs_' + year + '.csv'
# paths["urbs_storage"] = paths["urbs"] + 'Storage_urbs_' + year + '.csv'
# paths["urbs_transmission"] = paths["urbs"] + 'Transmission_urbs_' + year + '.csv'
# paths["urbs_model"] = paths["urbs"] + 'urbs_' + \
# param["region_name"] + '_' + param["subregions_name"] + '_' + year + '.xlsx'

# # evrys
# paths["evrys"] = pathtemp + 'evrys' + fs
# if not os.path.isdir(paths["evrys"]):
# os.makedirs(paths["evrys"])
# paths["evrys_demand"] = paths["evrys"] + 'Demand_evrys_' + year + '.csv'
# paths["evrys_commodity"] = paths["evrys"] + 'Commodity_evrys_' + year + '.csv'
# paths["evrys_process"] = paths["evrys"] + 'Process_evrys_' + year + '.csv'
# paths["evrys_storage"] = paths["evrys"] + 'Storage_evrys_' + year + '.csv'
# paths["evrys_transmission"] = paths["evrys"] + 'Transmission_evrys_' + year + '.csv'
# paths["evrys_model"] = paths["evrys"] + 'evrys_' + \
# param["region_name"] + '_' + param["subregions_name"] + '_' + year + '.xlsx'

# del root, PathTemp, fs
