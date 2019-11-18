import os
from pathlib import Path
import numpy as np


def configuration():
    """
    This function is the main configuration function that calls all the other modules in the code.

    :return (paths, param): The dictionary param containing all the user preferences, and the dictionary path containing all the paths to inputs and outputs.
    :rtype: tuple of dict
    """
    paths, param = general_settings()
    paths, param = scope_paths_and_parameters(paths, param)

    param = resolution_parameters(param)
    param = load_parameters(param)
    param = grid_parameters(param)
    param = processes_parameters(param)
    param = renewable_time_series_parameters(param)

    paths = global_maps_input_paths(paths)
    paths = assumption_paths(paths)
    paths = grid_input_paths(paths)
    paths = load_input_paths(paths, param)
    paths = renewable_time_series_paths(paths, param)
    paths = processes_input_paths(paths, param)
    paths = output_folders(paths, param)
    paths = output_paths(paths, param)
    paths = local_maps_paths(paths, param)

    return paths, param


def general_settings():
    """
    This function creates and initializes the dictionaries param and paths. It also creates global variables for the root folder ``root``
    and the system-dependent file separator ``fs``.

    :return (paths, param): The empty dictionary paths, and the dictionary param including some general information.
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
    root = str(Path(current_folder).parent.parent.parent)
    # For use at TUM ENS
    if root[-1] != fs:
        root = root + fs + "Database_KS" + fs
    else:
        root = root + "Database_KS" + fs

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
    
    *year* defines the year of the weather/input data, and *model_year* refers to the year to be modeled (could be the same as *year*, or in the future).
	
	*technology* is a dictionary of the technologies (*Storage*, *Prcess*) to be used in the model. The names of the technologies should match the names
	which are used in assumptions_flows.csv, assumptions_processes.csv and assumptions_storage.csv.
    
    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the user preferences.
    :type param: dict
	
    :return (paths, param): The updated dictionaries paths and param.
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
    param["year"] = 2015  # Data
    param["model_year"] = 2015  # Model

    # Technologies
    param["technology"] = {
        "Storage": ["Battery", "PumSt"],
        "Process": ["Bioenergy", "Coal", "Gas", "Geothermal", "Hydro", "Lignite", "Nuclear", "OilOther", "Solar", "WindOff", "WindOn"],
    }

    return paths, param


def resolution_parameters(param):
    """
    This function defines the resolution of weather data (low resolution), and the desired resolution of output rasters (high resolution).
    Both are numpy array with two numbers. The first number is the resolution in the vertical dimension (in degrees of latitude),
    the second is for the horizontal dimension (in degrees of longitude).

    :param param: Dictionary including the user preferences.
    :type param: dict
	
    :return param: The updated dictionary param.
    :rtype: dict
    """

    param["res_weather"] = np.array([1 / 2, 5 / 8])
    param["res_desired"] = np.array([1 / 240, 1 / 240])
    return param


def load_parameters(param):
    """
	This function defines the user preferences which are related to the load/demand.
	
	  * *sectors* are the sectors to be considered.
	  * *sectors_eurostat* is a dictionary for identifying the sectors to be considered, which have different names.
	  * *default_sec_shares* is the code name of the country to be used as a default, if the sector shares are missing for another region.
	  
	:param param: Dictionary including the user preferences.
    :type param: dict
	
    :return param: The updated dictionary param.
    :rtype: dict
    """

    param["load"] = {"default_sec_shares": "DEU"}
    return param


def renewable_time_series_parameters(param):
    """
    This function defines parameters relating to the renewable time series to be used in the models. In particular, the user can decide which
	`modes` to use from the files of the time series, provided they exist. See the repository tum-ens/renewable-timeseries for more information.
	
    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """

    param["ren_potential"] = {"WindOn": ["all", "mid"], "WindOff": ["all"], "PV": ["all"], "CSP": ["all"]}  # "Technology":[list of modes]

    return param


def grid_parameters(param):
    """
    """

    param["grid"] = {
        "quality": {"voltage": 1, "wires": 0, "cables": 0.5, "frequency": 0},
        "default": {"voltage": 220000, "wires": 1, "cables": 3, "frequency": 50},
    }

    return param


def processes_parameters(param):
    """
    """

    param["dist_ren"] = {
        "units": {"Solar": 5, "WindOn": 10, "WindOff": 20, "Bioenergy": 10, "Hydro": 50},
        "randomness": 0.99,
        "default_pa_type": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "default_pa_availability": np.array([1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 1.00, 1.00, 1.00, 1.00]),
    }
    param["process"] = {"cohorts": 5}  # 5 means 5-year steps, if no cohorts needed type 1
    return param


def global_maps_input_paths(paths):
    """
    This function defines the paths where the global maps are saved:
    
      * *LU_global* for the land use raster
      * *Pop_tiles* for the population tiles (rasters)
      * *Protected* for the shapefile of protected areas
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
    paths["Pop_global"] = PathTemp + "Population" + fs + "gpw_v4_population_count_rev10_2015_30_sec.tif"
    paths["Protected"] = PathTemp + "Protected Areas" + fs + "WDPA_Nov2018-shapefile-polygons.shp"
    paths["Countries"] = PathTemp + "Countries" + fs + "gadm36_0.shp"
    paths["EEZ_global"] = PathTemp + "EEZ" + fs + "eez_v10.shp"

    return paths


def assumption_paths(paths):
    """
    """
    global fs

    current_folder = os.path.dirname(os.path.abspath(__file__))
    PathTemp = str(Path(current_folder).parent)
    if PathTemp[-1] != fs:
        PathTemp = PathTemp + fs
    PathTemp = PathTemp + "assumptions"

    paths["assumptions_landuse"] = PathTemp + fs + "assumptions_landuse.csv"
    paths["assumptions_flows"] = PathTemp + fs + "assumptions_flows.csv"
    paths["assumptions_processes"] = PathTemp + fs + "assumptions_processes.csv"
    paths["assumptions_storage"] = PathTemp + fs + "assumptions_storage.csv"
    paths["assumptions_commodities"] = PathTemp + fs + "assumptions_commodities.csv"
    paths["assumptions_transmission"] = PathTemp + fs + "assumptions_transmission.csv"
    paths["dict_season"] = PathTemp + fs + "dict_season_north.csv"
    paths["dict_daytype"] = PathTemp + fs + "dict_day_type.csv"
    paths["dict_sectors"] = PathTemp + fs + "dict_sectors.csv"
    paths["dict_countries"] = PathTemp + fs + "dict_countries.csv"
    paths["dict_line_voltage"] = PathTemp + fs + "dict_line_voltage.csv"
    paths["dict_lines_costs"] = PathTemp + fs + "dict_lines_costs.csv"
    paths["dict_technologies"] = PathTemp + fs + "dict_technologies.csv"

    return paths


def load_input_paths(paths, param):
    """
    """
    global root
    global fs

    region = param["region_name"]

    # Raw Inputs
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

    # Cleaned load profiles
    PathTemp = root + "03 Intermediate files" + fs + "Files " + region + fs + "Load" + fs
    paths["cleaned_profiles"] = {
        "RES": PathTemp + "Residence_Load_profiles.csv",
        "IND": PathTemp + "Industry_Load_profiles.csv",
        "COM": PathTemp + "Commercial_Load_profiles.csv",
        "AGR": PathTemp + "Agriculture_Load_profiles.csv",
        "STR": PathTemp + "Streetlight_Load_profiles.csv",
    }

    return paths


def renewable_time_series_paths(paths, param):

    """

    :param paths:
    :param param:
    :return:
    """
    global root
    global fs

    region = param["region_name"]
    subregions = param["subregions_name"]
    year = str(param["year"])

    paths["region"] = root + "03 Intermediate files" + fs + "Files " + region + fs

    paths["TS_ren"] = {}
    PathTemp = paths["region"] + "Renewable energy" + fs + "Regional analysis" + fs + subregions + fs + "Regression outputs" + fs

    paths["TS_ren"] = {
        "WindOn": PathTemp + "Geothermal_WGC_WindOn_reg_TimeSeries_80_100_120_2015.csv",
        "WindOff": PathTemp + "Geothermal_WGC_WindOff_reg_TimeSeries_80_100_120_2015.csv",
        "PV": PathTemp + "Geothermal_WGC_PV_reg_TimeSeries_0_180_-90_90_2015.csv",
        "CSP": PathTemp + "",
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
        }
    }

    PathTemp = root + "01 Raw inputs" + fs + "Power plants and storage" + fs
    paths["FRESNA"] = PathTemp + "EU_Powerplants" + fs + "FRESNA2" + fs + "Matched_CARMA_ENTSOE_ESE_GEO_GPD_OPSD_reduced.csv"

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
    paths["potential_ren"] = paths["proc"] + "Renewables_potential.csv"

    # Other processes and storage
    paths["process_raw"] = paths["proc"] + "processes_and_storage_agg_bef_cleaning.csv"
    paths["process_filtered"] = paths["proc"] + "processes_and_storage_filtered.csv"
    paths["process_joined"] = paths["proc"] + "processes_and_storage_including_ren.csv"
    paths["process_completed"] = paths["proc"] + "processes_and_storage_completed.csv"
    paths["process_cleaned"] = paths["proc"] + "processes_and_storage_cleaned.shp"
    paths["process_regions"] = paths["proc_sub"] + "processes.csv"
    paths["storage_regions"] = paths["proc_sub"] + "storage.csv"
    paths["commodities_regions"] = paths["proc_sub"] + "commodities.csv"

    # Framework models
    paths["urbs_model"] = paths["urbs"] + region + "_" + subregions + "_" + year + ".xlsx"
    paths["evrys_model"] = paths["evrys"] + region + "_" + subregions + "_" + year + ".xlsx"

    return paths


def local_maps_paths(paths, param):
    """
    This function defines the paths where the local maps will be saved:
    
      * *LAND* for the raster of land areas within the scope
      * *EEZ* for the raster of sea areas within the scope
      * *LU* for the land use raster within the scope
      * *PA* for the raster of protected areas within the scope
      * *POP* for the population raster within the scope
    
    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the user preferences.
    :type param: dict
    
    :return paths: The updated dictionary paths.
    :rtype: dict
    """

    # Local maps
    PathTemp = paths["local_maps"] + param["region_name"]
    paths["LAND"] = PathTemp + "_Land.tif"  # Land pixels
    paths["EEZ"] = PathTemp + "_EEZ.tif"  # Sea pixels
    paths["LU"] = PathTemp + "_Landuse.tif"  # Land use types
    paths["PA"] = PathTemp + "_Protected_areas.tif"  # Protected areas
    paths["POP"] = PathTemp + "_Population.tif"  # Population

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


# ##################################
# #           Input files          #
# ##################################

# # Process and storage data
# PathTemp = root + '01 Raw inputs' + fs + 'Power plants and storage' + fs
# paths["database_Cal"] = PathTemp + 'CA_Powerplants' + fs + 'april_generator2017 (original data).xlsx'
