import os
from pathlib import Path
import numpy as np


def configuration():
    """
    This function is the main configuration function that calls all the other modules in the code.

    :return (paths, param): The dictionary param containing all the user preferences, and the dictionary path containing all the paths to inputs and outputs.
    :rtype: tuple(dict, dict)
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
    paths = load_input_paths(paths)
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
    :rtype: tuple(dict, dict)
    """
    # These variables will be initialized here, then read in other modules without modifying them.
    global fs
    global root

    param = {}
    param["author"] = "User"  # the name of the person running the script
    param["comment"] = "Tutorial"

    paths = {}
    fs = os.path.sep
    current_folder = os.path.dirname(os.path.abspath(__file__))
    root = str(Path(current_folder).parent.parent)
    # For use at TUM ENS
    if root[-1] != fs:
        root = root + fs + "Database" + fs
    else:
        root = root + "Database" + fs

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

    *technology* is a dictionary of the technologies (*Storage*, *Process*) to be used in the model. The names of the technologies should match the names
    which are used in assumptions_flows.csv, assumptions_processes.csv and assumptions_storage.csv.

    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the user preferences.
    :type param: dict

    :return (paths, param): The updated dictionaries paths and param.
    :rtype: tuple of dict
    """

    # Paths to the shapefiles
    PathTemp = root + "02 Shapefiles for regions" + fs 
    paths["spatial_scope"] = PathTemp + "User-defined" + fs + "gadm36_AUT_0.shp"# "Europe_NUTS0_wo_Balkans_with_EEZ.shp" # #
    paths["subregions"] = PathTemp + "Clustering outputs" + fs + "Austria" + fs + "Wind_FLH - Solar_FLH" + fs + "05 final_output" + fs + "final_result.shp"

    # Name tags for the scope and the subregions
    param["region_name"] = "Austria"  # Name tag of the spatial scope
    param["subregions_name"] = "Austria"  # Name tag of the subregions

    # Year
    param["year"] = 2015  # Data
    param["model_year"] = 2015  # Model

    # Technologies
    param["technology"] = {
        "Storage": ["Battery", "PumSt"],
        "Process": ["Bioenergy", "Coal", "Gas", "Geothermal", "Hydro", "Lignite", "Nuclear", "OilOther", "Solar", "WindOn"],
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
    This function defines the user preferences which are related to the load/demand. Currently, only one parameter is used, namely
    *default_sec_shares*, which sets the reference region to be used in case data for other regions is missing.

    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """

    param["load"] = {"default_sec_shares": "DEU"}
    return param


def renewable_time_series_parameters(param):
    """
    This function defines parameters related to the renewable time series to be used in the models. In particular, the user can decide which
    `modes` to use from the files of the time series, provided they exist. See the repository tum-ens/renewable-timeseries for more information.

    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """

    param["ren_potential"] = {"WindOn": ["all"], "WindOff": ["all"], "PV": ["all"], "CSP": ["all"]}  # "Technology":[list of modes]

    return param


def grid_parameters(param):
    """
    This function defines parameters related to the grid to be used while cleaning the data.
    
      * *quality* is a user assessment of the quality of the data. If the data is trustworthy, use 1, if it is not trustworthy at all, use 0. You can use values inbetween.
      * *default* is a collection of default values for voltage, wires, cables, and frequency, to use when these data are missing.

    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """

    param["grid"] = {
        "quality": {"voltage": 1, "wires": 0, "cables": 0.5, "frequency": 0},
        "default": {"voltage": 220000, "wires": 1, "cables": 3, "frequency": 50},
    }

    return param


def processes_parameters(param):
    """
    This function defines parameters related to the processes in general, and to distributed renewable capacities in particular.
    
    For *process*, only the parameter *cohorts* is currently used. It defines how power plants should be grouped according to their construction period.
    If *cohorts* is 5, then you will have groups of coal power plants from 1960, then another from 1965, and so on. If you do not wish to group the power plants,
    use the value 1.
    
    For distributed renewable capacities, *dist_ren*, the following parameters are needed:

      * *units* is a dictionary defining the standard power plant size for each distributed renewable technology in MW.
      * *randomness* is a value between 0 and 1, defining the randomness of the spatial distribution of renewable capacities. The complementary value (1 - randomness)
        is affected by the values of the potential raster used for the distribution. When using a high resolution map, set *randomness* at a high level (close to 1),
        otherwise all the power plants will be located in a small area of high potential, close to each other.
      * *default_pa_type* and *default_pa_availability* are two arrays defining the availability for each type of protected land. These arrays are used as default, along
        with the protected areas raster, in case no potential map is available for a distributed renewable technology.

    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """

    param["process"] = {"cohorts": 1}  # 5 means 5-year steps, if no cohorts needed type 1

    param["dist_ren"] = {
        "units": {"Solar": 5, "WindOn": 10, "Bioenergy": 10, "Hydro": 50},
        "randomness": 0.99,
        "default_pa_type": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "default_pa_availability": np.array([1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 1.00, 1.00, 1.00, 1.00]),
    }

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
    This function defines the paths for the assumption files and the dictionaries.
    
      * *assumptions_landuse* is a table with land use types as rows and sectors as columns. The table is filled with values between 0 and 1, so that each row
        has a total of 0 (no sectoral load there) or 1 (if there is a load, it will be distributed according to the shares of each sector).
      * *assumptions_flows* is a table with the following columns:
        
        * `year`: data reference year.
        * `Process/Storage`: name of the process or storage type.
        * `Direction`: either ``In`` or ``Out``. You can have multiple inputs and outputs, each one in a separate row.
        * `Commodity`: name of the input or output commodity.
        * `ratio`: ratio to the throughput, which is an intermediate level between input and output. It could be any positive value. The ratio of the output to the input corresponds to the efficiency. 
        * `ratio-min` similar to `ratio`, but at partial load.
        
      * *assumptions_processes* is a table with the following columns:
      
        * `year`: data reference year.
        * `Process`:  name of the process usually given as the technology type.
        * `cap-lo`: minimum power capacity.
        * `cap-up`: maximum power capacity.
        * `max-grad`: maximum allowed power gradient (1/h) relative to power capacity.
        * `min-fraction`: minimum load fraction at which the process can run at.
        * `inv-cost`: total investment cost per power capacity (Euro/MW). It will be annualized in the model using an annuity factor derived from the wacc and depreciation period.
        * `fix-cost`: annual operation independent or fix cost (Euro/MW/a)
        * `var-cost`: variable cost per throughput energy unit(Euro/MWh) but excludes fuel costs.
        * `start-cost`: startup cost when the process is switch on from the off condition.
        * `wacc`: weighted average cost of capital. Percentage of cost of capital after taxes.
        * `depreciation`: deprecation period in years.
        * `lifetime`: lifetime of already installed capacity in years.
        * `area-per-cap`: area required per power capacity (m²/MW).
        * `act-up`: maximal load (per unit).
        * `act-lo`: minimal load (per unit).
        * `on-off`: binary variable, 1 for controllable power plants, otherwise 0 (must-run).
        * `reserve-cost`: cost of power reserves (Euro/MW) (to be verified).
        * `ru`: ramp-up capacity (MW/MWp/min).
        * `rd`: ramp-down capacity (MW/MWp/min).
        * `rumax`: maximal ramp-up (MW/MWp/h).
        * `rdmax`: minimal ramp-up (MW/MWp/h).
        * `detail`: level of detail for modeling thermal power plants, modes 1-5.
        * `lambda`: cooling coefficient (to be verified).
        * `heatmax`: maximal heating capacity (to be verified).
        * `maxdeltaT`: maximal temperature gradient (to be verified).
        * `heatupcost`: costs of heating up (Euro/MWh_th) (to be verified).
        * `su`: ramp-up at start (to be verified).
        * `sd`: ramp-down at switch-off (to be verified).
        * `pdt`: (to be verified).
        * `hotstart`: (to be verified).
        * `pot`: (to be verified).
        * `pretemp`: temperature at the initial time step, per unit of the maximum operating temperature.
        * `preheat`: heat content at the initial time step, per unit of the maximum operating heat content.
        * `prestate`: operating state at the initial time step (binary).
        * `prepow`: available power at the initial time step (MW) (to be verified).
        * `precaponline`: online capacity at the initial time step (MW).
        * `year_mu`: average construction year for that type of power plants.
        * `year_stdev`: standard deviation from the average construction year for that type of power plants.

      * *assumptions_storage* is a table with the following columns:

        * `year`: data reference year.
        * `Storage`: name of the storage usually given as the technology type.
        * `ep-ratio`: fixed energy to power ratio (hours).
        * `cap-up-c`: maximum allowed energy capacity (MWh)
        * `cap-up-p`: maximum allowed power capacity (MW)
        * `inv-cost-p`: total investment cost per power capacity (Euro/MW). It will be annualized in the model using an annuity factor derived from the wacc and depreciation period.
        * `inv-cost-c`: total investment cost per energy capacity (Euro/MWh). It will be annualized in the model using an annuity factor derived from the wacc and depreciation period.
        * `fix-cost-p`: annual operation independent or fix cost per power capacity (Euro/MW/a)
        * `fix-cost-c`: annual operation independent or fix cost per energy capacity (Euro/MWh/a)
        * `var-cost-p`: opertion dependent costs for input and output of energy per MWh_out stored or retreived (euro/MWh)
        * `var-cost-c`: operation dependent costs per MWh stored. This value can used to model technologies that have increased wear and tear proportional to the amount of stored energy.
        * `lifetime`: lifetime of already installed capacity in years.
        * `depreciation`: deprecation period in years.
        * `wacc`: weighted average cost of capital. Percentage of cost of capital after taxes.
        * `init`: initial storage content. Fraction of storage capacity that is full at the simulation start. This level has to be reached in the final timestep.
        * `var-cost-pi`: variable costs for charing (Euro/MW).
        * `var-cost-po`: variable costs for discharing (Euro/MW).
        * `act-lo-pi`: minimal share of active capacity for charging (per unit).
        * `act-up-pi`: maximal share of active capacity for charging (per unit).
        * `act-lo-po`: minimal share of active capacity for discharging (per unit).
        * `act-up-po`: maximal share of active capacity for discharging (per unit).
        * `act-lo-c`: minimal share of storage capacity (per unit).
        * `act-up-c`: maximal share of storage capacity (per unit).
        * `precont`: energy content of the storage unit at the initial time step (MWh) (to be verified).
        * `prepowin`: energy stored at the initial time step (MW).
        * `prepowout`: energy discharged at the initial time step (MW).
        * `ru`: ramp-up capacity (MW/MWp/min).
        * `rd`: ramp-down capacity (MW/MWp/min).
        * `rumax`: maximal ramp-up (MW/MWp/h).
        * `rdmax`: minimal ramp-up (MW/MWp/h).
        * `seasonal`: binary variable, 1 for seasonal storage.
        * `ctr`: binary variable, 1 if can be used for secondary reserve.
        * `discharge`: energy losses due to self-discharge per hour as a percentage of the energy capacity.
        * `year_mu`: average construction year for that type of storage.
        * `year_stdev`: standard deviation from the average construction year for that type of storage.

      * *assumptions_commodities* is a table with the following columns:

        * `year`: data reference year.
        * `Commodity`: name of the commodity.
        * `Type_urbs`: type of the commodity according to urbs' terminology.
        * `Type_evrys`: type of the commodity according to evrys' terminology.
        * `price`: commodity price (euro/MWh).
        * `max`: maximum annual commodity use (MWh).
        * `maxperhour`: maximum commodity use per hour (MW).
        * `annual`: total value per year (MWh).
        * `losses`: losses (to be verified).

      * *assumptions_transmission* is a table with the following columns:

        * `Type`: type of transmission.
        * `length_limit_km`: maximum length of the transmission line in km, for which the assumptions are valid.
        * `year`: data reference year.
        * `Commodity`: name of the commodity to be transported along the transmission line.
        * `eff_per_1000km`: transmission efficiency after 1000km in percent.
        * `inv-cost-fix`: length independent investment cost (euro).
        * `inv-cost-length`: length dependent investment cost (euro/km).
        * `fix-cost-length`: fixed annual cost dependent on the length of the line (euro/km/a).
        * `var-cost`: variable costs per energy unit transmitted (euro/MWh)
        * `cap-lo`: minimum required power capacity (MW).
        * `cap-up`: maximum allowed power capacity (MW).
        * `wacc`: weighted average cost of capital. Percentage of cost of capital after taxes.
        * `depreciation`: deprecation period in years.
        * `act-lo`: minimum capacity (MW/MWp).
        * `act-up`: maximum capacity (MW/MWp).
        * `angle-up`: maximum phase angle ramp-up (to be verified).
        * `PSTmax`: maximum phase angle difference.

      * *dict_season_north* is a table with the following columns:

        * `Month`: number of the month (1-12).
        * `Season`: corresponding season.

      * *dict_daytype* is a table with the following columns:

        * `Weak day`: name of the weekday (Monday-Sunday).
        * `Type`: either `Working day`, `Saturday`, or `Sunday`.

      * *dict_sectors* is a table with the following columns:

        * `EUROSTAT`: name of the entry in the EUROSTAT table.
        * `Model_sectors`: corresponding sector (leave empty if irrelevant).

      * *dict_counties* is a table with the following columns:

        * `IRENA`: names of countries in IRENA database.
        * `Counties shapefile`: names of countries in the countries shapefile.
        * `NAME_SHORT`: code names for the countries as used by the code.
        * `ENTSO-E`: names of countries in the ENTSO-E dataset.
        * `EUROSTAT`: names of countries in the EUROSTAT table.

      * *dict_line_voltage* is a table with the following columns:

        * `voltage_kV`: sorted values of possible line voltages.
        * `specific_impedance_Ohm_per_km`: specific impedance (leave empty if unknown).
        * `loadability`: loadability factor according to the St Clair's curve (leave empty if unknown).
        * `SIL_MWh`: corresponding surge impedance load (leave empty if unknown).

      * *dict_technologies* is a table with the following columns:

        * `IRENA`: names of technologies in the IRENA database.
        * `FRESNA`: names of technologies in the FRESNA database.
        * `Model names`: names of technologies as used in the model.

    
    :param paths: Dictionary including the paths.
    :type paths: dict
    
    :return: The updated dictionary paths.
    :rtype: dict
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


def load_input_paths(paths):
    """
    This function defines the paths where the load related inputs are saved:
    
      * *sector_shares* for the sectoral shares in the annual electricity demand.
      * *load_ts* for the load time series.
      * *profiles* for the sectoral load profiles.
    
    :param paths: Dictionary including the paths.
    :type paths: dict

    :return paths: The updated dictionary paths.
    :rtype: dict
    """
    global root
    global fs

    # Raw Inputs
    PathTemp = root + "01 Raw inputs" + fs + "Load" + fs
    paths["sector_shares"] = PathTemp + "Eurostat" + fs + "nrg_105a_1_Data.csv"
    paths["load_ts"] = PathTemp + "ENTSOE" + fs + "Monthly-hourly-load-values_2006-2015.xlsx"  # CA: "CA_Load Profiles_11 Regions_correct names.csv"
    paths["profiles"] = {
        "RES": PathTemp + "Load profiles" + fs + "Lastprofil_Haushalt_H0.xlsx",
        # CA: "Residential Load Profile_2017_SCE.csv"
        "IND": PathTemp + "Load profiles" + fs + "Lastprofil_Industrie_Tag.xlsx",
        # CA: "Medium Commercial and Industrial_Load Profile_2017 SCE.xlsx"
        "COM": PathTemp
        + "Load profiles"
        + fs
        + "VDEW-Lastprofile-Gewerbe-Landwirtschaft_G0.csv",  # CA: "Small Commercial_Load Profile_2017 SCE.xlsx"
        "AGR": PathTemp + "Load profiles" + fs + "VDEW-Lastprofile-Landwirtschaft_L0.csv",
        # CA: "VDEW-Lastprofile-Landwirtschaft_L0.csv"
        "STR": PathTemp + "Load profiles" + fs + "Lastprofil_Strassenbeleuchtung_S0.xlsx",
        # CA: "Lastprofil_Strassenbeleuchtung_S0.xlsx"
    }

    return paths


def renewable_time_series_paths(paths, param):
    """
    This function defines the paths where the renewable time series (inputs) are located. *TS_ren* is itself a dictionary
    with the keys *WindOn*, *WindOff*, *PV*, *CSP* pointing to the individual files for each technology.
    
    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the parameters *region_name*, *subregions_name*, and *year*.
    :type param: dict

    :return paths: The updated dictionary paths.
    :rtype: dict
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
        "WindOn": PathTemp + "Austria_WindOn_reg_TimeSeries_80_2015.csv",
        "WindOff": PathTemp + "",
        "PV": PathTemp + "Austria_PV_reg_TimeSeries_0_2015.csv",
        "CSP": PathTemp + "",
    }

    return paths


def grid_input_paths(paths):
    """
    This function defines the paths where the transmission lines (inputs) are located.
    
    :param paths: Dictionary including the paths.
    :type paths: dict

    :return paths: The updated dictionary paths.
    :rtype: dict
    """
    global root
    global fs

    PathTemp = root + "01 Raw inputs" + fs + "Grid" + fs
    paths["transmission_lines"] = PathTemp + "gridkit_europe" + fs + "gridkit_europe-highvoltage-links.csv"

    return paths


def processes_input_paths(paths, param):
    """
    This function defines the paths where the process-related inputs are located:
    
      * *IRENA*: IRENA electricity statistics (useful to derive installed capacities of renewable energy technologies).
      * *dist_ren*: dictionary of paths to rasters defining how the potential for the renewable energy is spatially distributed. The rasters have to be the same size as the spatial scope.
      * *FRESNA*: path to the locally saved FRESNA database.
    
    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the parameter *year*.
    :type param: dict

    :return paths: The updated dictionary paths.
    :rtype: dict
    """
    global root
    global fs

    year = str(param["year"])

    PathTemp = root + "01 Raw inputs" + fs + "Renewable energy" + fs
    paths["IRENA"] = PathTemp + "IRENA" + fs + "IRENA_RE_electricity_statistics_allcountries_alltech_" + year + ".csv"

    PathTemp = root + "03 Intermediate files" + fs + "Files Austria" + fs + "Renewable energy" + fs + "Potential" + fs
    paths["dist_ren"] = {
        "rasters": {
            "Solar": PathTemp + "Austria_PV_0_FLH_mask_2015.tif",
            "WindOn": PathTemp + "Austria_WindOn_100_FLH_mask_2015.tif",
            #"WindOff": PathTemp + "Austria_WindOff_80_FLH_mask_2015.tif",
            "Bioenergy": PathTemp + "Austria_PV_0_FLH_mask_2015.tif",
            "Hydro": PathTemp + "Austria_PV_0_FLH_mask_2015.tif",
        }
    }

    PathTemp = root + "01 Raw inputs" + fs + "Power plants and storage" + fs
    paths["FRESNA"] = PathTemp + "EU_Powerplants" + fs + "FRESNA2" + fs + "Matched_CARMA_ENTSOE_ESE_GEO_GPD_OPSD_reduced.csv"

    return paths


def output_folders(paths, param):
    """
    This function defines the paths to multiple output folders:
    
      * *region* is the main output folder.
      * *local_maps* is the output folder for the local maps of the spatial scope.
      * *sites* is the output folder for the files related to the modeled sites.
      * *load* is the output folder for the subregions-independent, load-related intermediate files.
      * *load_sub* is the output folder for the subregions-dependent, load-related intermediate files.
      * *grid* is the output folder for the subregions-independent, grid-related intermediate files.
      * *grid_sub* is the output folder for the subregions-dependent, grid-related intermediate files.
      * *regional_analysis* is the output folder for the regional analysis of renewable energy.
      * *proc* is the output folder for the subregions-independent, process-related intermediate files.
      * *proc_sub* is the output folder for the subregions-dependent, process-related intermediate files.
      * *urbs* is the output folder for the urbs model input file.
      * *evrys* is the output folder for the evrys model input files.
      
    All the folders are created at the beginning of the calculation, if they do not already exist.
    
    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the user preferences *region_name* and *subregions_name*.
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
    This function defines the paths to multiple output files.
    
    Sites:
      * *sites_sub* is the CSV output file listing the modeled sites and their attributes.
      
    Load:
      * *stats_countries* is the CSV output file listing some load statistics on a country level.
      * *load_ts_clean* is the CSV output file with cleaned load time series on a country level.
      * *cleaned_profiles* is a dictionary of paths to the CSV file with cleaned load profiles for each sector.
      * *df_sector* is the CSV output file with load time series for each sector on a country level.
      * *load_sector* is the CSV output file with yearly electricity demand for each sector and country.
      * *load_landuse* is the CSV output file with load time series for each land use type on a country level.
      * *intersection_subregions_countries* is a shapefile where the polygons are the outcome of the intersection between the countries and the subregions.
      * *stats_country_parts* is the CSV output file listing some load statistics on the level of country parts.
      * *load_ts_clean* is the CSV output file with load time series on the level of subregions.
    
    Grid:
      * *grid_expanded* is a CSV file including a reformatted table of transmission lines.
      * *grid_filtered* is a CSV file obtained after filtering out erronous/useless data points.
      * *grid_corrected* is a CSV file obtained after correcting erronous data points.
      * *grid_filled* is a CSV file obtained after filling missing data with default values.
      * *grid_cleaned* is a CSV file obtained after cleaning the data and reformatting the table.
      * *grid_shp* is a shapefile of the transmission lines.
      * *grid_completed* is a CSV file containing the aggregated transmission lines between the subregions and their attributes.
      
    Renewable processes:
      * *IRENA_summary* is a CSV file with a summary of renewable energy statistics for the countries within the scope.
      * *locations_ren* is a dictionary of paths pointing to shapefiles of possible spatial distributions of renewable power plants.
      * *potential_ren* is a CSV file with renewable potentials.
      
    Other processes and storage:
      * *process_raw* is a CSV file including aggregated information about the power plants before processing it.
      * *process_filtered* is a CSV file obtained after filtering out erronous/useless data points.
      * *process_joined* is a CSV file obtained after joining the table with default attribute assumptions (like costs).
      * *process_completed* is a CSV file obtained after filling missing data with default values.
      * *process_cleaned* is a CSV file obtained after cleaning the data and reformatting the table.
      * *process_regions* is a CSV file containing the power plants for each subregion.
      * *storage_regions* is a CSV file containing the storage devices for each subregion.
      * *commodities_regions* is a CSV file containing the commodities for each subregion.
      
    Framework models:
      * *urbs_model* is the urbs model input file.
      * *evrys_model* is the evrys model input file.
    
    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the user preferences *region_name*, *subregions_name*, and *year*.
    :type param: dict
    :return: The updated dictionary paths.
    :rtype: dict
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

    # Cleaned load profiles
    paths["cleaned_profiles"] = {
        "RES": paths["load"] + "Residential_Load_profiles.csv",
        "IND": paths["load"] + "Industry_Load_profiles.csv",
        "COM": paths["load"] + "Commercial_Load_profiles.csv",
        "AGR": paths["load"] + "Agriculture_Load_profiles.csv",
        "STR": paths["load"] + "Streetlight_Load_profiles.csv",
    }

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
        #"WindOff": paths["proc"] + "WindOff.shp",
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
