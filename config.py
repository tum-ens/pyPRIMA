import datetime
import numpy as np
import os
from sys import platform

###########################
#### User preferences #####
###########################

param = {}
param["region"] = 'Europe'
param["model_regions"] = 'NUTS0_wo_Balkans'
param["year"] = year = 2015
param["technology"] = ['WindOn']  # 'WindOff', 'PV', 'CSP'
param["hub_heights"] = {'WindOn': np.array([60, 80, 100, 120, 140]),
                        'WindOff': np.array([135])}


# Models input file Sheets
param["urbs_model_sheets"] = ['Global', 'Site', 'Commodity', 'Process', 'Process-Commodity', 'Transmission', 'Storage',
                              'DSM', 'Demand', 'Suplm', 'Buy-Sell-Price']
param["evrys_model_sheets"] = ['Flags', 'Sites', 'Commodity', 'Process', 'Transmission', 'Storage', 'DSM', 'Demand']

# urbs Global paramters
urbs_global = {"Support timeframe": param["year"],
               "Discount rate": 0.03,
               "CO2 limit": 'inf',
               "Cost budget": 6.5e11,
               "CO2 budget": 'inf'
               }
param["urbs_global"] = urbs_global

# Load

load = {"dict_season": {1: 'Winter', 2: 'Winter', 3: 'Spring/Fall', 4: 'Spring/Fall', 5: 'Spring/Fall', 6: 'Summer',
                        7: 'Summer', 8: 'Summer', 9: 'Spring/Fall', 10: 'Spring/Fall', 11: 'Spring/Fall', 12: 'Winter'},
        "dict_daytype": {'Monday': 'Working day', 'Tuesday': 'Working day', 'Wednesday': 'Working day',
                         'Thursday': 'Working day', 'Friday': 'Working day', 'Saturday': 'Saturday',
                         'Sunday': 'Sunday'},
        # If the data source of the load time series uses different country codes
        "dict_countries": {'AT': 'AT', 'BA': 'BA', 'BE': 'BE', 'BG': 'BG', 'CH': 'CH', 'CY': 'CY', 'CZ': 'CZ',
                           'DE': 'DE',
                           'DK': 'DK', 'EE': 'EE', 'ES': 'ES', 'FI': 'FI', 'FR': 'FR', 'GB': 'UK', 'GR': 'EL',
                           'HR': 'HR',
                           'HU': 'HU', 'IE': 'IE', 'IS': 'IS', 'IT': 'IT', 'LT': 'LT', 'LU': 'LU', 'LV': 'LV',
                           'ME': 'ME',
                           'MK': 'MK', 'NI': 'UK', 'NL': 'NL', 'NO': 'NO', 'PL': 'PL', 'PT': 'PT', 'RO': 'RO',
                           'RS': 'RS',
                           'SE': 'SE', 'SI': 'SI', 'SK': 'SK'},
        # Replacement of missing countries with profile (replacement) and annual values
        "replacement": 'MK',
        "missing_countries": {'AL': 6376000, 'KS': 2887000},  # in MWh
        # Additional load from traffic and heating sector
        "degree_of_elec_traff": 1,
        "degree_of_elec_heat": 1,
        # Efficiency measures
        "degree_of_eff": 1,
        # Distribution factors for load from country to regional level
        "distribution_type": 'sectors_landuse',  # 'sectors_landuse' or 'population_GPD'
        "sectors": ['COM', 'IND', 'AGR', 'STR'],
        # Load sectorial share for California
        "sector_shares_Cal": {'IND': 0.175, 'RES': 0.327, 'COM': 0.416, 'AGR': 0.077, 'STR': 0.005}
        }

param["load"] = load

# Distribution_renewables

dist_ren = {"country_names": {'Albania': 'AL',
                              'Andorra': 'AD',
                              'Austria': 'AT',
                              'Belarus': 'BY',
                              'Belgium': 'BE',
                              'Bosnia Herzg': 'BA',
                              'Bulgaria': 'BG',
                              'Croatia': 'HR',
                              'Cyprus': 'CY',
                              'Czechia': 'CZ',
                              'Denmark': 'DK',
                              'Estonia': 'EE',
                              'Faroe Islands': 'FO',
                              'Finland': 'FI',
                              'France': 'FR',
                              'FYR Macedonia': 'MK',
                              'Germany': 'DE',
                              'Gibraltar': 'GI',
                              'Greece': 'EL',
                              'Holy See': 'HS',
                              'Hungary': 'HU',
                              'Iceland': 'IS',
                              'Ireland': 'IE',
                              'Italy': 'IT',
                              'Kosovo': 'KS',
                              'Latvia': 'LV',
                              'Liechtenstein': 'LI',
                              'Lithuania': 'LT',
                              'Luxembourg': 'LU',
                              'Malta': 'MT',
                              'Moldova Rep': 'MD',
                              'Monaco': 'MC',
                              'Montenegro': 'ME',
                              'Netherlands': 'NL',
                              'Norway': 'NO',
                              'Poland': 'PL',
                              'Portugal': 'PT',
                              'Romania': 'RO',
                              'San Marino': 'SM',
                              'Serbia': 'RS',
                              'Slovakia': 'SK',
                              'Slovenia': 'SI',
                              'Spain': 'ES',
                              'Sweden': 'SE',
                              'Switzerland': 'CH',
                              'Ukraine': 'UA',
                              'Czech Republic': 'CZ',
                              'United Kingdom': 'UK'
                              },
            "renewables": {'Onshore wind energy': 'WindOn',
                           'Offshore wind energy': 'WindOff',
                           'Solar': 'Solar',
                           'Biogas': 'Biogas',
                           'Liquid biofuels': 'Liquid biofuels',
                           'Other solid biofuels': 'Biomass',
                           'Hydro <1 MW': 'Hydro_Small',
                           'Hydro 1-10 MW': 'Hydro_Large',
                           'Hydro 10+ MW': 'Hydro_Large'},
            "p_landuse": {0: 0, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2, 6: 0.2, 7: 0.2, 8: 0.1, 9: 0.1, 10: 0.5, 11: 0,
                          12: 1, 13: 0, 14: 1, 15: 0, 16: 0},
            "units": {'Solar': 1, 'WindOn': 2.5, 'WindOff': 2.5, 'Biomass': 5, 'Biogas': 5, 'Liquid biofuels': 5,
                      'Hydro_Small': 0.2, 'Hydro_Large': 200},  # MW
            "randomness": 0.4,
            "cap_lo": 0,
            "cap_up": np.inf,
            "drop_params": ['Site', 'inst-cap', 'cap_lo', 'cap-up', 'year'],
            "res_weather": np.array([1 / 2, 5 / 8]),
            "res_desired": np.array([1 / 240, 1 / 240])
            }

param["dist_ren"] = dist_ren

# Clean Process and Storage data, Process, and Storage
pro_sto = {"year_ref": 2015,
           "proc_dict": {'Hard Coal': 'Coal',
                         'Hydro': 'Hydro_Small',
                         # Later, we will define Hydro_Large as power plants with capacity > 30MW
                         'Nuclear': 'Nuclear',
                         'Natural Gas': 'Gas',
                         'Lignite': 'Lignite',
                         'Oil': 'Oil',
                         'Bioenergy': 'Biomass',
                         'Other': 'Waste',
                         'Waste': 'Waste',
                         'Wind': 'WindOn',
                         'Geothermal': 'Geothermal',
                         'Solar': 'Solar'},
           "storage": ['PumSt', 'Battery'],
           "renewable_powerplants": ['Hydro_Large', 'Hydro_Small', 'WindOn', 'WindOff',
                                     'Solar', 'Biomass', 'Biogas', 'Liquid biofuels'],
           "agg_thres": 20,
           "wacc": 0.07
           }

param["pro_sto"] = pro_sto

# Processes and Storages in California

Cal_urbs = {'cap_lo': {'Biomass': 0, 'Coal': 0, 'Gas': 0, 'Geothermal': 0, 'Hydro_Large': 0, 'Hydro_Small': 0,
                       'Nuclear': 0, 'Oil': 0, 'Slack': 999999, 'Solar': 0, 'Waste': 0, 'WindOn': 0},
            'cap_up': {'Biomass': 100, 'Coal': 0, 'Gas': 10000, 'Geothermal': 0, 'Hydro_Large': 0, 'Hydro_Small': 0,
                       'Nuclear': 0, 'Oil': 0, 'Slack': 999999, 'Solar': np.inf, 'Waste': 0, 'WindOn': np.inf},
            'max_grad': {'Biomass': 1.2, 'Coal': 0.6, 'Gas': 4.8, 'Geothermal': 0.6, 'Hydro_Large': np.inf,
                         'Hydro_Small': np.inf,
                         'Nuclear': 0.3, 'Oil': 3, 'Slack': np.inf, 'Solar': np.inf, 'Waste': 1.2, 'WindOn': np.inf},
            'min_fraction': {'Biomass': 0, 'Coal': 0.5, 'Gas': 0.25, 'Geothermal': 0, 'Hydro_Large': 0,
                             'Hydro_Small': 0,
                             'Nuclear': 0.7, 'Oil': 0.4, 'Slack': 0, 'Solar': 0, 'Waste': 0, 'WindOn': 0},
            'inv_cost': {'Biomass': 875000, 'Coal': 600000, 'Gas': 450000, 'Geothermal': 1000000,
                         'Hydro_Large': 1600000, 'Hydro_Small': 320000,
                         'Nuclear': 1600000, 'Oil': 600000, 'Slack': 0, 'Solar': 600000, 'Waste': 800000,
                         'WindOn': 900000},
            'fix_cost': {'Biomass': 28000, 'Coal': 18000, 'Gas': 6000, 'Geothermal': 10000, 'Hydro_Large': 20000,
                         'Hydro_Small': 1000,
                         'Nuclear': 50000, 'Oil': 9000, 'Slack': 0, 'Solar': 25000, 'Waste': 4000, 'WindOn': 30000},
            'var_cost': {'Biomass': 5, 'Coal': 14, 'Gas': 25, 'Geothermal': 5, 'Hydro_Large': 3, 'Hydro_Small': 3,
                         'Nuclear': 10, 'Oil': 35, 'Slack': 200, 'Solar': 0, 'Waste': 2, 'WindOn': 0},
            'startup_cost': {'Biomass': 0, 'Coal': 90, 'Gas': 40, 'Geothermal': 0, 'Hydro_Large': 0, 'Hydro_Small': 0,
                             'Nuclear': 150, 'Oil': 40, 'Slack': 0, 'Solar': 0, 'Waste': 0, 'WindOn': 0},
            'wacc': 0.07,
            'depreciation': {'Biomass': 25, 'Coal': 40, 'Gas': 30, 'Geothermal': 100, 'Hydro_Large': 100,
                             'Hydro_Small': 30,
                             'Nuclear': 60, 'Oil': 30, 'Slack': 1, 'Solar': 25, 'Waste': 25, 'WindOn': 25},
            'area_per_cap': {'Biomass': np.nan, 'Coal': np.nan, 'Gas': np.nan, 'Geothermal': np.nan,
                             'Hydro_Large': np.nan, 'Hydro_Small': np.nan,
                             'Nuclear': np.nan, 'Oil': np.nan, 'Slack': np.nan, 'Solar': 14000, 'Waste': np.nan,
                             'WindOn': np.nan}
            }

pro_sto_Cal = {'proc_dict': {'AB': 'Biomass', 'BLQ': 'Biomass', 'OBG': 'Biomass', 'WDS': 'Biomass',
                             'BIT': 'Coal', 'RC': 'Coal',
                             'DFO': 'Oil', 'JF': 'Oil', 'PG': 'Oil', 'PC': 'Oil',
                             'LFG': 'Gas', 'NG': 'Gas', 'OG': 'Gas', 'PUR': 'Gas', 'WH': 'Gas',
                             'GEO': 'Geothermal',
                             'WAT': 'Hydro_Small',
                             # Later, we will define Hydro_Large as power plants with capacity > 30MW
                             'MWH': 'Battery',
                             'NUC': 'Nuclear',
                             'SUN': 'Solar',
                             'MSW': 'Waste',
                             'WND': 'WindOn'},
               'storage': ['PumSt', 'Battery'],
               'status': ['(OP) Operating', '(SB) Standby/Backup: available for service but not normally used'],
               'states': ['CA'],
               'Cal_urbs': Cal_urbs
               }

param["pro_sto_Cal"] = pro_sto_Cal

# Clean Grid
loadability = {"80": 3,
               "100": 2.75,
               "150": 2.5,
               "200": 2,
               "250": 1.75,
               "300": 1.5,
               "350": 1.375,
               "400": 1.25,
               "450": 1.125,
               "500": 1,
               "550": 0.9,
               "600": 0.85,
               "650": 0.8,
               "700": 0.77,
               "750": 0.6}
param["grid"] = {"depreciation": 40,
                 "loadability": loadability}

###########################
##### Define Paths ########
###########################

fs = os.path.sep
git_folder = os.path.dirname(os.path.abspath(__file__))

git_RT_folder = os.path.dirname(os.path.abspath(__file__))

if platform.startswith('win'):
    # Windows Root Folder
    from pathlib import Path

    root = str(Path(git_RT_folder).parent.parent) + fs + "Database_KS" + fs
elif platform.startswith('linux'):
    # Linux Root Folder
    root = git_RT_folder + fs + ".." + fs + ".." + fs + "Database_KS" + fs

region = param["region"]
model_regions = param["model_regions"]

paths = {}

##################################
#           Input files          #
##################################

# Shapefiles
PathTemp = root + "01 Raw inputs" + fs + "Maps" + fs + "Shapefiles" + fs
# paths["SHP"] = PathTemp + "Germany_with_EEZ.shp"
# paths["Countries"] = PathTemp + "gadm36_DEU_0.shp"  # No EEZ!
paths["Countries"] = PathTemp + "Europe_NUTS0_wo_Balkans.shp"
paths["SHP"] = paths["Countries"]
if param["region"] == "California":
    paths["SHP"] = PathTemp + "CA_Regions_EM_1.shp"
    paths["Countries"] = PathTemp + "CA_Load Profiles_Regions_EM.shp"
    paths["regions_SHP"] = PathTemp + "CA_regions_geographic.shp"

# Rasters
PathTemp = root + "02 Intermediate files" + fs + "Files " + region + fs + "Maps" + fs + region
paths["LU"] = PathTemp + "_Landuse.tif"  # Land use types
paths["POP"] = PathTemp + "_Population.tif"  # Population

# Assumptions
paths["assumptions"] = root + "00 Assumptions" + fs + "assumptions_const.xlsx"
# paths["assumptions"] = root + "00 Assumptions" + fs + "assumptions_const_v04_4NEMO.xlsx"


# Load
PathTemp = root + "01 Raw inputs" + fs + "Load" + fs
paths["sector_shares"] = PathTemp + "Eurostat" + fs + "SectorShares_20181206.csv"
paths["load_ts"] = PathTemp + "ENTSOE" + fs + "Monthly-hourly-load-values_2006-2015.xlsx"
paths["load_ts_ca"] = PathTemp + "CA_Load Profiles_11 Regions_correct names.csv"
paths["profiles"] = {'RES': PathTemp + "Load profiles" + fs + "Lastprofil_Haushalt_H0.xlsx",
                     'IND': PathTemp + "Load profiles" + fs + "Lastprofil_Industrie_Tag.xlsx",
                     'COM': PathTemp + "Load profiles" + fs + "VDEW-Lastprofile-Gewerbe-Landwirtschaft_G0.csv",
                     'AGR': PathTemp + "Load profiles" + fs + "VDEW-Lastprofile-Landwirtschaft_L0.csv",
                     'STR': PathTemp + "Load profiles" + fs + "Lastprofil_Strassenbeleuchtung_S0.xlsx"
                     }
paths["profiles_ca"] = {'RES': PathTemp + "Load profiles" + fs + "Residential Load Profile_2017_SCE.csv",
                        'IND': PathTemp + "Load profiles" + fs + "Medium Commercial and Industrial_Load Profile_2017 SCE.xlsx",
                        'COM': PathTemp + "Load profiles" + fs + "Small Commercial_Load Profile_2017 SCE.xlsx",
                        'AGR': PathTemp + "Load profiles" + fs + "VDEW-Lastprofile-Landwirtschaft_L0.csv",
                        'STR': PathTemp + "Load profiles" + fs + "Lastprofil_Strassenbeleuchtung_S0.xlsx"
                        }

# Process and storage data
PathTemp = root + '01 Raw inputs' + fs + 'Power plants and storage' + fs
paths["database"] = PathTemp + 'EU_Powerplants' + fs + 'Matched_CARMA_ENTSOE_GEO_OPSD_WRI_reduced.csv'
paths["database_FRESNA"] = PathTemp + 'EU_Powerplants' + fs + 'FRESNA2' + fs + \
                           'Matched_CARMA_ENTSOE_ESE_GEO_GPD_OPSD_reduced.csv'
paths["database_Cal"] = PathTemp + 'CA_Powerplants' + fs + 'april_generator2017 (original data).xlsx'

# Grid
paths["grid"] = root + '01 Raw inputs' + fs + 'Grid' + fs + 'gridkit_europe' + fs + \
                'gridkit_europe-highvoltage-links.csv'
# paths["grid"] = root + '01 Raw inputs' + fs + 'Grid' + fs + 'gridkit_US' + fs + \
#                 'gridkit_north_america-highvoltage-links.csv'


# ## Renewable Capacities
# Rasters for wind and solar
timestamp = '20190617T142740'
pathtemp = root + "02 Intermediate files" + fs + "Files " + region + fs + "Renewable energy" + fs + timestamp + fs
rasters = {'WindOn': pathtemp + 'Europe_WindOn_FLH_mask_2015.tif',
           'WindOff': pathtemp + 'Europe_WindOff_FLH_mask_2015.tif',
           'Solar': pathtemp + 'Europe_PV_FLH_mask_2015.tif',
           'Biomass': pathtemp + 'Europe_Biomass.tif',
           'Liquid biofuels': pathtemp + 'Europe_Biomass.tif',
           'Biogas': pathtemp + 'Europe_Biomass.tif'}

paths["rasters"] = rasters

# IRENA Data
paths["IRENA"] = root + '01 Raw inputs' + fs + 'Renewable energy' + fs + 'Renewables.xlsx'
# outputs
paths["map_power_plants"] = root + '01 Raw inputs' + fs + 'maps' + fs
paths["map_grid_plants"] = root + '01 Raw inputs' + fs + 'maps' + fs + 'random_points.shp'

# Intermittent Supply Timeseries
paths["raw_TS"] = {}
paths["reg_coef"] = {}
paths["regression_out"] = pathtemp + "Regression_Outputs" + fs
for tech in param["technology"]:
    paths["raw_TS"][tech] = {}
    if tech in ['WindOn', 'WindOff']:
        for height in param["hub_heights"][tech]:
            paths["raw_TS"][tech][str(height)] = \
                pathtemp + region + '_' + tech + '_' + str(height) + '_TS_' + str(year) + '.csv'
    else:
        paths["raw_TS"][tech][''] = pathtemp + region + '_' + tech + '_TS_' + str(year) + '.csv'

    paths["reg_coef"][tech] = \
        paths["regression_out"] + region + '_' + tech + '_reg_coefficients.csv'

##################################
#     General Ouputs Folders     #
##################################

pathtemp = root + '02 Intermediate files' + fs + 'Files ' + region + fs

# 02 - load
paths["load"] = pathtemp + 'Load' + fs
paths["model_regions"] = pathtemp + model_regions + fs
paths["sites"] = pathtemp + model_regions + fs + 'Sites.csv'
paths["load_EU"] = pathtemp + 'Load' + fs + 'Load_EU' + '%04d' % (param["year"]) + '.csv'

# 02 - Intermittent Supply Timeseries
paths["suplm_TS"] = paths["model_regions"] + 'intermittent_supply_timeseries_' + str(year) + '.csv'

# 02 - process and storage
paths["pro_sto"] = pathtemp + 'Processes_and_Storage_' + str(param["year"]) + '.shp'
paths["PPs_"] = root + '02 Intermediate files' + fs + 'Files ' + region + fs
paths["process_raw"] = pathtemp + 'Processes_raw_FRESNA2_2.csv'
paths["Process_agg"] = pathtemp + 'Processes_agg_FRESNA2_2.csv'
paths["Process_agg_bis"] = pathtemp + 'Processes_agg_FRESNA2_3.csv'

# 02 - Grid
paths["grid_shp"] = pathtemp + 'Grid' + fs + 'grid_cleaned_shape.shp'
paths["grid_cleaned"] = pathtemp + 'Grid' + fs + 'GridKit_cleaned.csv'

# urbs
paths["urbs"] = paths["model_regions"] + "urbs" + fs
paths["urbs_sites"] = paths["urbs"] + 'Site_urbs_' + str(year) + '.csv'
paths["urbs_suplm"] = paths["urbs"] + 'Suplm_urbs_' + str(year) + '.csv'
paths["urbs_demand"] = paths["urbs"] + 'Demand_urbs_' + str(year) + '.csv'
paths["urbs_commodity"] = paths["urbs"] + 'Commodity_urbs_' + str(year) + '.csv'
paths["urbs_process"] = paths["urbs"] + 'Process_urbs_' + str(year) + '.csv'
paths["urbs_storage"] = paths["urbs"] + 'Storage_urbs_' + str(year) + '.csv'
paths["urbs_transmission"] = paths["urbs"] + 'Transmission_urbs_' + str(year) + '.csv'
paths["urbs_model"] = paths["urbs"] + 'urbs_' + \
                      str(param["region"]) + '_' + str(param["model_regions"]) + '_' + str(param["year"]) + '.xlsx'

# evrys
paths["evrys"] = paths["model_regions"] + "evrys" + fs
paths["evrys_sites"] = paths["evrys"] + 'Sites_evrys_' + str(year) + '.csv'
paths["evrys_demand"] = paths["evrys"] + 'Demand_evrys_' + str(year) + '.csv'
paths["evrys_commodity"] = paths["evrys"] + 'Commodity_evrys_' + str(year) + '.csv'
paths["evrys_process"] = paths["evrys"] + 'Process_evrys_' + str(year) + '.csv'
paths["evrys_storage"] = paths["evrys"] + 'Storage_evrys_' + str(year) + '.csv'
paths["evrys_transmission"] = paths["evrys"] + 'Transmission_evrys_' + str(year) + '.csv'
paths["evrys_model"] = paths["evrys"] + 'evrys_' + \
                       str(param["region"]) + '_' + str(param["model_regions"]) + '_' + str(param["year"]) + '.xlsx'

if not os.path.isdir(paths["urbs"]):
    os.mkdir(paths["urbs"])
if not os.path.isdir(paths["evrys"]):
    os.mkdir(paths["evrys"])

del root, PathTemp, fs
