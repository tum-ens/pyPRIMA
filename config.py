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
param["year"] = 2015

# Models input file Sheets
param["urbs_model_sheets"] = ['Global', 'Site', 'Commodity', 'Process', 'Process-Commodity', 'Transmission', 'Storage',
                              'DSM', 'Demand', 'Suplm', 'Buy-Sell-Price']
param["evrys_model_sheets"] = ['Flags', 'Sites', 'Commodities', 'Process', 'Transmission', 'Storage', 'DSM', 'Demand']

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
        "distribution_type": 'sectors_landuse',
        "sectors": ['COM', 'IND', 'AGR']
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

if platform.startswith('win'):
    # Windows Root Folder
    from pathlib import Path

    root = str(Path(git_folder).parent) + fs

elif platform.startswith('linux'):
    # Linux Root Folder
    root = git_folder + fs + ".." + fs

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

# Rasters
PathTemp = root + "02 Intermediate files" + fs + "Files " + region + fs + "Maps" + fs + region
paths["LU"] = PathTemp + "_Landuse.tif"  # Land use types
paths["POP"] = PathTemp + "_Population.tif"  # Population

# Assumptions
paths["assumptions"] = root + "00 Assumptions" + fs + "assumptions_const.xlsx"

# Load
PathTemp = root + "01 Raw inputs" + fs + "Load" + fs
paths["sector_shares"] = PathTemp + "Eurostat" + fs + "SectorShares_20181206.csv"
paths["load_ts"] = PathTemp + "ENTSOE" + fs + "Monthly-hourly-load-values_2006-2015.xlsx"
paths["profiles"] = {'RES': PathTemp + "Load profiles" + fs + "Lastprofil_Haushalt_H0.xlsx",
                     'IND': PathTemp + "Load profiles" + fs + "Lastprofil_Industrie_Tag.xlsx",
                     'COM': PathTemp + "Load profiles" + fs + "VDEW-Lastprofile-Gewerbe-Landwirtschaft_G0.csv",
                     'AGR': PathTemp + "Load profiles" + fs + "VDEW-Lastprofile-Landwirtschaft_L0.csv",
                     'STR': PathTemp + "Load profiles" + fs + "Lastprofil_Strassenbeleuchtung_S0.xlsx",
                     }

# Process and storage data
paths["database"] = root + '01 Raw inputs' + fs + 'Power plants and storage' + fs + 'EU_Powerplants' + fs + \
                    'Matched_CARMA_ENTSOE_GEO_OPSD_WRI_reduced.csv'
paths["database_FRESNA"] = root + '01 Raw inputs' + fs + 'Power plants and storage' + fs + 'EU_Powerplants' + fs + \
                    'FRESNA2' + fs + 'Matched_CARMA_ENTSOE_GEO_OPSD_WRI_reduced.csv'

# Grid
paths["grid"] = root + '01 Raw inputs' + fs + 'Grid' + fs + 'gridkit_europe' + fs + \
                'gridkit_europe-highvoltage-links.csv'
# paths["grid"] = root + '01 Raw inputs' + fs + 'Grid' + fs + 'gridkit_US' + fs + \
#                 'gridkit_north_america-highvoltage-links.csv'


# ## Renewable Capacities
# Rasters for wind and solar
timestamp = '20190502 Referenzszenario'
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

##################################
#     General Ouputs Folders     #
##################################

pathtemp = root + '02 Intermediate files' + fs + 'Files ' + region + fs
# 02 - load
paths["load"] = pathtemp + 'Load' + fs
paths["model_regions"] = pathtemp + model_regions + fs
paths["sites"] = pathtemp + model_regions + fs + 'Sites.csv'
paths["load_EU"] = pathtemp + 'Load' + fs + 'Load_EU' + '%04d' % (param["year"]) + '.csv'

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
paths["urbs_sites"] = paths["urbs"] + 'Site_urbs' + ' %04d' % (param["year"]) + '.csv'
paths["urbs_demand"] = paths["urbs"] + 'Demand_urbs' + ' %04d' % (param["year"]) + '.csv'
paths["urbs_commodities"] = paths["urbs"] + 'Commodities_urbs' + ' %04d' % (param["year"]) + '.csv'
paths["urbs_process"] = paths["urbs"] + 'Process_urbs' + ' %04d' % (param["year"]) + '.csv'
paths["urbs_storage"] = paths["urbs"] + 'Storage_urbs' + ' %04d' % (param["year"]) + '.csv'
paths["urbs_transmission"] = paths["urbs"] + 'Transmission_urbs' + ' %04d' % (param["year"]) + '.csv'
paths["urbs_model"] = paths["urbs"] + 'urbs_' + \
                      str(param["region"]) + '_' + str(param["model_regions"]) + '_' + str(param["year"]) + '.xlsx'

# evrys
paths["evrys"] = paths["model_regions"] + "evrys" + fs
paths["evrys_sites"] = paths["evrys"] + 'Sites_evrys' + ' %04d' % (param["year"]) + '.csv'
paths["evrys_demand"] = paths["evrys"] + 'Demand_evrys' + '%04d' % (param["year"]) + '.csv'
paths["evrys_commodities"] = paths["evrys"] + 'Commodities_evrys' + ' %04d' % (param["year"]) + '.csv'
paths["evrys_process"] = paths["evrys"] + 'Process_evrys' + ' %04d' % (param["year"]) + '.csv'
paths["evrys_storage"] = paths["evrys"] + 'Storage_evrys' + ' %04d' % (param["year"]) + '.csv'
paths["evrys_transmission"] = paths["evrys"] + 'Transmission_evrys' + ' %04d' % (param["year"]) + '.csv'
paths["evrys_model"] = paths["evrys"] + 'evrys_' + \
                       str(param["region"]) + '_' + str(param["model_regions"]) + '_' + str(param["year"]) + '.xlsx'

if not os.path.isdir(paths["urbs"]):
    os.mkdir(paths["urbs"])
if not os.path.isdir(paths["evrys"]):
    os.mkdir(paths["evrys"])

del root, PathTemp, fs
