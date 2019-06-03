# import datetime
# import numpy as np
import os
from sys import platform

###########################
#### User preferences #####
###########################

param = {}
param["region"] = 'Europe'
param["model_regions"] = 'NUTS0_wo_Balkans'
param["year"] = 2015

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
        "missing_countries": {'AL': 6376000, 'KS': 2887000, 'UK': 6376000, 'EL': 2887000},  # in MWh
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
region = param["region"]
model_regions = param["model_regions"]

paths = {}

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

# Ouput Folders
paths["load"] = root + "02 Intermediate files" + fs + "Files " + region + fs + "Load" + fs + "load_Files.hdf"
paths["model_regions"] = root + "02 Intermediate files" + fs + "Files " + region + fs + model_regions + fs
paths["urbs"] = paths["model_regions"] + "urbs" + fs
paths["evrys"] = paths["model_regions"] + "evrys" + fs
paths["load_EU"] = root + "02 Intermediate files" + fs + "Files " + region + fs + "Load" + fs + 'Load_EU' + '%04d' % (param["year"]) + '.csv'
paths["df_evrys"] = paths["evrys"] + 'demand_evrys' + '%04d' % (param["year"]) + '.csv'
paths["df_urbs"] = paths["urbs"] + 'demand_urbs' + '%04d' % (param["year"]) + '.csv'

if not os.path.isdir(paths["urbs"]):
    os.mkdir(paths["urbs"])
if not os.path.isdir(paths["evrys"]):
    os.mkdir(paths["evrys"])
# if technology == "Wind":
# paths["OUT"] = root + "OUTPUT" + fs + region + fs + str(turbine["hub_height"]) + "m_" + str(correction) + "corr_" + timestamp
# else:
# paths["OUT"] = root + "OUTPUT" + fs + region + fs + str(pv["tracking"]) + "axis_" + timestamp


del root, PathTemp, fs
