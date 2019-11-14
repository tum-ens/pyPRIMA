# from osgeo import gdal, ogr, gdalnumeric
import pandas as pd
# from pandas import ExcelWriter
# import fiona
import geopandas as gpd
import numpy as np
from shapely import geometry
from shapely.geometry import Polygon, Point  # , mapping
import shapefile as shp
import pysal as ps
from geopy import distance
import sys
import datetime
import inspect
import os
# import glob
# import shutil
import math
import rasterio
from rasterio import MemoryFile, mask, windows
# from scipy.ndimage import convolve
# import osr
import json

import warnings
from warnings import warn

# gdal.PushErrorHandler('CPLQuietErrorHandler')
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def get_sectoral_profiles(paths, param):
    """
    This function reads the raw standard load profiles, repeats them to obtain a full year, normalizes them so that the
    sum is equal to 1, and stores the obtained load profile for each sector in the dataframe *profiles*.

    :param paths: Dictionary containing the paths to *dict_daytype*, *dict_season*, and to the raw standard load profiles.
    :type paths: dict
    :param param: Dictionary containing the *year* and load-related assumptions.
    :type param: dict

    :return profiles: The normalized load profiles for the sectors.
    :rtype: pandas dataframe
    """
    timecheck("Start")
    profiles_paths = paths["cleaned_profiles"]

    # Prepare the dataframe for the yearly load per sector
    profiles = pd.DataFrame(columns=param["load"]["sectors"])

    # Residential load
    if "RES" in param["load"]["sectors"]:
        profiles["RES"] = pd.read_csv(profiles_paths["RES"], sep=";", decimal=",", header=[0], index=[0]).to_numpy()

    # Industrial load
    if "IND" in param["load"]["sectors"]:
        profiles["IND"] = pd.read_csv(profiles_paths["IND"], sep=";", decimal=",", header=[0], index=[0]).to_numpy()

    # Commercial load
    if "COM" in param["load"]["sectors"]:
        profiles["COM"] = pd.read_csv(profiles_paths["COM"], sep=";", decimal=",", header=[0], index=[0]).to_numpy()

    # Agricultural load
    if "AGR" in param["load"]["sectors"]:
        profiles["AGR"] = pd.read_csv(profiles_paths["AGR"], sep=";", decimal=",", header=[0], index=[0]).to_numpy()

    # Street lights
    if "STR" in param["load"]["sectors"]:
        profiles["STR"] = pd.read_csv(profiles_paths["STR"], sep=";", decimal=",", header=[0], index=[0]).to_numpy()
    timecheck("End")
    return profiles


def timecheck(*args):
    """
    This function prints information about the progress of the script by displaying the function currently running, and optionally
    an input message, with a corresponding timestamp. If more than one argument is passed to the function, it will raise an exception.
    
    :param args: Message to be displayed with the function name and the timestamp (optional).
    :type args: string
    
    :return: The time stamp is printed.
    :rtype: None
    :raise: Too many arguments have been passed to the function, the maximum is only one string.
    """
    if len(args) == 0:
        print(inspect.stack()[1].function + str(datetime.datetime.now().strftime(": %H:%M:%S:%f")) + "\n")

    elif len(args) == 1:
        print(inspect.stack()[1].function + " - " + str(args[0]) + str(datetime.datetime.now().strftime(": %H:%M:%S:%f")) + "\n")

    else:
        raise Exception("Too many arguments have been passed.\nExpected: zero or one \nPassed: " + format(len(args)))


def display_progress(message, progress_stat):
    """
    This function displays a progress bar for long computations. To be used as part of a loop or with multiprocessing.
    
    :param message: Message to be displayed with the progress bar.
    :type message: string
    :param progress_stat: Tuple containing the total length of the calculation and the current status or progress.
    :type progress_stat: tuple(int, int)
    
    :return: The status bar is printed.
    :rtype: None
    """
    length = progress_stat[0]
    status = progress_stat[1]
    sys.stdout.write("\r")
    sys.stdout.write(message + " " + "[%-50s] %d%%" % ("=" * ((status * 50) // length), (status * 100) // length))
    sys.stdout.flush()
    if status == length:
        print("\n")


def reverse_lines(df):
    """
    This function reverses the line direction if the starting point is alphabetically
    after the end point.

    :param df: Dataframe with columns 'Region_start' and 'Region_end'.
    :type df: pandas dataframe

    :returns df_final: The same dataframe after the line direction has been reversed.
    :rtype: pandas dataframe
    """
    for idx in df.index:
        if df.Region_start[idx] > df.Region_end[idx]:
            df.loc[idx, "Region_start"], df.loc[idx, "Region_end"] = df.loc[idx, "Region_end"], df.loc[idx, "Region_start"]
    df_final = df

    return df_final


def expand_dataframe(df, column_names):
    """
    This function reads a dataframe where columns with known *column_names* have multiple values separated by a
    semicolon in each entry. It expands the dataframe by creating a row for each value in each of these columns.
    
    :param df: The original dataframe, with multiple values in some entries.
    :type df: pandas dataframe
    :param column_names: Names of columns where multiple values have to be separated.
    :type column_names: list
    
    :return df_final: The expanded dataframe, where each row contains only one value per column.
    :rtype: pandas dataframe
    """

    list_col = list(df.columns)
    df_dict = {}
    n_col = 0

    for col in column_names:
        df_dict[col] = df[col].str.split(";").apply(pd.Series)
        n_col = max(n_col, len(df_dict[col].columns))
        for cc in range(len(df_dict[col].columns), n_col):
            df_dict[col][cc] = np.nan
        list_col.remove(col)

    # Concatenate expanded columns into a dataframe of tuples
    df_concat = pd.concat(df_dict.values()).groupby(level=0).apply(lambda x: tuple(map(tuple, x.values.T)))
    df_concat = pd.DataFrame(list(df_concat))

    # Merge with original dataframe
    df_merged = (
        df_concat.merge(df, left_index=True, right_index=True)
        .drop(column_names, axis=1)
        .melt(id_vars=list_col, value_name="Combi")
        .drop(["variable"], axis=1)
    )

    # Replace column of tuples with individual columns
    df_final = df_merged.copy()
    df_final[column_names] = pd.DataFrame(df_merged["Combi"].tolist(), index=df_merged.index)
    df_final = df_final.drop(["Combi"], axis=1)

    # Columns should contain floats
    df_final[column_names] = df_final[column_names].astype(float)

    return df_final


# def add_suffix(df, suffix):
# # Check whether there is only one copy of the initial row, or more
# if str(df.index_old.iloc[1]).find('_') > 0:  # There are more than one copy of the row
# # Increment the suffix and replace the old one
# suffix = suffix + 1
# df.index_old.iloc[1] = df.index_old.iloc[1].replace('_' + str(suffix - 1), '_' + str(suffix))
# else:  # No other copy has been created so far
# # Reinitialize the suffix and concatenate it at the end of the old index
# suffix = 1
# df.index_old.iloc[1] = str(df.index_old.iloc[1]) + '_' + str(suffix)
# return (df, suffix)


def assign_values_based_on_series(series, dict):
    """
    This function fills a series based on the values of another series and a dictionary.
    The dictionary does not have to be sorted, it will be sorted before assigning the values.
    However, it must contain a key that is greater than any value in the series.
    It is equivalent to a function that maps ranges to discrete values.
    
    :param series: Series with input values that will be mapped.
    :type series: pandas series
    :param dict: Dictionary defining the limits of the ranges that will be mapped.
    :type dict: dictionary
    
    :return result: Series with the mapped discrete values.
    :rtype: pandas series
    """
    dict_sorted = sorted(list(dict.keys()), reverse=True)

    result = series.copy()
    for key in dict_sorted:
        result[series <= key] = dict[key]

    return result


# def read_assumptions_process(assumptions):
# process = {
# "cap_lo": dict(zip(assumptions['Process'], assumptions['cap-lo'].astype(float))),
# "cap_up": dict(zip(assumptions['Process'], assumptions['cap-up'].astype(float))),
# "max_grad": dict(zip(assumptions['Process'], assumptions['max-grad'].astype(float))),
# "min_fraction": dict(zip(assumptions['Process'], assumptions['min-fraction'].astype(float))),
# "inv_cost": dict(zip(assumptions['Process'], assumptions['inv-cost'].astype(float))),
# "fix_cost": dict(zip(assumptions['Process'], assumptions['fix-cost'].astype(float))),
# "var_cost": dict(zip(assumptions['Process'], assumptions['var-cost'].astype(float))),
# "startup_cost": dict(zip(assumptions['Process'], assumptions['startup-cost'].astype(float))),
# "depreciation": dict(zip(assumptions['Process'], assumptions['depreciation'].astype(float))),
# "area_per_cap": dict(zip(assumptions['Process'], assumptions['area-per-cap'].astype(float))),
# "year_my": dict(zip(assumptions['Process'], assumptions['year_mu'].astype(float))),
# "eff": dict(zip(assumptions['Process'], assumptions['eff'].astype(float))),
# "effmin": dict(zip(assumptions['Process'], assumptions['effmin'].astype(float))),
# "act_up": dict(zip(assumptions['Process'], assumptions['act-up'].astype(float))),
# "act_lo": dict(zip(assumptions['Process'], assumptions['act-lo'].astype(float))),
# "on_off": dict(zip(assumptions['Process'], assumptions['on-off'].astype(float))),
# "start_cost": dict(zip(assumptions['Process'], assumptions['start-cost'].astype(float))),
# "reserve_cost": dict(zip(assumptions['Process'], assumptions['reserve-cost'].astype(float))),
# "ru": dict(zip(assumptions['Process'], assumptions['ru'].astype(float))),
# "rd": dict(zip(assumptions['Process'], assumptions['rd'].astype(float))),
# "rumax": dict(zip(assumptions['Process'], assumptions['rumax'].astype(float))),
# "rdmax": dict(zip(assumptions['Process'], assumptions['rdmax'].astype(float))),
# "cotwo": dict(zip(assumptions['Process'], assumptions['cotwo'].astype(float))),
# "detail": dict(zip(assumptions['Process'], assumptions['detail'].astype(float))),
# "lambda_": dict(zip(assumptions['Process'], assumptions['lambda'].astype(float))),
# "heatmax": dict(zip(assumptions['Process'], assumptions['heatmax'].astype(float))),
# "maxdeltaT": dict(zip(assumptions['Process'], assumptions['maxdeltaT'].astype(float))),
# "heatupcost": dict(zip(assumptions['Process'], assumptions['heatupcost'].astype(float))),
# "su": dict(zip(assumptions['Process'], assumptions['su'].astype(float))),
# "sd": dict(zip(assumptions['Process'], assumptions['sd'].astype(float))),
# "pdt": dict(zip(assumptions['Process'], assumptions['pdt'].astype(float))),
# "hotstart": dict(zip(assumptions['Process'], assumptions['hotstart'].astype(float))),
# "pot": dict(zip(assumptions['Process'], assumptions['pot'].astype(float))),
# "pretemp": dict(zip(assumptions['Process'], assumptions['pretemp'].astype(float))),
# "preheat": dict(zip(assumptions['Process'], assumptions['preheat'].astype(float))),
# "prestate": dict(zip(assumptions['Process'], assumptions['prestate'].astype(float))),
# "year_mu": dict(zip(assumptions['Process'], assumptions['year_mu'].astype(float))),
# "year_stdev": dict(zip(assumptions['Process'], assumptions['year_stdev'].astype(float)))
# }

# return process


# def read_assumptions_storage(assumptions):
# storage = {
# "cap_lo_c": dict(zip(assumptions['Storage'], assumptions['cap-lo-c'].astype(float))),
# "cap_lo_p": dict(zip(assumptions['Storage'], assumptions['cap-lo-p'].astype(float))),
# "cap_up_c": dict(zip(assumptions['Storage'], assumptions['cap-up-c'].astype(float))),
# "cap_up_p": dict(zip(assumptions['Storage'], assumptions['cap-up-p'].astype(float))),
# "inv_cost_c": dict(zip(assumptions['Storage'], assumptions['inv-cost-c'].astype(float))),
# "fix_cost_c": dict(zip(assumptions['Storage'], assumptions['fix-cost-c'].astype(float))),
# "var_cost_c": dict(zip(assumptions['Storage'], assumptions['var-cost-c'].astype(float))),
# "inv_cost_p": dict(zip(assumptions['Storage'], assumptions['inv-cost-p'].astype(float))),
# "fix_cost_p": dict(zip(assumptions['Storage'], assumptions['fix-cost-p'].astype(float))),
# "var_cost_p": dict(zip(assumptions['Storage'], assumptions['var-cost-p'].astype(float))),
# "depreciation": dict(zip(assumptions['Storage'], assumptions['depreciation'].astype(float))),
# "init": dict(zip(assumptions['Storage'], assumptions['init'].astype(float))),
# "eff_in": dict(zip(assumptions['Storage'], assumptions['eff-in'].astype(float))),
# "eff_out": dict(zip(assumptions['Storage'], assumptions['eff-out'].astype(float))),
# "var_cost_pi": dict(zip(assumptions['Storage'], assumptions['var-cost-pi'].astype(float))),
# "var_cost_po": dict(zip(assumptions['Storage'], assumptions['var-cost-po'].astype(float))),
# "act_up_pi": dict(zip(assumptions['Storage'], assumptions['act-up-pi'].astype(float))),
# "act_lo_pi": dict(zip(assumptions['Storage'], assumptions['act-lo-pi'].astype(float))),
# "act_up_po": dict(zip(assumptions['Storage'], assumptions['act-up-po'].astype(float))),
# "act_lo_po": dict(zip(assumptions['Storage'], assumptions['act-lo-po'].astype(float))),
# "act_lo_c": dict(zip(assumptions['Storage'], assumptions['act-lo-c'].astype(float))),
# "act_up_c": dict(zip(assumptions['Storage'], assumptions['act-up-c'].astype(float))),
# "prepowin": dict(zip(assumptions['Storage'], assumptions['prepowin'].astype(float))),
# "prepowout": dict(zip(assumptions['Storage'], assumptions['prepowout'].astype(float))),
# "ru": dict(zip(assumptions['Storage'], assumptions['ru'].astype(float))),
# "rd": dict(zip(assumptions['Storage'], assumptions['rd'].astype(float))),
# "rumax": dict(zip(assumptions['Storage'], assumptions['rumax'].astype(float))),
# "rdmax": dict(zip(assumptions['Storage'], assumptions['rdmax'].astype(float))),
# "seasonal": dict(zip(assumptions['Storage'], assumptions['seasonal'].astype(float))),
# "ctr": dict(zip(assumptions['Storage'], assumptions['ctr'].astype(float))),
# "year_mu": dict(zip(assumptions['Storage'], assumptions['year_mu'].astype(float))),
# "year_stdev": dict(zip(assumptions['Storage'], assumptions['year_stdev'].astype(float))),
# "discharge": dict(zip(assumptions['Storage'], assumptions['discharge'].astype(float)))
# }

# return storage


def changem(A, newval, oldval):
    """
    This function replaces existing values *oldval* in a data array *A* by new values *newval*.
    
    *oldval* and *newval* must have the same size.

    :param A: Input matrix.
    :type A: numpy array
    :param newval: Vector of new values to be set.
    :type newval: numpy array
    :param oldval: Vector of old values to be replaced.
    :type oldval: numpy array

    :return Out: The updated array.
    :rtype: numpy array
    """
    Out = np.zeros(A.shape)
    z = np.array((oldval, newval)).T
    for i, j in z:
        np.place(Out, A == i, j)
    return Out


def create_json(filepath, param, param_keys, paths, paths_keys):
    """
    This function creates a metadata JSON file containing information about the file in filepath by storing the relevant keys from
    both the param and path dictionaries.
    
    :param filepath: Path to the file for which the JSON file will be created.
    :type filepath: string
    :param param: Dictionary of dictionaries containing the user input parameters and intermediate outputs.
    :type param: dict
    :param param_keys: Keys of the parameters to be extracted from the *param* dictionary and saved into the JSON file.
    :type param_keys: list of strings
    :param paths: Dictionary of dictionaries containing the paths for all files.
    :type paths: dict
    :param paths_keys: Keys of the paths to be extracted from the *paths* dictionary and saved into the JSON file.
    :type paths_keys: list of strings
    
    :return: The JSON file will be saved in the desired path *filepath*.
    :rtype: None
    """
    new_file = os.path.splitext(filepath)[0] + ".json"
    new_dict = {}
    # Add standard keys
    param_keys = param_keys + ["author", "comment"]
    for key in param_keys:
        new_dict[key] = param[key]
        if type(param[key]) == np.ndarray:
            new_dict[key] = param[key].tolist()
        if type(param[key]) == tuple:
            param[key] = list(param[key])
            c = 0
            for e in param[key]:
                if type(e) == np.ndarray:
                    new_dict[key][c] = e.tolist()
                c += 1
        if type(param[key]) == dict:
            for k, v in param[key].items():
                if type(v) == np.ndarray:
                    new_dict[key][k] = v.tolist()
                if type(v) == tuple:
                    param[key][k] = list(param[key][k])
                    c = 0
                    for e in param[key][k]:
                        if type(e) == np.ndarray:
                            new_dict[key][k][c] = e.tolist()
                        c += 1
                if type(v) == dict:
                    for k2, v2 in v.items():
                        if type(v2) == np.ndarray:
                            new_dict[key][k][k2] = v2.tolist()
                        if type(v2) == tuple:
                            param[key][k][k2] = list(param[key][k][k2])
                            c = 0
                            for e in param[key][k][k2]:
                                if type(e) == np.ndarray:
                                    new_dict[key][k][k2][c] = e.tolist()
                                c += 1
                        if type(v2) == dict:
                            for k3, v3 in v.items():
                                if type(v3) == np.ndarray:
                                    new_dict[key][k][k2][k3] = v3.tolist()
                                if type(v3) == tuple:
                                    param[key][k][k2][k3] = list(param[key][k][k2][k3])
                                    c = 0
                                    for e in param[key][k][k2][k3]:
                                        if type(e) == np.ndarray:
                                            new_dict[key][k][k2][k3][c] = e.tolist()
                                        c += 1

    for key in paths_keys:
        new_dict[key] = paths[key]
    # Add timestamp
    new_dict["timestamp"] = str(datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
    # Add caller function's name
    new_dict["function"] = inspect.stack()[1][3]
    with open(new_file, "w") as json_file:
        json.dump(new_dict, json_file)
    print("File saved: " + new_file)
