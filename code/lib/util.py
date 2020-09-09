import sys
import os
import warnings
from warnings import warn
import inspect
import datetime
import math
import numpy as np
from numpy.matlib import repmat, reshape
import pandas as pd
from osgeo import gdal, ogr, osr, gdal_array
import rasterio
from rasterio import MemoryFile, mask, windows
from geopy import distance
import shapefile as shp
import pysal as ps
from shapely import geometry
from shapely.geometry import Polygon, Point
import geopandas as gpd
import re
import json

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
    
    profiles = pd.DataFrame(columns=paths["profiles"].keys())
    #import pdb; pdb.set_trace()
    
    for key in paths["profiles"].keys():
        profiles[key] = pd.read_csv(profiles_paths[key], sep=";", decimal=",", header=[0], index_col=[0])[key]

    timecheck("End")
    return profiles


def resizem(A_in, row_new, col_new):
    """
    This function resizes regular data grid, by copying and pasting parts of the original array.

    :param A_in: Input matrix.
    :type A_in: numpy array
    :param row_new: New number of rows.
    :type row_new: integer
    :param col_new: New number of columns.
    :type col_new: integer

    :return A_out: Resized matrix.
    :rtype: numpy array
    """
    row_rep = row_new // np.shape(A_in)[0]
    col_rep = col_new // np.shape(A_in)[1]
    A_inf = A_in.flatten(order="F")[np.newaxis]
    A_out = reshape(
        repmat(
            reshape(reshape(repmat((A_in.flatten(order="F")[np.newaxis]), row_rep, 1), (row_new, -1), order="F").T, (-1, 1), order="F"), 1, col_rep
        ).T,
        (col_new, row_new),
        order="F",
    ).T

    return A_out


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


def field_exists(field_name, shp_path):
    """
    This function returns whether the specified field exists or not in the shapefile linked by a path.

    :param field_name: Name of the field to be checked for.
    :type field_name: str
    :param shp_path: Path to the shapefile.
    :type shp_path: str

    :return: ``True`` if it exists or ``False`` if it doesn't exist.
    :rtype: bool
    """
    shp = ogr.Open(shp_path, 0)
    lyr = shp.GetLayer()
    lyr_dfn = lyr.GetLayerDefn()

    exists = False
    for i in range(lyr_dfn.GetFieldCount()):
        exists = exists or (field_name == lyr_dfn.GetFieldDefn(i).GetName())
    return exists


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
        if key != "inf":
            result[series <= key] = dict[key]
        else:
            result[series] = dict[key]

    return result


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
