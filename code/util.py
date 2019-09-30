from osgeo import gdal, ogr, gdalnumeric
import pandas as pd
from pandas import ExcelWriter
import geopandas as gpd
import numpy as np
from shapely import geometry
from shapely.geometry import Point, Polygon
import shapefile as shp
import pysal as ps
from geopy import distance
import sys
import datetime
import inspect
import os
import glob
import shutil
import math
import rasterio
from rasterio import windows, mask, MemoryFile
from scipy.ndimage import convolve
import osr
import json
import warnings
from warnings import warn
gdal.PushErrorHandler('CPLQuietErrorHandler')
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def char_range(c1, c2):
    """
    This function creates a generator to iterate between the characters *c1* and *c2*, including the latter.
    
    :param c1: First character in the iteration.
    :type c1: char
    :param c2: Last character in the iteration (included).
    :type c2: char
    :return: Generator to iterate between the characters *c1* and *c2*.
    :rtype: python generator
    """
    for c in range(ord(c1), ord(c2) + 1):
        yield chr(c)


def timecheck(*args):
    """
    Print information about the progress of the script by displaying the function currently running, and optionally
    an input message, with a corresponding timestamp.

    :param args: Message to be displayed with the function name and the timestamp.
    :type args: string (``optional``)
    :return: None
    """
    if len(args) == 0:
        print(inspect.stack()[1].function + str(datetime.datetime.now().strftime(": %H:%M:%S:%f")))

    elif len(args) == 1:
        print(inspect.stack()[1].function + ' - ' + str(args[0])
              + str(datetime.datetime.now().strftime(": %H:%M:%S:%f")))

    else:
        raise Exception('Too many arguments have been passed.\nExpected: zero or one \nPassed: ' + format(len(args)))


def display_progress(message, progress_stat):
    length = progress_stat[0]
    status = progress_stat[1]
    sys.stdout.write('\r')
    sys.stdout.write(message + ' ' + '[%-50s] %d%%' % ('=' * ((status * 50) // length), (status * 100) // length))
    sys.stdout.flush()
    if status == length:
        print('\n')