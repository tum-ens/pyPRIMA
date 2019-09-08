from osgeo import gdal, ogr, gdalnumeric
from osgeo.gdalconst import GA_ReadOnly
import pandas as pd
from pandas import ExcelWriter
import geopandas as gpd
import numpy as np
from shapely import geometry
from shapely.geometry import Point
import shapefile as shp
import pysal as ps
from geopy import distance
import sys
import datetime
import inspect
import os
import glob
import shutil
from scipy.ndimage import convolve



gdal.PushErrorHandler('CPLQuietErrorHandler')


def clean_load_data(paths, param, countries):
    """

    :param paths:
    :param param:
    :param countries:
    :return:
    """
    timecheck('Start')

    # Read country load timeseries
    df_raw = pd.read_excel(paths["load_ts"], header=0, skiprows=[0, 1, 2], sep=',', decimal='.')

    # Filter by year
    df_year = df_raw.loc[df_raw['Year'] == param["year"]]

    # Scale based on coverage ratio
    df_scaled = df_year.copy()
    a = df_year.iloc[:, 5:].values
    b = df_year.iloc[:, 4].values
    c = a / b[:, np.newaxis] * 100
    df_scaled.iloc[:, 5:] = c
    del a, b, c

    # Reshape so that rows correspond to hours and columns to countries
    data = np.reshape(df_scaled.iloc[:, 5:].values.T, (-1, len(df_scaled['Country'].unique())), order='F')
    # Create dataframe where rows correspond to hours and columns to countries
    df_reshaped = pd.DataFrame(data, index=np.arange(data.shape[0]), columns=df_scaled['Country'].unique())

    # Rename countries
    df_renamed = df_reshaped.T.rename(index=param["load"]["dict_countries"])
    df_renamed = df_renamed.reset_index().rename(columns={'index': 'Country'})
    df_renamed = df_renamed.groupby(['Country']).sum()
    df_renamed.reset_index(inplace=True)

    # Reshape Renamed_df
    df_reshaped_renamed = pd.DataFrame(df_renamed.loc[:, df_renamed.columns != 'Country'].T.to_numpy(),
                                       columns=df_renamed['Country'])

    # Create time series for missing countries
    df_completed = df_reshaped_renamed.copy()
    missing_countries = param["load"]["missing_countries"]
    replacement = param["load"]["replacement"]
    for i in missing_countries.keys():
        df_completed.loc[:, i] = df_completed.loc[:, replacement] / df_completed[replacement].sum() * missing_countries[
            i]

    # Select only countries needed

    df_filtered = df_completed[countries['Country'].unique()]

    # Fill missing data by values from the day before, adjusted based on the trend of the previous five hours
    df_filled = df_filtered.copy()
    for i, j in np.argwhere(np.isnan(df_filtered.values)):
        df_filled.iloc[i, j] = df_filled.iloc[i - 5:i, j].sum() / df_filled.iloc[i - 5 - 24:i - 24, j].sum() * \
                               df_filled.iloc[i - 24, j].sum()

    timecheck('End')
    return df_filled


def get_sectoral_profiles(paths, param):
    '''
    Read and store the load profile for each sector in the table 'profiles'.
    '''
    timecheck('Start')
    dict_daytype = param["load"]["dict_daytype"]
    dict_season = param["load"]["dict_season"]

    # Prepare the dataframe for the daily load:
    start = datetime.datetime(param["year"], 1, 1)
    end = datetime.datetime(param["year"], 12, 31)
    hours = [str(x) for x in list(range(0, 24))]
    time_series = pd.DataFrame(data=np.zeros((365, 27)), index=None, columns=['Date', 'Day', 'Season'] + hours)
    time_series['Date'] = pd.date_range(start, end)
    time_series['Day'] = [dict_daytype[time_series.loc[i, 'Date'].day_name()] for i in time_series.index]
    time_series['Season'] = [dict_season[time_series.loc[i, 'Date'].month] for i in time_series.index]

    # Residential load
    if param["Region"] == "California":
        residential_profile_raw = pd.read_excel(paths["profiles_ca"]["RES"], header=[0])
        residential_profile = residential_profile_raw.ilco[:365, 2:].values
        residential_profile = np.reshape(residential_profile, (8760, 1))

    else:
        residential_profile_raw = pd.read_excel(paths["profiles"]["RES"], header=[3, 4], skipinitialspace=True)
        residential_profile_raw.rename(columns={'Übergangszeit': 'Spring/Fall', 'Sommer': 'Summer',
                                                'Werktag': 'Working day', 'Sonntag/Feiertag': 'Sunday',
                                                'Samstag': 'Saturday'}, inplace=True)
        residential_profile = time_series.copy()
        for i in residential_profile.index:
            residential_profile.loc[i, hours] = list(
                residential_profile_raw[(residential_profile.loc[i, 'Season'], residential_profile.loc[i, 'Day'])])
        # Reshape the hourly load in one vector, where the rows are the hours of the year
        residential_profile = np.reshape(residential_profile.loc[:, hours].values, -1, order='C')
    profiles = pd.DataFrame(residential_profile / residential_profile.sum(), columns=['RES'])

    # Industrial load
    if 'IND' in param["load"]["sectors"]:
        if param["Region"] == "California":
            industrial_profile_raw = pd.read_excel(paths["profiles_ca"]["IND"], header=0)
            industrial_profile = industrial_profile_raw.iloc[:365, 2:].values
            industrial_profile = np.reshape(industrial_profile, (8760, 1))
        else:
            industrial_profile_raw = pd.read_excel(paths["profiles"]["IND"], header=0)
            industrial_profile_raw.rename(columns={'Stunde': 'Hour', 'Last': 'Load'}, inplace=True)
            # Reshape the hourly load in one vector, where the rows are the hours of the year
            industrial_profile = np.tile(industrial_profile_raw['Load'].values, 365)
        profiles['IND'] = industrial_profile / industrial_profile.sum()

    # Commercial load
    if 'COM' in param["load"]["sectors"]:
        if param["Region"] == "California":
            commercial_profile_raw = pd.read_excel(paths["profiles_ca"]["COM"], header=0)
            commercial_profile = commercial_profile_raw.iloc[:365, 2:].values
            commercial_profile = np.reshape(commercial_profile, (8760, 1))
        else:
            commercial_profile_raw = pd.read_csv(paths["profiles"]["COM"], sep='[;]', engine='python', decimal=',',
                                                 skiprows=[0, 99], header=[0, 1],
                                                 skipinitialspace=True)
            # commercial_profile_raw.rename(columns={'Übergangszeit': 'Spring/Fall', 'Sommer': 'Summer',
            #                                     'Werktag': 'Working day', 'Sonntag': 'Sunday', 'Samstag': 'Saturday'},
            #                               inplace=True)
            commercial_profile_raw.rename(columns={'Ãœbergangszeit': 'Spring/Fall', 'Sommer': 'Summer',
                                                   'Werktag': 'Working day', 'Sonntag': 'Sunday',
                                                   'Samstag': 'Saturday'},
                                          inplace=True)
            # Aggregate from 15 min --> hourly load
            commercial_profile_raw[('Hour', 'All')] = [int(str(commercial_profile_raw.loc[i, ('G0', '[W]')])[:2])
                                                       for i in commercial_profile_raw.index]
            commercial_profile_raw = commercial_profile_raw.groupby([('Hour', 'All')]).sum()
            commercial_profile_raw.reset_index(inplace=True)
            commercial_profile = time_series.copy()
            for i in commercial_profile.index:
                commercial_profile.loc[i, hours] = list(
                    commercial_profile_raw[(commercial_profile.loc[i, 'Season'], commercial_profile.loc[i, 'Day'])])
            # Reshape the hourly load in one vector, where the rows are the hours of the year
            commercial_profile = np.reshape(commercial_profile.loc[:, hours].values, -1, order='C')
        profiles['COM'] = commercial_profile / commercial_profile.sum()

    # Agricultural load
    if 'AGR' in param["load"]["sectors"]:
        agricultural_profile_raw = pd.read_csv(paths["profiles"]["AGR"], sep='[;]', engine='python', decimal=',',
                                               skiprows=[0, 99], header=[0, 1],
                                               skipinitialspace=True)
        # agricultural_profile_raw.rename(columns={'Übergangszeit': 'Spring/Fall', 'Sommer': 'Summer',
        #                                         'Werktag': 'Working day', 'Sonntag': 'Sunday', 'Samstag': 'Saturday'},
        #                                 inplace=True)
        agricultural_profile_raw.rename(columns={'Ãœbergangszeit': 'Spring/Fall', 'Sommer': 'Summer',
                                                 'Werktag': 'Working day', 'Sonntag': 'Sunday', 'Samstag': 'Saturday'},
                                        inplace=True)
        # Aggregate from 15 min --> hourly load
        agricultural_profile_raw['Hour'] = [int(str(agricultural_profile_raw.loc[i, ('L0', '[W]')])[:2]) for i in
                                            agricultural_profile_raw.index]
        agricultural_profile_raw = agricultural_profile_raw.groupby(['Hour']).sum()
        agricultural_profile = time_series.copy()
        for i in agricultural_profile.index:
            agricultural_profile.loc[i, hours] = list(
                agricultural_profile_raw[(agricultural_profile.loc[i, 'Season'], agricultural_profile.loc[i, 'Day'])])
        # Reshape the hourly load in one vector, where the rows are the hours of the year
        agricultural_profile = np.reshape(agricultural_profile.loc[:, hours].values, -1, order='C')
        profiles['AGR'] = agricultural_profile / agricultural_profile.sum()

    # Street lights
    if 'STR' in param["load"]["sectors"]:
        streets_profile_raw = pd.read_excel(paths["profiles"]["STR"], header=[4], skipinitialspace=True,
                                            usecols=[0, 1, 2])
        # Aggregate from 15 min --> hourly load
        streets_profile_raw['Hour'] = [int(str(streets_profile_raw.loc[i, 'Uhrzeit'])[:2]) for i in
                                       streets_profile_raw.index]
        streets_profile_raw = streets_profile_raw.groupby(['Datum', 'Hour']).sum()
        streets_profile_raw.iloc[0] = streets_profile_raw.iloc[0] + streets_profile_raw.iloc[-1]
        streets_profile_raw = streets_profile_raw.iloc[:-1]
        # Reshape the hourly load in one vector, where the rows are the hours of the year
        streets_profile = streets_profile_raw.values
        # Normalize the load over the year, ei. integral over the year of all loads for each individual sector is 1
        profiles['STR'] = streets_profile / streets_profile.sum()

    timecheck('End')
    return profiles


def intersection_regions_countries(paths):
    '''
    description
    '''

    # load shapefiles, and create spatial indexes for both files
    border_region = gpd.GeoDataFrame.from_file(paths["SHP"])
    border_region['geometry'] = border_region.buffer(0)
    border_country = gpd.GeoDataFrame.from_file(paths["Countries"])
    data = []
    for index, region in border_region.iterrows():
        for index2, country in border_country.iterrows():
            if (region.Population > 0):
                if region['geometry'].intersects(country['geometry']):
                    data.append({'geometry': region['geometry'].intersection(country['geometry']),
                                 'NAME_SHORT': region['NAME_SHORT'] + '_' + country['NAME_SHORT']})

    # Clean data
    i = 0
    list_length = len(data)
    while i < list_length:
        if data[i]['geometry'].geom_type == 'Polygon':
            data[i]['geometry'] = geometry.multipolygon.MultiPolygon([data[i]['geometry']])
        if not (data[i]['geometry'].geom_type == 'Polygon' or data[i]['geometry'].geom_type == 'MultiPolygon'):
            del data[i]
            list_length = list_length - 1
        else:
            i = i + 1

    # Create GeoDataFrame
    intersection = gpd.GeoDataFrame(data, columns=['geometry', 'NAME_SHORT'])
    intersection.to_file(paths["model_regions"] + 'intersection.shp')


def bbox_to_pixel_offsets(gt, bbox):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width) + 1

    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height) + 1

    xsize = x2 - x1
    ysize = y2 - y1

    return x1, y1, xsize, ysize


def zonal_stats(vector_path, raster_path, raster_type, nodata_value=None, global_src_extent=False):
    """
    Zonal Statistics
    Vector-Raster Analysis
    
    Copyright 2013 Matthew Perry
    
    Usage:
      zonal_stats.py VECTOR RASTER
      zonal_stats.py -h | --help
      zonal_stats.py --version
    
    Options:
      -h --help     Show this screen.
      --version     Show version.
    """

    rds = gdal.Open(raster_path, GA_ReadOnly)
    assert (rds)
    rb = rds.GetRasterBand(1)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(vector_path, GA_ReadOnly)  # TODO maybe open update if we want to write stats
    assert (vds)
    vlyr = vds.GetLayer(0)

    # create an in-memory numpy array of the source raster data
    # covering the whole extent of the vector layer
    if global_src_extent:
        # use global source extent
        # useful only when disk IO or raster scanning inefficiencies are your limiting factor
        # advantage: reads raster data in one pass
        # disadvantage: large vector extents may have big memory requirements
        src_offset = bbox_to_pixel_offsets(rgt, vlyr.GetExtent())
        src_array = rb.ReadAsArray(*src_offset)

        # calculate new geotransform of the layer subset
        new_gt = (
            (rgt[0] + (src_offset[0] * rgt[1])),
            rgt[1],
            0.0,
            (rgt[3] + (src_offset[1] * rgt[5])),
            0.0,
            rgt[5]
        )

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors
    stats = []
    feat = vlyr.GetNextFeature()
    while feat is not None:

        if not global_src_extent:
            # use local source extent
            # fastest option when you have fast disks and well indexed raster (ie tiled Geotiff)
            # advantage: each feature uses the smallest raster chunk
            # disadvantage: lots of reads on the source raster
            src_offset = bbox_to_pixel_offsets(rgt, feat.geometry().GetEnvelope())
            src_array = rb.ReadAsArray(*src_offset)

            # calculate new geotransform of the feature subset
            new_gt = (
                (rgt[0] + (src_offset[0] * rgt[1])),
                rgt[1],
                0.0,
                (rgt[3] + (src_offset[1] * rgt[5])),
                0.0,
                rgt[5]
            )

        # Create a temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())

        # Rasterize it
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()

        # Mask the source data array with our current feature
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explicitly
        masked = np.ma.MaskedArray(
            src_array,
            mask=np.logical_or(
                src_array == nodata_value,
                np.logical_not(rv_array)
            )
        )

        if raster_type == 'landuse':
            unique, counts = np.unique(masked, return_counts=True)
            unique2 = [str(i) for i in unique.astype(int)]
            count = dict(zip(unique2, counts.astype(int)))
            feature_stats = {
                # 'sum': float(masked.sum()),
                'NAME_SHORT': str(feat.GetField('NAME_SHORT'))}
            feature_stats.update(count)
        elif raster_type == 'population':
            feature_stats = {
                # 'max': float(masked.max()),
                'sum': float(masked.sum()),
                # 'count': int(masked.count()),
                # 'fid': int(feat.GetFID()),
                'NAME_SHORT': str(feat.GetField('NAME_SHORT'))}
        elif raster_type == 'renewable':
            feature_stats = {
                'max': float(masked.max()),
                # 'sum': float(masked.sum()),
                # 'count': int(masked.count()),
                # 'fid': int(feat.GetFID()),
                'NAME_SHORT': str(feat.GetField('NAME_SHORT'))}

        stats.append(feature_stats)

        rvds = None
        mem_ds = None
        feat = vlyr.GetNextFeature()

    vds = None
    rds = None

    return stats


def zonal_weighting(paths, df_load, df_stat, s):
    shp_path = paths["Countries"]
    raster_path = paths["LU"]
    shp = ogr.Open(shp_path, 1)
    raster = gdal.Open(raster_path)
    lyr = shp.GetLayer()

    # Create memory target raster
    target_ds = gdal.GetDriverByName('GTiff').Create(paths["load"] + 'Europe_' + s + '_load_pax.tif',
                                                     raster.RasterXSize,
                                                     raster.RasterYSize,
                                                     1, gdal.GDT_Float32)
    target_ds.SetGeoTransform(raster.GetGeoTransform())
    target_ds.SetProjection(raster.GetProjection())

    # NoData value
    mem_band = target_ds.GetRasterBand(1)
    mem_band.Fill(0)
    mem_band.SetNoDataValue(0)

    # Add a new field
    if not field_exists('Weight_' + s, shp_path):
        new_field = ogr.FieldDefn('Weight_' + s, ogr.OFTReal)
        lyr.CreateField(new_field)

    for feat in lyr:
        country = feat.GetField('NAME_SHORT')[:2]
        if s == 'RES':
            feat.SetField('Weight_' + s, df_load[country, s] / df_stat.loc[country, 'RES'])
        else:
            feat.SetField('Weight_' + s, df_load[country, s] / df_stat.loc[country, s])
        lyr.SetFeature(feat)
        feat = None

    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr, None, None, [0], ['ALL_TOUCHED=FALSE', 'ATTRIBUTE=Weight_' + s[:3]])


def field_exists(field_name, shp_path):
    shp = ogr.Open(shp_path, 0)
    lyr = shp.GetLayer()
    lyr_dfn = lyr.GetLayerDefn()

    exists = False
    for i in range(lyr_dfn.GetFieldCount()):
        exists = exists or (field_name == lyr_dfn.GetFieldDefn(i).GetName())

    return exists


# 05a_Distribution_Renewable_powerplants

# ## Functions:

# https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#clip-a-geotiff-with-shapefile

def world2Pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line)


def rasclip(raster_path, shapefile_path, counter):
    # Load the source data as a gdalnumeric array
    srcArray = gdalnumeric.LoadFile(raster_path)

    # Also load as a gdal image to get geotransform
    # (world file) info
    srcImage = gdal.Open(raster_path)
    geoTrans = srcImage.GetGeoTransform()

    # Create an OGR layer from a boundary shapefile
    shapef = ogr.Open(shapefile_path)
    lyr = shapef.GetLayer(os.path.split(os.path.splitext(shapefile_path)[0])[1])

    # Filter based on FID
    lyr.SetAttributeFilter("FID = {}".format(counter))
    poly = lyr.GetNextFeature()

    # Convert the polygon extent to image pixel coordinates
    minX, maxX, minY, maxY = poly.GetGeometryRef().GetEnvelope()
    ulX, ulY = world2Pixel(geoTrans, minX, maxY)
    lrX, lrY = world2Pixel(geoTrans, maxX, minY)

    # Calculate the pixel size of the new image
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)

    clip = srcArray[ulY:lrY, ulX:lrX]

    # Create pixel offset to pass to new image Projection info
    xoffset = ulX
    yoffset = ulY
    # print("Xoffset, Yoffset = ( %f, %f )" % ( xoffset, yoffset ))

    # Create a second (modified) layer
    outdriver = ogr.GetDriverByName('MEMORY')
    source = outdriver.CreateDataSource('memData')
    # outdriver = ogr.GetDriverByName('ESRI Shapefile')
    # source = outdriver.CreateDataSource(mypath+'00 Inputs/maps/dummy.shp')
    lyr2 = source.CopyLayer(lyr, 'dummy', ['OVERWRITE=YES'])
    featureDefn = lyr2.GetLayerDefn()
    # create a new ogr geometry
    geom = poly.GetGeometryRef().Buffer(-1 / 240)
    # write the new feature
    newFeature = ogr.Feature(featureDefn)
    newFeature.SetGeometryDirectly(geom)
    lyr2.CreateFeature(newFeature)
    # here you can place layer.SyncToDisk() if you want
    newFeature.Destroy()
    # lyr2 = source.CopyLayer(lyr,'dummy',['OVERWRITE=YES'])
    lyr2.ResetReading()
    poly_old = lyr2.GetNextFeature()
    lyr2.DeleteFeature(poly_old.GetFID())

    # Create memory target raster
    target_ds = gdal.GetDriverByName('MEM').Create('', srcImage.RasterXSize, srcImage.RasterYSize, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geoTrans)
    target_ds.SetProjection(srcImage.GetProjection())

    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr2, None, None, [1], ['ALL_TOUCHED=FALSE'])
    mask = target_ds.ReadAsArray()
    mask = mask[ulY:lrY, ulX:lrX]

    # Clip the image using the mask
    clip = np.multiply(clip, mask).astype(gdalnumeric.float64)
    return poly.GetField('NAME_SHORT'), xoffset, yoffset, clip


def map_power_plants(p, x, y, c, outSHPfn):
    # Create the output shapefile
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outSHPfn):
        shpDriver.DeleteDataSource(outSHPfn)
    outDataSource = shpDriver.CreateDataSource(outSHPfn)
    outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint)

    # create point geometry
    point = ogr.Geometry(ogr.wkbPoint)
    # create a field
    idField = ogr.FieldDefn('CapacityMW', ogr.OFTReal)
    outLayer.CreateField(idField)
    # Create the feature
    featureDefn = outLayer.GetLayerDefn()

    # Set values
    for i in range(0, len(x)):
        point.AddPoint(x[i], y[i])
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(point)
        outFeature.SetField('CapacityMW', c[i])
        outLayer.CreateFeature(outFeature)
    outFeature = None
    print("File Saved: " + outSHPfn)


def map_grid_plants(x, y, paths):
    outSHPfn = paths["map_grid_plants"]

    # Create the output shapefile
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outSHPfn):
        shpDriver.DeleteDataSource(outSHPfn)
    outDataSource = shpDriver.CreateDataSource(outSHPfn)
    outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint)

    # create point geometry
    point = ogr.Geometry(ogr.wkbPoint)
    # Create the feature
    featureDefn = outLayer.GetLayerDefn()

    # Set values
    for i in range(0, len(x)):
        point.AddPoint(x[i], y[i])
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(point)
        outLayer.CreateFeature(outFeature)
    outFeature = None


def timecheck(*args):
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


def crd_merra(Crd_regions, res_weather):
    ''' description '''
    Crd = np.array([(np.ceil((Crd_regions[:, 0] - res_weather[0] / 2) / res_weather[0])
                     * res_weather[0] + res_weather[0] / 2),
                    (np.ceil((Crd_regions[:, 1] - res_weather[1] / 2) / res_weather[1])
                     * res_weather[1] + res_weather[1] / 2),
                    (np.floor((Crd_regions[:, 2] + res_weather[0] / 2) / res_weather[0])
                     * res_weather[0] - res_weather[0] / 2),
                    (np.floor((Crd_regions[:, 3] + res_weather[1] / 2) / res_weather[1])
                     * res_weather[1] - res_weather[1] / 2)])
    Crd = Crd.T
    return Crd


def filter_life_time(param, raw, depreciation):
    if param["year"] > param["pro_sto"]["year_ref"]:
        # Set depreciation period
        for c in raw["CoIn"].unique():
            raw.loc[raw["CoIn"] == c, "lifetime"] = depreciation[c]
        lifetimeleft = raw["lifetime"] + raw["year"]
        current = raw.drop(raw.loc[lifetimeleft < param["year"]].index)
        print('Already depreciated units:\n')
        print(str(len(raw) - len(current)) + '# Units have been removed')
    else:
        current = raw.copy()
        print('Number of current units: ' + str(len(current)))
    return current


def get_sites(current, paths):
    # Get regions from shapefile
    regions = gpd.read_file(paths["SHP"])
    regions["geometry"] = regions.buffer(0)

    # Spacial join
    current.crs = regions[["NAME_SHORT", "geometry"]].crs
    located = gpd.sjoin(current, regions[["NAME_SHORT", "geometry"]], how='left', op='intersects')
    located.rename(columns={'NAME_SHORT': 'Site'}, inplace=True)

    # Remove duplicates that lie in the border between land and sea
    located.drop_duplicates(subset=["CoIn", "Pro", "inst-cap", "year", "Site"], inplace=True)

    # Remove duplicates that lie in two different zones
    located = located.loc[~located.index.duplicated(keep='last')]

    located.dropna(axis=0, subset=["Site"], inplace=True)

    return located, regions


def closest_polygon(geom, polygons):
    """Returns polygon from polygons that is closest to geom.

    Args:
        geom: shapely geometry (used here: a point)
        polygons: GeoDataFrame of non-overlapping (!) polygons

    Returns:
        The polygon from 'polygons' which is closest to 'geom'.
    """
    dist = np.inf
    for poly in polygons.index:
        if polygons.loc[poly].geometry.convex_hull.exterior.distance(geom) < dist:
            dist = polygons.loc[poly].geometry.convex_hull.exterior.distance(geom)
            closest = polygons.loc[poly]
    return closest


def containing_polygon(geom, polygons):
    """Returns polygon from polygons that contains geom.

    Args:
        geom: shapely geometry (used here: a point)
        polygons: GeoDataFrame of non-overlapping (!) polygons

    Returns:
        The polygon from 'polygons' which contains (in
        the way shapely implements it) 'geom'. Throws
        an error if more than one polygon contain 'geom'.
        Returns 'None' if no polygon contains it.
    """
    try:
        containing_polygons = polygons[polygons.contains(geom)]
    except:
        containing_polygons = []
    if len(containing_polygons) == 0:
        return closest_polygon(geom, polygons)
    if len(containing_polygons) > 1:
        print(containing_polygons)
        # raise ValueError('geom lies in more than one polygon!')
    return containing_polygons.iloc[0]


def reverse_lines(df):
    """Reverses the line direction if the starting point is alphabetically
    after the end point.

    Args:
        df: dataframe with columns 'Region_start' and 'Region_end'.

    Returns:
        The same dataframe after the line direction has been reversed.
    """
    for idx in df.index:
        if df.Region_start[idx] > df.Region_end[idx]:
            df.loc[idx, 'Region_start'], df.loc[idx, 'Region_end'] = df.loc[idx, 'Region_end'], df.loc[
                idx, 'Region_start']
    df_final = df
    return df_final


def string_to_int(mylist):
    """This function converts list entries from strings to integers.

    Args:
        mylist: list eventually containing some integers interpreted
        as string elements.

    Returns:
        The same list after the strings where converted to integers.
    """
    result = [int(i) for i in mylist]
    return result


def zero_free(mylist):
    """This function deletes zero entries from a list.

    Args:
        mylist: list eventually containing zero entries.

    Returns:
        The same list after the zero entries where removed.
    """
    result = []
    for j in np.arange(len(mylist)):
        if mylist[j] > 0:
            result = result + [mylist[j]]
    return result


def add_suffix(df, suffix):
    # Check whether there is only one copy of the initial row, or more
    if str(df.index_old.iloc[1]).find('_') > 0:  # There are more than one copy of the row
        # Increment the suffix and replace the old one
        suffix = suffix + 1
        df.index_old.iloc[1] = df.index_old.iloc[1].replace('_' + str(suffix - 1), '_' + str(suffix))
    else:  # No other copy has been created so far
        # Reinitialize the suffix and concatenate it at the end of the old index
        suffix = 1
        df.index_old.iloc[1] = str(df.index_old.iloc[1]) + '_' + str(suffix)
    return (df, suffix)


def deduplicate_lines(df):
    """ Aggregate bidirectional lines to single lines.

    Given a th"""
    # aggregate val of rows with (a,b,t) == (b,a,t)
    idx = 0
    while idx < len(df) - 1:
        if (df.iloc[idx, 0] == df.iloc[idx + 1, 0]) & (df.iloc[idx, 1] == df.iloc[idx + 1, 1]) & (
                df.iloc[idx, 2] == df.iloc[idx + 1, 2]):
            df.iloc[idx, 4] = df.iloc[idx, 4] + df.iloc[idx + 1, 4]  # Capacity MVA
            df.iloc[idx, 9] = 1 / (1 / df.iloc[idx, 9] + 1 / df.iloc[idx + 1, 9])  # Specific resistance Ohm/km
            # df.iloc[idx,13] = df.iloc[idx,13] + df.iloc[idx+1,13] # Length
            df = df.drop(df.index[idx + 1])
        else:
            idx += 1

    df_final = df
    return df_final


def match_wire_voltages(grid_sorted):
    """
    the columns 'voltage' and 'wires' may contain multiple values separated with a semicolon. The goal is to assign
    a voltage to every circuit, whenever possible.

    Algorithm:

    [Case #1] If (n_voltages_count = 1), then every circuit is on that voltage level. We can replace the list
    entries in 'wires' with their sum;
    Else:
    [Case #2] If (n_circuits_count = n_voltages_count), then update the value in the list 'wires' so that each
    voltage level has only one circuit;
    [Case #3] If (n_circuits_count < n_voltages_count), then ignore the exceeding voltages and update the value in
    the list 'Circuits' so that each voltage level has only one circuit;
    [Case #4] If (n_voltages_count < n_circuits), then assign the highest voltage to the rest of the circuits;
    [Case #5] If (n_circuits < n_voltages_count) and (n_voltages_count < n_circuits_count), then ignore
    the exceeding voltages so that each voltage level has as many circuits as in the list entries of 'wires'.

    :param grid_sorted:
    :return:
    """

    timecheck('Start')

    n_circuits = pd.Series(map(string_to_int, grid_sorted.wires.str.split(';')))
    n_circuits_count = pd.Series(map(sum, n_circuits), index=grid_sorted.index)
    n_circuits = pd.Series(map(len, n_circuits), index=grid_sorted.index)
    n_voltages = pd.Series(map(zero_free, map(string_to_int, grid_sorted.voltage.str.split(';'))))
    n_voltages_count = pd.Series(map(len, n_voltages), index=grid_sorted.index)
    n_voltages = pd.Series(n_voltages, index=grid_sorted.index)
    grid_sorted.voltage = n_voltages

    # Case 1: (n_voltages_count = 1)
    ind_excerpt = grid_sorted[n_voltages_count == 1].index
    grid_clean = grid_sorted.loc[ind_excerpt]
    grid_dirty = grid_sorted.loc[grid_sorted[n_voltages_count != 1].index]
    n_circuits.loc[ind_excerpt] = n_circuits_count.loc[ind_excerpt]
    grid_clean.loc[:, 'wires'] = n_circuits.loc[ind_excerpt]

    # Reindex in order to avoid user warnings later
    n_circuits_count = n_circuits_count.reindex(grid_dirty.index)
    n_circuits = n_circuits.reindex(grid_dirty.index)
    n_voltages_count = n_voltages_count.reindex(grid_dirty.index)
    n_voltages = n_voltages.reindex(grid_dirty.index)

    # Case 2: (n_circuits_count = n_voltages_count)
    ind_excerpt = grid_dirty[n_circuits_count == n_voltages_count].index
    n_circuits.loc[ind_excerpt] = n_circuits_count.loc[ind_excerpt]
    grid_dirty.loc[ind_excerpt, 'wires'] = [';'.join(['1'] * n_circuits_count.loc[i]) for i in ind_excerpt]

    # Case 3: (n_circuits_count < n_voltages_count)
    ind_excerpt = grid_dirty[n_circuits_count < n_voltages_count].index
    n_circuits.loc[ind_excerpt] = n_circuits_count.loc[ind_excerpt]
    n_voltages_count.loc[ind_excerpt] = n_circuits.loc[ind_excerpt]
    n_voltages.loc[ind_excerpt] = [grid_dirty.loc[i, 'voltage'][:n_circuits_count.loc[i]] for i in ind_excerpt]
    grid_dirty.loc[ind_excerpt, 'voltage'] = n_voltages.loc[ind_excerpt]
    grid_dirty.loc[ind_excerpt, 'wires'] = [';'.join(['1'] * n_circuits_count.loc[i]) for i in ind_excerpt]

    # Case 4: (n_voltages_count < n_circuits)
    ind_excerpt = grid_dirty[(n_voltages_count < n_circuits) & (n_voltages_count > 0)].index
    missing_voltages = n_circuits.loc[ind_excerpt] - n_voltages_count.loc[ind_excerpt]
    n_voltages_count.loc[ind_excerpt] = n_circuits.loc[ind_excerpt]
    for i in ind_excerpt:
        for j in np.arange(missing_voltages[i]):
            n_voltages.loc[i].append(max(grid_dirty.loc[i, 'voltage']))
            grid_dirty.loc[i, 'voltage'].append(max(grid_dirty.loc[i, 'voltage']))

    # Case 5: (n_circuits < n_voltages_count) and (n_voltages_count < n_circuits_count)
    ind_excerpt = grid_dirty[(n_circuits < n_voltages_count) & (n_voltages_count < n_circuits_count)].index
    n_voltages_count.loc[ind_excerpt] = n_circuits.loc[ind_excerpt]
    n_voltages.loc[ind_excerpt] = [grid_dirty.loc[i, 'voltage'][:n_circuits.loc[i]] for i in ind_excerpt]
    grid_dirty.loc[ind_excerpt, 'voltage'] = n_voltages.loc[ind_excerpt]

    # By now n_circuits = n_voltages_count, so that we can split the list entries of 'voltage' and 'wires'
    # in exactly the same amount of rows:

    suffix = 1  # When we create a new row, we will add a suffix to the old index
    status = 0
    count = len(grid_dirty)
    while len(grid_dirty):
        status = count - len(grid_dirty) + 1
        display_progress("Cleaning GridKit progress: ", (count, status))
        # In case the first line is clean
        if grid_dirty.wires.iloc[0].count(';') == 0:
            grid_clean = grid_clean.append(grid_dirty.iloc[0], ignore_index=True)
            grid_dirty = grid_dirty.drop(grid_dirty.index[[0]])
        else:
            # Append a copy of the first row of grid_dirty at the top of the same dataframe
            grid_dirty = grid_dirty.iloc[0].to_frame().transpose().append(grid_dirty, ignore_index=True)
            # Extract the first number of circuits from that row and remove the rest of the string
            grid_dirty.wires.iloc[0] = grid_dirty.wires.iloc[0][:grid_dirty.wires.iloc[0].find(';')]
            # Extract the first voltage level from that row and remove the rest of the list
            grid_dirty.voltage.iloc[0] = grid_dirty.voltage.iloc[0][:1]

            # Add the right suffix
            grid_dirty, suffix = add_suffix(grid_dirty, suffix)

            # Update the string in the original row
            grid_dirty.wires.iloc[1] = grid_dirty.wires.iloc[1][grid_dirty.wires.iloc[1].find(';') + 1:]
            grid_dirty.voltage.iloc[1] = grid_dirty.voltage.iloc[1][1:]

            # Move the 'clean' row to grid_clean, and drop it from grid_dirty
            grid_clean = grid_clean.append(grid_dirty.iloc[0], ignore_index=True)
            grid_dirty = grid_dirty.drop(grid_dirty.index[[0]])

    # Express voltage in kV
    grid_clean.voltage = pd.Series([grid_clean.loc[i, 'voltage'][0] / 1000 for i in grid_clean.index],
                                   index=grid_clean.index)
    timecheck('End')
    return grid_clean


def set_loadability(grid_filled, param):
    loadability = param["grid"]["loadability"]
    grid_filled.loc[grid_filled[grid_filled.length_m <= float(80)].index, 'loadability_c'] = loadability["80"]
    grid_filled.loc[grid_filled[(grid_filled.length_m > 80) & (grid_filled.length_m <= 100)].index, 'loadability_c'] \
        = loadability["100"]
    grid_filled.loc[grid_filled[(grid_filled.length_m > 100) & (grid_filled.length_m <= 150)].index, 'loadability_c'] \
        = loadability["150"]
    grid_filled.loc[grid_filled[(grid_filled.length_m > 150) & (grid_filled.length_m <= 200)].index, 'loadability_c'] \
        = loadability["200"]
    grid_filled.loc[grid_filled[(grid_filled.length_m > 200) & (grid_filled.length_m <= 250)].index, 'loadability_c'] \
        = loadability["250"]
    grid_filled.loc[grid_filled[(grid_filled.length_m > 250) & (grid_filled.length_m <= 300)].index, 'loadability_c'] \
        = loadability["300"]
    grid_filled.loc[grid_filled[(grid_filled.length_m > 300) & (grid_filled.length_m <= 350)].index, 'loadability_c'] \
        = loadability["350"]
    grid_filled.loc[grid_filled[(grid_filled.length_m > 350) & (grid_filled.length_m <= 400)].index, 'loadability_c'] \
        = loadability["400"]
    grid_filled.loc[grid_filled[(grid_filled.length_m > 400) & (grid_filled.length_m <= 450)].index, 'loadability_c'] \
        = loadability["450"]
    grid_filled.loc[grid_filled[(grid_filled.length_m > 450) & (grid_filled.length_m <= 500)].index, 'loadability_c'] \
        = loadability["500"]
    grid_filled.loc[grid_filled[(grid_filled.length_m > 500) & (grid_filled.length_m <= 550)].index, 'loadability_c'] \
        = loadability["550"]
    grid_filled.loc[grid_filled[(grid_filled.length_m > 550) & (grid_filled.length_m <= 600)].index, 'loadability_c'] \
        = loadability["600"]
    grid_filled.loc[grid_filled[(grid_filled.length_m > 600) & (grid_filled.length_m <= 650)].index, 'loadability_c'] \
        = loadability["650"]
    grid_filled.loc[grid_filled[(grid_filled.length_m > 650) & (grid_filled.length_m <= 700)].index, 'loadability_c'] \
        = loadability["700"]
    grid_filled.loc[grid_filled[grid_filled["length_m"] > 700].index, 'loadability_c'] = loadability["750"]

    return grid_filled


def format_process_model(process_compact, param):
    assump = param["assumptions"]

    # evrys
    output_pro_evrys = process_compact.copy()
    output_pro_evrys.drop(['on-off'], axis=1, inplace=True)

    output_pro_evrys = output_pro_evrys.join(pd.DataFrame([], columns=['eff', 'effmin', 'act-up', 'act-lo', 'on-off',
                                                                       'start-cost', 'reserve-cost', 'ru', 'rd',
                                                                       'rumax',
                                                                       'rdmax', 'cotwo', 'detail', 'lambda', 'heatmax',
                                                                       'maxdeltaT', 'heatupcost', 'su', 'sd', 'pdt',
                                                                       'hotstart',
                                                                       'pot', 'pretemp', 'preheat', 'prestate',
                                                                       'prepow',
                                                                       'precaponline']), how='outer')
    for c in output_pro_evrys['CoIn'].unique():
        output_pro_evrys.loc[output_pro_evrys.CoIn == c,
                             ['eff', 'effmin', 'act-up', 'act-lo', 'on-off', 'start-cost',
                              'reserve-cost', 'ru', 'rd', 'rumax', 'rdmax', 'cotwo',
                              'detail', 'lambda', 'heatmax', 'maxdeltaT', 'heatupcost',
                              'su', 'sd', 'pdt', 'hotstart', 'pot', 'pretemp',
                              'preheat', 'prestate']] = [assump["eff"][c], assump["effmin"][c], assump["act_up"][c],
                                                         assump["act_lo"][c], assump["on_off"][c],
                                                         assump["start_cost"][c], assump["reserve_cost"][c],
                                                         assump["ru"][c], assump["rd"][c], assump["rumax"][c],
                                                         assump["rdmax"][c], assump["cotwo"][c], assump["detail"][c],
                                                         assump["lambda_"][c], assump["heatmax"][c],
                                                         assump["maxdeltaT"][c], assump["heatupcost"][c],
                                                         assump["su"][c], assump["sd"][c], assump["pdt"][c],
                                                         assump["hotstart"][c], assump["pot"][c],
                                                         assump["pretemp"][c], assump["preheat"][c],
                                                         assump["prestate"][c]]

    ind = output_pro_evrys['CoIn'] == 'Coal'
    output_pro_evrys.loc[ind, 'eff'] = 0.35 + 0.1 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (
            param["pro_sto"]["year_ref"] - 1960)
    output_pro_evrys.loc[ind, 'effmin'] = 0.92 * output_pro_evrys.loc[ind, 'eff']

    ind = output_pro_evrys['CoIn'] == 'Lignite'
    output_pro_evrys.loc[ind, 'eff'] = 0.33 + 0.1 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (
            param["pro_sto"]["year_ref"] - 1960)
    output_pro_evrys.loc[ind, 'effmin'] = 0.9 * output_pro_evrys.loc[ind, 'eff']

    ind = ((output_pro_evrys['CoIn'] == 'Gas') & (output_pro_evrys['inst-cap'] <= 100))
    output_pro_evrys.loc[ind, 'eff'] = 0.3 + 0.15 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (
            param["pro_sto"]["year_ref"] - 1960)
    output_pro_evrys.loc[ind, 'effmin'] = 0.65 * output_pro_evrys.loc[ind, 'eff']
    output_pro_evrys.loc[ind, 'act-lo'] = 0.3
    output_pro_evrys.loc[ind, 'ru'] = 0.01
    output_pro_evrys.loc[ind, 'lambda'] = 0.3
    output_pro_evrys.loc[ind, 'heatupcost'] = 20
    output_pro_evrys.loc[ind, 'su'] = 0.9

    ind = output_pro_evrys['CoIn'] == 'Oil'
    output_pro_evrys.loc[ind, 'eff'] = 0.25 + 0.15 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (
            param["pro_sto"]["year_ref"] - 1960)
    output_pro_evrys.loc[ind, 'effmin'] = 0.65 * output_pro_evrys.loc[ind, 'eff']

    ind = output_pro_evrys['CoIn'] == 'Nuclear'
    output_pro_evrys.loc[ind, 'eff'] = 0.3 + 0.05 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (
            param["pro_sto"]["year_ref"] - 1960)
    output_pro_evrys.loc[ind, 'effmin'] = 0.95 * output_pro_evrys.loc[ind, 'eff']

    output_pro_evrys['prepow'] = output_pro_evrys['inst-cap'] * output_pro_evrys['act-lo']
    output_pro_evrys['precaponline'] = output_pro_evrys['prepow']

    # Change the order of the columns
    output_pro_evrys = output_pro_evrys[
        ['Site', 'Pro', 'CoIn', 'CoOut', 'inst-cap', 'eff', 'effmin', 'act-lo', 'act-up',
         'on-off', 'start-cost', 'reserve-cost', 'ru', 'rd', 'rumax', 'rdmax', 'cotwo',
         'detail', 'lambda', 'heatmax', 'maxdeltaT', 'heatupcost', 'su', 'sd', 'pdt',
         'hotstart', 'pot', 'prepow', 'pretemp', 'preheat', 'prestate', 'precaponline', 'year']]
    output_pro_evrys.iloc[:, 4:] = output_pro_evrys.iloc[:, 4:].astype(float)

    # function to remove non-ASCII
    def remove_non_ascii(text):
        return ''.join(i for i in text if ord(i) < 128)

    # function to shorten names
    def shorten_labels(text):
        return text[:63]

    output_pro_evrys.loc[:, 'Pro'] = output_pro_evrys.loc[:, 'Pro'].apply(remove_non_ascii)
    output_pro_evrys.loc[:, 'Pro'] = output_pro_evrys.loc[:, 'Pro'].apply(shorten_labels)

    # urbs

    # Take excerpt from the evrys table and group by tuple of sites and commodity
    process_grouped = output_pro_evrys[['Site', 'CoIn', 'inst-cap', 'act-lo', 'start-cost', 'ru']].apply(pd.to_numeric,
                                                                                                         errors='ignore')
    process_grouped.rename(columns={'CoIn': 'Process'}, inplace=True)
    process_grouped = process_grouped.groupby(['Site', 'Process'])

    inst_cap0 = process_grouped['inst-cap'].sum().rename('inst-cap')
    max_grad0 = process_grouped['ru'].mean().rename('max-grad') * 60
    max_grad0[max_grad0 == 60] = float('Inf')
    min_fraction0 = process_grouped['act-lo'].mean().rename('min-fraction')
    startup_cost0 = process_grouped['start-cost'].mean().rename('startup-cost')

    # Combine the list of series into a dataframe
    process_existant = pd.DataFrame([inst_cap0, max_grad0, min_fraction0, startup_cost0]).transpose()

    # Get the possible commodities and add Slacks
    commodity = list(output_pro_evrys.CoIn.unique())
    commodity.append('Slack')
    commodity.append('Shunt')

    # Create a dataframe to store all the possible combinations of sites and commodities
    df = pd.DataFrame(index=pd.MultiIndex.from_product([output_pro_evrys.Site.unique(), commodity],
                                                       names=['Site', 'Process']))

    # Get the capacities of existing processes
    df_joined = df.join(process_existant, how='outer')

    # Set the capacity of inexistant processes to zero
    df_joined.loc[np.isnan(df_joined['inst-cap']), 'inst-cap'] = 0

    output_pro_urbs = df_joined.reset_index(drop=False)
    output_pro_urbs = output_pro_urbs.join(pd.DataFrame([], columns=['cap-lo', 'cap-up', 'inv-cost', 'fix-cost',
                                                                     'var-cost', 'wacc', 'depreciation',
                                                                     'area-per-cap']), how='outer')
    for c in output_pro_urbs['Process'].unique():
        output_pro_urbs.loc[
            output_pro_urbs['Process'] == c, ['cap-lo', 'cap-up', 'max-grad',
                                              'min-fraction', 'inv-cost', 'fix-cost',
                                              'var-cost', 'startup-cost', 'wacc',
                                              'depreciation', 'area-per-cap']] = [
            assump["cap_lo"][c], assump["cap_up"][c], assump["max_grad"][c],
            assump["min_fraction"][c], assump["inv_cost"][c], assump["fix_cost"][c],
            assump["var_cost"][c], assump["startup_cost"][c], param["pro_sto"]["wacc"],
            assump["depreciation"][c], assump["area_per_cap"][c]]

    # Cap-up must be greater than inst-cap
    output_pro_urbs.loc[output_pro_urbs['cap-up'] < output_pro_urbs['inst-cap'], 'cap-up'] = output_pro_urbs.loc[
        output_pro_urbs['cap-up'] < output_pro_urbs['inst-cap'], 'inst-cap']

    # inst-cap must be greater than cap-lo
    output_pro_urbs.loc[output_pro_urbs['inst-cap'] < output_pro_urbs['cap-lo'], 'inst-cap'] = output_pro_urbs.loc[
        output_pro_urbs['inst-cap'] < output_pro_urbs['cap-lo'], 'cap-lo']

    # Cap-up must be of type float
    output_pro_urbs[['cap-up']] = output_pro_urbs[['cap-up']].astype(float)

    # Delete rows where cap-up is zero
    output_pro_urbs = output_pro_urbs[output_pro_urbs['cap-up'] != 0]

    # Change the order of the columns
    output_pro_urbs = output_pro_urbs[
        ['Site', 'Process', 'inst-cap', 'cap-lo', 'cap-up', 'max-grad', 'min-fraction', 'inv-cost',
         'fix-cost', 'var-cost', 'startup-cost', 'wacc', 'depreciation', 'area-per-cap']]
    output_pro_urbs = output_pro_urbs.fillna(0)

    return output_pro_evrys, output_pro_urbs


def format_storage_model(storage_compact, param):
    assump = param["assumptions"]

    # evrys
    output_sto_evrys = storage_compact.copy()

    output_sto_evrys = output_sto_evrys.join(pd.DataFrame([], columns=['inst-cap-po', 'inst-cap-c', 'eff-in', 'eff-out',
                                                                       'var-cost-pi', 'var-cost-po', 'var-cost-c',
                                                                       'act-up-pi',
                                                                       'act-up-po', 'act-lo-pi', 'act-lo-po',
                                                                       'act-lo-c',
                                                                       'act-up-c', 'precont', 'prepowin', 'prepowout',
                                                                       'ru', 'rd', 'rumax', 'rdmax', 'seasonal',
                                                                       'ctr']), how='outer')
    for c in output_sto_evrys.Sto:
        output_sto_evrys.loc[
            output_sto_evrys.Sto == c, ['eff-in', 'eff-out', 'var-cost-pi', 'var-cost-po', 'var-cost-c',
                                        'act-up-pi', 'act-up-po', 'act-lo-pi', 'act-lo-po', 'act-lo-c',
                                        'act-up-c', 'prepowin', 'prepowout', 'ru', 'rd', 'rumax',
                                        'rdmax', 'seasonal', 'ctr']] = [
            assump["eff_in"][c], assump["eff_out"][c], assump["var_cost_pi"][c],
            assump["var_cost_po"][c], assump["var_cost_c"][c], assump["act_up_pi"][c],
            assump["act_up_po"][c], assump["act_lo_pi"][c],
            assump["act_lo_po"][c], assump["act_lo_c"][c], assump["act_up_c"][c],
            assump["prepowin"][c], assump["prepowout"][c],
            assump["ru"][c], assump["rd"][c], assump["rumax"][c], assump["rdmax"][c],
            assump["seasonal"][c], assump["ctr"][c]]

    output_sto_evrys['inst-cap-po'] = output_sto_evrys['inst-cap-pi']
    output_sto_evrys.loc[output_sto_evrys['Sto'] == 'PumSt', 'inst-cap-c'] = 6 * output_sto_evrys.loc[
        output_sto_evrys['Sto'] == 'PumSt', 'inst-cap-pi']
    output_sto_evrys.loc[output_sto_evrys['Sto'] == 'Battery', 'inst-cap-c'] = 2 * output_sto_evrys.loc[
        output_sto_evrys['Sto'] == 'Battery', 'inst-cap-pi']
    output_sto_evrys['precont'] = 0.5 * output_sto_evrys['inst-cap-c']

    # Change the order of the columns
    output_sto_evrys = output_sto_evrys[
        ['Site', 'Sto', 'Co', 'inst-cap-pi', 'inst-cap-po', 'inst-cap-c', 'eff-in', 'eff-out',
         'var-cost-pi', 'var-cost-po', 'var-cost-c', 'act-lo-pi', 'act-up-pi', 'act-lo-po',
         'act-up-po', 'act-lo-c', 'act-up-c', 'precont', 'prepowin', 'prepowout', 'ru', 'rd',
         'rumax', 'rdmax', 'seasonal', 'ctr']]
    output_sto_evrys = output_sto_evrys.iloc[:, :3].join(output_sto_evrys.iloc[:, 3:].astype(float))

    # urbs
    # Create a dataframe to store all the possible combinations of sites and commodities
    df = pd.DataFrame(index=pd.MultiIndex.from_product([param["regions"]['NAME_SHORT'].unique(),
                                                        param["pro_sto"]["storage"]],
                                                       names=['Site', 'Storage']))

    # Take excerpt from the evrys table and group by tuple of sites and commodity
    storage_existant = output_sto_evrys[['Site', 'Sto', 'Co', 'inst-cap-c', 'inst-cap-pi']].rename(
        columns={'Sto': 'Storage', 'Co': 'Commodity', 'inst-cap-pi': 'inst-cap-p'})

    # Get the capacities of existing processes
    df_joined = df.join(storage_existant.set_index(['Site', 'Storage']), how='outer')

    # Set the capacity of inexistant processes to zero
    df_joined['Commodity'].fillna('Elec', inplace=True)
    df_joined.fillna(0, inplace=True)

    output_sto_urbs = df_joined.reset_index()
    output_sto_urbs = output_sto_urbs.join(pd.DataFrame([], columns=['cap-lo-c', 'cap-up-c', 'cap-lo-p', 'cap-up-p',
                                                                     'eff-in', 'eff-out', 'inv-cost-p', 'inv-cost-c',
                                                                     'fix-cost-p', 'fix-cost-c', 'var-cost-p',
                                                                     'var-cost-c',
                                                                     'wacc', 'depreciation', 'init', 'discharge']),
                                           how='outer')
    for c in output_sto_urbs.Storage:
        output_sto_urbs.loc[
            output_sto_urbs.Storage == c, ['cap-lo-c', 'cap-up-c', 'cap-lo-p',
                                           'cap-up-p', 'eff-in', 'eff-out',
                                           'inv-cost-p', 'inv-cost-c', 'fix-cost-p',
                                           'fix-cost-c', 'var-cost-p', 'var-cost-c',
                                           'wacc', 'depreciation', 'init', 'discharge']] = [
            assump["cap_lo_c"][c], assump["cap_up_c"][c], assump["cap_lo_p"][c],
            assump["cap_up_p"][c], assump["eff_in"][c], assump["eff_out"][c],
            assump["inv_cost_p"][c], assump["inv_cost_c"][c], assump["fix_cost_p"][c],
            assump["fix_cost_c"][c], assump["var_cost_p"][c], assump["var_cost_c"][c],
            param["pro_sto"]["wacc"], assump["depreciation"][c], assump["init"][c], assump["discharge"][c]]

    output_sto_urbs.loc[output_sto_urbs['Storage'] == 'PumSt', 'cap-up-c'] = output_sto_urbs.loc[
        output_sto_urbs['Storage'] == 'PumSt', 'inst-cap-c']
    output_sto_urbs.loc[output_sto_urbs['Storage'] == 'PumSt', 'cap-up-p'] = output_sto_urbs.loc[
        output_sto_urbs['Storage'] == 'PumSt', 'inst-cap-p']

    # Change the order of the columns
    output_sto_urbs = output_sto_urbs[
        ['Site', 'Storage', 'Commodity', 'inst-cap-c', 'cap-lo-c', 'cap-up-c', 'inst-cap-p', 'cap-lo-p', 'cap-up-p',
         'eff-in', 'eff-out', 'inv-cost-p', 'inv-cost-c', 'fix-cost-p', 'fix-cost-c', 'var-cost-p', 'var-cost-c',
         'wacc', 'depreciation', 'init', 'discharge']]

    output_sto_urbs.iloc[:, 3:] = output_sto_urbs.iloc[:, 3:].astype(float)

    return output_sto_evrys, output_sto_urbs


def format_process_model_California(process_compact, process_small, param):
    # evrys
    output_pro_evrys = process_compact.copy()
    output_pro_evrys['eff'] = 1  # Will be changed for thermal power plants
    output_pro_evrys['effmin'] = 1  # Will be changed for thermal power plants
    output_pro_evrys['act-up'] = 1
    output_pro_evrys['act-lo'] = 0  # Will be changed for most conventional power plants
    output_pro_evrys['on-off'] = 1  # Will be changed to 0 for SupIm commodities
    output_pro_evrys['start-cost'] = 0  # Will be changed for most conventional power plants
    output_pro_evrys['reserve-cost'] = 0
    output_pro_evrys['ru'] = 1  # Will be changed for thermal power plants
    output_pro_evrys['cotwo'] = 0  # Will be changed for most conventional power plants
    output_pro_evrys['detail'] = 1  # 5: thermal modeling, 1: simple modeling, will be changed for thermal power plants
    output_pro_evrys['lambda'] = 0  # Will be changed for most conventional power plants
    output_pro_evrys['heatmax'] = 1  # Will be changed for most conventional power plants
    output_pro_evrys['maxdeltaT'] = 1
    output_pro_evrys['heatupcost'] = 0  # Will be changed for most conventional power plants
    output_pro_evrys['su'] = 1  # Will be changed for most conventional power plants
    output_pro_evrys['pdt'] = 0
    output_pro_evrys['hotstart'] = 0
    output_pro_evrys['pot'] = 0
    output_pro_evrys['pretemp'] = 1
    output_pro_evrys['preheat'] = 0
    output_pro_evrys['prestate'] = 1

    ind = output_pro_evrys['CoIn'] == 'Coal'
    output_pro_evrys.loc[ind, 'eff'] = 0.35 + 0.1 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (param["year"] - 1960)
    output_pro_evrys.loc[ind, 'effmin'] = 0.92 * output_pro_evrys.loc[ind, 'eff']
    output_pro_evrys.loc[ind, 'act-lo'] = 0.4
    output_pro_evrys.loc[ind, 'start-cost'] = 90
    output_pro_evrys.loc[ind, 'ru'] = 0.03
    output_pro_evrys.loc[ind, 'cotwo'] = 0.33
    output_pro_evrys.loc[ind, 'detail'] = 5
    output_pro_evrys.loc[ind, 'lambda'] = 0.06
    output_pro_evrys.loc[ind, 'heatmax'] = 0.15
    output_pro_evrys.loc[ind, 'heatupcost'] = 110
    output_pro_evrys.loc[ind, 'su'] = 0.5

    ind = output_pro_evrys['CoIn'] == 'Lignite'
    output_pro_evrys.loc[ind, 'eff'] = 0.33 + 0.1 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (param["year"] - 1960)
    output_pro_evrys.loc[ind, 'effmin'] = 0.9 * output_pro_evrys.loc[ind, 'eff']
    output_pro_evrys.loc[ind, 'act-lo'] = 0.45
    output_pro_evrys.loc[ind, 'start-cost'] = 110
    output_pro_evrys.loc[ind, 'ru'] = 0.02
    output_pro_evrys.loc[ind, 'cotwo'] = 0.40
    output_pro_evrys.loc[ind, 'detail'] = 5
    output_pro_evrys.loc[ind, 'lambda'] = 0.04
    output_pro_evrys.loc[ind, 'heatmax'] = 0.12
    output_pro_evrys.loc[ind, 'heatupcost'] = 130
    output_pro_evrys.loc[ind, 'su'] = 0.5

    ind = (output_pro_evrys['CoIn'] == 'Gas') & (output_pro_evrys['inst-cap'] > 100) & (
            output_pro_evrys.index < len(process_compact) - len(process_small))
    output_pro_evrys.loc[ind, 'eff'] = 0.45 + 0.15 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (param["year"] - 1960)
    output_pro_evrys.loc[ind, 'effmin'] = 0.82 * output_pro_evrys.loc[ind, 'eff']
    output_pro_evrys.loc[ind, 'act-lo'] = 0.45
    output_pro_evrys.loc[ind, 'start-cost'] = 40
    output_pro_evrys.loc[ind, 'ru'] = 0.05
    output_pro_evrys.loc[ind, 'cotwo'] = 0.20
    output_pro_evrys.loc[ind, 'detail'] = 5
    output_pro_evrys.loc[ind, 'lambda'] = 0.1
    output_pro_evrys.loc[ind, 'heatmax'] = 0.2
    output_pro_evrys.loc[ind, 'heatupcost'] = 60
    output_pro_evrys.loc[ind, 'su'] = 0.5

    ind = (output_pro_evrys['CoIn'] == 'Gas') & ((output_pro_evrys['inst-cap'] <= 100) | (
            output_pro_evrys.index >= len(process_compact) - len(process_small)))
    output_pro_evrys.loc[ind, 'eff'] = 0.3 + 0.15 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (param["year"] - 1960)
    output_pro_evrys.loc[ind, 'effmin'] = 0.65 * output_pro_evrys.loc[ind, 'eff']
    output_pro_evrys.loc[ind, 'act-lo'] = 0.3
    output_pro_evrys.loc[ind, 'start-cost'] = 40
    output_pro_evrys.loc[ind, 'ru'] = 0.01
    output_pro_evrys.loc[ind, 'cotwo'] = 0.20
    output_pro_evrys.loc[ind, 'detail'] = 5
    output_pro_evrys.loc[ind, 'lambda'] = 0.3
    output_pro_evrys.loc[ind, 'heatupcost'] = 20
    output_pro_evrys.loc[ind, 'su'] = 0.9

    ind = output_pro_evrys['CoIn'] == 'Oil'
    output_pro_evrys.loc[ind, 'eff'] = 0.25 + 0.15 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (param["year"] - 1960)
    output_pro_evrys.loc[ind, 'effmin'] = 0.65 * output_pro_evrys.loc[ind, 'eff']
    output_pro_evrys.loc[ind, 'act-lo'] = 0.4
    output_pro_evrys.loc[ind, 'start-cost'] = 40
    output_pro_evrys.loc[ind, 'ru'] = 0.05
    output_pro_evrys.loc[ind, 'cotwo'] = 0.30
    output_pro_evrys.loc[ind, 'detail'] = 5
    output_pro_evrys.loc[ind, 'lambda'] = 0.3
    output_pro_evrys.loc[ind, 'heatupcost'] = 20
    output_pro_evrys.loc[ind, 'su'] = 0.7

    ind = output_pro_evrys['CoIn'] == 'Nuclear'
    output_pro_evrys.loc[ind, 'eff'] = 0.3 + 0.05 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (param["year"] - 1960)
    output_pro_evrys.loc[ind, 'effmin'] = 0.95 * output_pro_evrys.loc[ind, 'eff']
    output_pro_evrys.loc[ind, 'act-lo'] = 0.45
    output_pro_evrys.loc[ind, 'start-cost'] = 150
    output_pro_evrys.loc[ind, 'ru'] = 0.04
    output_pro_evrys.loc[ind, 'detail'] = 5
    output_pro_evrys.loc[ind, 'lambda'] = 0.03
    output_pro_evrys.loc[ind, 'heatmax'] = 0.1
    output_pro_evrys.loc[ind, 'heatupcost'] = 100
    output_pro_evrys.loc[ind, 'su'] = 0.45

    ind = output_pro_evrys['CoIn'].isin(['Biomass', 'Waste'])
    output_pro_evrys.loc[ind, 'eff'] = 0.3
    output_pro_evrys.loc[ind, 'effmin'] = 0.3
    output_pro_evrys.loc[ind, 'ru'] = 0.05

    ind = output_pro_evrys['CoIn'].isin(['Solar', 'WindOn', 'WindOff', 'Hydro_large', 'Hydro_Small'])
    output_pro_evrys.loc[ind, 'on-off'] = 0

    output_pro_evrys['rd'] = output_pro_evrys['ru']
    output_pro_evrys['rumax'] = np.minimum(output_pro_evrys['ru'] * 60, 1)
    output_pro_evrys['rdmax'] = output_pro_evrys['rumax']
    output_pro_evrys['sd'] = output_pro_evrys['su']
    output_pro_evrys['prepow'] = output_pro_evrys['inst-cap'] * output_pro_evrys['act-lo']
    output_pro_evrys['precaponline'] = output_pro_evrys['prepow']
    # Change the order of the columns
    output_pro_evrys = output_pro_evrys[
        ['Site', 'Pro', 'CoIn', 'CoOut', 'inst-cap', 'eff', 'effmin', 'act-lo', 'act-up',
         'on-off', 'start-cost', 'reserve-cost', 'ru', 'rd', 'rumax', 'rdmax', 'cotwo',
         'detail', 'lambda', 'heatmax', 'maxdeltaT', 'heatupcost', 'su', 'sd', 'pdt',
         'hotstart', 'pot', 'prepow', 'pretemp', 'preheat', 'prestate', 'precaponline', 'year']]

    # urbs
    # Take excerpt from the evrys table and group by tuple of sites and commodity
    process_grouped = output_pro_evrys[['Site', 'CoIn', 'inst-cap', 'act-lo', 'start-cost', 'ru']].groupby(
        ['Site', 'CoIn'])

    inst_cap0 = process_grouped['inst-cap'].sum().rename('inst-cap')
    max_grad0 = process_grouped['ru'].mean().rename('max-grad') * 60
    max_grad0[max_grad0 == 60] = float('Inf')
    min_fraction0 = process_grouped['act-lo'].mean().rename('min-fraction')
    startup_cost0 = process_grouped['start-cost'].mean().rename('startup-cost')

    # Combine the list of series into a dataframe
    process_existant = pd.DataFrame([inst_cap0, max_grad0, min_fraction0, startup_cost0]).transpose()

    # Get the possible commodities and add Slacks
    commodity = list(output_pro_evrys.CoIn.unique())
    commodity.append('Slack')

    # Create a dataframe to store all the possible combinations of sites and commodities
    df = pd.DataFrame(index=pd.MultiIndex.from_product([output_pro_evrys.Site.unique(), commodity],
                                                       names=['Site', 'CoIn']))
    # Get the capacities of existing processes
    df_joined = df.join(process_existant, how='outer')

    # Set the capacity of inexistant processes to zero
    df_joined.loc[np.isnan(df_joined['inst-cap']), 'inst-cap'] = 0

    output_pro_urbs = df_joined.reset_index(drop=False)
    output_pro_urbs = output_pro_urbs.join(pd.DataFrame([], columns=['cap-lo', 'cap-up', 'inv-cost', 'fix-cost',
                                                                     'var-cost', 'wacc', 'depreciation',
                                                                     'area-per-cap']), how='outer')

    for c in output_pro_urbs.CoIn:
        output_pro_urbs.loc[
            output_pro_urbs.CoIn == c, ['cap-lo', 'cap-up', 'max-grad', 'min-fraction', 'inv-cost', 'fix-cost',
                                        'var-cost']] = [param["pro_sto_Cal"]["Cal_urbs"]["cap_lo"][c],
                                                        param["pro_sto_Cal"]["Cal_urbs"]["cap_up"][c],
                                                        param["pro_sto_Cal"]["Cal_urbs"]["max_grad"][c],
                                                        param["pro_sto_Cal"]["Cal_urbs"]["min_fraction"][c],
                                                        param["pro_sto_Cal"]["Cal_urbs"]["inv_cost"][c],
                                                        param["pro_sto_Cal"]["Cal_urbs"]["fix_cost"][c],
                                                        param["pro_sto_Cal"]["Cal_urbs"]["var_cost"][c]]
        output_pro_urbs.loc[output_pro_urbs.CoIn == c, 'startup-cost'] = \
            param["pro_sto_Cal"]["Cal_urbs"]["startup_cost"][c]
        output_pro_urbs.loc[output_pro_urbs.CoIn == c, 'wacc'] = \
            param["pro_sto_Cal"]["Cal_urbs"]["wacc"]
        output_pro_urbs.loc[output_pro_urbs.CoIn == c, ['depreciation', 'area-per-cap']] = \
            [param["pro_sto_Cal"]["Cal_urbs"]["depreciation"][c],
             param["pro_sto_Cal"]["Cal_urbs"]["area_per_cap"][c]]

    # Cap-up must be greater than inst-cap
    output_pro_urbs.loc[output_pro_urbs['cap-up'] < output_pro_urbs['inst-cap'], 'cap-up'] = output_pro_urbs.loc[
        output_pro_urbs['cap-up'] < output_pro_urbs['inst-cap'], 'inst-cap']

    # inst-cap must be greater than cap-lo
    output_pro_urbs.loc[output_pro_urbs['inst-cap'] < output_pro_urbs['cap-lo'], 'inst-cap'] = output_pro_urbs.loc[
        output_pro_urbs['inst-cap'] < output_pro_urbs['cap-lo'], 'cap-lo']

    # Cap-up must be of type float
    output_pro_urbs[['cap-up']] = output_pro_urbs[['cap-up']].astype(float)

    # Delete rows where cap-up is zero
    output_pro_urbs = output_pro_urbs[output_pro_urbs['cap-up'] != 0]

    # Change the order of the columns
    output_pro_urbs = output_pro_urbs[
        ['Site', 'CoIn', 'inst-cap', 'cap-lo', 'cap-up', 'max-grad', 'min-fraction', 'inv-cost',
         'fix-cost', 'var-cost', 'startup-cost', 'wacc', 'depreciation', 'area-per-cap']]

    return output_pro_evrys, output_pro_urbs


def format_storage_model_California(storage_raw, param):
    # evrys
    # Take the raw storage table and group by tuple of sites and storage type
    sto_evrys = storage_raw[['Site', 'CoIn', 'CoOut', 'inst-cap']].rename(columns={'CoIn': 'Sto', 'CoOut': 'Co'})
    sto_grouped = sto_evrys.groupby(['Site', 'Sto'])

    inst_cap0 = sto_grouped['inst-cap'].sum().rename('inst-cap-pi')
    co0 = sto_grouped['Co'].first()

    # Combine the list of series into a dataframe
    sto_existant = pd.DataFrame([inst_cap0, co0]).transpose()
    output_sto_evrys = sto_existant.reset_index()
    output_sto_evrys['inst-cap-po'] = output_sto_evrys['inst-cap-pi']
    output_sto_evrys['var-cost-pi'] = 0.05
    output_sto_evrys['var-cost-po'] = 0.05
    output_sto_evrys['var-cost-c'] = -0.01
    output_sto_evrys['act-lo-pi'] = 0
    output_sto_evrys['act-up-pi'] = 1
    output_sto_evrys['act-lo-po'] = 0
    output_sto_evrys['act-up-po'] = 1
    output_sto_evrys['act-lo-c'] = 0
    output_sto_evrys['act-up-c'] = 1
    output_sto_evrys['prepowin'] = 0
    output_sto_evrys['prepowout'] = 0
    output_sto_evrys['ru'] = 0.1
    output_sto_evrys['rd'] = 0.1
    output_sto_evrys['rumax'] = 1
    output_sto_evrys['rdmax'] = 1
    output_sto_evrys['seasonal'] = 0
    output_sto_evrys['ctr'] = 1

    ind = (output_sto_evrys['Sto'] == 'PumSt')
    output_sto_evrys.loc[ind, 'inst-cap-c'] = 8 * output_sto_evrys.loc[ind, 'inst-cap-pi']
    output_sto_evrys.loc[ind, 'eff-in'] = 0.92
    output_sto_evrys.loc[ind, 'eff-out'] = 0.92

    ind = (output_sto_evrys['Sto'] == 'Battery')
    output_sto_evrys.loc[ind, 'inst-cap-c'] = 2 * output_sto_evrys.loc[ind, 'inst-cap-pi']
    output_sto_evrys.loc[ind, 'eff-in'] = 0.94
    output_sto_evrys.loc[ind, 'eff-out'] = 0.94
    output_sto_evrys['precont'] = output_sto_evrys['inst-cap-c'] / 2

    # Change the order of the columns
    output_sto_evrys = output_sto_evrys[
        ['Site', 'Sto', 'Co', 'inst-cap-pi', 'inst-cap-po', 'inst-cap-c', 'eff-in', 'eff-out',
         'var-cost-pi', 'var-cost-po', 'var-cost-c', 'act-lo-pi', 'act-up-pi', 'act-lo-po',
         'act-up-po', 'act-lo-c', 'act-up-c', 'precont', 'prepowin', 'prepowout', 'ru', 'rd',
         'rumax', 'rdmax', 'seasonal', 'ctr']]

    # urbs
    # Create a dataframe to store all the possible combinations of sites and commodities
    df = pd.DataFrame(index=pd.MultiIndex.from_product([param["sites_evrys_unique"], output_sto_evrys.Sto.unique()],
                                                       names=['Site', 'Storage']))
    # Take excerpt from the evrys table and group by tuple of sites and commodity
    storage_existant = output_sto_evrys[['Site', 'Sto', 'Co', 'inst-cap-c', 'inst-cap-pi', 'precont']].rename(
        columns={'Sto': 'Storage', 'Co': 'Commodity', 'inst-cap-pi': 'inst-cap-p'})

    # Get the capacities of existing processes
    df_joined = df.join(storage_existant.set_index(['Site', 'Storage']), how='outer')

    # Set the capacity of inexistant processes to zero
    df_joined['Commodity'].fillna('Elec', inplace=True)
    df_joined.fillna(0, inplace=True)
    out_sto_urbs = df_joined.reset_index()
    out_sto_urbs['cap-lo-c'] = 0
    out_sto_urbs['cap-up-c'] = out_sto_urbs['inst-cap-c']
    out_sto_urbs['cap-lo-p'] = 0
    out_sto_urbs['cap-up-p'] = out_sto_urbs['inst-cap-p']
    out_sto_urbs['var-cost-p'] = 0
    out_sto_urbs['var-cost-c'] = 0
    out_sto_urbs['wacc'] = 0
    out_sto_urbs['init'] = 0.5

    ind = out_sto_urbs['Storage'] == 'PumSt'
    out_sto_urbs.loc[ind, 'eff-in'] = 0.92
    out_sto_urbs.loc[ind, 'eff-out'] = 0.92
    out_sto_urbs.loc[ind, 'inv-cost-p'] = 275000
    out_sto_urbs.loc[ind, 'inv-cost-c'] = 0
    out_sto_urbs.loc[ind, 'fix-cost-p'] = 4125
    out_sto_urbs.loc[ind, 'fix-cost-c'] = 0
    out_sto_urbs.loc[ind, 'depreciation'] = 50
    ind = out_sto_urbs['Storage'] == 'Battery'
    out_sto_urbs.loc[ind, 'cap-up-c'] = np.inf
    out_sto_urbs.loc[ind, 'cap-up-p'] = np.inf
    out_sto_urbs.loc[ind, 'eff-in'] = 0.94
    out_sto_urbs.loc[ind, 'eff-out'] = 0.94
    out_sto_urbs.loc[ind, 'inv-cost-p'] = 75000
    out_sto_urbs.loc[ind, 'inv-cost-c'] = 200000
    out_sto_urbs.loc[ind, 'fix-cost-p'] = 3750
    out_sto_urbs.loc[ind, 'fix-cost-c'] = 10000
    out_sto_urbs.loc[ind, 'depreciation'] = 10

    # Change the order of the columns
    out_sto_urbs = out_sto_urbs[
        ['Site', 'Storage', 'Commodity', 'inst-cap-c', 'cap-lo-c', 'cap-up-c', 'inst-cap-p', 'cap-lo-p',
         'cap-up-p', 'eff-in', 'eff-out', 'inv-cost-p', 'inv-cost-c', 'fix-cost-p', 'fix-cost-c',
         'var-cost-p', 'var-cost-c', 'depreciation', 'wacc', 'init']]

    return output_sto_evrys, out_sto_urbs


def format_transmission_model(icl_final, paths, param):
    # evrys
    output_evrys = pd.DataFrame(icl_final,
                                columns=['SitIn', 'SitOut', 'Co', 'var-cost', 'inst-cap', 'act-lo', 'act-up',
                                         'reactance',
                                         'cap-up-therm', 'angle-up', 'length', 'tr_type', 'PSTmax', 'idx'])

    output_evrys['SitIn'] = icl_final['Region_start']
    output_evrys['SitOut'] = icl_final['Region_end']
    output_evrys['Co'] = 'Elec'
    output_evrys['var-cost'] = 0
    output_evrys['inst-cap'] = output_evrys['cap-up-therm'] = icl_final['Capacity_MVA']
    output_evrys['act-lo'] = 0
    output_evrys['act-up'] = 1
    output_evrys['reactance'] = icl_final['X_ohm'].astype(float)
    output_evrys['angle-up'] = 45
    output_evrys['PSTmax'] = 0
    output_evrys['idx'] = np.arange(1, len(output_evrys) + 1)

    # Length of lines based on distance between centroids
    coord = pd.read_csv(paths["sites"], sep=';', decimal=',').set_index('Site')
    coord = coord[coord['Population'] > 0]
    output_evrys = output_evrys.join(coord[['Longitude', 'Latitude']], on='SitIn', rsuffix='_1', how='inner')
    output_evrys = output_evrys.join(coord[['Longitude', 'Latitude']], on='SitOut', rsuffix='_2', how='inner')
    output_evrys.reset_index(inplace=True)
    output_evrys['length'] = [distance.distance(tuple(output_evrys.loc[i, ['Latitude', 'Longitude']].astype(float)),
                                                tuple(output_evrys.loc[i, ['Latitude_2', 'Longitude_2']].astype(
                                                    float))).km
                              for i in output_evrys.index]
    output_evrys.drop(['Longitude', 'Latitude', 'Longitude_2', 'Latitude_2', 'index'], axis=1, inplace=True)
    output_evrys = output_evrys.set_index(['SitIn', 'SitOut'])

    # Create a dataframe to store all the possible combinations of pairs of 1st order neighbors
    df = pd.DataFrame(columns=['SitIn', 'SitOut'])
    zones = param["zones"]
    weights = param["weights"]
    for z in range(len(zones)):
        for n in weights.neighbors[z]:
            if (zones[z] < zones[n]) & ~(zones[z].endswith('_off') | zones[n].endswith('_off')):
                df = df.append(pd.DataFrame([[zones[z], zones[n]]], columns=['SitIn', 'SitOut']), ignore_index=True)

    # urbs

    # Set SitIn and SitOut as index
    df.set_index(['SitIn', 'SitOut'], inplace=True)

    # Get the capacities of existing lines
    df_joined = df.join(output_evrys, how='outer')

    # Set the capacity of inexistant lines to zero
    df_joined['inst-cap'].fillna(0, inplace=True)

    # Reset the index
    df_joined.reset_index(drop=False, inplace=True)

    # Length of lines based on distance between centroids
    coord = pd.read_csv(paths["sites"], sep=';', decimal=',').set_index('Site')
    coord = coord[coord['Population'] > 0]
    df_joined = df_joined.join(coord[['Longitude', 'Latitude']], on='SitIn', rsuffix='_1', how='inner')
    df_joined = df_joined.join(coord[['Longitude', 'Latitude']], on='SitOut', rsuffix='_2', how='inner')
    df_joined['length'] = [distance.distance(tuple(df_joined.loc[i, ['Latitude', 'Longitude']].astype(float)),
                                             tuple(df_joined.loc[i, ['Latitude_2', 'Longitude_2']].astype(
                                                 float))).km for i in df_joined.index]
    df_joined.drop(['Longitude', 'Latitude', 'Longitude_2', 'Latitude_2'], axis=1, inplace=True)

    output_urbs = df_joined.rename(columns={'SitIn': 'Site In', 'SitOut': 'Site Out', 'Co': 'Commodity'})
    output_urbs['tr_type'].fillna('AC_OHL', inplace=True)
    output_urbs.loc[output_urbs['tr_type'] == 'AC_OHL', 'Transmission'] = 'AC_OHL'
    output_urbs.loc[~(output_urbs['tr_type'] == 'AC_OHL'), 'Transmission'] = 'DC_CAB'
    output_urbs['Commodity'].fillna('Elec', inplace=True)

    # Use the length between the centroids [in km]
    output_urbs.loc[output_urbs['tr_type'] == 'AC_OHL', 'eff'] = 0.92 ** (
            output_urbs.loc[output_urbs['tr_type'] == 'AC_OHL', 'length'] / 1000)
    output_urbs.loc[output_urbs['tr_type'] == 'AC_CAB', 'eff'] = 0.9 ** (
            output_urbs.loc[output_urbs['tr_type'] == 'AC_CAB', 'length'] / 1000)
    output_urbs.loc[output_urbs['tr_type'] == 'DC_OHL', 'eff'] = 0.95 ** (
            output_urbs.loc[output_urbs['tr_type'] == 'DC_OHL', 'length'] / 1000)
    output_urbs.loc[output_urbs['tr_type'] == 'DC_CAB', 'eff'] = 0.95 ** (
            output_urbs.loc[output_urbs['tr_type'] == 'DC_CAB', 'length'] / 1000)

    output_urbs.loc[(output_urbs['tr_type'] == 'AC_OHL')
                    & (output_urbs['length'] < 150), 'inv-cost'] = \
        300 * output_urbs.loc[(output_urbs['tr_type'] == 'AC_OHL')
                              & (output_urbs['length'] < 150), 'length']

    output_urbs.loc[(output_urbs['tr_type'] == 'AC_OHL')
                    & (output_urbs['length'] >= 150), 'inv-cost'] = \
        770 * output_urbs.loc[(output_urbs['tr_type'] == 'AC_OHL')
                              & (output_urbs['length'] >= 150), 'length'] - 70000

    output_urbs.loc[(output_urbs['tr_type'] == 'AC_CAB')
                    & (output_urbs['length'] < 150), 'inv-cost'] = \
        1200 * output_urbs.loc[(output_urbs['tr_type'] == 'AC_CAB')
                               & (output_urbs['length'] < 150), 'length']

    output_urbs.loc[(output_urbs['tr_type'] == 'AC_CAB')
                    & (output_urbs['length'] >= 150), 'inv-cost'] = \
        3080 * output_urbs.loc[(output_urbs['tr_type'] == 'AC_CAB')
                               & (output_urbs['length'] >= 150), 'length'] - 280000

    output_urbs.loc[output_urbs['tr_type'] == 'DC_OHL', 'inv-cost'] = 288 * output_urbs.loc[
        output_urbs['tr_type'] == 'DC_OHL', 'length'] + 160000
    output_urbs.loc[output_urbs['tr_type'] == 'DC_CAB', 'inv-cost'] = 1152 * output_urbs.loc[
        output_urbs['tr_type'] == 'DC_CAB', 'length'] + 160000

    output_urbs.loc[(output_urbs['tr_type'] == 'AC_OHL')
                    | (output_urbs['tr_type'] == 'DC_OHL'), 'fix-cost'] = \
        42 * output_urbs.loc[(output_urbs['tr_type'] == 'AC_OHL')
                             | (output_urbs['tr_type'] == 'DC_OHL'), 'length']

    output_urbs.loc[(output_urbs['tr_type'] == 'AC_CAB')
                    | (output_urbs['tr_type'] == 'DC_CAB'), 'fix-cost'] = \
        21 * output_urbs.loc[(output_urbs['tr_type'] == 'AC_CAB')
                             | (output_urbs['tr_type'] == 'DC_CAB'), 'length']

    output_urbs['var-cost'].fillna(0, inplace=True)
    output_urbs['cap-lo'] = param["dist_ren"]["cap_lo"]
    output_urbs['cap-up'] = output_urbs['inst-cap']
    output_urbs['cap-up'].fillna(0, inplace=True)
    output_urbs['wacc'] = param["pro_sto"]["wacc"]
    output_urbs['depreciation'] = param["grid"]["depreciation"]

    # Change the order of the columns
    output_urbs = output_urbs[
        ['Site In', 'Site Out', 'Transmission', 'Commodity', 'eff', 'inv-cost', 'fix-cost', 'var-cost',
         'inst-cap', 'cap-lo', 'cap-up', 'wacc', 'depreciation']]
    output_urbs.iloc[:, 4:] = output_urbs.iloc[:, 4:].astype('float64')
    return output_evrys, output_urbs


def read_assumptions_process(assumptions):
    process = {
        "cap_lo": dict(zip(assumptions['Process'], assumptions['cap-lo'].astype(float))),
        "cap_up": dict(zip(assumptions['Process'], assumptions['cap-up'].astype(float))),
        "max_grad": dict(zip(assumptions['Process'], assumptions['max-grad'].astype(float))),
        "min_fraction": dict(zip(assumptions['Process'], assumptions['min-fraction'].astype(float))),
        "inv_cost": dict(zip(assumptions['Process'], assumptions['inv-cost'].astype(float))),
        "fix_cost": dict(zip(assumptions['Process'], assumptions['fix-cost'].astype(float))),
        "var_cost": dict(zip(assumptions['Process'], assumptions['var-cost'].astype(float))),
        "startup_cost": dict(zip(assumptions['Process'], assumptions['startup-cost'].astype(float))),
        "depreciation": dict(zip(assumptions['Process'], assumptions['depreciation'].astype(float))),
        "area_per_cap": dict(zip(assumptions['Process'], assumptions['area-per-cap'].astype(float))),
        "year_my": dict(zip(assumptions['Process'], assumptions['year_mu'].astype(float))),
        "eff": dict(zip(assumptions['Process'], assumptions['eff'].astype(float))),
        "effmin": dict(zip(assumptions['Process'], assumptions['effmin'].astype(float))),
        "act_up": dict(zip(assumptions['Process'], assumptions['act-up'].astype(float))),
        "act_lo": dict(zip(assumptions['Process'], assumptions['act-lo'].astype(float))),
        "on_off": dict(zip(assumptions['Process'], assumptions['on-off'].astype(float))),
        "start_cost": dict(zip(assumptions['Process'], assumptions['start-cost'].astype(float))),
        "reserve_cost": dict(zip(assumptions['Process'], assumptions['reserve-cost'].astype(float))),
        "ru": dict(zip(assumptions['Process'], assumptions['ru'].astype(float))),
        "rd": dict(zip(assumptions['Process'], assumptions['rd'].astype(float))),
        "rumax": dict(zip(assumptions['Process'], assumptions['rumax'].astype(float))),
        "rdmax": dict(zip(assumptions['Process'], assumptions['rdmax'].astype(float))),
        "cotwo": dict(zip(assumptions['Process'], assumptions['cotwo'].astype(float))),
        "detail": dict(zip(assumptions['Process'], assumptions['detail'].astype(float))),
        "lambda_": dict(zip(assumptions['Process'], assumptions['lambda'].astype(float))),
        "heatmax": dict(zip(assumptions['Process'], assumptions['heatmax'].astype(float))),
        "maxdeltaT": dict(zip(assumptions['Process'], assumptions['maxdeltaT'].astype(float))),
        "heatupcost": dict(zip(assumptions['Process'], assumptions['heatupcost'].astype(float))),
        "su": dict(zip(assumptions['Process'], assumptions['su'].astype(float))),
        "sd": dict(zip(assumptions['Process'], assumptions['sd'].astype(float))),
        "pdt": dict(zip(assumptions['Process'], assumptions['pdt'].astype(float))),
        "hotstart": dict(zip(assumptions['Process'], assumptions['hotstart'].astype(float))),
        "pot": dict(zip(assumptions['Process'], assumptions['pot'].astype(float))),
        "pretemp": dict(zip(assumptions['Process'], assumptions['pretemp'].astype(float))),
        "preheat": dict(zip(assumptions['Process'], assumptions['preheat'].astype(float))),
        "prestate": dict(zip(assumptions['Process'], assumptions['prestate'].astype(float))),
        "year_mu": dict(zip(assumptions['Process'], assumptions['year_mu'].astype(float))),
        "year_stdev": dict(zip(assumptions['Process'], assumptions['year_stdev'].astype(float)))
    }

    return process


def read_assumptions_storage(assumptions):
    storage = {
        "cap_lo_c": dict(zip(assumptions['Storage'], assumptions['cap-lo-c'].astype(float))),
        "cap_lo_p": dict(zip(assumptions['Storage'], assumptions['cap-lo-p'].astype(float))),
        "cap_up_c": dict(zip(assumptions['Storage'], assumptions['cap-up-c'].astype(float))),
        "cap_up_p": dict(zip(assumptions['Storage'], assumptions['cap-up-p'].astype(float))),
        "inv_cost_c": dict(zip(assumptions['Storage'], assumptions['inv-cost-c'].astype(float))),
        "fix_cost_c": dict(zip(assumptions['Storage'], assumptions['fix-cost-c'].astype(float))),
        "var_cost_c": dict(zip(assumptions['Storage'], assumptions['var-cost-c'].astype(float))),
        "inv_cost_p": dict(zip(assumptions['Storage'], assumptions['inv-cost-p'].astype(float))),
        "fix_cost_p": dict(zip(assumptions['Storage'], assumptions['fix-cost-p'].astype(float))),
        "var_cost_p": dict(zip(assumptions['Storage'], assumptions['var-cost-p'].astype(float))),
        "depreciation": dict(zip(assumptions['Storage'], assumptions['depreciation'].astype(float))),
        "init": dict(zip(assumptions['Storage'], assumptions['init'].astype(float))),
        "eff_in": dict(zip(assumptions['Storage'], assumptions['eff-in'].astype(float))),
        "eff_out": dict(zip(assumptions['Storage'], assumptions['eff-out'].astype(float))),
        "var_cost_pi": dict(zip(assumptions['Storage'], assumptions['var-cost-pi'].astype(float))),
        "var_cost_po": dict(zip(assumptions['Storage'], assumptions['var-cost-po'].astype(float))),
        "act_up_pi": dict(zip(assumptions['Storage'], assumptions['act-up-pi'].astype(float))),
        "act_lo_pi": dict(zip(assumptions['Storage'], assumptions['act-lo-pi'].astype(float))),
        "act_up_po": dict(zip(assumptions['Storage'], assumptions['act-up-po'].astype(float))),
        "act_lo_po": dict(zip(assumptions['Storage'], assumptions['act-lo-po'].astype(float))),
        "act_lo_c": dict(zip(assumptions['Storage'], assumptions['act-lo-c'].astype(float))),
        "act_up_c": dict(zip(assumptions['Storage'], assumptions['act-up-c'].astype(float))),
        "prepowin": dict(zip(assumptions['Storage'], assumptions['prepowin'].astype(float))),
        "prepowout": dict(zip(assumptions['Storage'], assumptions['prepowout'].astype(float))),
        "ru": dict(zip(assumptions['Storage'], assumptions['ru'].astype(float))),
        "rd": dict(zip(assumptions['Storage'], assumptions['rd'].astype(float))),
        "rumax": dict(zip(assumptions['Storage'], assumptions['rumax'].astype(float))),
        "rdmax": dict(zip(assumptions['Storage'], assumptions['rdmax'].astype(float))),
        "seasonal": dict(zip(assumptions['Storage'], assumptions['seasonal'].astype(float))),
        "ctr": dict(zip(assumptions['Storage'], assumptions['ctr'].astype(float))),
        "year_mu": dict(zip(assumptions['Storage'], assumptions['year_mu'].astype(float))),
        "year_stdev": dict(zip(assumptions['Storage'], assumptions['year_stdev'].astype(float))),
        "discharge": dict(zip(assumptions['Storage'], assumptions['discharge'].astype(float)))
    }

    return storage
