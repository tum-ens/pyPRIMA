from osgeo import gdal, ogr
from osgeo.gdalconst import GA_ReadOnly
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
from shapely import geometry
import sys
import datetime
import inspect
import os

gdal.PushErrorHandler('CPLQuietErrorHandler')


def clean_load_data(paths, param, countries):
    """

    :param paths:
    :param param:
    :param countries:
    :return:
    """
    timecheck('Start')
    # avoid reading the excel file each time(to be removed)
    if not os.path.isfile('savetimeseries_temp.hdf'):
        # Get dataframe with timeseries
        print('reading excel file (might take a bit of time)\n')
        df_raw = pd.read_excel(paths["load_ts"], header=0, skiprows=[0, 1, 2], sep=',', decimal='.')
        print('done')
        # Filter by year
        df_year = df_raw.loc[df_raw['Year'] == param["year"]]
    else:
        df_year = pd.read_hdf('savetimeseries_temp.hdf', 'df')

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

    # Create time series for missing countries
    df_completed = df_reshaped.copy()
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
        industrial_profile_raw = pd.read_excel(paths["profiles"]["IND"], header=0)
        industrial_profile_raw.rename(columns={'Stunde': 'Hour', 'Last': 'Load'}, inplace=True)
        # Reshape the hourly load in one vector, where the rows are the hours of the year
        industrial_profile = np.tile(industrial_profile_raw['Load'].values, 365)
        profiles['IND'] = industrial_profile / industrial_profile.sum()

    # Commercial load
    if 'COM' in param["load"]["sectors"]:
        commercial_profile_raw = pd.read_csv(paths["profiles"]["COM"], sep='[;]', engine='python', decimal=',',
                                             skiprows=[0, 99], header=[0, 1],
                                             skipinitialspace=True)
        # commercial_profile_raw.rename(columns={'Übergangszeit': 'Spring/Fall', 'Sommer': 'Summer',
        #                                        'Werktag': 'Working day', 'Sonntag': 'Sunday', 'Samstag': 'Saturday'},
        #                               inplace=True)
        commercial_profile_raw.rename(columns={'Ãœbergangszeit': 'Spring/Fall', 'Sommer': 'Summer',
                                               'Werktag': 'Working day', 'Sonntag': 'Sunday', 'Samstag': 'Saturday'},
                                      inplace=True)
        # Aggregate from 15 min --> hourly load
        commercial_profile_raw[('Hour', 'All')] = [int(str(commercial_profile_raw.loc[i, ('G0', '[W]')])[:2]) for i in
                                                   commercial_profile_raw.index]
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


def zonal_weighting(shp_path, raster_path, df_load, df_stat, s):

    shp = ogr.Open(shp_path, 1)
    raster = gdal.Open(raster_path)
    lyr = shp.GetLayer()

    # Create memory target raster
    target_ds = gdal.GetDriverByName('GTiff').Create("something",
                                                     raster.RasterXsize,
                                                     raster.RasterYsize,
                                                     1, gdal.GDT_Float32)
    target_ds.SetGeoTransform(raster.GeoTransform())
    target_ds.setProjection(raster.GetProjection())

    # NoData Value
    mem_band = target_ds.GetRasterBand(1)
    mem_band.Fill(0)
    mem_band.SetNoDataValue(0)

    # Add a new field
    if not field_exists('Weight_'+ s, shp_path):
        new_field = ogr.FieldDefn('Weight_' + s, ogr.OFTReal)
        lyr.CreateField(new_field)

    for feat in lyr:
        country = feat.GetField('NAME_SHORT')[:2]
        if s == 'RES':
            feat.SetField('Weight_' + s, df_load[country, s] / df_stat.loc[country, 'RES'])
        else:
            feat.SetField('weight_' + s, df_load[country, s] / df_stat.loc[country, s])
        lyr.SetFeature(feat)
        feat = None

    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr, None, None, [0], ['ALL_TOUCHED=FALSE', 'ATTRIBUTE=Weight_' + s[:3]])

    return


def field_exists(field_name, shp_path):

    shp = ogr.Open(shp_path, 0)
    lyr = shp.GetLayer()
    lyr_dfn = lyr.GetLayerDefn()

    exists = False
    for i in range(lyr_dfn.GetFieldCount()):
        exists = exists or (field_name == lyr_dfn.GetFieldDefn(i).GetName())

    return exists


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