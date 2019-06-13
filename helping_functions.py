from osgeo import gdal, ogr, gdalnumeric
from osgeo.gdalconst import GA_ReadOnly
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import geometry
from shapely.geometry import Point
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
        df_year.to_hdf('savetimeseries_temp.hdf', 'df')
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

    # Reshape Renamed_df
    df_reshaped_renamed = pd.DataFrame(df_renamed.loc[:, df_renamed.columns != 'Country'].T.to_numpy(), columns=df_renamed['Country'])

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


def map_power_plants(p, x, y, c, paths):
    outSHPfn = paths["map_power_plants"] + p + '.shp'

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
        print('Already depreciated processes:\n')
        print(str(len(raw)-len(current)) + '# process have been removed')
    else:
        current = raw.copy()
        print('Number of current processes: ' + str(len(current)))
    return current


def get_sites(current, paths):

    # Get regions from shapefile
    regions = gpd.read_file(paths["SHP"])
    regions["geometry"] = regions.buffer(0)

    # Spacial join
    located = gpd.sjoin(current, regions[["NAME_SHORT", "geometry"]], how='left', op='intersects')
    located.rename(columns={'NAME_SHORT': 'Site'}, inplace=True)

    # Remove duplicates that lie in the border between land and sea
    located.drop_duplicates(subset=["CoIn", "Pro", "inst-cap", "year", "Site"], inplace=True)

    # Remove duplicates that lie in two different zones
    located = located.loc[~located.index.duplicated(keep='last')]

    located.dropna(axis=0, subset=["Site"], inplace=True)

    return located


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
            df.loc[idx, ('Region_start')], df.loc[idx, ('Region_end')] = df.loc[idx, ('Region_end')], df.loc[
                idx, ('Region_start')]
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
        if (df.iloc[idx, 0] == df.iloc[idx + 1, 0]) & (
                df.iloc[idx, 1] == df.iloc[idx + 1, 1]):  # & (df.iloc[idx,2] == df.iloc[idx+1,2]):
            df.iloc[idx, 26] = df.iloc[idx, 26] + df.iloc[idx + 1, 26]  # Capacity MVA
            df.iloc[idx, 23] = 1 / (1 / df.iloc[idx, 23] + 1 / df.iloc[idx + 1, 23])  # Specific resistance Ohm/km
            df.iloc[idx, 13] = df.iloc[idx, 13] + df.iloc[idx + 1, 13]  # Length
            df = df.drop(df.index[idx + 1])
        else:
            idx += 1

    df_final = df
    return df_final
