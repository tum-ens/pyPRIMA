from lib.util import *

def define_spatial_scope(scope_shp):
    """
    This function reads the spatial scope shapefile and returns its bounding box.
    
    :param scope_shp: Spatial scope shapefile.
    :type scope_shp: Geopandas dataframe
    
    :return box: List of the bounding box coordinates.
    :rtype: list
    """
    scope_shp = scope_shp.to_crs({'init': 'epsg:4326'})
    r = scope_shp.total_bounds
    box = r[::-1][np.newaxis]
    return box


def crd_merra(Crd_regions, res_weather):
    """
    This function calculates coordinates of the bounding box covering MERRA-2 data.
    
    :param Crd_regions: Coordinates of the bounding boxes of the regions.
    :type Crd_regions: numpy array
    :param res_weather: Weather data resolution.
    :type res_weather: list
    
    :return Crd: Coordinates of the bounding box covering MERRA-2 data for each region.
    :rtype: numpy array
    """

    Crd = np.array(
        [np.ceil((Crd_regions[:, 0] + res_weather[0] / 2) / res_weather[0]) * res_weather[0] - res_weather[0] / 2,
         np.ceil(Crd_regions[:, 1] / res_weather[1]) * res_weather[1],
         np.floor((Crd_regions[:, 2] + res_weather[0] / 2) / res_weather[0]) * res_weather[0] - res_weather[0] / 2,
         np.floor(Crd_regions[:, 3] / res_weather[1]) * res_weather[1]])
    Crd = Crd.T
    return Crd


def ind_merra(Crd, Crd_all, res):
    """
    This function converts longitude and latitude coordinates into indices within the spatial scope of MERRA-2 data.
    
    :param Crd: Coordinates to be converted into indices.
    :type Crd: numpy array
    :param Crd_all: Coordinates of the bounding box of the spatial scope.
    :type Crd_all: numpy array
    :param res: Resolution of the data, for which the indices are produced.
    :type res: list
    
    :return Ind: Indices within the spatial scope of MERRA-2 data.
    :rtype: numpy array
    """
    if len(Crd.shape) == 1:
        Crd = Crd[np.newaxis]
    Ind = np.array([(Crd[:, 0] - Crd_all[2]) / res[0],
                    (Crd[:, 1] - Crd_all[3]) / res[1],
                    (Crd[:, 2] - Crd_all[2]) / res[0] + 1,
                    (Crd[:, 3] - Crd_all[3]) / res[1] + 1])
    Ind = np.transpose(Ind.astype(int))
    return Ind


def calc_geotiff(Crd_all, res_desired):
    """
    This function returns a dictionary containing the georeferencing parameters for geotiff creation,
    based on the desired extent and resolution.
    
    :param Crd_all: Coordinates of the bounding box of the spatial scope.
    :type Crd_all: numpy array
    :param res_desired: Desired data resolution in the vertical and horizontal dimensions.
    :type res_desired: list
    
    :return GeoRef: Georeference dictionary containing *RasterOrigin*, *RasterOrigin_alt*, *pixelWidth*, and *pixelHeight*.
    :rtype: dict
    """
    GeoRef = {"RasterOrigin": [Crd_all[3], Crd_all[0]],
              "RasterOrigin_alt": [Crd_all[3], Crd_all[2]],
              "pixelWidth": res_desired[1],
              "pixelHeight": -res_desired[0]}
    return GeoRef


def calc_region(region, Crd_reg, res_desired, GeoRef):
    """
    This function reads the region geometry, and returns a masking raster equal to 1 for pixels within and 0 outside of
    the region.
    
    :param region: Region geometry
    :type region: Geopandas series
    :param Crd_reg: Coordinates of the region
    :type Crd_reg: list
    :param res_desired: Desired high resolution of the output raster
    :type res_desired: list
    :param GeoRef: Georeference dictionary containing *RasterOrigin*, *RasterOrigin_alt*, *pixelWidth*, and *pixelHeight*.
    :type GeoRef: dict
    
    :return A_region: Masking raster of the region.
    :rtype: numpy array
    """
    latlim = Crd_reg[2] - Crd_reg[0]
    lonlim = Crd_reg[3] - Crd_reg[1]
    M = int(math.fabs(latlim) / res_desired[0])
    N = int(math.fabs(lonlim) / res_desired[1])
    A_region = np.ones((M, N))
    origin = [Crd_reg[3], Crd_reg[2]]

    if region.geometry.geom_type == 'MultiPolygon':
        features = [feature for feature in region.geometry]
    else:
        features = [region.geometry]
    west = origin[0]
    south = origin[1]
    profile = {'driver': 'GTiff',
               'height': M,
               'width': N,
               'count': 1,
               'dtype': rasterio.float64,
               'crs': 'EPSG:4326',
               'transform': rasterio.transform.from_origin(west, south, GeoRef["pixelWidth"], GeoRef["pixelHeight"])}

    with MemoryFile() as memfile:
        with memfile.open(**profile) as f:
            f.write(A_region, 1)
            out_image, out_transform = mask.mask(f, features, crop=False, nodata=0, all_touched=False, filled=True)
        A_region = out_image[0]

    return A_region
    
    
# def ind_global(Crd, res_desired):
    # """
    # This function converts longitude and latitude coordinates into indices on a global data scope, where the origin is at (-90, -180).

    # :param Crd: Coordinates to be converted into indices.
    # :type Crd: numpy array
    # :param res_desired: Desired resolution in the vertical and horizontal dimensions.
    # :type res_desired: list
    
    # :return Ind: Indices on a global data scope.
    # :rtype: numpy array
    # """
    # if len(Crd.shape) == 1:
        # Crd = Crd[np.newaxis]
    # Ind = np.array([np.round((90 - Crd[:, 0]) / res_desired[0]) + 1,
                    # np.round((180 + Crd[:, 1]) / res_desired[1]),
                    # np.round((90 - Crd[:, 2]) / res_desired[0]),
                    # np.round((180 + Crd[:, 3]) / res_desired[1]) + 1])
    # Ind = np.transpose(Ind.astype(int))
    # return Ind
    
def intersection_subregions_countries(paths, param):
    """
    This function reads two geodataframes, and creates a third one that is made of their intersection. The features
    are the different pieces that you would obtain by overlaying the features from the two input layers.
    
    :param paths: Dictionary containing the path to the shapefile that is obtained.
    :type paths: dict
    :param param: Dictionary containing the two input geodataframes (for countries, *regions_land*, and for subregions, *regions_sub*).
    :type param: dict
    
    :return intersection: Geodataframe with the features resulting from the intersection.
    :rtype: geodataframe
    """

    # load shapefiles, and create spatial indexes for both files
    subregions = param["regions_sub"]
    subregions['geometry'] = subregions.buffer(0)
    countries = param["regions_land"]
    data = []
    for index, subregion in subregions.iterrows():
        for index2, country in countries.iterrows():
            if subregion['geometry'].intersects(country['geometry']):
                data.append({'geometry': subregion['geometry'].intersection(country['geometry']),
                             'NAME_SHORT': subregion['NAME_SHORT'] + '_' + country['GID_0']})

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
    intersection.to_file(paths["intersection_subregions_countries"])
    create_json(paths["intersection_subregions_countries"], param, ["region_name", "subregions_name", "Crd_all"],
                paths, ["spatial_scope", "Countries", "subregions"])
    return intersection


# def bbox_to_pixel_offsets(gt, bbox):
    # originX = gt[0]
    # originY = gt[3]
    # pixel_width = gt[1]
    # pixel_height = gt[5]
    # x1 = int((bbox[0] - originX) / pixel_width)
    # x2 = int((bbox[1] - originX) / pixel_width) + 1

    # y1 = int((bbox[3] - originY) / pixel_height)
    # y2 = int((bbox[2] - originY) / pixel_height) + 1

    # xsize = x2 - x1
    # ysize = y2 - y1

    # return x1, y1, xsize, ysize
    
    
def zonal_stats(regions_shp, raster_dict, param):
    """
    This function calculates the zonal statistics for a given shapefile and a dictionary of rasters:
    
      * Population: the sum is calculated.
      * Landuse: the pixels for each land use type are counted.
      * other_keys: the maximum is returned.
      
    :param regions_shp: Geodataframe containing the regions for which the statistics should be calculated.
    :type regions_shp: geodataframe
    :param raster_dict: Dictionary with keys *Population*, *Landuse*, or any other string, and with paths of rasters as values.
    :type raster_dict: dict
    :param param: Dictionary containing *landuse_types*.
    :type param: dict
    
    :return df: Dataframe containing the zonal statistics, with the names of the regions as index and the keys of the statistics as columns.
    :rtype: pandas dataframe
    """
    timecheck('Start')
    Crd_all = param["Crd_all"]
    res_desired = param["res_desired"]
    GeoRef = param["GeoRef"]
    nRegions = len(regions_shp)
    
    # Read rasters
    for key, val in raster_dict.items():
        with rasterio.open(val) as src:
            raster_dict[key] = np.flipud(src.read(1))
        
    # Prepare output
    df_columns = []
    other_keys = list(raster_dict.keys())
    if 'Population' in raster_dict.keys():
        df_columns = df_columns + ['RES']
        other_keys.remove('Population')
    if 'Landuse' in raster_dict.keys():
        df_columns = df_columns + param["landuse_types"]
        other_keys.remove('Landuse')
    if len(other_keys):
        df_columns = df_columns + other_keys
    df = pd.DataFrame(0, index = regions_shp.index, columns = df_columns)
    
    status = 0
    for reg in range(0, nRegions):
        # Show status bar
        status = status + 1
        sys.stdout.write('\r')
        sys.stdout.write('Calculating statistics ' + '[%-50s] %d%%' % (
            '=' * ((status * 50) // nRegions), (status * 100) // nRegions))
        sys.stdout.flush()

        # Calculate A_region_extended
        A_region_extended = calc_region(regions_shp.loc[reg], Crd_all, res_desired, GeoRef)
        A_region_extended[A_region_extended == 0] = np.nan
        
        if 'Population' in raster_dict.keys():
            df.loc[reg, 'RES'] = np.nansum(A_region_extended * raster_dict['Population'])
            
        if 'Landuse' in raster_dict.keys():
            A_data = A_region_extended * raster_dict['Landuse'].astype(int)
            unique, counts = np.unique(A_data[~np.isnan(A_data)], return_counts=True)
            for element in range(0, len(unique)):
                df.loc[reg, str(int(unique[element]))] = int(counts[element])

        for key in other_keys:
            df.loc[reg, key] = np.nanmax(A_region_extended * raster_dict[key])
            
    timecheck('End')
    return df
    
    
# # ## Functions:

# # https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#clip-a-geotiff-with-shapefile

# def world2Pixel(geoMatrix, x, y):
    # """
    # Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    # the pixel location of a geospatial coordinate
    # """
    # ulX = geoMatrix[0]
    # ulY = geoMatrix[3]
    # xDist = geoMatrix[1]
    # yDist = geoMatrix[5]
    # rtnX = geoMatrix[2]
    # rtnY = geoMatrix[4]
    # pixel = int((x - ulX) / xDist)
    # line = int((ulY - y) / xDist)
    # return (pixel, line)


# def rasclip(raster_path, shapefile_path, counter):
    # # Load the source data as a gdalnumeric array
    # srcArray = gdalnumeric.LoadFile(raster_path)

    # # Also load as a gdal image to get geotransform
    # # (world file) info
    # srcImage = gdal.Open(raster_path)
    # geoTrans = srcImage.GetGeoTransform()

    # # Create an OGR layer from a boundary shapefile
    # shapef = ogr.Open(shapefile_path)
    # lyr = shapef.GetLayer(os.path.split(os.path.splitext(shapefile_path)[0])[1])

    # # Filter based on FID
    # lyr.SetAttributeFilter("FID = {}".format(counter))
    # poly = lyr.GetNextFeature()

    # # Convert the polygon extent to image pixel coordinates
    # minX, maxX, minY, maxY = poly.GetGeometryRef().GetEnvelope()
    # ulX, ulY = world2Pixel(geoTrans, minX, maxY)
    # lrX, lrY = world2Pixel(geoTrans, maxX, minY)

    # # Calculate the pixel size of the new image
    # pxWidth = int(lrX - ulX)
    # pxHeight = int(lrY - ulY)

    # clip = srcArray[ulY:lrY, ulX:lrX]

    # # Create pixel offset to pass to new image Projection info
    # xoffset = ulX
    # yoffset = ulY
    # # print("Xoffset, Yoffset = ( %f, %f )" % ( xoffset, yoffset ))

    # # Create a second (modified) layer
    # outdriver = ogr.GetDriverByName('MEMORY')
    # source = outdriver.CreateDataSource('memData')
    # # outdriver = ogr.GetDriverByName('ESRI Shapefile')
    # # source = outdriver.CreateDataSource(mypath+'00 Inputs/maps/dummy.shp')
    # lyr2 = source.CopyLayer(lyr, 'dummy', ['OVERWRITE=YES'])
    # featureDefn = lyr2.GetLayerDefn()
    # # create a new ogr geometry
    # geom = poly.GetGeometryRef().Buffer(-1 / 240)
    # # write the new feature
    # newFeature = ogr.Feature(featureDefn)
    # newFeature.SetGeometryDirectly(geom)
    # lyr2.CreateFeature(newFeature)
    # # here you can place layer.SyncToDisk() if you want
    # newFeature.Destroy()
    # # lyr2 = source.CopyLayer(lyr,'dummy',['OVERWRITE=YES'])
    # lyr2.ResetReading()
    # poly_old = lyr2.GetNextFeature()
    # lyr2.DeleteFeature(poly_old.GetFID())

    # # Create memory target raster
    # target_ds = gdal.GetDriverByName('MEM').Create('', srcImage.RasterXSize, srcImage.RasterYSize, 1, gdal.GDT_Byte)
    # target_ds.SetGeoTransform(geoTrans)
    # target_ds.SetProjection(srcImage.GetProjection())

    # # Rasterize zone polygon to raster
    # gdal.RasterizeLayer(target_ds, [1], lyr2, None, None, [1], ['ALL_TOUCHED=FALSE'])
    # mask = target_ds.ReadAsArray()
    # mask = mask[ulY:lrY, ulX:lrX]

    # # Clip the image using the mask
    # clip = np.multiply(clip, mask).astype(gdalnumeric.float64)
    # return poly.GetField('NAME_SHORT'), xoffset, yoffset, clip
    
    
# def map_power_plants(p, x, y, c, outSHPfn):
    # # Create the output shapefile
    # shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    # if os.path.exists(outSHPfn):
        # shpDriver.DeleteDataSource(outSHPfn)
    # outDataSource = shpDriver.CreateDataSource(outSHPfn)
    # outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint)

    # # create point geometry
    # point = ogr.Geometry(ogr.wkbPoint)
    # # create a field
    # idField = ogr.FieldDefn('CapacityMW', ogr.OFTReal)
    # outLayer.CreateField(idField)
    # # Create the feature
    # featureDefn = outLayer.GetLayerDefn()

    # # Set values
    # for i in range(0, len(x)):
        # point.AddPoint(x[i], y[i])
        # outFeature = ogr.Feature(featureDefn)
        # outFeature.SetGeometry(point)
        # outFeature.SetField('CapacityMW', c[i])
        # outLayer.CreateFeature(outFeature)
    # outFeature = None
    # print("File Saved: " + outSHPfn)


# def map_grid_plants(x, y, paths):
    # outSHPfn = paths["map_grid_plants"]

    # # Create the output shapefile
    # shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    # if os.path.exists(outSHPfn):
        # shpDriver.DeleteDataSource(outSHPfn)
    # outDataSource = shpDriver.CreateDataSource(outSHPfn)
    # outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint)

    # # create point geometry
    # point = ogr.Geometry(ogr.wkbPoint)
    # # Create the feature
    # featureDefn = outLayer.GetLayerDefn()

    # # Set values
    # for i in range(0, len(x)):
        # point.AddPoint(x[i], y[i])
        # outFeature = ogr.Feature(featureDefn)
        # outFeature.SetGeometry(point)
        # outLayer.CreateFeature(outFeature)
    # outFeature = None





# def closest_polygon(geom, polygons):
    # """Returns polygon from polygons that is closest to geom.

    # Args:
        # geom: shapely geometry (used here: a point)
        # polygons: GeoDataFrame of non-overlapping (!) polygons

    # Returns:
        # The polygon from 'polygons' which is closest to 'geom'.
    # """
    # dist = np.inf
    # for poly in polygons.index:
        # if polygons.loc[poly].geometry.convex_hull.exterior.distance(geom) < dist:
            # dist = polygons.loc[poly].geometry.convex_hull.exterior.distance(geom)
            # closest = polygons.loc[poly]
    # return closest


# def containing_polygon(geom, polygons):
    # """Returns polygon from polygons that contains geom.

    # Args:
        # geom: shapely geometry (used here: a point)
        # polygons: GeoDataFrame of non-overlapping (!) polygons

    # Returns:
        # The polygon from 'polygons' which contains (in
        # the way shapely implements it) 'geom'. Throws
        # an error if more than one polygon contain 'geom'.
        # Returns 'None' if no polygon contains it.
    # """
    # try:
        # containing_polygons = polygons[polygons.contains(geom)]
    # except:
        # containing_polygons = []
    # if len(containing_polygons) == 0:
        # return closest_polygon(geom, polygons)
    # if len(containing_polygons) > 1:
        # print(containing_polygons)
        # # raise ValueError('geom lies in more than one polygon!')
    # return containing_polygons.iloc[0]
    
    
# def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array):
    # """
    # Saves array to geotiff raster format based on EPSG 4326.

    # :param newRasterfn: Output path of the raster.
    # :type newRasterfn: string
    # :param rasterOrigin: Latitude and longitude of the Northwestern corner of the raster.
    # :type rasterOrigin: list of two floats
    # :param pixelWidth:  Pixel width (might be negative).
    # :type pixelWidth: integer
    # :param pixelHeight: Pixel height (might be negative).
    # :type pixelHeight: integer
    # :param array: Array to be converted into a raster.
    # :type array: numpy array
    # :return: The raster file will be saved in the desired path *newRasterfn*.
    # :rtype: None
    # """
    # cols = array.shape[1]
    # rows = array.shape[0]
    # originX = rasterOrigin[0]
    # originY = rasterOrigin[1]

    # driver = gdal.GetDriverByName('GTiff')
    # outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float64, ['COMPRESS=PACKBITS'])
    # outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    # outRasterSRS = osr.SpatialReference()
    # outRasterSRS.ImportFromEPSG(4326)
    # outRaster.SetProjection(outRasterSRS.ExportToWkt())
    # outband = outRaster.GetRasterBand(1)
    # outband.WriteArray(np.flipud(array))
    # outband.FlushCache()
    # outband = None