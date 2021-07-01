from .input_maps import generate_protected_areas
from .util import *


def define_spatial_scope(scope_shp):
    """
    This function reads the spatial scope shapefile and returns its bounding box.
    
    :param scope_shp: Spatial scope shapefile.
    :type scope_shp: Geopandas dataframe
    
    :return box: List of the bounding box coordinates.
    :rtype: list
    """
    scope_shp = scope_shp.to_crs({"init": "epsg:4326"})
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
        [
            np.ceil((Crd_regions[:, 0] + res_weather[0] / 2) / res_weather[0]) * res_weather[0] - res_weather[0] / 2,
            np.ceil(Crd_regions[:, 1] / res_weather[1]) * res_weather[1],
            np.floor((Crd_regions[:, 2] + res_weather[0] / 2) / res_weather[0]) * res_weather[0] - res_weather[0] / 2,
            np.floor(Crd_regions[:, 3] / res_weather[1]) * res_weather[1],
        ]
    )
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
    Ind = np.array(
        [
            (Crd[:, 0] - Crd_all[2]) / res[0],
            (Crd[:, 1] - Crd_all[3]) / res[1],
            (Crd[:, 2] - Crd_all[2]) / res[0] + 1,
            (Crd[:, 3] - Crd_all[3]) / res[1] + 1,
        ]
    )
    Ind = np.transpose(Ind.astype(int))
    return Ind


def ind_global(Crd, res_desired):
    """
    This function converts longitude and latitude coordinates into indices on a global data scope, where the origin is at (-90, -180).

    :param Crd: Coordinates to be converted into indices.
    :type Crd: numpy array
    :param res_desired: Desired resolution in the vertical and horizontal dimensions.
    :type res_desired: list

    :return Ind: Indices on a global data scope.
    :rtype: numpy array
    """
    if len(Crd.shape) == 1:
        Crd = Crd[np.newaxis]
    Ind = np.array(
        [
            np.round((90 - Crd[:, 0]) / res_desired[0]) + 1,
            np.round((180 + Crd[:, 1]) / res_desired[1]),
            np.round((90 - Crd[:, 2]) / res_desired[0]),
            np.round((180 + Crd[:, 3]) / res_desired[1]) + 1,
        ]
    )
    Ind = np.transpose(Ind.astype(int))
    return Ind


def crd_exact_points(Ind_points, Crd_all, res):
    """
    This function converts indices of points in high resolution rasters into longitude and latitude coordinates.

    :param Ind_points: Tuple of arrays of indices in the vertical and horizontal axes.
    :type Ind_points: tuple of arrays
    :param Crd_all: Array of coordinates of the bounding box of the spatial scope.
    :type Crd_all: numpy array
    :param res: Data resolution in the vertical and horizontal dimensions.
    :type res: list
    
    :return Crd_points: Coordinates of the points in the vertical and horizontal dimensions.
    :rtype: list of arrays
    """
    Crd_points = [Ind_points[0] * res[0] + Crd_all[2], Ind_points[1] * res[1] + Crd_all[3]]
    return Crd_points


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
    GeoRef = {
        "RasterOrigin": [Crd_all[3], Crd_all[0]],
        "RasterOrigin_alt": [Crd_all[3], Crd_all[2]],
        "pixelWidth": res_desired[1],
        "pixelHeight": -res_desired[0],
    }
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

    if region.geometry.geom_type == "MultiPolygon":
        features = [feature for feature in region.geometry]
    else:
        features = [region.geometry]
    west = origin[0]
    south = origin[1]
    profile = {
        "driver": "GTiff",
        "height": M,
        "width": N,
        "count": 1,
        "dtype": rasterio.float64,
        "crs": "EPSG:4326",
        "transform": rasterio.transform.from_origin(west, south, GeoRef["pixelWidth"], GeoRef["pixelHeight"]),
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as f:
            f.write(A_region, 1)
            out_image, out_transform = mask.mask(f, features, crop=False, nodata=0, all_touched=False, filled=True)

        A_region = out_image[0]

    return A_region


def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array):
    """
    This function saves array to geotiff raster format based on EPSG 4326.

    :param newRasterfn: Output path of the raster.
    :type newRasterfn: string
    :param rasterOrigin: Latitude and longitude of the Northwestern corner of the raster.
    :type rasterOrigin: list of two floats
    :param pixelWidth:  Pixel width (might be negative).
    :type pixelWidth: integer
    :param pixelHeight: Pixel height (might be negative).
    :type pixelHeight: integer
    :param array: Array to be converted into a raster.
    :type array: numpy array

    :return: The raster file will be saved in the desired path *newRasterfn*.
    :rtype: None
    """
    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName("GTiff")
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float64, ["COMPRESS=PACKBITS"])
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(np.flipud(array))
    outband.FlushCache()
    outband = None


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
    subregions["geometry"] = subregions.buffer(0)
    countries = param["regions_land"]
    data = []
    for index, subregion in subregions.iterrows():
        for index2, country in countries.iterrows():
            if subregion["geometry"].intersects(country["geometry"]):
                data.append(
                    {
                        "geometry": subregion["geometry"].intersection(country["geometry"]),
                        "NAME_SHORT": subregion["NAME_SHORT"] + "_" + country["GID_0"],
                    }
                )

    # Clean data
    i = 0
    list_length = len(data)
    while i < list_length:
        if data[i]["geometry"].geom_type == "Polygon":
            data[i]["geometry"] = geometry.multipolygon.MultiPolygon([data[i]["geometry"]])
        if not (data[i]["geometry"].geom_type == "Polygon" or data[i]["geometry"].geom_type == "MultiPolygon"):
            del data[i]
            list_length = list_length - 1
        else:
            i = i + 1

    # Create GeoDataFrame
    intersection = gpd.GeoDataFrame(data, columns=["geometry", "NAME_SHORT"])
    intersection.to_file(paths["intersection_subregions_countries"])
    create_json(
        paths["intersection_subregions_countries"],
        param,
        ["region_name", "subregions_name", "Crd_all"],
        paths,
        ["spatial_scope", "Countries", "subregions"],
    )
    return intersection


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
    timecheck("Start")
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
    if "Population" in raster_dict.keys():
        df_columns = df_columns + ["RES"]
        other_keys.remove("Population")
    if "Landuse" in raster_dict.keys():
        df_columns = df_columns + param["landuse_types"]
        other_keys.remove("Landuse")
    if len(other_keys):
        df_columns = df_columns + other_keys
    df = pd.DataFrame(0, index=regions_shp.index, columns=df_columns)

    status = 0
    for reg in range(0, nRegions):
        # Show status bar
        display_progress("Calculating statistics ", (nRegions, status))

        # Calculate A_region_extended
        A_region_extended = calc_region(regions_shp.loc[reg], Crd_all, res_desired, GeoRef)
        A_region_extended[A_region_extended == 0] = np.nan

        if "Population" in raster_dict.keys():
            df.loc[reg, "RES"] = np.nansum(A_region_extended * raster_dict["Population"])

        if "Landuse" in raster_dict.keys():
            A_data = A_region_extended * raster_dict["Landuse"].astype(int)
            unique, counts = np.unique(A_data[~np.isnan(A_data)], return_counts=True)
            for element in range(0, len(unique)):
                df.loc[reg, str(int(unique[element]))] = int(counts[element])

        for key in other_keys:
            df.loc[reg, key] = np.nanmax(A_region_extended * raster_dict[key])

        status = status + 1
    timecheck("End")
    return df


def create_shapefiles_of_ren_power_plants(paths, param, inst_cap, tech):
    """
    This module iterates over the countries in the IRENA summary report, applies a mask of each country on a raster of potential of the technology *tech* that
    spans over the whole geographic scope, and calculates a probability distribution for that country which takes into account the potential but also a random factor.
    It selects the pixels with the highest probabilities, such that the number of pixels is equal to the number of units in that country and for that technology.
    After deriving the coordinates of those pixels, it saves them into a shapefile of points for each technology.
    
    :param paths: Dictionary containing the paths for the potential maps of each technology. If a map is missing, the map of protected areas *PA* is used per default.
      It contains also the paths to the final shapefiles *locations_ren*.
    :type paths: dict
    :param param: Dictionary containing information about the coordinates of the bounding box, the resolution, the georeferencing dictionary, geodataframes of land and sea areas,
      several parameters related to the distribution of renewable capacities.
    :type param: dict
    :param inst_cap: Dataframe of the IRENA report after some processing in the module :mod:`distribute_renewable_capacities_IRENA`.
    :type inst_cap: pandas dataframe
    :param tech: Name of the renewable technology, as used in the dictionary *dist_ren* and in the dataframe *inst_cap*.
    :type tech: string
    
    :return: The shapefile of points corresponding to the locations of power plants for the renewable energy technology *tech* is saved in the desired path, along with
      its corresponding metadata in a JSON file.
    :rtype: None
    """
    timecheck(tech + " - Start")

    Crd_all = param["Crd_all"]
    res_desired = param["res_desired"]
    GeoRef = param["GeoRef"]
    nRegions = len(inst_cap["Country/area"].unique())
    raster_path = paths["dist_ren"]["rasters"][tech]
    units = param["dist_ren"]["units"]

    # Read rasters
    try:
        with rasterio.open(raster_path) as src:
            raster = np.flipud(src.read(1))
    except:
        # Use a mask of protected areas
        if not os.path.exists(paths["PA"]):
            generate_protected_areas(paths, param)
        with rasterio.open(paths["PA"]) as src:
            A_protect = np.flipud(src.read(1)).astype(int)
        raster = changem(A_protect, param["dist_ren"]["default_pa_availability"], param["dist_ren"]["default_pa_type"]).astype(float)
    status = 0
    ind_needed = {}
    x = y = p = c = []
    length = len(inst_cap["Country/area"].unique())
    for reg in inst_cap["Country/area"].unique():
        # Show status bar
        display_progress("Distribution for " + tech + ": ", (length, status))

        if not inst_cap.loc[(inst_cap["Country/area"] == reg) & (inst_cap["Technology"] == tech), "Units"].values[0]:
            # Show status bar
            status = status + 1
            display_progress("Distribution for " + tech + ": ", (length, status))
            continue

        # Calculate A_region_extended
        if tech == "WindOff":
            regions_shp = param["regions_sea"]
            mask = regions_shp.loc[regions_shp["ISO_Ter1"] == reg].dissolve(by="ISO_Ter1").squeeze()
            A_region_extended = calc_region(mask, Crd_all, res_desired, GeoRef)
        else:
            regions_shp = param["regions_land"]
            mask = regions_shp.loc[regions_shp["GID_0"] == reg].squeeze()
            A_region_extended = calc_region(mask, Crd_all, res_desired, GeoRef)
        A_region_extended[A_region_extended == 0] = np.nan

        # Calculate potential distribution
        distribution = A_region_extended * raster
        potential = distribution.flatten()

        # Calculate the part of the probability that is based on the potential
        potential_nan = np.isnan(potential) | (potential == 0)
        if np.nanmax(potential) - np.nanmin(potential):
            potential = (potential - np.nanmin(potential)) / (np.nanmax(potential) - np.nanmin(potential))
        else:
            potential = np.zeros(potential.shape)
        potential[potential_nan] = 0

        # Calculate the random part of the probability
        potential_random = np.random.random_sample(potential.shape)
        potential_random[potential_nan] = 0

        # Combine the two parts
        potential_new = (1 - param["dist_ren"]["randomness"]) * potential + param["dist_ren"]["randomness"] * potential_random

        # Sort elements based on their probability and keep the indices
        ind_sort = np.argsort(potential_new, axis=None)  # Ascending
        ind_needed = ind_sort[-int(inst_cap.loc[(inst_cap["Country/area"] == reg) & (inst_cap["Technology"] == tech), "Units"].values) :]

        # Get the coordinates of the power plants and their respective capacities
        power_plants = [units[tech]] * len(ind_needed)
        if inst_cap.loc[(inst_cap["Country/area"] == reg) & (inst_cap["Technology"] == tech), "inst-cap (MW)"].values % units[tech] > 0:
            power_plants[-1] = (
                inst_cap.loc[(inst_cap["Country/area"] == reg) & (inst_cap["Technology"] == tech), "inst-cap (MW)"].values % units[tech]
            )[0]
        y_pp, x_pp = np.unravel_index(ind_needed, distribution.shape)

        Crd_y_pp, Crd_x_pp = crd_exact_points((y_pp, x_pp), Crd_all, res_desired)

        x = x + Crd_x_pp.tolist()
        y = y + Crd_y_pp.tolist()
        p = p + power_plants
        c = c + potential_new[ind_needed].tolist()  # Power_plants

        # Show status bar
        status = status + 1
        display_progress("Distribution for " + tech + ": ", (length, status))
    # Format point locations
    points = [(x[i], y[i]) for i in range(0, len(y))]

    # Create shapefile
    locations_ren = pd.DataFrame()
    locations_ren["geometry"] = [Point(points[i]) for i in range(0, len(points))]
    locations_ren["Technology"] = tech
    locations_ren["Capacity"] = p
    locations_ren["Prob"] = c
    locations_ren = gpd.GeoDataFrame(locations_ren, geometry="geometry", crs={"init": "epsg:4326"})
    locations_ren.to_file(driver="ESRI Shapefile", filename=paths["locations_ren"][tech])
    print("\n")
    create_json(
        paths["locations_ren"][tech],
        param,
        ["region_name", "year", "dist_ren", "Crd_all", "res_desired", "GeoRef"],
        paths,
        ["dist_ren", "PA", "IRENA_summary", "dict_technologies"],
    )
    print("File saved: " + paths["locations_ren"][tech])
    print("\n")
    timecheck(tech + " - End")


def get_sites(points_shp, param):
    """
    This function reads a shapefile of points, then performs a spatial join with a shapefile of regions to associate the names
    of the regions to the attributes of the points. It also removes duplicates and points that lie outside the subregions.
    
    :param points_shp: A shapefile of points (power plants, storage devices, etc.)
    :type points_shp: Geopandas dataframe
    :param param: Dictionary of user-defined parameters, including the shapefile *regions_sub*.
    :type param: dict
    
    :return located: The shapefile of points that are located within the subregions, with the name of the subregion as an attribute.
    :rtype: Geopandas dataframe
    """
    regions = param["regions_sub"]

    # Spatial join
    points_shp = points_shp.to_crs(regions.crs)
    located = gpd.sjoin(points_shp, regions[["NAME_SHORT", "geometry"]], how="left", op="intersects")
    located.rename(columns={"NAME_SHORT": "Site"}, inplace=True)
    located.drop(columns=["index_right"], inplace=True)

    # Remove duplicates that lie in the border between two regions
    located = located.drop_duplicates(subset=["Name"], inplace=False)

    # Remove features that do not lie in any subregion
    located.dropna(axis=0, subset=["Site"], inplace=True)

    return located
