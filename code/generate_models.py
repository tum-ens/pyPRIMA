from helping_functions import *
# from config import ts_paths


def initialization():
    ''' documentation '''
    timecheck('Start')

    # import param and paths
    from config import configuration
    paths, param = configuration()

    # Read shapefile of scope
    scope_shp = gpd.read_file(paths["spatial_scope"])
    param["spatial_scope"] = define_spatial_scope(scope_shp)

    res_weather = param["res_weather"]
    res_desired = param["res_desired"]
    Crd_all = crd_merra(param["spatial_scope"], res_weather)[0]
    param["Crd_all"] = Crd_all
    ymax, xmax, ymin, xmin = Crd_all
    bounds_box = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

    timecheck('Read shapefile of countries')
    # Extract land areas
    countries_shp = gpd.read_file(paths["Countries"], bbox=scope_shp)
    countries_shp = countries_shp.to_crs({'init': 'epsg:4326'})
    
    # Crop all polygons and take the part inside the bounding box
    countries_shp['geometry'] = countries_shp['geometry'].intersection(bounds_box)
    countries_shp = countries_shp[countries_shp.geometry.area > 0]
    param["regions_land"] = countries_shp
    param["nRegions_land"] = len(param["regions_land"])
    
    if not (os.path.exists(paths["LAND"]) and os.path.exists(paths["EEZ"])):
        Crd_regions_land = np.zeros((param["nRegions_land"], 4))
        
        for reg in range(0, param["nRegions_land"]):
            # Box coordinates for MERRA2 data
            r = countries_shp.bounds.iloc[reg]
            box = np.array([r["maxy"], r["maxx"], r["miny"], r["minx"]])[np.newaxis]
            Crd_regions_land[reg, :] = crd_merra(box, res_weather)
        
        timecheck('Read shapefile of EEZ')
        # Extract sea areas
        eez_shp = gpd.read_file(paths["EEZ_global"], bbox=scope_shp)
        eez_shp = eez_shp.to_crs({'init': 'epsg:4326'})
        
        # Crop all polygons and take the part inside the bounding box
        eez_shp['geometry'] = eez_shp['geometry'].intersection(bounds_box)
        eez_shp = eez_shp[eez_shp.geometry.area > 0]
        param["regions_sea"] = eez_shp
        param["nRegions_sea"] = len(param["regions_sea"])
        Crd_regions_sea = np.zeros((param["nRegions_sea"], 4))
        
        for reg in range(0, param["nRegions_sea"]):
            # Box coordinates for MERRA2 data
            r = eez_shp.bounds.iloc[reg]
            box = np.array([r["maxy"], r["maxx"], r["miny"], r["minx"]])[np.newaxis]
            Crd_regions_sea[reg, :] = crd_merra(box, res_weather)
            
        # Saving parameters
        param["Crd_regions"] = np.concatenate((Crd_regions_land, Crd_regions_sea), axis=0)

    timecheck('Read shapefile of subregions')
    # Read shapefile of regions
    regions_shp = gpd.read_file(paths["subregions"], bbox=scope_shp)
    regions_shp = regions_shp.to_crs({'init': 'epsg:4326'})

    # Crop all polygons and take the part inside the bounding box
    regions_shp['geometry'] = regions_shp['geometry'].intersection(bounds_box)
    regions_shp = regions_shp[regions_shp.geometry.area > 0]
    regions_shp.sort_values(by=['NAME_SHORT'], inplace=True)
    regions_shp = regions_shp.reset_index().rename(columns={'index': 'original_index'})
    param["regions_sub"] = regions_shp
    param["nRegions_sub"] = len(param["regions_sub"])
    Crd_regions_sub = np.zeros((param["nRegions_sub"], 4))

    for reg in range(0, param["nRegions_sub"]):
        # Box coordinates for MERRA2 data
        r = regions_shp.bounds.iloc[reg]
        box = np.array([r["maxy"], r["maxx"], r["miny"], r["minx"]])[np.newaxis]
        Crd_regions_sub[reg, :] = crd_merra(box, res_weather)

    # Saving parameters
    param["Crd_subregions"] = Crd_regions_sub
    
    # Indices and matrix dimensions
    Ind_all_low = ind_merra(Crd_all, Crd_all, res_weather)
    Ind_all_high = ind_merra(Crd_all, Crd_all, res_desired)

    param["m_high"] = int((Ind_all_high[:, 0] - Ind_all_high[:, 2] + 1)[0])  # number of rows
    param["n_high"] = int((Ind_all_high[:, 1] - Ind_all_high[:, 3] + 1)[0])  # number of columns
    param["m_low"] = int((Ind_all_low[:, 0] - Ind_all_low[:, 2] + 1)[0])  # number of rows
    param["n_low"] = int((Ind_all_low[:, 1] - Ind_all_low[:, 3] + 1)[0])  # number of columns
    param["GeoRef"] = calc_geotiff(Crd_all, res_desired)
    
    if not (os.path.exists(paths["LAND"]) and os.path.exists(paths["EEZ"])):
        # Generate land and sea rasters
        generate_landsea(paths, param)
        
    timecheck('End')

    # Display initial information
    print('\nRegion: ' + param["subregions_name"] + ' - Year: ' + str(param["year"]))
    print('Folder Path: ' + paths["region"] + '\n')
    return paths, param


def generate_sites_from_shapefile(paths):
    '''
    description
    '''
    timecheck('Start')

    # Initialize region masking parameters
    Crd_all = param["Crd_all"]
    GeoRef = param["GeoRef"]
    res_desired = param["res_desired"]
    nRegions = param["nRegions_sub"]
    regions_shp = param["regions_sub"]
    
    # Initialize dataframe
    regions = pd.DataFrame(0, index=range(0, nRegions),
                           columns=['Name', 'Index_shapefile', 'Area_m2', 'Longitude', 'Latitude',
                                    'slacknode', 'syncarea', 'ctrarea', 'primpos', 'primneg',
                                    'secpos', 'secneg', 'terpos', 'terneg'])
                                    
    # Read masks
    with rasterio.open(paths["LAND"]) as src:
        A_land = src.read(1)
        A_land = np.flipud(A_land).astype(int)
    with rasterio.open(paths["EEZ"]) as src:
        A_sea = src.read(1)
        A_sea = np.flipud(A_sea).astype(int)
    
    status = 0
    for reg in range(0, nRegions):
        # Display Progress
        status += 1
        display_progress('Generating sites ', (nRegions, status))

        # Compute region_mask
        A_region_extended = calc_region(regions_shp.loc[reg], Crd_all, res_desired, GeoRef)
        
        # Get name of region
        if np.nansum(A_region_extended * A_land) > np.nansum(A_region_extended * A_sea):
            regions.loc[reg, "Name"] = regions_shp.loc[reg]["NAME_SHORT"]
        else:
            regions.loc[reg, "Name"] = regions_shp.loc[reg]["NAME_SHORT"] + "_offshore"
    
        # Calculate longitude and latitude of centroids
        regions.loc[reg, 'Longitude'] = regions_shp.geometry.centroid.loc[reg].x
        regions.loc[reg, 'Latitude'] = regions_shp.geometry.centroid.loc[reg].y
        
    import pdb; pdb.set_trace()
    # Calculate area using Lambert Cylindrical Equal Area EPSG:9835
    regions_shp = regions_shp.to_crs('+proj=cea')
    regions['Area_m2'] = regions_shp.geometry.area
    regions_shp = regions_shp.to_crs({'init': 'epsg:4326'})
    
    # Get original index in shapefile
    regions['Index_shapefile'] = regions_shp['original_index']
    
    # Assign slack node
    regions['slacknode'] = 0
    regions.loc[0, 'slacknode'] = 1
    
    # Define synchronous areas and control areas
    regions['syncharea'] = 1
    regions['ctrarea'] = 1
    
    # Define reserves
    regions['primpos'] = 0
    regions['primneg'] = 0
    regions['secpos'] = 0
    regions['secneg'] = 0
    regions['terpos'] = 0
    regions['terneg'] = 0

    # Export model-independent list of regions
    regions.to_csv(paths["sites_sub"], index=False, sep=';', decimal=',')

    # # Preparing output for evrys
    # zones_evrys = zones[['Site', 'Latitude', 'Longitude']].rename(columns={'Latitude': 'lat', 'Longitude': 'long'})
    # zones_evrys['slacknode'] = 0
    # zones_evrys.loc[0, 'slacknode'] = 1
    # zones_evrys['syncharea'] = 1
    # zones_evrys['ctrarea'] = 1
    # zones_evrys['primpos'] = 0
    # zones_evrys['primneg'] = 0
    # zones_evrys['secpos'] = 0
    # zones_evrys['secneg'] = 0
    # zones_evrys['terpos'] = 0
    # zones_evrys['terneg'] = 0
    # zones_evrys = zones_evrys[
        # ['Site', 'slacknode', 'syncharea', 'lat', 'long', 'ctrarea', 'primpos', 'primneg', 'secpos', 'secneg', 'terpos',
         # 'terneg']]

    # # Preparing output for urbs
    # zones_urbs = zones[['Site', 'Area']].rename(columns={'Site': 'Name', 'Area': 'area'})
    # zones_urbs['area'] = zones_urbs['area'] * 1000000  # in m²

    # zones_evrys.to_csv(paths["evrys_sites"], index=False, sep=';', decimal=',')
    # print("File Saved: " + paths["evrys_sites"])
    # zones_urbs.to_csv(paths["urbs_sites"], index=False, sep=';', decimal=',')
    # print("File Saved: " + paths["urbs_sites"])

    timecheck('End')


def generate_intermittent_supply_timeseries(paths, param):
    '''
    description
    '''
    timecheck('Start')
    Timeseries = None

    # Loop over the technologies understudy
    for tech in param["technology"]:
        # Read coefs
        if os.path.isfile(paths["reg_coef"][tech]):
            Coef = pd.read_csv(paths["reg_coef"][tech], sep=';', decimal=',', index_col=[0])
        else:
            print("No regression Coefficients found for " + tech)
            continue

        # Extract hub heights and find the required TS
        hub_heights = pd.Series(Coef.columns).str.slice(3).unique()
        regions = pd.Series(Coef.columns).str.slice(0, 2).unique()
        quantiles = pd.Series(Coef.index)

        # Read the timeseries
        TS = {}
        hh = ''
        paths = ts_paths(hub_heights, tech, paths)
        for height in hub_heights:
            TS[height] = pd.read_csv(paths["raw_TS"][tech][height],
                                     sep=';', decimal=',', header=[0, 1], index_col=[0], dtype=np.float)

        # Prepare Dataframe to be filled
        TS_tech = pd.DataFrame(np.zeros((8760, len(regions))), columns=regions + '.' + tech)
        for reg in regions:
            for height in hub_heights:
                for quan in quantiles:
                    if height != '':
                        TS_tech[reg + '.' + tech] = TS_tech[reg + '.' + tech] + \
                                                    (TS[height][reg, 'q' + str(quan)] * Coef[reg + '_' + height].loc[
                                                        quan])
                    else:
                        TS_tech[reg + '.' + tech] = TS_tech[reg + '.' + tech] + \
                                                    (TS[height][reg, 'q' + str(quan)] * Coef[reg].loc[quan])
        TS_tech.set_index(np.arange(1, 8761), inplace=True)
        if Timeseries is None:
            Timeseries = TS_tech.copy()
        else:
            Timeseries = pd.concat([Timeseries, TS_tech], axis=1)

    Timeseries.to_csv(paths["suplm_TS"], sep=';', decimal=',')
    print("File Saved: " + paths["suplm_TS"])
    Timeseries.to_csv(paths["urbs_suplm"], sep=';', decimal=',')
    print("File Saved: " + paths["urbs_suplm"])
    timecheck('End')


def generate_load_timeseries(paths, param):
    """
    """

    timecheck('Start')

    # Sector land use allocation
    sector_lu = pd.read_csv(paths["assumptions_landuse"], index_col=0, sep=";", decimal=",")
    shared_sectors = set(sector_lu.columns).intersection(set(param["load"]["sectors"]))
    sector_lu = sector_lu[sorted(list(shared_sectors))]
    shared_sectors.add('RES')
    if not shared_sectors == set(param["load"]["sectors"]):
        warn('The following sectors are not included in ' + paths["assumptions_landuse"] + ": " + str(set(param["load"]["sectors"]) - shared_sectors), UserWarning)
    
    landuse_types = [str(i) for i in sector_lu.index]
    param["landuse_types"] = landuse_types
    # Normalize the land use coefficients found in the assumptions table over each sector
    sector_lu = sector_lu.transpose().div(np.repeat(sector_lu.sum(axis=0)[:, None], len(sector_lu), axis=1))
    sec = [str(i) for i in sector_lu.index]
    

    # Share of sectors in electricity demand
    sec_share = clean_sector_shares(paths, param)
    
    # Create landuse and population maps, if they do not exist already
    if not os.path.exists(paths["LU"]):
        generate_landuse(paths, param)
    if not os.path.exists(paths["POP"]):
        generate_population(paths, param)
    
    # Count pixels of each land use type and create weighting factors for each country
    if not os.path.exists(paths["stats_countries"]):
        df = zonal_stats2(param["regions_land"],
                          {'Population': paths["POP"],
                           'Landuse': paths["LU"]},
                           param)
        stat = param["regions_land"][["GID_0"]].rename(columns={'GID_0': 'Country'}).join(df).set_index('Country')
        stat.to_csv(paths["stats_countries"], sep=";", decimal=",", index=True)
    else:
        stat = pd.read_csv(paths["stats_countries"], sep=";", decimal=",", index_col=0)
    
    # Weighting by sector
    for s in sec:
        stat.loc[:, s] = np.dot(stat.loc[:, landuse_types], sector_lu.loc[s])

    if not (os.path.isfile(paths["df_sector"]) and
            os.path.isfile(paths["load_sector"]) and
            os.path.isfile(paths["load_landuse"])):

        # Get dataframe with cleaned timeseries for countries
        df_load_countries = clean_load_data(paths, param)
        countries = param["regions_land"].rename(columns={'GID_0': 'Country'})
        
        # Get sectoral normalized profiles
        profiles = get_sectoral_profiles(paths, param)
        
        # Prepare an empty table of the hourly load for the five sectors in each countries.
        df_sectors = pd.DataFrame(0, index=df_load_countries.index, columns=pd.MultiIndex.from_product(
            [df_load_countries.columns.tolist(), param["load"]["sectors"]], names=['Country', 'Sector']))
        
        # Copy the load profiles for each sector in the columns of each country, and multiply each sector by the share
        # defined in 'sec_share'. Note that at the moment the values are the same for all countries
        for c in df_load_countries.columns:
            for s in param["load"]["sectors"]:
                try:
                    df_sectors.loc[:, (c, s)] = profiles[s] * sec_share.loc[c, s]
                except KeyError:
                    df_sectors.loc[:, (c, s)] = profiles[s] * sec_share.loc[param["load"]["default_sec_shares"], s]
        
        # Normalize the load profiles over all sectors by the hour so that the sum of the loads of all sectors = 1
        # for each hour, then multiply with the actual hourly loads for each country
        df_scaling = df_sectors.groupby(level=0, axis=1).sum()
        for c in df_load_countries.columns:
            for s in sec + ['RES']:
                df_sectors.loc[:, (c, s)] = df_sectors.loc[:, (c, s)] / df_scaling[c] * df_load_countries[c]
        
        # Calculate the yearly load per sector and country
        load_sector = df_sectors.sum(axis=0).rename('Load in MWh')
        
        # Prepare dataframe load_landuse, that calculates the hourly load for each land use unit in each country
        rows = landuse_types.copy()
        rows.append('RES')
        countries = sorted(list(set(stat.index.tolist()).intersection(set(df_load_countries.columns))))
        m_index = pd.MultiIndex.from_product([countries, rows], names=['Country', 'Land use'])
        load_landuse = pd.DataFrame(0, index=m_index, columns=df_sectors.index)
        
        status = 0
        length = len(countries) * len(landuse_types) * len(sec)
        display_progress("Computing regions load", (length, status))
        for c in countries:  # Countries
            load_landuse.loc[c, 'RES'] = load_landuse.loc[c, 'RES'] \
                                         + df_sectors[(c, 'RES')] \
                                         / stat.loc[c, 'RES']
            for lu in landuse_types:  # Land use types
                for s in sec:  # other sectors
                    load_landuse.loc[c, lu] = load_landuse.loc[c, lu] \
                                              + sector_lu.loc[s, int(lu)] \
                                              * df_sectors[(c, s)] \
                                              / stat.loc[c, s]
                    status = status + 1
                    display_progress("Computing regions load", (length, status))
        
        # Save the data into HDF5 files for faster execution
        df_sectors.to_csv(paths["df_sector"], sep=';', decimal=',', index=False, header=True)
        print("Dataframe with time series for each country and sector saved: " + paths["df_sector"])
        load_sector.to_csv(paths["load_sector"], sep=';', decimal=',', index=True, header=True)
        print("Dataframe with yearly demand for each country and sector saved: " + paths["load_sector"])
        load_landuse.to_csv(paths["load_landuse"], sep=';', decimal=',', index=True)
        print("Dataframe with time series for each land use pixel saved: " + paths["load_landuse"])

    # Read CSV files
    df_sectors = pd.read_csv(paths["df_sector"], sep=';', decimal=',', header=[0, 1])
    load_sector = pd.read_csv(paths["load_sector"], sep=';', decimal=',', index_col=[0, 1])["Load in MWh"]
    load_landuse = pd.read_csv(paths["load_landuse"], sep=';', decimal=',', index_col=[0, 1])

    # Split subregions into country parts
    # (a subregion can overlap with many countries, but a country part belongs to only one country)
    reg_intersection = intersection_subregions_countries(paths, param)

    # Count number of pixels for each country part
    if not os.path.exists(paths["stats_country_parts"]):
        df = zonal_stats2(reg_intersection,
                          {'Population': paths["POP"],
                           'Landuse': paths["LU"]},
                           param)
        stat_sub = reg_intersection[["NAME_SHORT"]].rename(columns={'NAME_SHORT': 'Country_part'}).join(df).set_index('Country_part')
        stat_sub.to_csv(paths["stats_country_parts"], sep=";", decimal=",", index=True)
    else:
        stat_sub = pd.read_csv(paths["stats_country_parts"], sep=";", decimal=",", index_col=0)

    # Add attributes for country/region
    stat_sub['Region'] = 0
    stat_sub['Country'] = 0
    for i in stat_sub.index:
        stat_sub.loc[i, ['Region', 'Country']] = i.split('_')
        if stat_sub.loc[i, 'Country'] not in list(df_sectors.columns.get_level_values(0).unique()):
            stat_sub.drop(index=i, inplace=True)

    # Prepare dataframe to save the hourly load in each country part
    load_country_part = pd.DataFrame(0, index=stat_sub.index,
                                     columns=df_sectors.index.tolist() + ['Region', 'Country'])
    load_country_part[['Region', 'Country']] = stat_sub[['Region', 'Country']]
    
    # Calculate the hourly load for each subregion
    status = 0
    length = len(load_country_part.index) * len(landuse_types)
    display_progress("Computing sub regions load:", (length, status))
    for cp in load_country_part.index:
        c = load_country_part.loc[cp, 'Country']
        # For residential:
        load_country_part.loc[cp, df_sectors.index.tolist()] = load_country_part.loc[cp, df_sectors.index.tolist()] \
                                                             + stat_sub.loc[cp, 'RES'] \
                                                             * load_landuse.loc[c, 'RES'].to_numpy()
        for lu in landuse_types:
            load_country_part.loc[cp, df_sectors.index.tolist()] = load_country_part.loc[cp, df_sectors.index.tolist()] \
                                                                 + stat_sub.loc[cp, lu] \
                                                                 * load_landuse.loc[c, lu].to_numpy()
            # show_progress
            status = status + 1
            display_progress("Computing load in country parts", (length, status))

    # Aggregate into subregions
    load_regions = load_country_part.groupby(['Region', 'Country']).sum()
    load_regions.reset_index(inplace=True)
    load_regions = load_regions.groupby(['Region']).sum().T
    
    # Output
    load_regions.to_csv(paths["load_regions"], sep=';', decimal=',', index=True)
    print("File saved: " + paths["load_regions"])

    timecheck('End')


def generate_commodity(paths, param):
    ''' documentation '''
    timecheck('Start')

    assumptions = pd.read_excel(paths["assumptions"], sheet_name='Commodity')
    commodity = list(assumptions['Commodity'].unique())

    dict_price_instate = dict(zip(assumptions['Commodity'], assumptions['price mid']))
    dict_price_outofstate = dict(zip(assumptions['Commodity'], assumptions['price out-of-state']))
    dict_type_evrys = dict(zip(assumptions['Commodity'], assumptions['Type_evrys']))
    dict_type_urbs = dict(zip(assumptions['Commodity'], assumptions['Type_urbs']))
    dict_annual = dict(zip(assumptions['Commodity'], assumptions['annual']))
    dict_co_max = dict(zip(assumptions['Commodity'], assumptions['max']))
    dict_maxperstep = dict(zip(assumptions['Commodity'], assumptions['maxperstep']))

    # Read the CSV containing the list of sites
    sites = pd.read_csv(paths["sites"], sep=';', decimal=',')

    # Read the CSV containing the annual load
    load = pd.read_csv(paths["annual_load"], index_col=['sit'])

    # Prepare output tables for evrys and urbs

    output_evrys = pd.DataFrame(columns=['Site', 'Co', 'price', 'annual', 'losses', 'type'], dtype=np.float64)
    output_urbs = pd.DataFrame(columns=['Site', 'Commodity', 'Type', 'price', 'max', 'maxperhour'])

    # Fill tables
    for s in sites["Site"]:
        for c in commodity:
            if c == 'Elec':
                if s in load.index:
                    annual = load.loc[s][0]
                else:
                    annual = 0
            else:
                annual = dict_annual[c]
            if len(s) >= 2:
                output_evrys = output_evrys.append(
                    {'Site': s, 'Co': c, 'price': dict_price_instate[c], 'annual': annual, 'losses': 0,
                     'type': dict_type_evrys[c]}, ignore_index=True)
                output_urbs = output_urbs.append(
                    {'Site': s, 'Commodity': c, 'Type': dict_type_urbs[c], 'price': dict_price_instate[c],
                     'max': dict_co_max[c], 'maxperhour': dict_maxperstep[c]}, ignore_index=True)
            else:
                output_evrys = output_evrys.append(
                    {'Site': s, 'Co': c, 'price': dict_price_outofstate[c], 'annual': annual, 'losses': 0,
                     'type': dict_type_evrys[c]}, ignore_index=True)
                output_urbs = output_urbs.append(
                    {'Site': s, 'Commodity': c, 'Type': dict_type_urbs[c], 'price': dict_price_outofstate[c],
                     'max': dict_co_max[c], 'maxperhour': dict_maxperstep[c]}, ignore_index=True)

    output_urbs.to_csv(paths["urbs_commodity"], index=False, sep=';', decimal=',')
    print("File Saved: " + paths["urbs_commodity"])

    output_evrys.to_csv(paths["evrys_commodity"], index=False, sep=';', decimal=',')
    print("File Saved: " + paths["evrys_commodity"])

    timecheck('End')


def distribute_renewable_capacities(paths, param):
    ''' documentation '''
    timecheck("Start")

    # Shapefile with countries
    countries = gpd.read_file(paths["Countries"])

    # Countries to be considered
    sites = pd.DataFrame(countries[['NAME_SHORT']].rename(columns={'NAME_SHORT': 'Site'}))
    sites = sites.sort_values(by=['Site'], axis=0)['Site'].unique()

    # Read input file, extracted from IRENA
    data_raw = pd.read_excel(paths["IRENA"], skiprows=[0, 1, 2, 3, 4, 5, 6])

    # Add missing country names
    for i in np.arange(1, len(data_raw.index)):
        if data_raw.isnull().loc[i, 'Country/area']:
            data_raw.loc[i, 'Country/area'] = data_raw.loc[i - 1, 'Country/area']

    # Select technologies needed in urbs and rename them
    data_raw = data_raw.loc[data_raw["Technology"].isin(param["dist_ren"]["renewables"])].reset_index(drop=True)
    data_raw["Technology"] = data_raw["Technology"].replace(param["dist_ren"]["renewables"])
    data_raw = data_raw.rename(columns={'Country/area': 'Site', 'Technology': 'Process', 2015: 'inst-cap'})

    # Create new dataframe with needed information, rename sites and extract chosen sites
    data = data_raw[["Site", "Process", "inst-cap"]]
    data = data.replace({"Site": param["dist_ren"]["country_names"]}).fillna(value=0)
    data = data.loc[data["Site"].isin(sites)].reset_index(drop=True)

    # Group by and sum
    data = data.groupby(["Site", "Process"]).sum().reset_index()

    # Estimate number of units
    units = param["dist_ren"]["units"]
    for p in data["Process"].unique():
        data.loc[data["Process"] == p, "Unit"] = data.loc[data["Process"] == p, "inst-cap"] // units[p] \
                                                 + (data.loc[data["Process"] == p, "inst-cap"] % units[p] > 0)
    for p in data["Process"].unique():
        x = y = c = []
        for counter in range(0, len(countries) - 1):
            print(counter)
            if float(data.loc[(data["Site"] == countries.loc[counter, "NAME_SHORT"]) & (
                    data["Process"] == p), 'inst-cap']) == 0:
                continue
            if (countries.loc[counter, "Population"]) & (p == 'WindOff'):
                continue
            if (countries.loc[counter, "Population"] == 0) & (p != 'WindOff'):
                continue
            name, x_off, y_off, potential = rasclip(paths["rasters"][p], paths["Countries"], counter)
            raster_shape = potential.shape
            potential = potential.flatten()

            # Calculate the part of the probability that is based on the potential
            potential_nan = np.isnan(potential) | (potential == 0)
            potential = (potential - np.nanmin(potential)) / (np.nanmax(potential) - np.nanmin(potential))
            potential[potential_nan] = 0

            # Calculate the random part of the probability
            potential_random = np.random.random_sample(potential.shape)
            potential_random[potential_nan] = 0

            # Combine the two parts
            potential_new = (1 - param["dist_ren"]["randomness"]) * potential \
                            + param["dist_ren"]["randomness"] * potential_random

            # Sort elements based on their probability and keep the indices
            ind_sort = np.argsort(potential_new, axis=None)  # Ascending
            ind_needed = ind_sort[-int(data.loc[(data["Site"] == name) & (data["Process"] == p), "Unit"].values):]

            # Free memory
            del ind_sort, potential, potential_nan, potential_random

            # Get the coordinates of the power plants and their respective capacities
            power_plants = [units[p]] * len(ind_needed)
            if data.loc[(data["Site"] == name) & (data["Process"] == p), "inst-cap"].values % units[p] > 0:
                power_plants[-1] = data.loc[(data["Site"] == name) & (data["Process"] == p), "inst-cap"].values % units[
                    p]
            y_pp, x_pp = np.unravel_index(ind_needed, raster_shape)
            x = x + ((x_pp + x_off + 0.5) * param["res_desired"][1] + param["Crd_all"][3]).tolist()
            y = y + (param["Crd_all"][0] - (y_pp + y_off + 0.5) * param["res_desired"][0]).tolist()
            c = c + potential_new[ind_needed].tolist()  # Power_plants

            del potential_new

        # Create map
        import pdb;
        pdb.set_trace()
        map_power_plants(p, x, y, c, paths["map_power_plants"] + p + '.shp')


def clean_processes_and_storage_data(paths, param):
    ''' documentation '''
    timecheck("Start")

    assumptions = pd.read_excel(paths["assumptions"], sheet_name='Process')

    depreciation = dict(zip(assumptions['Process'], assumptions['depreciation'].astype(float)))
    year_mu = dict(zip(assumptions['Process'], assumptions['year_mu'].astype(float)))
    year_stdev = dict(zip(assumptions['Process'], assumptions['year_stdev'].astype(float)))

    # Get data from fresna database
    Process = pd.read_csv(paths["database"], header=0, skipinitialspace=True,
                          usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Process.rename(columns={'Capacity': 'inst-cap', 'YearCommissioned': 'year', 'lat': 'Latitude', 'lon': 'Longitude'},
                   inplace=True)
    print('Number of power plants: ', len(Process))

    # ### Process name

    # Use the name of the processes in OPSD as a standard name
    Process['Pro'] = Process['OPSD']

    # If the name is missing in OPSD, use the name in other databases
    Process.loc[Process['Pro'].isnull(), 'Pro'] = Process.loc[Process['Pro'].isnull(), 'CARMA']
    Process.loc[Process['Pro'].isnull(), 'Pro'] = Process.loc[Process['Pro'].isnull(), 'ENTSOE']
    Process.loc[Process['Pro'].isnull(), 'Pro'] = Process.loc[Process['Pro'].isnull(), 'GEO']
    Process.loc[Process['Pro'].isnull(), 'Pro'] = Process.loc[Process['Pro'].isnull(), 'WRI']

    # Add suffix to duplicate names
    Process['Pro'] = Process['Pro'] + Process.groupby(['Pro']).cumcount().astype(str).replace('0', '')

    # Remove spaces from the name and replace them with underscores
    Process['Pro'] = [Process.loc[i, 'Pro'].replace(' ', '_') for i in Process.index]

    # Remove useless columns
    Process.drop(['CARMA', 'ENTSOE', 'GEO', 'OPSD', 'WRI'], axis=1, inplace=True)

    print('Number of power plants with distinct names: ', len(Process['Pro'].unique()))

    # ### Type

    Process['CoIn'] = np.nan
    for i in Process.index:
        # Get the pumped storage facilities
        if Process.loc[i, 'Technology'] in ['Pumped Storage', 'Pumped Storage With Natural Inflow',
                                            'Pumped Storage, Pumped Storage With Natural Inflow, Reservoir',
                                            'Pumped Storage, Reservoir', 'Pumped Storage, Run-Of-River']:
            Process.loc[i, 'CoIn'] = 'PumSt'

        # Get the tidal power plants
        if Process.loc[i, 'Technology'] == 'Tidal':
            Process.loc[i, 'CoIn'] = 'Tidal'

        # Assign an input commodity
        if pd.isnull(Process.loc[i, 'CoIn']):
            Process.loc[i, 'CoIn'] = param["pro_sto"]["proc_dict"][Process.loc[i, 'Fueltype']]

        # Distinguish between small and large hydro
        if (Process.loc[i, 'CoIn'] == 'Hydro_Small') and (Process.loc[i, 'inst-cap'] > 30):
            Process.loc[i, 'CoIn'] = 'Hydro_Large'

    # Remove useless columns
    Process.drop(['Fueltype', 'Technology', 'Set'], axis=1, inplace=True)

    # Remove renewable power plants (except Tidal and Geothermal)
    Process.set_index('CoIn', inplace=True)
    Process.drop(list(set(Process.index.unique()) & set(param["pro_sto"]["renewable_powerplants"])),
                 axis=0, inplace=True)

    Process.reset_index(inplace=True)

    print('Possible processes: ', Process['CoIn'].unique())

    # ### Include renewable power plants

    for pp in param["pro_sto"]["renewable_powerplants"]:
        # Shapefile with power plants
        pp_shapefile = gpd.read_file(paths["PPs_"] + pp + '.shp')
        pp_df = pd.DataFrame(pp_shapefile.rename(columns={'CapacityMW': 'inst-cap'}))
        pp_df['Longitude'] = [pp_df.loc[i, 'geometry'].x for i in pp_df.index]
        pp_df['Latitude'] = [pp_df.loc[i, 'geometry'].y for i in pp_df.index]
        pp_df['CoIn'] = pp
        pp_df['Pro'] = [pp + '_' + str(i) for i in pp_df.index]
        pp_df.drop(['geometry'], axis=1, inplace=True)
        Process = Process.append(pp_df, ignore_index=True, sort=True)

    # ### Year

    # Assign a dummy year for missing entries (will be changed later)
    for c in Process['CoIn'].unique():
        if c in param["pro_sto"]["storage"]:
            Process.loc[(Process['CoIn'] == c) & (Process['year'].isnull()), 'year_mu'] = 1980
            Process.loc[(Process['CoIn'] == c) & (Process['year'].isnull()), 'year_stdev'] = 5
        else:
            Process.loc[(Process['CoIn'] == c) & (Process['year'].isnull()), 'year_mu'] = year_mu[c]
            Process.loc[(Process['CoIn'] == c) & (Process['year'].isnull()), 'year_stdev'] = year_stdev[c]

    Process.loc[Process['year'].isnull(), 'year'] = np.floor(
        np.random.normal(Process.loc[Process['year'].isnull(), 'year_mu'],
                         Process.loc[Process['year'].isnull(), 'year_stdev']))

    # Drop recently built plants (after the reference year)
    Process = Process[(Process['year'] <= param["year"])]

    # ### Coordinates

    P_missing = Process[Process['Longitude'].isnull()].copy()
    P_located = Process[~Process['Longitude'].isnull()].copy()

    # Assign dummy coordinates within the same country (will be changed later)
    for country in P_missing['Country'].unique():
        P_missing.loc[P_missing['Country'] == country, 'Latitude'] = P_located[P_located['Country'] == country].iloc[
            0, 2]
        P_missing.loc[P_missing['Country'] == country, 'Longitude'] = P_located[P_located['Country'] == country].iloc[
            0, 3]

    Process = P_located.append(P_missing)
    Process = Process[Process['Longitude'] > -11]

    # ### Consider lifetime of power plants

    if param["year"] > param["pro_sto"]["year_ref"]:
        for c in Process['CoIn'].unique():
            if c in param["pro_sto"]["storage"]:
                Process.loc[Process['CoIn'] == c, 'lifetime'] = 60
            else:
                Process.loc[Process['CoIn'] == c, 'lifetime'] = depreciation[c]
        print(len(Process.loc[(Process['lifetime'] + Process['year']) < param["year"]]), ' processes will be deleted')
        Process.drop(Process.loc[(Process['lifetime'] + Process['year']) < param["year"]].index, inplace=True)

    # ### Site

    # Create point geometries (shapely)
    Process['geometry'] = list(zip(Process.Longitude, Process.Latitude))
    Process['geometry'] = Process['geometry'].apply(Point)

    P_located = gpd.GeoDataFrame(Process, geometry='geometry', crs='')
    P_located.crs = {'init': 'epsg:4326'}

    # Define the output commodity
    P_located['CoOut'] = 'Elec'
    P_located.drop(['Latitude', 'Longitude', 'year_mu', 'year_stdev'], axis=1, inplace=True)
    P_located = P_located[['Pro', 'CoIn', 'CoOut', 'inst-cap', 'Country', 'year', 'geometry']]

    # Save the GeoDataFrame
    if not os.path.isfile(paths["pro_sto"]):
        P_located.to_file(driver='ESRI Shapefile', filename=paths["pro_sto"])
    else:
        os.remove(paths["pro_sto"])
        P_located.to_file(driver='ESRI Shapefile', filename=paths["pro_sto"])

    print("File Saved: " + paths["pro_sto"])
    timecheck("End")


def clean_processes_and_storage_data_FRESNA(paths, param):
    ''' documentation '''
    timecheck("Start")

    assumptions = pd.read_excel(paths["assumptions"], sheet_name='Process')
    depreciation = dict(zip(assumptions['Process'], assumptions['depreciation'].astype(float)))
    year_mu = dict(zip(assumptions['Process'], assumptions['year_mu'].astype(float)))
    year_stdev = dict(zip(assumptions['Process'], assumptions['year_stdev'].astype(float)))

    # Get data from fresna database
    Process = pd.read_csv(paths["database_FRESNA"], header=0, skipinitialspace=True,
                          usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Process.rename(columns={'Capacity': 'inst-cap', 'lat': 'Latitude', 'lon': 'Longitude'},
                   inplace=True)
    print('Number of power plants: ', len(Process))

    Process['Technology'].fillna('NaN', inplace=True)
    Process['inst-cap'].fillna(0, inplace=True)
    Process[['Fueltype', 'Technology', 'Set']].drop_duplicates()
    Process[(Process['Technology'] == 'Run-Of-River') & (Process['Fueltype'] == 'Hydro')].groupby(['Country']).sum()
    Process.groupby(['Fueltype', 'Technology', 'Set']).sum().to_csv(paths["process_raw"], sep=';',
                                                                    decimal=',', index=True)
    # Type
    Process['CoIn'] = np.nan

    for i in Process.index:
        # Get the coal
        if Process.loc[i, 'Fueltype'] == 'Hard Coal':
            Process.loc[i, 'CoIn'] = 'Coal'
        # Get the gas
        if tuple(Process.loc[i, ['Fueltype', 'Technology']]) in [('Natural Gas', 'CCGT'),
                                                                 ('Natural Gas', 'CCGT, Thermal'),
                                                                 ('Natural Gas', 'Gas Engines'), ('Natural Gas', 'NaN'),
                                                                 ('Natural Gas', 'Pv')]:
            Process.loc[i, 'CoIn'] = 'Gas_CCGT'
        if tuple(Process.loc[i, ['Fueltype', 'Technology']]) in [('Natural Gas', 'OCGT')]:
            Process.loc[i, 'CoIn'] = 'Gas_OCGT'
        if tuple(Process.loc[i, ['Fueltype', 'Technology']]) in [('Natural Gas', 'Steam Turbine')]:
            Process.loc[i, 'CoIn'] = 'Gas_ST'
        # Get lignite and nuclear
        if Process.loc[i, 'Fueltype'] in ['Lignite', 'Nuclear']:
            Process.loc[i, 'CoIn'] = Process.loc[i, 'Fueltype']
        # Get oil and other
        if Process.loc[i, 'Fueltype'] in ['Oil', 'Other', 'Waste']:
            Process.loc[i, 'CoIn'] = 'Oil/Other'
        # Get the unconventional storage
        if tuple(Process.loc[i, ['Fueltype', 'Technology']]) in [('Natural Gas', 'Storage Technologies'),
                                                                 ('Natural Gas', 'Caes')]:
            Process.loc[i, 'CoIn'] = 'Storage_ST'
        # Get the pumped storage facilities
        if Process.loc[i, 'Technology'] in ['Pumped Storage', 'Pumped Storage With Natural Inflow']:
            Process.loc[i, 'CoIn'] = 'Storage_Long-Term'
        if tuple(Process.loc[i, ['Fueltype', 'Technology']]) in [('Hydro', 'Reservoir'), ('Hydro', 'NaN'),
                                                                 ('Hydro', 'Pv'), ('Hydro', 'Run-Of-River')]:
            Process.loc[i, 'CoIn'] = 'Hydro'
        Process.loc[i, 'Site'] = param["dist_ren"]["country_names"][Process.loc[i, 'Country']]

    # Remove useless rows
    Process.dropna(subset=['CoIn'], inplace=True)

    # year
    for i in Process.index:
        Process.loc[i, 'Year'] = max(Process.loc[i, 'YearCommissioned'], Process.loc[i, 'Retrofit'])
        Process.loc[i, 'Cohort'] = max((Process.loc[i, 'YearCommissioned'] // 5) * 5, 1960)
        Process.loc[i, 'Cohort_new'] = max((Process.loc[i, 'Year'] // 5) * 5, 1960)
        if np.isnan(Process.loc[i, 'Cohort']):
            Process.loc[i, 'Cohort'] = 'NaN'
        else:
            Process.loc[i, 'Cohort'] = str(int(Process.loc[i, 'Cohort']))
        if np.isnan(Process.loc[i, 'Cohort_new']):
            Process.loc[i, 'Cohort_new'] = 'NaN'
        else:
            Process.loc[i, 'Cohort_new'] = str(int(Process.loc[i, 'Cohort_new']))

    Process_agg = Process.groupby(['Site', 'CoIn', 'Cohort']).sum() / 1000

    Process_agg = Process_agg[['inst-cap']]

    full_ind = pd.MultiIndex.from_product([Process['Site'].unique(),
                                           Process['CoIn'].unique(),
                                           # ['Lignite', 'Coal', 'Gas_CCGT', 'Gas_ST', 'Gas_OCGT', 'Oil/Other'],
                                           Process['Cohort'].unique()])

    table_empty = pd.DataFrame(0, index=full_ind, columns=['inst-cap'])
    for i in Process_agg.index:
        table_empty.loc[i, 'inst-cap'] = Process_agg.loc[i, 'inst-cap']

    Process_agg = table_empty.reset_index().rename(columns={'level_0': 'Site', 'level_1': 'CoIn', 'level_2': 'Cohort'})

    Process_agg.set_index(['CoIn', 'Cohort'], inplace=True)
    Process_agg.pivot(columns='Site').to_csv(paths["Process_agg"], sep=';', decimal=',', index=True)

    Process.drop(Process[(Process['YearCommissioned'] > param["year"])].index, axis=0, inplace=True)

    # Assign a dummy year for missing entries (will be changed later)
    for c in Process['CoIn'].unique():
        if c == 'Storage_Long-Term':
            Process.loc[(Process['CoIn'] == c) & (Process['YearCommissioned'].isnull()), 'year_mu'] = 1980
            Process.loc[(Process['CoIn'] == c) & (Process['YearCommissioned'].isnull()), 'year_stdev'] = 5
        elif c == 'Storage_ST':
            Process.loc[(Process['CoIn'] == c) & (Process['YearCommissioned'].isnull()), 'year_mu'] = 2010
            Process.loc[(Process['CoIn'] == c) & (Process['YearCommissioned'].isnull()), 'year_stdev'] = 5
        else:
            Process.loc[(Process['CoIn'] == c) & (Process['YearCommissioned'].isnull()), 'year_mu'] = year_mu[c]
            Process.loc[(Process['CoIn'] == c) & (Process['YearCommissioned'].isnull()), 'year_stdev'] = year_stdev[c]

    Process.loc[Process['YearCommissioned'].isnull(), 'YearCommissioned'] = np.floor(
        np.random.normal(Process.loc[Process['YearCommissioned'].isnull(), 'year_mu'],
                         Process.loc[Process['YearCommissioned'].isnull(), 'year_stdev']))

    Process.loc[Process['YearCommissioned'] > param["year"], 'YearCommissioned'] = param["year"]

    # Recalculate cohorts
    for i in Process.index:
        Process.loc[i, 'Year'] = max(Process.loc[i, 'YearCommissioned'], Process.loc[i, 'Retrofit'])
        Process.loc[i, 'Cohort'] = max((Process.loc[i, 'YearCommissioned'] // 5) * 5, 1960)
        Process.loc[i, 'Cohort_new'] = max((Process.loc[i, 'Year'] // 5) * 5, 1960)
        if np.isnan(Process.loc[i, 'Cohort']):
            Process.loc[i, 'Cohort'] = 'NaN'
        else:
            Process.loc[i, 'Cohort'] = str(int(Process.loc[i, 'Cohort']))
        if np.isnan(Process.loc[i, 'Cohort_new']):
            Process.loc[i, 'Cohort_new'] = 'NaN'
        else:
            Process.loc[i, 'Cohort_new'] = str(int(Process.loc[i, 'Cohort_new']))

    Process_agg2 = Process.groupby(['Site', 'CoIn', 'Cohort']).sum() / 1000

    Process_agg2 = Process_agg2[['inst-cap']]

    full_ind = pd.MultiIndex.from_product([Process['Site'].unique(),
                                           Process['CoIn'].unique(),
                                           # ['Lignite', 'Coal', 'Gas_CCGT', 'Gas_ST', 'Gas_OCGT', 'Oil/Other'],
                                           Process['Cohort'].unique()])

    table_empty = pd.DataFrame(0, index=full_ind, columns=['inst-cap'])
    for i in Process_agg2.index:
        table_empty.loc[i, 'inst-cap'] = Process_agg2.loc[i, 'inst-cap']

    Process_agg2 = table_empty.reset_index().rename(columns={'level_0': 'Site', 'level_1': 'CoIn', 'level_2': 'Cohort'})

    Process_agg2.set_index(['CoIn', 'Cohort'], inplace=True)
    Process_agg2.pivot(columns='Site').to_csv(paths["Process_agg_bis"], sep=';', decimal=',', index=True)

    # Process name
    # Use the name of the processes in OPSD as a standard name
    Process['Pro'] = Process['Name']
    Process['Pro'].fillna('unnamed', inplace=True)

    # Add suffix to duplicate names
    Process['Pro'] = Process['Pro'] + Process.groupby(['Pro']).cumcount().astype(str).replace('0', '')

    # Remove spaces from the name and replace them with underscores
    Process['Pro'] = [Process.loc[i, 'Pro'].replace(' ', '_') for i in Process.index]

    # Show except
    print('Number of power plants with distinct names: ', len(Process['Pro'].unique()))

    # Coordinates
    P_missing = Process[Process['Longitude'].isnull()].copy()
    P_located = Process[~Process['Longitude'].isnull()].copy()

    # Assign dummy coordinates within the same country (will be changed later)
    for country in P_missing['Country'].unique():
        P_missing.loc[P_missing['Country'] == country, 'Latitude'] = P_located[P_located['Country'] == country].iloc[
            0, 9]
        P_missing.loc[P_missing['Country'] == country, 'Longitude'] = P_located[P_located['Country'] == country].iloc[
            0, 10]

    Process = P_located.append(P_missing)
    Process = Process[Process['Longitude'] > -11]

    # Sites
    # Create point geometries (shapely)
    Process['geometry'] = list(zip(Process.Longitude, Process.Latitude))
    Process['geometry'] = Process['geometry'].apply(Point)

    P_located = gpd.GeoDataFrame(Process, geometry='geometry', crs='')
    P_located.crs = {'init': 'epsg:4326'}

    # Define the output commodity
    P_located['CoOut'] = 'Elec'

    P_located.drop(['Latitude', 'Longitude', 'year_mu', 'year_stdev'], axis=1, inplace=True)

    P_located = P_located[['Pro', 'CoIn', 'CoOut', 'inst-cap', 'Country', 'Year', 'geometry']]

    # Save the GeoDataFrame
    if not os.path.isfile(paths["pro_sto"]):
        P_located.to_file(driver='ESRI Shapefile', filename=paths["pro_sto"])
    else:
        os.remove(paths["pro_sto"])
        P_located.to_file(driver='ESRI Shapefile', filename=paths["pro_sto"])

    print("File Saved: " + paths["pro_sto"])
    timecheck("End")


def generate_processes(paths, param):
    ''' documentation '''
    timecheck("Start")

    assumptions = pd.read_excel(paths["assumptions"], sheet_name='Process')
    # Only use the assumptions of that particular year
    assumptions = assumptions[assumptions['year'] == param["year"]]

    param["assumptions"] = read_assumptions_process(assumptions)

    depreciation = param["assumptions"]["depreciation"]
    on_off = param["assumptions"]["on_off"]

    # Get data from the shapefile
    pro_and_sto = gpd.read_file(paths["pro_sto"])

    # Split the storage from the processes
    process_raw = pro_and_sto[~pro_and_sto["CoIn"].isin(param["pro_sto"]["storage"])]
    print('Number of processes read: ' + str(len(process_raw)))

    # Consider the lifetime of power plants
    process_current = filter_life_time(param, process_raw, depreciation)

    # Get Sites
    process_located, _ = get_sites(process_current, paths)
    print('Number of processes after duplicate removal: ' + str(len(process_located)))

    # Reduce the number of processes by aggregating the small and must-run power plants
    process_compact = process_located.copy()
    for c in process_compact["CoIn"].unique():
        process_compact.loc[process_compact["CoIn"] == c, "on-off"] = on_off[c]

    # Select small processes and group them
    process_group = process_compact[(process_compact["inst-cap"] < param["pro_sto"]["agg_thres"])
                                    | (process_compact["on-off"] == 0)]
    process_group = process_group.groupby(["Site", "CoIn"])

    # Define the attributes of the aggregates
    small_cap = pd.DataFrame(process_group["inst-cap"].sum())
    small_pro = pd.DataFrame(process_group["Pro"].first() + '_agg')
    small_coout = pd.DataFrame(process_group["CoOut"].first())
    small_year = pd.DataFrame(process_group["year"].min())

    # Aggregate the small processes
    process_small = small_cap.join([small_pro, small_coout, small_year]).reset_index()

    # Recombine big processes with the aggregated small ones
    process_compact = process_compact[(process_compact["inst-cap"] >= param["pro_sto"]["agg_thres"])
                                      & (process_compact["on-off"] == 1)]
    process_compact = process_compact.append(process_small, ignore_index=True, sort=True)
    print("Number of compacted processes: " + str(len(process_compact)))

    # Process evrys, urbs
    evrys_process, urbs_process = format_process_model(process_compact, param)

    # Output
    urbs_process.to_csv(paths["urbs_process"], index=False, sep=';', decimal=',')
    print("File Saved: " + paths["urbs_process"])
    evrys_process.to_csv(paths["evrys_process"], index=False, sep=';', decimal=',', encoding='ascii')
    print("File Saved: " + paths["evrys_process"])

    timecheck("End")


def generate_storage(paths, param):
    ''' documentation '''
    timecheck("Start")

    # Read required assumptions
    assumptions = pd.read_excel(paths["assumptions"], sheet_name='Storage')

    # Only use the assumptions of that particular year
    assumptions = assumptions[assumptions['year'] == param["year"]]

    param["assumptions"] = read_assumptions_storage(assumptions)

    depreciation = param["assumptions"]["depreciation"]

    # Get data from the shapefile
    pro_and_sto = gpd.read_file(paths["pro_sto"])

    # Split the storages from the processes
    storage_raw = pro_and_sto[pro_and_sto["CoIn"].isin(param["pro_sto"]["storage"])]
    print('Number of storage units read: ' + str(len(storage_raw)))

    # Consider lifetime of storage units
    storage_current = filter_life_time(param, storage_raw, depreciation)

    # Get sites
    storage_located, regions = get_sites(storage_current, paths)
    param["regions"] = regions
    print('Number of storage units after duplicate removal: ' + str(len(storage_located)))

    # Reduce number of storage units by aggregating the small storage units
    storage_compact = storage_located.copy()

    # Select small processes and group them
    storage_group = storage_compact[storage_compact["inst-cap"] < param["pro_sto"]["agg_thres"]].groupby(
        ["Site", "CoIn"])

    # Define the attributes of the aggregates
    small_cap = pd.DataFrame(storage_group["inst-cap"].sum())
    small_pro = pd.DataFrame(storage_group["Pro"].first() + '_agg')
    small_coout = pd.DataFrame(storage_group["CoOut"].first())
    small_year = pd.DataFrame(storage_group["year"].min())

    # Aggregate the small storage units
    storage_small = small_cap.join([small_pro, small_coout, small_year]).reset_index()

    # Recombine big storage units with the aggregated small ones
    storage_compact = storage_compact[storage_compact["inst-cap"] >= param["pro_sto"]["agg_thres"]]
    storage_compact = storage_compact.append(storage_small, ignore_index=True, sort=True)
    print("Number of compacted storage units: " + str(len(storage_compact)))

    # Take the raw storage table and group by tuple of sites and storage type
    storage_compact = storage_compact[["Site", "CoIn", "CoOut", "inst-cap"]].copy()
    storage_compact.rename(columns={'CoIn': 'Sto', 'CoOut': 'Co'}, inplace=True)
    storage_group = storage_compact.groupby(["Site", "Sto"])

    # Define the attributes of the aggregates
    inst_cap0 = storage_group["inst-cap"].sum().rename('inst-cap-pi')

    co0 = storage_group["Co"].first()

    # Combine the list of series into a dataframe
    storage_compact = pd.DataFrame([inst_cap0, co0]).transpose().reset_index()

    # Storage evrys, urbs
    evrys_storage, urbs_storage = format_storage_model(storage_compact, param)

    # Output
    urbs_storage.to_csv(paths["urbs_storage"], index=False, sep=';', decimal=',')
    print("File Saved: " + paths["urbs_storage"])
    evrys_storage.to_csv(paths["evrys_storage"], index=False, sep=';', decimal=',', encoding='ascii')
    print("File Saved: " + paths["evrys_storage"])

    timecheck("End")


def generate_processes_and_storage_california(paths, param):
    timecheck('Start')
    Process = pd.read_excel(paths["database_Cal"], sheet_name='Operating', header=1, skipinitialspace=True,
                            usecols=[0, 2, 5, 6, 7, 10, 11, 14, 17, 25, 26],
                            dtype={'Entity ID': np.unicode_, 'Plant ID': np.unicode_})
    Process.rename(columns={'\nNameplate Capacity (MW)': 'inst-cap', 'Operating Year': 'year'}, inplace=True)
    regions = gpd.read_file(paths["regions_SHP"])

    # Drop recently built plants (after the reference year),
    # non-operating plants, and plants outside the geographic scope
    Process = Process[(Process['year'] <= param["year"])
                      & (Process['Status'].isin(param["pro_sto_Cal"]["status"]))
                      & (Process['Plant State'].isin(param["pro_sto_Cal"]["states"]))]

    for i in Process.index:
        # Define a unique ID for the processes
        Process.loc[i, 'Pro'] = (Process.loc[i, 'Plant State'] + '_' +
                                 Process.loc[i, 'Entity ID'] + '_' +
                                 Process.loc[i, 'Plant ID'] + '_' +
                                 Process.loc[i, 'Generator ID'])

        # Define the input commodity
        Process.loc[i, 'CoIn'] = param["pro_sto_Cal"]["proc_dict"][Process.loc[i, 'Energy Source Code']]
        if Process.loc[i, 'Technology'] == 'Hydroelectric Pumped Storage':
            Process.loc[i, 'CoIn'] = 'PumSt'
        if (Process.loc[i, 'CoIn'] == 'Hydro_Small') and (Process.loc[i, 'inst-cap'] > 30):
            Process.loc[i, 'CoIn'] = 'Hydro_Large'

        # Define the location of the process
        if Process.loc[i, 'Pro'] == 'CA_50045_56284_EPG':  # Manual correction
            Process.loc[i, 'Site'] = 'LAX'
        else:
            Process.loc[i, 'Site'] = \
                containing_polygon(Point(Process.loc[i, 'Longitude'], Process.loc[i, 'Latitude']), regions)[
                    'NAME_SHORT']

    # Define the output commodity
    Process['CoOut'] = 'Elec'

    # Select columns to be used
    Process = Process[['Site', 'Pro', 'CoIn', 'CoOut', 'inst-cap', 'year']]
    print('Number of Entries: ' + str(len(Process)))

    # Split the storages from the processes
    process_raw = Process[~Process['CoIn'].isin(param["pro_sto_Cal"]["storage"])]
    storage_raw = Process[Process['CoIn'].isin(param["pro_sto_Cal"]["storage"])]
    print('Number of Processes: ' + str(len(process_raw)))
    print('Number of Storage systems: ' + str(len(storage_raw)))

    # Processes
    # Reduce the number of processes by aggregating the small ones
    # Select small processes and group them
    process_group = process_raw[process_raw['inst-cap'] < 10].groupby(['Site', 'CoIn'])
    # Define the attributes of the aggregates
    small_cap = pd.DataFrame(process_group['inst-cap'].sum())
    small_pro = pd.DataFrame(process_group['Pro'].first() + '_agg')
    small_coout = pd.DataFrame(process_group['CoOut'].first())
    small_year = pd.DataFrame(process_group['year'].min())
    # Aggregate the small processes
    process_small = small_cap.join([small_pro, small_coout, small_year]).reset_index()

    # Recombine big processes with the aggregated small ones
    process_compact = process_raw[process_raw['inst-cap'] >= 10].append(process_small, ignore_index=True)
    print('Number of Processes after agregation: ' + str(len(process_compact)))
    evrys_process, urbs_process = format_process_model_California(process_compact, process_small, param)

    # Output
    urbs_process.to_csv(paths["urbs_process"], index=False, sep=';', decimal=',')
    print("File Saved: " + paths["urbs_process"])
    evrys_process.to_csv(paths["evrys_process"], index=False, sep=';', decimal=',', encoding='ascii')
    print("File Saved: " + paths["evrys_process"])

    # Storage Systems
    param["sites_evrys_unique"] = evrys_process.Sites.unique()
    evrys_storage, urbs_storage = format_storage_model_California(storage_raw, param)

    # Output
    urbs_storage.to_csv(paths["urbs_storage"], index=False, sep=';', decimal=',')
    print("File Saved: " + paths["urbs_storage"])
    evrys_storage.to_csv(paths["evrys_storage"], index=False, sep=';', decimal=',', encoding='ascii')
    print("File Saved: " + paths["evrys_storage"])

    timecheck('End')


def clean_grid_data(paths, param):
    """
    @all-contributors please add @kais-siala for bug, code, doc, ideas, maintenance, review, test and talk
    :param paths:
    :param param:
    :return:
    """
    timecheck("Start")

    # Read CSV file containing the lines data
    grid_raw = pd.read_csv(paths["transmission_lines"], header=0, sep=',', decimal='.')
    #import pdb; pdb.set_trace()

    # Extract the string with the coordinates from the last column
    grid_raw["wkt_srid_4326"] = pd.Series(map(lambda s: s[21:-1], grid_raw["wkt_srid_4326"]), grid_raw.index)

    # Extract the coordinates into a new dataframe with four columns for each coordinate
    coordinates = pd.DataFrame(grid_raw["wkt_srid_4326"].str.split(' |,').tolist(),
                               columns=['V1_long', 'V1_lat', 'V2_long', 'V2_lat'])

    # Merge the original dataframe (grid_raw) with the one for the coordinates
    grid_raw = grid_raw.merge(coordinates, how='outer', left_index=True, right_index=True)

    # Drop the old column and the coordinates dataframe
    grid_raw.drop('wkt_srid_4326', axis=1, inplace=True)
    del coordinates
    
    # Expand columns with multiple values
    grid_expanded = grid_raw.copy()
    grid_expanded = expand_dataframe(grid_expanded, ["voltage", "wires", "cables", "frequency"])
    grid_expanded.to_csv(paths["grid_expanded"], index=False, sep=';', decimal=',')
    
    # If data is trustworthy, remove NaN values
    grid_filtered = grid_expanded.copy()
    for col in ["voltage", "wires", "cables", "frequency"]:
        if param["grid"]["quality"][col] == 1:
            grid_filtered = grid_filtered[~grid_filtered[col].isnull()]
    grid_filtered.to_csv(paths["grid_filtered"], index=False, sep=';', decimal=',')
    
    # Fill missing data with most common value
    grid_corrected = grid_filtered.copy()
    for col in ["voltage", "wires", "cables", "frequency"]:
        grid_corrected.loc[grid_corrected[col].isnull(), col] = grid_corrected[col].value_counts().index[0]
    
    # Replace voltage = 0 with most common value
    grid_corrected.loc[grid_corrected["voltage"] == 0, "voltage"] = grid_corrected["voltage"].value_counts().index[0]
    
    # Eventually overwrite the values in 'wires' using 'cables'
    if param["grid"]["quality"]["cables"] > param["grid"]["quality"]["wires"]:
        grid_corrected.loc[:, "wires"] = np.minimum(grid_corrected.loc[:, "cables"] // 3, 1)
    grid_corrected.to_csv(paths["grid_corrected"], index=False, sep=';', decimal=',')
    
    # Complete missing information
    grid_filled = grid_corrected.copy()
    grid_filled["length_m"] = grid_filled["length_m"].astype(float)
    grid_filled["x_ohmkm"] = assign_values_based_on_series(grid_filled["voltage"] / 1000, param["grid"]["specific_reactance"])
    grid_filled["X_ohm"] = grid_filled['x_ohmkm'] * grid_filled['length_m'] / 1000 / grid_filled['wires']
    grid_filled["loadability"] = assign_values_based_on_series(grid_filled["length_m"] / 1000, param["grid"]["loadability"])
    grid_filled["SIL_MW"] = assign_values_based_on_series(grid_filled["voltage"] / 1000, param["grid"]["SIL"])
    grid_filled["Capacity_MVA"] = grid_filled["SIL_MW"] * grid_filled["loadability"] * grid_filled["wires"]
    grid_filled["Y_mho_ref_380kV"] = 1 / (grid_filled["X_ohm"] * ((380000 / grid_filled["voltage"]) ** 2))
    grid_filled.loc[grid_filled['frequency'] == 0, 'tr_type'] = 'DC_CAB'
    grid_filled.loc[~(grid_filled['frequency'] == 0), 'tr_type'] = 'AC_OHL'
    grid_filled.to_csv(paths["grid_filled"], index=False, sep=';', decimal=',')
    
    # Group lines with same IDs
    grid_grouped = grid_filled[["l_id", "tr_type", "Capacity_MVA", "Y_mho_ref_380kV", "V1_long", "V1_lat", "V2_long", "V2_lat"]] \
                              .groupby(["l_id", "tr_type", "V1_long", "V1_lat", "V2_long", "V2_lat"]).sum()
    grid_grouped.reset_index(inplace=True)
    grid_grouped.loc[:, ['V1_long', 'V1_lat', 'V2_long', 'V2_lat']] = grid_grouped.loc[:, ['V1_long', 'V1_lat', 'V2_long', 'V2_lat']].astype(float)
    #import pdb; pdb.set_trace()
    grid_grouped.to_csv(paths["grid_cleaned"], index=False, sep=';', decimal=',')
    print("File Saved: " + paths["grid_cleaned"])

    #import pdb; pdb.set_trace()

    # Writing to shapefile
    with shp.Writer(paths["grid_shp"], shapeType=3) as w:
        w.autoBalance = 1
        w.field('ID', 'N', 6, 0)
        w.field('Cap_MVA', 'N', 8, 2)
        w.field('Type', 'C', 6, 0)
        count = len(grid_grouped.index)
        status = 0
        for i in grid_grouped.index:
            status += 1
            display_progress("Writing grid to shapefile: ", (count, status))
            w.line([[grid_grouped.loc[i, ['V1_long', 'V1_lat']].astype(float),
                     grid_grouped.loc[i, ['V2_long', 'V2_lat']].astype(float)]])
            w.record(grid_grouped.loc[i, 'l_id'], grid_grouped.loc[i, 'Capacity_MVA'], grid_grouped.loc[i, 'tr_type'])

    print("File Saved: " + paths["grid_shp"])
    timecheck("End")


def generate_aggregated_grid(paths, param):
    """

    :param paths:
    :param param:
    :return:
    """
    timecheck("Start")

    # Read the shapefile containing the map data
    regions = gpd.read_file(paths["SHP"])
    regions["Geometry"] = regions.buffer(0)

    # Read the shapefile and get the weights of the neighboring zones
    weights = ps.lib.weights.Queen.from_shapefile(paths["SHP"])
    param["weights"] = weights

    # Get the names of the modeled zones from the database file
    zones = regions["NAME_SHORT"]
    for i in regions.index:
        if regions.loc[i, 'Population'] == 0:
            zones.loc[i] = zones.loc[i] + '_off'
    param["zones"] = zones

    # Read the cleaned GridKit dataset
    grid_cleaned = pd.read_csv(paths["grid_cleaned"], header=0, sep=';', decimal=',')

    # Create point geometries
    grid_cleaned['V1'] = list(zip(grid_cleaned.V1_long, grid_cleaned.V1_lat))
    grid_cleaned['V1'] = grid_cleaned['V1'].apply(Point)
    grid_cleaned['V2'] = list(zip(grid_cleaned.V2_long, grid_cleaned.V2_lat))
    grid_cleaned['V2'] = grid_cleaned['V2'].apply(Point)

    # Create a dataframe for the start regions
    Region_start = gpd.GeoDataFrame(grid_cleaned[['l_id', 'V1']], geometry='V1', crs='').rename(
        columns={'V1': 'geometry'})
    Region_start.crs = {'init': 'epsg:4326'}
    Region_start.crs = regions[['NAME_SHORT', 'geometry']].crs
    Region_start = gpd.sjoin(Region_start, regions[['NAME_SHORT', 'geometry']], how='left', op='intersects')[
        ['NAME_SHORT']].rename(columns={'NAME_SHORT': 'Region_start'})

    # Create a dataframe for the end regions
    Region_end = gpd.GeoDataFrame(grid_cleaned[['l_id', 'V2']], geometry='V2', crs='').rename(
        columns={'V2': 'geometry'})
    Region_end.crs = {'init': 'epsg:4326'}
    # regions is in GRS80 which is almost the same as WGS 84 also known as epsg:4326
    Region_end.crs = regions[['NAME_SHORT', 'geometry']].crs
    Region_end = gpd.sjoin(Region_end, regions[['NAME_SHORT', 'geometry']], how='left', op='intersects')[
        ['NAME_SHORT']].rename(columns={'NAME_SHORT': 'Region_end'})

    # Join dataframes
    grid_regions = grid_cleaned.copy()
    grid_regions.drop(['V1', 'V2'], axis=1, inplace=True)
    grid_regions = grid_regions.join([Region_start, Region_end])

    intra = len(grid_regions.loc[(grid_regions['Region_start'] == grid_regions['Region_end']) & ~(
        grid_regions['Region_start'].isnull())])
    extra = len(grid_regions.loc[grid_regions['Region_start'].isnull() | grid_regions['Region_end'].isnull()])
    inter = len(grid_regions) - intra - extra

    # Show numbers of intraregional, interregional and extraregional lines
    print("\nLinetypes : ")
    print((("intraregional", intra), ("interregional", inter), ("extraregional", extra)))

    # Remove intraregional and extraregional lines
    icls_concatenated = grid_regions.loc[
        (grid_regions['Region_start'] != grid_regions['Region_end']) &
        ~(grid_regions['Region_start'].isnull() | grid_regions['Region_end'].isnull())
        ].copy()

    # Sort alphabetically and reindex
    icls_reversed = reverse_lines(icls_concatenated)
    icls_reversed.sort_values(['Region_start', 'Region_end', 'tr_type'], inplace=True)
    icl = icls_reversed.set_index(['Region_start', 'Region_end', 'tr_type'])
    icl = icl.reset_index()

    icl_final = deduplicate_lines(icl)

    # Transmission evrys, urbs
    evrys_transmission, urbs_transmission = format_transmission_model(icl_final, paths, param)

    # Ouput
    urbs_transmission.to_csv(paths["urbs_transmission"], sep=';', decimal=',', index=False)
    print("File Saved: " + paths["urbs_transmission"])

    evrys_transmission.to_csv(paths["evrys_transmission"], sep=';', decimal=',')
    print("File Saved: " + paths["evrys_transmission"])

    timecheck("End")


def generate_urbs_model(paths, param):
    """
    Read model's .csv files, and create relevant dataframes.
    Writes dataframes to urbs input excel file.
    """
    timecheck('Start')

    urbs_model = {}
    
    # Read sites
    if os.path.exists(paths["sites_sub"]):
        sites = pd.read_csv(paths["sites_sub"], sep=';', decimal=',')
        sites = sites[['Name', 'Area_m2']].rename(columns = {'Area_m2': 'area'})
        urbs_model["Site"] = sites
        
    # Read electricity demand
    if os.path.exists(paths["load_regions"]):
        demand = pd.read_csv(paths["load_regions"], sep=';', decimal=',', index_col=0)
        demand.columns = demand.columns + '.Elec'
        demand.index.name = "t"
        demand.index = range(1,8761)
        demand.loc[0] = 0
        demand.sort_index(inplace=True)
        urbs_model["Demand"] = demand
        
    # # List all files present in urbs folder
    # urbs_paths = glob.glob(paths["urbs"] + '*.csv')
    # # create empty dictionary
    # urbs_model = {}
    # # read .csv files and associate them with relevant sheet
    # for name in urbs_paths:
        # # clean input names and associate them with the relevant dataframe
        # sheet = os.path.basename(name).replace('_urbs_' + str(param["year"]) + '.csv', '')
        # urbs_model[sheet] = pd.read_csv(name, sep=';', decimal=',')

    # # Add global parameters
    # urbs_model["Global"] = pd.read_excel(paths["assumptions"], sheet_name='Global')

    # # Add Process-Commodity parameters
    # urbs_model["Process-Commodity"] = pd.read_excel(paths["assumptions"], sheet_name='Process-Commodity').fillna(0)

    # # Filter processes if not in Process-Commodity
    # urbs_model["Process"] = urbs_model["Process"].loc[
        # urbs_model["Process"]["Process"].isin(urbs_model["Process-Commodity"]["Process"].unique())]

    # # Verify Commodity
    # missing_commodities = urbs_model["Process-Commodity"]["Commodity"].loc[
        # ~urbs_model["Process-Commodity"]["Commodity"].isin(urbs_model["Commodity"]["Commodity"].unique())]
    # if len(missing_commodities) > 0:
        # print("Error: Missing Commodities from Process-Commodity: ")
        # print(missing_commodities)
        # return

    # # Add DSM and Buy-Sell-Price
    # DSM_header = ['Site', 'Commodity', 'delay', 'eff', 'recov', 'cap-max-do', 'cap-max-up']
    # urbs_model["DSM"] = pd.DataFrame(columns=DSM_header)
    # urbs_model["Buy-Sell-Price"] = pd.DataFrame(np.arange(0, 8761), columns=['t'])

    # Create ExcelWriter
    with ExcelWriter(paths["urbs_model"], mode='w') as writer:
        # populate excel file with available sheets
        status = 0
        for sheet in urbs_model.keys():
            urbs_model[sheet].to_excel(writer, sheet_name=sheet, index=False, header=True)
            status += 1
            display_progress("Writing to excel file in progress: ", (len(urbs_model.keys()), status))

    print("File saved: " + paths["urbs_model"])

    timecheck('End')

    # # Calculate the sum (yearly consumption) in a separate vector
    # yearly_load = load_regions.sum(axis=1)

    # # Calculte the ratio of the hourly load to the yearly load
    # df_normed = load_regions / np.tile(yearly_load.to_numpy(), (8760, 1)).transpose()

    # # Prepare the output in the desired format
    # df_output = pd.DataFrame(list(df_normed.index) * 8760, columns=['sit'])
    # df_output['value'] = np.reshape(df_normed.to_numpy(), -1, order='F')
    # df_output['t'] = df_output.index // len(df_normed) + 1
    # df_output = pd.concat([df_output, pd.DataFrame({'co': 'Elec'}, index=df_output.index)], axis=1)

    # df_evrys = df_output[['t', 'sit', 'co', 'value']]  # .rename(columns={'Region': 'sit'})

    # # Transform the yearly load into a dataframe
    # df_load = pd.DataFrame()

    # df_load['annual'] = yearly_load

    # # Preparation of dataframe
    # df_load = df_load.reset_index()
    # df_load = df_load.rename(columns={'Region': 'sit'})

    # # Merging load dataframes and calculation of total demand
    # df_load['total'] = param["load"]["degree_of_eff"] * df_load['annual']
    # df_merged = pd.merge(df_output, df_load, how='outer', on=['sit'])
    # df_merged['value_normal'] = df_merged['value'] * df_merged['total']

    # # Calculation of the absolute load per country
    # df_absolute = df_merged  # .reset_index()[['t','Countries','value_normal']]

    # # Rename the countries
    # df_absolute['sitco'] = df_absolute['sit'] + '.Elec'

    # df_urbs = df_absolute.pivot(index='t', columns='sitco', values='value_normal')
    # df_urbs = df_urbs.reset_index()

    # # Yearly consumption for each zone
    # annual_load = pd.DataFrame(df_absolute.groupby('sit').sum()['value_normal'].rename('Load'))

def generate_evrys_model(paths, param):
    """
    Read model's .csv files, and create relevant dataframes.
    Writes dataframes to evrys input excel file.
    """
    timecheck('Start')

    evrys_model = {}
    
    # Read sites
    if os.path.exists(paths["sites_sub"]):
        sites = pd.read_csv(paths["sites_sub"], sep=';', decimal=',')
        sites = sites[['Name', 'slacknode', 'syncharea', 'Latitude', 'Longitude', 'ctrarea', 'primpos', 'primneg', 'secpos', 'secneg', 'terpos', 'terneg']].rename(
            columns = {'Name': 'Site', 'Latitude': 'lat', 'Longitude': 'long'})
        evrys_model["Site"] = sites
    
    # # List all files present in urbs folder
    # evrys_paths = glob.glob(paths["evrys"] + '*.csv')
    # # create empty dictionary
    # evrys_model = {}
    # # read .csv files and associate them with relevant sheet
    # for name in evrys_paths:
        # # clean input names and associate them with the relevant dataframe
        # sheet = os.path.basename(name).replace('_evrys_' + str(param["year"]) + '.csv', '')
        # evrys_model[sheet] = pd.read_csv(name, sep=';', decimal=',')

    # Create ExcelWriter
    with ExcelWriter(paths["evrys_model"], mode='w') as writer:
        # populate excel file with available sheets
        status = 0
        for sheet in evrys_model.keys():
            evrys_model[sheet].to_excel(writer, sheet_name=sheet, index=False)
            status += 1
            display_progress("Writing to excel file in progress: ", (len(evrys_model.keys()), status))

    print("File saved: " + paths["evrys_model"])

    timecheck('End')


if __name__ == '__main__':
    paths, param = initialization()
    # generate_sites_from_shapefile(paths)
    # generate_load_timeseries(paths, param)
    clean_grid_data(paths, param)  # corresponds to 06a - done
    # generate_aggregated_grid(paths, param)  # corresponds to 06b - done
    
    # generate_intermittent_supply_timeseries(paths, param)  # separate module
    # generate_stratified_intermittent_supply_timeseries(paths, param)
    # generate_commodity(paths, param)  # corresponds to 04 - done
    # distribute_renewable_capacities(paths, param)  # corresponds to 05a - done
    # if param["region"] == 'California':
    #     generate_processes_and_storage_california(paths, param)  # done (Still needs testing)
    # else:
    #     clean_processes_and_storage_data(paths, param)  # corresponds to 05b I think - done
    #     clean_processes_and_storage_data_FRESNA(paths, param)  # Optional
    #     generate_processes(paths, param)  # corresponds to 05c - done
    #     generate_storage(paths, param)  # corresponds to 05d - done (Weird code at the end)
    
    # generate_urbs_model(paths, param)
    # generate_evrys_model(paths, param)
