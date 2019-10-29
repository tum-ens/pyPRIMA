from lib.util import *

# def filter_life_time(param, raw, depreciation):
# if param["year"] > param["pro_sto"]["year_ref"]:
# # Set depreciation period
# for c in raw["CoIn"].unique():
# raw.loc[raw["CoIn"] == c, "lifetime"] = depreciation[c]
# lifetimeleft = raw["lifetime"] + raw["year"]
# current = raw.drop(raw.loc[lifetimeleft < param["year"]].index)
# print('Already depreciated units:\n')
# print(str(len(raw) - len(current)) + '# Units have been removed')
# else:
# current = raw.copy()
# print('Number of current units: ' + str(len(current)))
# return current


# def get_sites(current, paths):
# # Get regions from shapefile
# regions = gpd.read_file(paths["SHP"])
# regions["geometry"] = regions.buffer(0)

# # Spacial join
# current.crs = regions[["NAME_SHORT", "geometry"]].crs
# located = gpd.sjoin(current, regions[["NAME_SHORT", "geometry"]], how='left', op='intersects')
# located.rename(columns={'NAME_SHORT': 'Site'}, inplace=True)

# # Remove duplicates that lie in the border between land and sea
# located.drop_duplicates(subset=["CoIn", "Pro", "inst-cap", "year", "Site"], inplace=True)

# # Remove duplicates that lie in two different zones
# located = located.loc[~located.index.duplicated(keep='last')]

# located.dropna(axis=0, subset=["Site"], inplace=True)

# return located, regions


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
    dict_daytype = pd.read_csv(paths["dict_daytype"], sep=";", decimal=",", index_col=["Week day"])["Type"].to_dict()
    dict_season = pd.read_csv(paths["dict_season"], sep=";", decimal=",", index_col=["Month"])["Season"].to_dict()
    profiles_paths = paths["profiles"]

    # Prepare the dataframe for the daily load
    start = datetime.datetime(param["year"], 1, 1)
    end = datetime.datetime(param["year"], 12, 31)
    hours = [str(x) for x in list(range(0, 24))]
    time_series = pd.DataFrame(data=np.zeros((365, 27)), index=None, columns=["Date", "Day", "Season"] + hours)
    time_series["Date"] = pd.date_range(start, end)
    time_series["Day"] = [dict_daytype[time_series.loc[i, "Date"].day_name()] for i in time_series.index]
    time_series["Season"] = [dict_season[time_series.loc[i, "Date"].month] for i in time_series.index]

    # Prepare the dataframe for the yearly load per sector
    profiles = pd.DataFrame(columns=param["load"]["sectors"])

    # Residential load
    if "RES" in param["load"]["sectors"]:
        residential_profile_raw = pd.read_excel(profiles_paths["RES"], header=[3, 4], skipinitialspace=True)
        residential_profile_raw.rename(
            columns={
                "Übergangszeit": "Spring/Fall",
                "Sommer": "Summer",
                "Werktag": "Working day",
                "Sonntag/Feiertag": "Sunday",
                "Samstag": "Saturday",
            },
            inplace=True,
        )
        residential_profile = time_series.copy()
        for i in residential_profile.index:
            residential_profile.loc[i, hours] = list(
                residential_profile_raw[(residential_profile.loc[i, "Season"], residential_profile.loc[i, "Day"])]
            )
        # Reshape the hourly load in one vector, where the rows are the hours of the year
        residential_profile = np.reshape(residential_profile.loc[:, hours].values, -1, order="C")
        profiles["RES"] = residential_profile / residential_profile.sum()

    # Industrial load
    if "IND" in param["load"]["sectors"]:
        industrial_profile_raw = pd.read_excel(profiles_paths["IND"], header=0)
        industrial_profile_raw.rename(columns={"Stunde": "Hour", "Last": "Load"}, inplace=True)
        # Reshape the hourly load in one vector, where the rows are the hours of the year
        industrial_profile = np.tile(industrial_profile_raw["Load"].values, 365)
        profiles["IND"] = industrial_profile / industrial_profile.sum()

    # Commercial load
    if "COM" in param["load"]["sectors"]:
        commercial_profile_raw = pd.read_csv(
            profiles_paths["COM"], sep="[;]", engine="python", decimal=",", skiprows=[0, 99], header=[0, 1], skipinitialspace=True
        )
        commercial_profile_raw.rename(
            columns={"Ãœbergangszeit": "Spring/Fall", "Sommer": "Summer", "Werktag": "Working day", "Sonntag": "Sunday", "Samstag": "Saturday"},
            inplace=True,
        )
        # Aggregate from 15 min --> hourly load
        commercial_profile_raw[("Hour", "All")] = [int(str(commercial_profile_raw.loc[i, ("G0", "[W]")])[:2]) for i in commercial_profile_raw.index]
        commercial_profile_raw = commercial_profile_raw.groupby([("Hour", "All")]).sum()
        commercial_profile_raw.reset_index(inplace=True)
        commercial_profile = time_series.copy()
        for i in commercial_profile.index:
            commercial_profile.loc[i, hours] = list(commercial_profile_raw[(commercial_profile.loc[i, "Season"], commercial_profile.loc[i, "Day"])])
        # Reshape the hourly load in one vector, where the rows are the hours of the year
        commercial_profile = np.reshape(commercial_profile.loc[:, hours].values, -1, order="C")
        profiles["COM"] = commercial_profile / commercial_profile.sum()

    # Agricultural load
    if "AGR" in param["load"]["sectors"]:
        agricultural_profile_raw = pd.read_csv(
            profiles_paths["AGR"], sep="[;]", engine="python", decimal=",", skiprows=[0, 99], header=[0, 1], skipinitialspace=True
        )
        agricultural_profile_raw.rename(
            columns={"Ãœbergangszeit": "Spring/Fall", "Sommer": "Summer", "Werktag": "Working day", "Sonntag": "Sunday", "Samstag": "Saturday"},
            inplace=True,
        )
        # Aggregate from 15 min --> hourly load
        agricultural_profile_raw["Hour"] = [int(str(agricultural_profile_raw.loc[i, ("L0", "[W]")])[:2]) for i in agricultural_profile_raw.index]
        agricultural_profile_raw = agricultural_profile_raw.groupby(["Hour"]).sum()
        agricultural_profile = time_series.copy()
        for i in agricultural_profile.index:
            agricultural_profile.loc[i, hours] = list(
                agricultural_profile_raw[(agricultural_profile.loc[i, "Season"], agricultural_profile.loc[i, "Day"])]
            )
        # Reshape the hourly load in one vector, where the rows are the hours of the year
        agricultural_profile = np.reshape(agricultural_profile.loc[:, hours].values, -1, order="C")
        profiles["AGR"] = agricultural_profile / agricultural_profile.sum()

    # Street lights
    if "STR" in param["load"]["sectors"]:
        streets_profile_raw = pd.read_excel(profiles_paths["STR"], header=[4], skipinitialspace=True, usecols=[0, 1, 2])
        # Aggregate from 15 min --> hourly load
        streets_profile_raw["Hour"] = [int(str(streets_profile_raw.loc[i, "Uhrzeit"])[:2]) for i in streets_profile_raw.index]
        streets_profile_raw = streets_profile_raw.groupby(["Datum", "Hour"]).sum()
        streets_profile_raw.iloc[0] = streets_profile_raw.iloc[0] + streets_profile_raw.iloc[-1]
        streets_profile_raw = streets_profile_raw.iloc[:-1]
        # Reshape the hourly load in one vector, where the rows are the hours of the year
        streets_profile = streets_profile_raw.values
        # Normalize the load over the year, ei. integral over the year of all loads for each individual sector is 1
        profiles["STR"] = streets_profile / streets_profile.sum()

    timecheck("End")
    return profiles


def clean_load_data_ENTSOE(paths, param):
    """
    This function reads the raw load time series from ENTSO-E, filters them for the desired year, scales them based on their coverage
    ratio, renames the countries based on *dict_countries*, and fills missing data by values from the day before (the magnitude is
    adjusted based on the trend of the previous five hours).
    
    :param paths: Dictionary containing the paths to the ENTSO-E input, to the dictionary of country names, and to the output.
    :type paths: dict
    :param param: Dictionary containing information about the year.
    :type param: dict
    
    :return: The result is saved directly in a CSV file in the desired path, along with its corresponding metadata.
    :rtype: None
    """
    timecheck("Start")

    # Read country load timeseries
    df_raw = pd.read_excel(paths["load_ts"], header=0, skiprows=[0, 1, 2], sep=",", decimal=".")

    # Filter by year
    df_year = df_raw.loc[df_raw["Year"] == param["year"]]

    # Scale based on coverage ratio
    df_scaled = df_year.copy()
    a = df_year.iloc[:, 5:].values
    b = df_year.iloc[:, 4].values
    c = a / b[:, np.newaxis] * 100
    df_scaled.iloc[:, 5:] = c
    del a, b, c

    # Reshape so that rows correspond to hours and columns to countries
    data = np.reshape(df_scaled.iloc[:, 5:].values.T, (-1, len(df_scaled["Country"].unique())), order="F")
    # Create dataframe where rows correspond to hours and columns to countries
    df_reshaped = pd.DataFrame(data, index=np.arange(data.shape[0]), columns=df_scaled["Country"].unique())

    # Rename countries
    dict_countries = pd.read_csv(paths["dict_countries"], sep=";", decimal=",", index_col=["ENTSO-E"], usecols=["ENTSO-E", "Countries shapefile"])
    dict_countries = dict_countries.loc[dict_countries.index.dropna()]["Countries shapefile"].to_dict()
    dict_countries_old = dict_countries.copy()
    for k, v in dict_countries_old.items():
        if ", " in k:
            keys = k.split(", ")
            for kk in keys:
                dict_countries[kk] = v
            del dict_countries[k]
    df_renamed = df_reshaped.rename(columns=dict_countries)

    # Group countries with same name
    df_grouped = df_renamed.T
    df_grouped = df_grouped.reset_index().rename(columns={"index": "Country"})
    df_grouped = df_grouped.groupby(["Country"]).sum().T
    df_grouped.reset_index(inplace=True, drop=True)

    # Fill missing data by values from the day before, adjusted based on the trend of the previous five hours
    df_filled = df_grouped.copy()
    for i, j in np.argwhere(df_filled.values == 0):
        df_filled.iloc[i, j] = df_filled.iloc[i - 5 : i, j].sum() / df_filled.iloc[i - 5 - 24 : i - 24, j].sum() * df_filled.iloc[i - 24, j].sum()

    df_filled.to_csv(paths["load_ts_clean"], index=False, sep=";", decimal=",")
    create_json(paths["load_ts_clean"], param, ["region_name", "year"], paths, ["dict_countries", "load_ts"])
    print("File saved: " + paths["load_ts_clean"])

    timecheck("End")


def clean_sector_shares_Eurostat(paths, param):
    """
    This function reads the CSV file with the sector shares from Eurostat (instructions on downloading it are in the documentation),
    filters it for the desired year and countries, reclassifies the country names and sectors, and normalizes the result.
    
    :param paths: Dictionary containing the paths to the Eurostat input, to the dictionary of country names, and to the output.
    :type paths: dict
    :param param: Dictionary containing information about the year and load-related assumptions.
    :type param: dict
    
    :return: The result is saved directly in a CSV file in the desired path, along with its corresponding metadata.
    :rtype: None
    
    MOVE THIS TO DOCUMENTATION ABOUT RECOMMENDED INPUTS
    For data from Eurostat, table: [nrg_105a]
    GEO: Choose all countries, but not EU
    INDIC_NRG: Choose all indices
    PRODUCT: Electrical energy (code 6000)
    TIME: Choose years
    UNIT: GWh
    Download in one single csv file
    """
    timecheck("Start")

    dict_countries = pd.read_csv(paths["dict_countries"], sep=";", decimal=",", index_col=["EUROSTAT"], usecols=["EUROSTAT", "Countries shapefile"])
    dict_countries = dict_countries.loc[dict_countries.index.dropna()]["Countries shapefile"].to_dict()
    dict_sectors = param["load"]["sectors_eurostat"]

    df_raw = pd.read_csv(
        paths["sector_shares"], sep=",", decimal=".", index_col=["TIME", "GEO", "INDIC_NRG"], usecols=["TIME", "GEO", "INDIC_NRG", "Value"]
    )

    # Filter the data
    filter_year = [param["year"]]
    filter_countries = list(dict_countries.keys())
    filter_indices = list(param["load"]["sectors_eurostat"].keys())
    filter_all = pd.MultiIndex.from_product([filter_year, filter_countries, filter_indices], names=["Year", "Country", "Sector"])
    df_raw.index.names = ["Year", "Country", "Sector"]
    df_filtered = df_raw.loc[df_raw.index.isin(filter_all)]

    # Reclassify
    df_reclassified = df_filtered.reset_index()
    df_reclassified.drop(["Year"], axis=1, inplace=True)
    for ind in df_reclassified.index:
        df_reclassified.loc[ind, "Country"] = dict_countries[df_reclassified.loc[ind, "Country"]]
        df_reclassified.loc[ind, "Sector"] = dict_sectors[df_reclassified.loc[ind, "Sector"]]
        try:
            df_reclassified.loc[ind, "Value"] = float(df_reclassified.loc[ind, "Value"].replace(" ", ""))
        except:
            df_reclassified.loc[ind, "Value"] = 0

    # Normalize
    total = df_reclassified.groupby(["Country"]).sum()["Value"]
    df_normalized = df_reclassified.groupby(["Country", "Sector"]).sum()
    df_normalized.reset_index(inplace=True)
    for ind in df_normalized.index:
        if not total[df_normalized.loc[ind, "Country"]]:
            df_normalized.loc[ind, "Value"] = 0
        else:
            df_normalized.loc[ind, "Value"] = df_normalized.loc[ind, "Value"] / total[df_normalized.loc[ind, "Country"]]

    # Reshape
    df_reshaped = df_normalized.pivot(index="Country", columns="Sector", values="Value")
    df_reshaped.to_csv(paths["sector_shares_clean"], index=True, sep=";", decimal=",")
    create_json(paths["sector_shares_clean"], param, ["region_name", "year", "load"], paths, ["dict_countries", "sector_shares"])
    print("File saved: " + paths["sector_shares_clean"])

    timecheck("End")


# def clean_processes_and_storage_data(paths, param):
# ''' documentation '''
# timecheck("Start")

# assumptions = pd.read_excel(paths["assumptions"], sheet_name='Process')

# depreciation = dict(zip(assumptions['Process'], assumptions['depreciation'].astype(float)))
# year_mu = dict(zip(assumptions['Process'], assumptions['year_mu'].astype(float)))
# year_stdev = dict(zip(assumptions['Process'], assumptions['year_stdev'].astype(float)))

# # Get data from fresna database
# Process = pd.read_csv(paths["database"], header=0, skipinitialspace=True,
# usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
# Process.rename(columns={'Capacity': 'inst-cap', 'YearCommissioned': 'year', 'lat': 'Latitude', 'lon': 'Longitude'},
# inplace=True)
# print('Number of power plants: ', len(Process))

# ## Process name

# # Use the name of the processes in OPSD as a standard name
# Process['Pro'] = Process['OPSD']

# # If the name is missing in OPSD, use the name in other databases
# Process.loc[Process['Pro'].isnull(), 'Pro'] = Process.loc[Process['Pro'].isnull(), 'CARMA']
# Process.loc[Process['Pro'].isnull(), 'Pro'] = Process.loc[Process['Pro'].isnull(), 'ENTSOE']
# Process.loc[Process['Pro'].isnull(), 'Pro'] = Process.loc[Process['Pro'].isnull(), 'GEO']
# Process.loc[Process['Pro'].isnull(), 'Pro'] = Process.loc[Process['Pro'].isnull(), 'WRI']

# # Add suffix to duplicate names
# Process['Pro'] = Process['Pro'] + Process.groupby(['Pro']).cumcount().astype(str).replace('0', '')

# # Remove spaces from the name and replace them with underscores
# Process['Pro'] = [Process.loc[i, 'Pro'].replace(' ', '_') for i in Process.index]

# # Remove useless columns
# Process.drop(['CARMA', 'ENTSOE', 'GEO', 'OPSD', 'WRI'], axis=1, inplace=True)

# print('Number of power plants with distinct names: ', len(Process['Pro'].unique()))

# ## Type

# Process['CoIn'] = np.nan
# for i in Process.index:
# # Get the pumped storage facilities
# if Process.loc[i, 'Technology'] in ['Pumped Storage', 'Pumped Storage With Natural Inflow',
# 'Pumped Storage, Pumped Storage With Natural Inflow, Reservoir',
# 'Pumped Storage, Reservoir', 'Pumped Storage, Run-Of-River']:
# Process.loc[i, 'CoIn'] = 'PumSt'

# # Get the tidal power plants
# if Process.loc[i, 'Technology'] == 'Tidal':
# Process.loc[i, 'CoIn'] = 'Tidal'

# # Assign an input commodity
# if pd.isnull(Process.loc[i, 'CoIn']):
# Process.loc[i, 'CoIn'] = param["pro_sto"]["proc_dict"][Process.loc[i, 'Fueltype']]

# # Distinguish between small and large hydro
# if (Process.loc[i, 'CoIn'] == 'Hydro_Small') and (Process.loc[i, 'inst-cap'] > 30):
# Process.loc[i, 'CoIn'] = 'Hydro_Large'

# # Remove useless columns
# Process.drop(['Fueltype', 'Technology', 'Set'], axis=1, inplace=True)

# # Remove renewable power plants (except Tidal and Geothermal)
# Process.set_index('CoIn', inplace=True)
# Process.drop(list(set(Process.index.unique()) & set(param["pro_sto"]["renewable_powerplants"])),
# axis=0, inplace=True)

# Process.reset_index(inplace=True)

# print('Possible processes: ', Process['CoIn'].unique())

# ## Include renewable power plants

# for pp in param["pro_sto"]["renewable_powerplants"]:
# # Shapefile with power plants
# pp_shapefile = gpd.read_file(paths["PPs_"] + pp + '.shp')
# pp_df = pd.DataFrame(pp_shapefile.rename(columns={'CapacityMW': 'inst-cap'}))
# pp_df['Longitude'] = [pp_df.loc[i, 'geometry'].x for i in pp_df.index]
# pp_df['Latitude'] = [pp_df.loc[i, 'geometry'].y for i in pp_df.index]
# pp_df['CoIn'] = pp
# pp_df['Pro'] = [pp + '_' + str(i) for i in pp_df.index]
# pp_df.drop(['geometry'], axis=1, inplace=True)
# Process = Process.append(pp_df, ignore_index=True, sort=True)

# ## Year

# # Assign a dummy year for missing entries (will be changed later)
# for c in Process['CoIn'].unique():
# if c in param["pro_sto"]["storage"]:
# Process.loc[(Process['CoIn'] == c) & (Process['year'].isnull()), 'year_mu'] = 1980
# Process.loc[(Process['CoIn'] == c) & (Process['year'].isnull()), 'year_stdev'] = 5
# else:
# Process.loc[(Process['CoIn'] == c) & (Process['year'].isnull()), 'year_mu'] = year_mu[c]
# Process.loc[(Process['CoIn'] == c) & (Process['year'].isnull()), 'year_stdev'] = year_stdev[c]

# Process.loc[Process['year'].isnull(), 'year'] = np.floor(
# np.random.normal(Process.loc[Process['year'].isnull(), 'year_mu'],
# Process.loc[Process['year'].isnull(), 'year_stdev']))

# # Drop recently built plants (after the reference year)
# Process = Process[(Process['year'] <= param["year"])]

# ## Coordinates

# P_missing = Process[Process['Longitude'].isnull()].copy()
# P_located = Process[~Process['Longitude'].isnull()].copy()

# # Assign dummy coordinates within the same country (will be changed later)
# for country in P_missing['Country'].unique():
# P_missing.loc[P_missing['Country'] == country, 'Latitude'] = P_located[P_located['Country'] == country].iloc[
# 0, 2]
# P_missing.loc[P_missing['Country'] == country, 'Longitude'] = P_located[P_located['Country'] == country].iloc[
# 0, 3]

# Process = P_located.append(P_missing)
# Process = Process[Process['Longitude'] > -11]

# ## Consider lifetime of power plants

# if param["year"] > param["pro_sto"]["year_ref"]:
# for c in Process['CoIn'].unique():
# if c in param["pro_sto"]["storage"]:
# Process.loc[Process['CoIn'] == c, 'lifetime'] = 60
# else:
# Process.loc[Process['CoIn'] == c, 'lifetime'] = depreciation[c]
# print(len(Process.loc[(Process['lifetime'] + Process['year']) < param["year"]]), ' processes will be deleted')
# Process.drop(Process.loc[(Process['lifetime'] + Process['year']) < param["year"]].index, inplace=True)

# ## Site

# # Create point geometries (shapely)
# Process['geometry'] = list(zip(Process.Longitude, Process.Latitude))
# Process['geometry'] = Process['geometry'].apply(Point)

# P_located = gpd.GeoDataFrame(Process, geometry='geometry', crs='')
# P_located.crs = {'init': 'epsg:4326'}

# # Define the output commodity
# P_located['CoOut'] = 'Elec'
# P_located.drop(['Latitude', 'Longitude', 'year_mu', 'year_stdev'], axis=1, inplace=True)
# P_located = P_located[['Pro', 'CoIn', 'CoOut', 'inst-cap', 'Country', 'year', 'geometry']]

# # Save the GeoDataFrame
# if not os.path.isfile(paths["pro_sto"]):
# P_located.to_file(driver='ESRI Shapefile', filename=paths["pro_sto"])
# else:
# os.remove(paths["pro_sto"])
# P_located.to_file(driver='ESRI Shapefile', filename=paths["pro_sto"])

# print("File Saved: " + paths["pro_sto"])
# timecheck("End")


# def clean_processes_and_storage_data_FRESNA(paths, param):
# ''' documentation '''
# timecheck("Start")

# assumptions = pd.read_excel(paths["assumptions"], sheet_name='Process')
# depreciation = dict(zip(assumptions['Process'], assumptions['depreciation'].astype(float)))
# year_mu = dict(zip(assumptions['Process'], assumptions['year_mu'].astype(float)))
# year_stdev = dict(zip(assumptions['Process'], assumptions['year_stdev'].astype(float)))

# # Get data from fresna database
# Process = pd.read_csv(paths["database_FRESNA"], header=0, skipinitialspace=True,
# usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
# Process.rename(columns={'Capacity': 'inst-cap', 'lat': 'Latitude', 'lon': 'Longitude'},
# inplace=True)
# print('Number of power plants: ', len(Process))

# Process['Technology'].fillna('NaN', inplace=True)
# Process['inst-cap'].fillna(0, inplace=True)
# Process[['Fueltype', 'Technology', 'Set']].drop_duplicates()
# Process[(Process['Technology'] == 'Run-Of-River') & (Process['Fueltype'] == 'Hydro')].groupby(['Country']).sum()
# Process.groupby(['Fueltype', 'Technology', 'Set']).sum().to_csv(paths["process_raw"], sep=';',
# decimal=',', index=True)
# # Type
# Process['CoIn'] = np.nan

# for i in Process.index:
# # Get the coal
# if Process.loc[i, 'Fueltype'] == 'Hard Coal':
# Process.loc[i, 'CoIn'] = 'Coal'
# # Get the gas
# if tuple(Process.loc[i, ['Fueltype', 'Technology']]) in [('Natural Gas', 'CCGT'),
# ('Natural Gas', 'CCGT, Thermal'),
# ('Natural Gas', 'Gas Engines'), ('Natural Gas', 'NaN'),
# ('Natural Gas', 'Pv')]:
# Process.loc[i, 'CoIn'] = 'Gas_CCGT'
# if tuple(Process.loc[i, ['Fueltype', 'Technology']]) in [('Natural Gas', 'OCGT')]:
# Process.loc[i, 'CoIn'] = 'Gas_OCGT'
# if tuple(Process.loc[i, ['Fueltype', 'Technology']]) in [('Natural Gas', 'Steam Turbine')]:
# Process.loc[i, 'CoIn'] = 'Gas_ST'
# # Get lignite and nuclear
# if Process.loc[i, 'Fueltype'] in ['Lignite', 'Nuclear']:
# Process.loc[i, 'CoIn'] = Process.loc[i, 'Fueltype']
# # Get oil and other
# if Process.loc[i, 'Fueltype'] in ['Oil', 'Other', 'Waste']:
# Process.loc[i, 'CoIn'] = 'Oil/Other'
# # Get the unconventional storage
# if tuple(Process.loc[i, ['Fueltype', 'Technology']]) in [('Natural Gas', 'Storage Technologies'),
# ('Natural Gas', 'Caes')]:
# Process.loc[i, 'CoIn'] = 'Storage_ST'
# # Get the pumped storage facilities
# if Process.loc[i, 'Technology'] in ['Pumped Storage', 'Pumped Storage With Natural Inflow']:
# Process.loc[i, 'CoIn'] = 'Storage_Long-Term'
# if tuple(Process.loc[i, ['Fueltype', 'Technology']]) in [('Hydro', 'Reservoir'), ('Hydro', 'NaN'),
# ('Hydro', 'Pv'), ('Hydro', 'Run-Of-River')]:
# Process.loc[i, 'CoIn'] = 'Hydro'
# Process.loc[i, 'Site'] = param["dist_ren"]["country_names"][Process.loc[i, 'Country']]

# # Remove useless rows
# Process.dropna(subset=['CoIn'], inplace=True)

# # year
# for i in Process.index:
# Process.loc[i, 'Year'] = max(Process.loc[i, 'YearCommissioned'], Process.loc[i, 'Retrofit'])
# Process.loc[i, 'Cohort'] = max((Process.loc[i, 'YearCommissioned'] // 5) * 5, 1960)
# Process.loc[i, 'Cohort_new'] = max((Process.loc[i, 'Year'] // 5) * 5, 1960)
# if np.isnan(Process.loc[i, 'Cohort']):
# Process.loc[i, 'Cohort'] = 'NaN'
# else:
# Process.loc[i, 'Cohort'] = str(int(Process.loc[i, 'Cohort']))
# if np.isnan(Process.loc[i, 'Cohort_new']):
# Process.loc[i, 'Cohort_new'] = 'NaN'
# else:
# Process.loc[i, 'Cohort_new'] = str(int(Process.loc[i, 'Cohort_new']))

# Process_agg = Process.groupby(['Site', 'CoIn', 'Cohort']).sum() / 1000

# Process_agg = Process_agg[['inst-cap']]

# full_ind = pd.MultiIndex.from_product([Process['Site'].unique(),
# Process['CoIn'].unique(),
# # ['Lignite', 'Coal', 'Gas_CCGT', 'Gas_ST', 'Gas_OCGT', 'Oil/Other'],
# Process['Cohort'].unique()])

# table_empty = pd.DataFrame(0, index=full_ind, columns=['inst-cap'])
# for i in Process_agg.index:
# table_empty.loc[i, 'inst-cap'] = Process_agg.loc[i, 'inst-cap']

# Process_agg = table_empty.reset_index().rename(columns={'level_0': 'Site', 'level_1': 'CoIn', 'level_2': 'Cohort'})

# Process_agg.set_index(['CoIn', 'Cohort'], inplace=True)
# Process_agg.pivot(columns='Site').to_csv(paths["Process_agg"], sep=';', decimal=',', index=True)

# Process.drop(Process[(Process['YearCommissioned'] > param["year"])].index, axis=0, inplace=True)

# # Assign a dummy year for missing entries (will be changed later)
# for c in Process['CoIn'].unique():
# if c == 'Storage_Long-Term':
# Process.loc[(Process['CoIn'] == c) & (Process['YearCommissioned'].isnull()), 'year_mu'] = 1980
# Process.loc[(Process['CoIn'] == c) & (Process['YearCommissioned'].isnull()), 'year_stdev'] = 5
# elif c == 'Storage_ST':
# Process.loc[(Process['CoIn'] == c) & (Process['YearCommissioned'].isnull()), 'year_mu'] = 2010
# Process.loc[(Process['CoIn'] == c) & (Process['YearCommissioned'].isnull()), 'year_stdev'] = 5
# else:
# Process.loc[(Process['CoIn'] == c) & (Process['YearCommissioned'].isnull()), 'year_mu'] = year_mu[c]
# Process.loc[(Process['CoIn'] == c) & (Process['YearCommissioned'].isnull()), 'year_stdev'] = year_stdev[c]

# Process.loc[Process['YearCommissioned'].isnull(), 'YearCommissioned'] = np.floor(
# np.random.normal(Process.loc[Process['YearCommissioned'].isnull(), 'year_mu'],
# Process.loc[Process['YearCommissioned'].isnull(), 'year_stdev']))

# Process.loc[Process['YearCommissioned'] > param["year"], 'YearCommissioned'] = param["year"]

# # Recalculate cohorts
# for i in Process.index:
# Process.loc[i, 'Year'] = max(Process.loc[i, 'YearCommissioned'], Process.loc[i, 'Retrofit'])
# Process.loc[i, 'Cohort'] = max((Process.loc[i, 'YearCommissioned'] // 5) * 5, 1960)
# Process.loc[i, 'Cohort_new'] = max((Process.loc[i, 'Year'] // 5) * 5, 1960)
# if np.isnan(Process.loc[i, 'Cohort']):
# Process.loc[i, 'Cohort'] = 'NaN'
# else:
# Process.loc[i, 'Cohort'] = str(int(Process.loc[i, 'Cohort']))
# if np.isnan(Process.loc[i, 'Cohort_new']):
# Process.loc[i, 'Cohort_new'] = 'NaN'
# else:
# Process.loc[i, 'Cohort_new'] = str(int(Process.loc[i, 'Cohort_new']))

# Process_agg2 = Process.groupby(['Site', 'CoIn', 'Cohort']).sum() / 1000

# Process_agg2 = Process_agg2[['inst-cap']]

# full_ind = pd.MultiIndex.from_product([Process['Site'].unique(),
# Process['CoIn'].unique(),
# # ['Lignite', 'Coal', 'Gas_CCGT', 'Gas_ST', 'Gas_OCGT', 'Oil/Other'],
# Process['Cohort'].unique()])

# table_empty = pd.DataFrame(0, index=full_ind, columns=['inst-cap'])
# for i in Process_agg2.index:
# table_empty.loc[i, 'inst-cap'] = Process_agg2.loc[i, 'inst-cap']

# Process_agg2 = table_empty.reset_index().rename(columns={'level_0': 'Site', 'level_1': 'CoIn', 'level_2': 'Cohort'})

# Process_agg2.set_index(['CoIn', 'Cohort'], inplace=True)
# Process_agg2.pivot(columns='Site').to_csv(paths["Process_agg_bis"], sep=';', decimal=',', index=True)

# # Process name
# # Use the name of the processes in OPSD as a standard name
# Process['Pro'] = Process['Name']
# Process['Pro'].fillna('unnamed', inplace=True)

# # Add suffix to duplicate names
# Process['Pro'] = Process['Pro'] + Process.groupby(['Pro']).cumcount().astype(str).replace('0', '')

# # Remove spaces from the name and replace them with underscores
# Process['Pro'] = [Process.loc[i, 'Pro'].replace(' ', '_') for i in Process.index]

# # Show except
# print('Number of power plants with distinct names: ', len(Process['Pro'].unique()))

# # Coordinates
# P_missing = Process[Process['Longitude'].isnull()].copy()
# P_located = Process[~Process['Longitude'].isnull()].copy()

# # Assign dummy coordinates within the same country (will be changed later)
# for country in P_missing['Country'].unique():
# P_missing.loc[P_missing['Country'] == country, 'Latitude'] = P_located[P_located['Country'] == country].iloc[
# 0, 9]
# P_missing.loc[P_missing['Country'] == country, 'Longitude'] = P_located[P_located['Country'] == country].iloc[
# 0, 10]

# Process = P_located.append(P_missing)
# Process = Process[Process['Longitude'] > -11]

# # Sites
# # Create point geometries (shapely)
# Process['geometry'] = list(zip(Process.Longitude, Process.Latitude))
# Process['geometry'] = Process['geometry'].apply(Point)

# P_located = gpd.GeoDataFrame(Process, geometry='geometry', crs='')
# P_located.crs = {'init': 'epsg:4326'}

# # Define the output commodity
# P_located['CoOut'] = 'Elec'

# P_located.drop(['Latitude', 'Longitude', 'year_mu', 'year_stdev'], axis=1, inplace=True)

# P_located = P_located[['Pro', 'CoIn', 'CoOut', 'inst-cap', 'Country', 'Year', 'geometry']]

# # Save the GeoDataFrame
# if not os.path.isfile(paths["pro_sto"]):
# P_located.to_file(driver='ESRI Shapefile', filename=paths["pro_sto"])
# else:
# os.remove(paths["pro_sto"])
# P_located.to_file(driver='ESRI Shapefile', filename=paths["pro_sto"])

# print("File Saved: " + paths["pro_sto"])
# timecheck("End")


def clean_GridKit_Europe(paths, param):
    """
    This function reads the raw data from GridKit (Europe). First, it converts the column of locations into separate columns
    of longitude and latitude of starting and ending points. Then, it expands the dataframe, so that every row would only have
    one entry in the columns for *voltage*, *wires*, *cables*, and *frequency*. Based on the user judgement of the *quality* of
    the data, the dataframe is filtered and rows with missing data are filled with most common value.
    Based on *length_m* and *voltage*, the values for the impedance *X_ohm*, the *loadability* and the surge impedance loading
    *SIL_MW* are determined.
    
    :param paths: Dictionary including the paths to the raw database *transmission_lines*, and to the desired intermediate and finally output locations.
    :type paths: dict
    :param param: Dictionary including the *grid* dictionary.
    :type param: dict

    :return: The cleaned database is saved as a CSV in the path *grid_cleaned* and as a shapefile in the path *grid_shp*, along with the corresponding metadata in JSON files.
    :rtype: None
    """
    timecheck("Start")

    # Read CSV file containing the lines data
    grid_raw = pd.read_csv(paths["transmission_lines"], header=0, sep=",", decimal=".")

    # Extract the string with the coordinates from the last column
    grid_raw["wkt_srid_4326"] = pd.Series(map(lambda s: s[21:-1], grid_raw["wkt_srid_4326"]), grid_raw.index)

    # Extract the coordinates into a new dataframe with four columns for each coordinate
    coordinates = pd.DataFrame(grid_raw["wkt_srid_4326"].str.split(" |,").tolist(), columns=["V1_long", "V1_lat", "V2_long", "V2_lat"])

    # Merge the original dataframe (grid_raw) with the one for the coordinates
    grid_raw = grid_raw.merge(coordinates, how="outer", left_index=True, right_index=True)

    # Drop the old column and the coordinates dataframe
    grid_raw.drop("wkt_srid_4326", axis=1, inplace=True)
    del coordinates

    # Expand columns with multiple values
    grid_expanded = grid_raw.copy()
    grid_expanded = expand_dataframe(grid_expanded, ["voltage", "wires", "cables", "frequency"])
    grid_expanded.to_csv(paths["grid_expanded"], index=False, sep=";", decimal=",")
    create_json(paths["grid_expanded"], param, [], paths, ["transmission_lines"])

    # If data is trustworthy, remove NaN values
    grid_filtered = grid_expanded.copy()
    for col in ["voltage", "wires", "cables", "frequency"]:
        if param["grid"]["quality"][col] == 1:
            grid_filtered = grid_filtered[~grid_filtered[col].isnull()]
    grid_filtered.to_csv(paths["grid_filtered"], index=False, sep=";", decimal=",")
    create_json(paths["grid_filtered"], param, ["grid"], paths, ["transmission_lines", "grid_expanded"])

    # Fill missing data with most common value
    grid_corrected = grid_filtered.copy()
    for col in ["voltage", "wires", "cables", "frequency"]:
        grid_corrected.loc[grid_corrected[col].isnull(), col] = grid_corrected[col].value_counts().index[0]

    # Replace voltage = 0 with most common value
    grid_corrected.loc[grid_corrected["voltage"] == 0, "voltage"] = grid_corrected["voltage"].value_counts().index[0]

    # Eventually overwrite the values in 'wires' using 'cables'
    if param["grid"]["quality"]["cables"] > param["grid"]["quality"]["wires"]:
        grid_corrected.loc[:, "wires"] = np.minimum(grid_corrected.loc[:, "cables"] // 3, 1)
    grid_corrected.to_csv(paths["grid_corrected"], index=False, sep=";", decimal=",")
    create_json(paths["grid_corrected"], param, ["grid"], paths, ["transmission_lines", "grid_expanded", "grid_filtered"])

    # Complete missing information
    grid_filled = grid_corrected.copy()
    grid_filled["length_m"] = grid_filled["length_m"].astype(float)
    grid_filled["x_ohmkm"] = assign_values_based_on_series(grid_filled["voltage"] / 1000, param["grid"]["specific_reactance"])
    grid_filled["X_ohm"] = grid_filled["x_ohmkm"] * grid_filled["length_m"] / 1000 / grid_filled["wires"]
    grid_filled["loadability"] = assign_values_based_on_series(grid_filled["length_m"] / 1000, param["grid"]["loadability"])
    grid_filled["SIL_MW"] = assign_values_based_on_series(grid_filled["voltage"] / 1000, param["grid"]["SIL"])
    grid_filled["Capacity_MVA"] = grid_filled["SIL_MW"] * grid_filled["loadability"] * grid_filled["wires"]
    grid_filled["Y_mho_ref_380kV"] = 1 / (grid_filled["X_ohm"] * ((380000 / grid_filled["voltage"]) ** 2))
    grid_filled.loc[grid_filled["frequency"] == 0, "tr_type"] = "DC_CAB"
    grid_filled.loc[~(grid_filled["frequency"] == 0), "tr_type"] = "AC_OHL"
    grid_filled.to_csv(paths["grid_filled"], index=False, sep=";", decimal=",")

    # Group lines with same IDs
    grid_grouped = (
        grid_filled[["l_id", "tr_type", "Capacity_MVA", "Y_mho_ref_380kV", "V1_long", "V1_lat", "V2_long", "V2_lat"]]
        .groupby(["l_id", "tr_type", "V1_long", "V1_lat", "V2_long", "V2_lat"])
        .sum()
    )
    grid_grouped.reset_index(inplace=True)
    grid_grouped.loc[:, ["V1_long", "V1_lat", "V2_long", "V2_lat"]] = grid_grouped.loc[:, ["V1_long", "V1_lat", "V2_long", "V2_lat"]].astype(float)
    grid_grouped.to_csv(paths["grid_cleaned"], index=False, sep=";", decimal=",")
    create_json(paths["grid_cleaned"], param, ["grid"], paths, ["transmission_lines", "grid_expanded", "grid_filtered", "grid_corrected"])
    print("File saved: " + paths["grid_cleaned"])

    # Writing to shapefile
    with shp.Writer(paths["grid_shp"], shapeType=3) as w:
        w.autoBalance = 1
        w.field("ID", "N", 6, 0)
        w.field("Cap_MVA", "N", 8, 2)
        w.field("Type", "C", 6, 0)
        count = len(grid_grouped.index)
        status = 0
        for i in grid_grouped.index:
            status += 1
            display_progress("Writing grid to shapefile: ", (count, status))
            w.line([[grid_grouped.loc[i, ["V1_long", "V1_lat"]].astype(float), grid_grouped.loc[i, ["V2_long", "V2_lat"]].astype(float)]])
            w.record(grid_grouped.loc[i, "l_id"], grid_grouped.loc[i, "Capacity_MVA"], grid_grouped.loc[i, "tr_type"])
    create_json(paths["grid_shp"], param, ["grid"], paths, ["transmission_lines", "grid_expanded", "grid_filtered", "grid_corrected"])
    print("File saved: " + paths["grid_shp"])
    timecheck("End")


# def distribute_renewable_capacities(paths, param):
# ''' documentation '''
# timecheck("Start")

# # Shapefile with countries
# countries = gpd.read_file(paths["Countries"])

# # Countries to be considered
# sites = pd.DataFrame(countries[['NAME_SHORT']].rename(columns={'NAME_SHORT': 'Site'}))
# sites = sites.sort_values(by=['Site'], axis=0)['Site'].unique()

# # Read input file, extracted from IRENA
# data_raw = pd.read_excel(paths["IRENA"], skiprows=[0, 1, 2, 3, 4, 5, 6])

# # Add missing country names
# for i in np.arange(1, len(data_raw.index)):
# if data_raw.isnull().loc[i, 'Country/area']:
# data_raw.loc[i, 'Country/area'] = data_raw.loc[i - 1, 'Country/area']

# # Select technologies needed in urbs and rename them
# data_raw = data_raw.loc[data_raw["Technology"].isin(param["dist_ren"]["renewables"])].reset_index(drop=True)
# data_raw["Technology"] = data_raw["Technology"].replace(param["dist_ren"]["renewables"])
# data_raw = data_raw.rename(columns={'Country/area': 'Site', 'Technology': 'Process', 2015: 'inst-cap'})

# # Create new dataframe with needed information, rename sites and extract chosen sites
# data = data_raw[["Site", "Process", "inst-cap"]]
# data = data.replace({"Site": param["dist_ren"]["country_names"]}).fillna(value=0)
# data = data.loc[data["Site"].isin(sites)].reset_index(drop=True)

# # Group by and sum
# data = data.groupby(["Site", "Process"]).sum().reset_index()

# # Estimate number of units
# units = param["dist_ren"]["units"]
# for p in data["Process"].unique():
# data.loc[data["Process"] == p, "Unit"] = data.loc[data["Process"] == p, "inst-cap"] // units[p] \
# + (data.loc[data["Process"] == p, "inst-cap"] % units[p] > 0)
# for p in data["Process"].unique():
# x = y = c = []
# for counter in range(0, len(countries) - 1):
# print(counter)
# if float(data.loc[(data["Site"] == countries.loc[counter, "NAME_SHORT"]) & (
# data["Process"] == p), 'inst-cap']) == 0:
# continue
# if (countries.loc[counter, "Population"]) & (p == 'WindOff'):
# continue
# if (countries.loc[counter, "Population"] == 0) & (p != 'WindOff'):
# continue
# name, x_off, y_off, potential = rasclip(paths["rasters"][p], paths["Countries"], counter)
# raster_shape = potential.shape
# potential = potential.flatten()

# # Calculate the part of the probability that is based on the potential
# potential_nan = np.isnan(potential) | (potential == 0)
# potential = (potential - np.nanmin(potential)) / (np.nanmax(potential) - np.nanmin(potential))
# potential[potential_nan] = 0

# # Calculate the random part of the probability
# potential_random = np.random.random_sample(potential.shape)
# potential_random[potential_nan] = 0

# # Combine the two parts
# potential_new = (1 - param["dist_ren"]["randomness"]) * potential \
# + param["dist_ren"]["randomness"] * potential_random

# # Sort elements based on their probability and keep the indices
# ind_sort = np.argsort(potential_new, axis=None)  # Ascending
# ind_needed = ind_sort[-int(data.loc[(data["Site"] == name) & (data["Process"] == p), "Unit"].values):]

# # Free memory
# del ind_sort, potential, potential_nan, potential_random

# # Get the coordinates of the power plants and their respective capacities
# power_plants = [units[p]] * len(ind_needed)
# if data.loc[(data["Site"] == name) & (data["Process"] == p), "inst-cap"].values % units[p] > 0:
# power_plants[-1] = data.loc[(data["Site"] == name) & (data["Process"] == p), "inst-cap"].values % units[
# p]
# y_pp, x_pp = np.unravel_index(ind_needed, raster_shape)
# x = x + ((x_pp + x_off + 0.5) * param["res_desired"][1] + param["Crd_all"][3]).tolist()
# y = y + (param["Crd_all"][0] - (y_pp + y_off + 0.5) * param["res_desired"][0]).tolist()
# c = c + potential_new[ind_needed].tolist()  # Power_plants

# del potential_new

# # Create map
# import pdb;
# pdb.set_trace()
# map_power_plants(p, x, y, c, paths["map_power_plants"] + p + '.shp')


# def format_process_model(process_compact, param):
# assump = param["assumptions"]

# # evrys
# output_pro_evrys = process_compact.copy()
# output_pro_evrys.drop(['on-off'], axis=1, inplace=True)

# output_pro_evrys = output_pro_evrys.join(pd.DataFrame([], columns=['eff', 'effmin', 'act-up', 'act-lo', 'on-off',
# 'start-cost', 'reserve-cost', 'ru', 'rd',
# 'rumax',
# 'rdmax', 'cotwo', 'detail', 'lambda', 'heatmax',
# 'maxdeltaT', 'heatupcost', 'su', 'sd', 'pdt',
# 'hotstart',
# 'pot', 'pretemp', 'preheat', 'prestate',
# 'prepow',
# 'precaponline']), how='outer')
# for c in output_pro_evrys['CoIn'].unique():
# output_pro_evrys.loc[output_pro_evrys.CoIn == c,
# ['eff', 'effmin', 'act-up', 'act-lo', 'on-off', 'start-cost',
# 'reserve-cost', 'ru', 'rd', 'rumax', 'rdmax', 'cotwo',
# 'detail', 'lambda', 'heatmax', 'maxdeltaT', 'heatupcost',
# 'su', 'sd', 'pdt', 'hotstart', 'pot', 'pretemp',
# 'preheat', 'prestate']] = [assump["eff"][c], assump["effmin"][c], assump["act_up"][c],
# assump["act_lo"][c], assump["on_off"][c],
# assump["start_cost"][c], assump["reserve_cost"][c],
# assump["ru"][c], assump["rd"][c], assump["rumax"][c],
# assump["rdmax"][c], assump["cotwo"][c], assump["detail"][c],
# assump["lambda_"][c], assump["heatmax"][c],
# assump["maxdeltaT"][c], assump["heatupcost"][c],
# assump["su"][c], assump["sd"][c], assump["pdt"][c],
# assump["hotstart"][c], assump["pot"][c],
# assump["pretemp"][c], assump["preheat"][c],
# assump["prestate"][c]]

# ind = output_pro_evrys['CoIn'] == 'Coal'
# output_pro_evrys.loc[ind, 'eff'] = 0.35 + 0.1 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (
# param["pro_sto"]["year_ref"] - 1960)
# output_pro_evrys.loc[ind, 'effmin'] = 0.92 * output_pro_evrys.loc[ind, 'eff']

# ind = output_pro_evrys['CoIn'] == 'Lignite'
# output_pro_evrys.loc[ind, 'eff'] = 0.33 + 0.1 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (
# param["pro_sto"]["year_ref"] - 1960)
# output_pro_evrys.loc[ind, 'effmin'] = 0.9 * output_pro_evrys.loc[ind, 'eff']

# ind = ((output_pro_evrys['CoIn'] == 'Gas') & (output_pro_evrys['inst-cap'] <= 100))
# output_pro_evrys.loc[ind, 'eff'] = 0.3 + 0.15 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (
# param["pro_sto"]["year_ref"] - 1960)
# output_pro_evrys.loc[ind, 'effmin'] = 0.65 * output_pro_evrys.loc[ind, 'eff']
# output_pro_evrys.loc[ind, 'act-lo'] = 0.3
# output_pro_evrys.loc[ind, 'ru'] = 0.01
# output_pro_evrys.loc[ind, 'lambda'] = 0.3
# output_pro_evrys.loc[ind, 'heatupcost'] = 20
# output_pro_evrys.loc[ind, 'su'] = 0.9

# ind = output_pro_evrys['CoIn'] == 'Oil'
# output_pro_evrys.loc[ind, 'eff'] = 0.25 + 0.15 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (
# param["pro_sto"]["year_ref"] - 1960)
# output_pro_evrys.loc[ind, 'effmin'] = 0.65 * output_pro_evrys.loc[ind, 'eff']

# ind = output_pro_evrys['CoIn'] == 'Nuclear'
# output_pro_evrys.loc[ind, 'eff'] = 0.3 + 0.05 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (
# param["pro_sto"]["year_ref"] - 1960)
# output_pro_evrys.loc[ind, 'effmin'] = 0.95 * output_pro_evrys.loc[ind, 'eff']

# output_pro_evrys['prepow'] = output_pro_evrys['inst-cap'] * output_pro_evrys['act-lo']
# output_pro_evrys['precaponline'] = output_pro_evrys['prepow']

# # Change the order of the columns
# output_pro_evrys = output_pro_evrys[
# ['Site', 'Pro', 'CoIn', 'CoOut', 'inst-cap', 'eff', 'effmin', 'act-lo', 'act-up',
# 'on-off', 'start-cost', 'reserve-cost', 'ru', 'rd', 'rumax', 'rdmax', 'cotwo',
# 'detail', 'lambda', 'heatmax', 'maxdeltaT', 'heatupcost', 'su', 'sd', 'pdt',
# 'hotstart', 'pot', 'prepow', 'pretemp', 'preheat', 'prestate', 'precaponline', 'year']]
# output_pro_evrys.iloc[:, 4:] = output_pro_evrys.iloc[:, 4:].astype(float)

# # function to remove non-ASCII
# def remove_non_ascii(text):
# return ''.join(i for i in text if ord(i) < 128)

# # function to shorten names
# def shorten_labels(text):
# return text[:63]

# output_pro_evrys.loc[:, 'Pro'] = output_pro_evrys.loc[:, 'Pro'].apply(remove_non_ascii)
# output_pro_evrys.loc[:, 'Pro'] = output_pro_evrys.loc[:, 'Pro'].apply(shorten_labels)

# # urbs

# # Take excerpt from the evrys table and group by tuple of sites and commodity
# process_grouped = output_pro_evrys[['Site', 'CoIn', 'inst-cap', 'act-lo', 'start-cost', 'ru']].apply(pd.to_numeric,
# errors='ignore')
# process_grouped.rename(columns={'CoIn': 'Process'}, inplace=True)
# process_grouped = process_grouped.groupby(['Site', 'Process'])

# inst_cap0 = process_grouped['inst-cap'].sum().rename('inst-cap')
# max_grad0 = process_grouped['ru'].mean().rename('max-grad') * 60
# max_grad0[max_grad0 == 60] = float('Inf')
# min_fraction0 = process_grouped['act-lo'].mean().rename('min-fraction')
# startup_cost0 = process_grouped['start-cost'].mean().rename('startup-cost')

# # Combine the list of series into a dataframe
# process_existant = pd.DataFrame([inst_cap0, max_grad0, min_fraction0, startup_cost0]).transpose()

# # Get the possible commodities and add Slacks
# commodity = list(output_pro_evrys.CoIn.unique())
# commodity.append('Slack')
# commodity.append('Shunt')

# # Create a dataframe to store all the possible combinations of sites and commodities
# df = pd.DataFrame(index=pd.MultiIndex.from_product([output_pro_evrys.Site.unique(), commodity],
# names=['Site', 'Process']))

# # Get the capacities of existing processes
# df_joined = df.join(process_existant, how='outer')

# # Set the capacity of inexistant processes to zero
# df_joined.loc[np.isnan(df_joined['inst-cap']), 'inst-cap'] = 0

# output_pro_urbs = df_joined.reset_index(drop=False)
# output_pro_urbs = output_pro_urbs.join(pd.DataFrame([], columns=['cap-lo', 'cap-up', 'inv-cost', 'fix-cost',
# 'var-cost', 'wacc', 'depreciation',
# 'area-per-cap']), how='outer')
# for c in output_pro_urbs['Process'].unique():
# output_pro_urbs.loc[
# output_pro_urbs['Process'] == c, ['cap-lo', 'cap-up', 'max-grad',
# 'min-fraction', 'inv-cost', 'fix-cost',
# 'var-cost', 'startup-cost', 'wacc',
# 'depreciation', 'area-per-cap']] = [
# assump["cap_lo"][c], assump["cap_up"][c], assump["max_grad"][c],
# assump["min_fraction"][c], assump["inv_cost"][c], assump["fix_cost"][c],
# assump["var_cost"][c], assump["startup_cost"][c], param["pro_sto"]["wacc"],
# assump["depreciation"][c], assump["area_per_cap"][c]]

# # Cap-up must be greater than inst-cap
# output_pro_urbs.loc[output_pro_urbs['cap-up'] < output_pro_urbs['inst-cap'], 'cap-up'] = output_pro_urbs.loc[
# output_pro_urbs['cap-up'] < output_pro_urbs['inst-cap'], 'inst-cap']

# # inst-cap must be greater than cap-lo
# output_pro_urbs.loc[output_pro_urbs['inst-cap'] < output_pro_urbs['cap-lo'], 'inst-cap'] = output_pro_urbs.loc[
# output_pro_urbs['inst-cap'] < output_pro_urbs['cap-lo'], 'cap-lo']

# # Cap-up must be of type float
# output_pro_urbs[['cap-up']] = output_pro_urbs[['cap-up']].astype(float)

# # Delete rows where cap-up is zero
# output_pro_urbs = output_pro_urbs[output_pro_urbs['cap-up'] != 0]

# # Change the order of the columns
# output_pro_urbs = output_pro_urbs[
# ['Site', 'Process', 'inst-cap', 'cap-lo', 'cap-up', 'max-grad', 'min-fraction', 'inv-cost',
# 'fix-cost', 'var-cost', 'startup-cost', 'wacc', 'depreciation', 'area-per-cap']]
# output_pro_urbs = output_pro_urbs.fillna(0)

# return output_pro_evrys, output_pro_urbs


# def format_storage_model(storage_compact, param):
# assump = param["assumptions"]

# # evrys
# output_sto_evrys = storage_compact.copy()

# output_sto_evrys = output_sto_evrys.join(pd.DataFrame([], columns=['inst-cap-po', 'inst-cap-c', 'eff-in', 'eff-out',
# 'var-cost-pi', 'var-cost-po', 'var-cost-c',
# 'act-up-pi',
# 'act-up-po', 'act-lo-pi', 'act-lo-po',
# 'act-lo-c',
# 'act-up-c', 'precont', 'prepowin', 'prepowout',
# 'ru', 'rd', 'rumax', 'rdmax', 'seasonal',
# 'ctr']), how='outer')
# for c in output_sto_evrys.Sto:
# output_sto_evrys.loc[
# output_sto_evrys.Sto == c, ['eff-in', 'eff-out', 'var-cost-pi', 'var-cost-po', 'var-cost-c',
# 'act-up-pi', 'act-up-po', 'act-lo-pi', 'act-lo-po', 'act-lo-c',
# 'act-up-c', 'prepowin', 'prepowout', 'ru', 'rd', 'rumax',
# 'rdmax', 'seasonal', 'ctr']] = [
# assump["eff_in"][c], assump["eff_out"][c], assump["var_cost_pi"][c],
# assump["var_cost_po"][c], assump["var_cost_c"][c], assump["act_up_pi"][c],
# assump["act_up_po"][c], assump["act_lo_pi"][c],
# assump["act_lo_po"][c], assump["act_lo_c"][c], assump["act_up_c"][c],
# assump["prepowin"][c], assump["prepowout"][c],
# assump["ru"][c], assump["rd"][c], assump["rumax"][c], assump["rdmax"][c],
# assump["seasonal"][c], assump["ctr"][c]]

# output_sto_evrys['inst-cap-po'] = output_sto_evrys['inst-cap-pi']
# output_sto_evrys.loc[output_sto_evrys['Sto'] == 'PumSt', 'inst-cap-c'] = 6 * output_sto_evrys.loc[
# output_sto_evrys['Sto'] == 'PumSt', 'inst-cap-pi']
# output_sto_evrys.loc[output_sto_evrys['Sto'] == 'Battery', 'inst-cap-c'] = 2 * output_sto_evrys.loc[
# output_sto_evrys['Sto'] == 'Battery', 'inst-cap-pi']
# output_sto_evrys['precont'] = 0.5 * output_sto_evrys['inst-cap-c']

# # Change the order of the columns
# output_sto_evrys = output_sto_evrys[
# ['Site', 'Sto', 'Co', 'inst-cap-pi', 'inst-cap-po', 'inst-cap-c', 'eff-in', 'eff-out',
# 'var-cost-pi', 'var-cost-po', 'var-cost-c', 'act-lo-pi', 'act-up-pi', 'act-lo-po',
# 'act-up-po', 'act-lo-c', 'act-up-c', 'precont', 'prepowin', 'prepowout', 'ru', 'rd',
# 'rumax', 'rdmax', 'seasonal', 'ctr']]
# output_sto_evrys = output_sto_evrys.iloc[:, :3].join(output_sto_evrys.iloc[:, 3:].astype(float))

# # urbs
# # Create a dataframe to store all the possible combinations of sites and commodities
# df = pd.DataFrame(index=pd.MultiIndex.from_product([param["regions"]['NAME_SHORT'].unique(),
# param["pro_sto"]["storage"]],
# names=['Site', 'Storage']))

# # Take excerpt from the evrys table and group by tuple of sites and commodity
# storage_existant = output_sto_evrys[['Site', 'Sto', 'Co', 'inst-cap-c', 'inst-cap-pi']].rename(
# columns={'Sto': 'Storage', 'Co': 'Commodity', 'inst-cap-pi': 'inst-cap-p'})

# # Get the capacities of existing processes
# df_joined = df.join(storage_existant.set_index(['Site', 'Storage']), how='outer')

# # Set the capacity of inexistant processes to zero
# df_joined['Commodity'].fillna('Elec', inplace=True)
# df_joined.fillna(0, inplace=True)

# output_sto_urbs = df_joined.reset_index()
# output_sto_urbs = output_sto_urbs.join(pd.DataFrame([], columns=['cap-lo-c', 'cap-up-c', 'cap-lo-p', 'cap-up-p',
# 'eff-in', 'eff-out', 'inv-cost-p', 'inv-cost-c',
# 'fix-cost-p', 'fix-cost-c', 'var-cost-p',
# 'var-cost-c',
# 'wacc', 'depreciation', 'init', 'discharge']),
# how='outer')
# for c in output_sto_urbs.Storage:
# output_sto_urbs.loc[
# output_sto_urbs.Storage == c, ['cap-lo-c', 'cap-up-c', 'cap-lo-p',
# 'cap-up-p', 'eff-in', 'eff-out',
# 'inv-cost-p', 'inv-cost-c', 'fix-cost-p',
# 'fix-cost-c', 'var-cost-p', 'var-cost-c',
# 'wacc', 'depreciation', 'init', 'discharge']] = [
# assump["cap_lo_c"][c], assump["cap_up_c"][c], assump["cap_lo_p"][c],
# assump["cap_up_p"][c], assump["eff_in"][c], assump["eff_out"][c],
# assump["inv_cost_p"][c], assump["inv_cost_c"][c], assump["fix_cost_p"][c],
# assump["fix_cost_c"][c], assump["var_cost_p"][c], assump["var_cost_c"][c],
# param["pro_sto"]["wacc"], assump["depreciation"][c], assump["init"][c], assump["discharge"][c]]

# output_sto_urbs.loc[output_sto_urbs['Storage'] == 'PumSt', 'cap-up-c'] = output_sto_urbs.loc[
# output_sto_urbs['Storage'] == 'PumSt', 'inst-cap-c']
# output_sto_urbs.loc[output_sto_urbs['Storage'] == 'PumSt', 'cap-up-p'] = output_sto_urbs.loc[
# output_sto_urbs['Storage'] == 'PumSt', 'inst-cap-p']

# # Change the order of the columns
# output_sto_urbs = output_sto_urbs[
# ['Site', 'Storage', 'Commodity', 'inst-cap-c', 'cap-lo-c', 'cap-up-c', 'inst-cap-p', 'cap-lo-p', 'cap-up-p',
# 'eff-in', 'eff-out', 'inv-cost-p', 'inv-cost-c', 'fix-cost-p', 'fix-cost-c', 'var-cost-p', 'var-cost-c',
# 'wacc', 'depreciation', 'init', 'discharge']]

# output_sto_urbs.iloc[:, 3:] = output_sto_urbs.iloc[:, 3:].astype(float)

# return output_sto_evrys, output_sto_urbs


# def format_process_model_California(process_compact, process_small, param):
# # evrys
# output_pro_evrys = process_compact.copy()
# output_pro_evrys['eff'] = 1  # Will be changed for thermal power plants
# output_pro_evrys['effmin'] = 1  # Will be changed for thermal power plants
# output_pro_evrys['act-up'] = 1
# output_pro_evrys['act-lo'] = 0  # Will be changed for most conventional power plants
# output_pro_evrys['on-off'] = 1  # Will be changed to 0 for SupIm commodities
# output_pro_evrys['start-cost'] = 0  # Will be changed for most conventional power plants
# output_pro_evrys['reserve-cost'] = 0
# output_pro_evrys['ru'] = 1  # Will be changed for thermal power plants
# output_pro_evrys['cotwo'] = 0  # Will be changed for most conventional power plants
# output_pro_evrys['detail'] = 1  # 5: thermal modeling, 1: simple modeling, will be changed for thermal power plants
# output_pro_evrys['lambda'] = 0  # Will be changed for most conventional power plants
# output_pro_evrys['heatmax'] = 1  # Will be changed for most conventional power plants
# output_pro_evrys['maxdeltaT'] = 1
# output_pro_evrys['heatupcost'] = 0  # Will be changed for most conventional power plants
# output_pro_evrys['su'] = 1  # Will be changed for most conventional power plants
# output_pro_evrys['pdt'] = 0
# output_pro_evrys['hotstart'] = 0
# output_pro_evrys['pot'] = 0
# output_pro_evrys['pretemp'] = 1
# output_pro_evrys['preheat'] = 0
# output_pro_evrys['prestate'] = 1

# ind = output_pro_evrys['CoIn'] == 'Coal'
# output_pro_evrys.loc[ind, 'eff'] = 0.35 + 0.1 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (param["year"] - 1960)
# output_pro_evrys.loc[ind, 'effmin'] = 0.92 * output_pro_evrys.loc[ind, 'eff']
# output_pro_evrys.loc[ind, 'act-lo'] = 0.4
# output_pro_evrys.loc[ind, 'start-cost'] = 90
# output_pro_evrys.loc[ind, 'ru'] = 0.03
# output_pro_evrys.loc[ind, 'cotwo'] = 0.33
# output_pro_evrys.loc[ind, 'detail'] = 5
# output_pro_evrys.loc[ind, 'lambda'] = 0.06
# output_pro_evrys.loc[ind, 'heatmax'] = 0.15
# output_pro_evrys.loc[ind, 'heatupcost'] = 110
# output_pro_evrys.loc[ind, 'su'] = 0.5

# ind = output_pro_evrys['CoIn'] == 'Lignite'
# output_pro_evrys.loc[ind, 'eff'] = 0.33 + 0.1 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (param["year"] - 1960)
# output_pro_evrys.loc[ind, 'effmin'] = 0.9 * output_pro_evrys.loc[ind, 'eff']
# output_pro_evrys.loc[ind, 'act-lo'] = 0.45
# output_pro_evrys.loc[ind, 'start-cost'] = 110
# output_pro_evrys.loc[ind, 'ru'] = 0.02
# output_pro_evrys.loc[ind, 'cotwo'] = 0.40
# output_pro_evrys.loc[ind, 'detail'] = 5
# output_pro_evrys.loc[ind, 'lambda'] = 0.04
# output_pro_evrys.loc[ind, 'heatmax'] = 0.12
# output_pro_evrys.loc[ind, 'heatupcost'] = 130
# output_pro_evrys.loc[ind, 'su'] = 0.5

# ind = (output_pro_evrys['CoIn'] == 'Gas') & (output_pro_evrys['inst-cap'] > 100) & (
# output_pro_evrys.index < len(process_compact) - len(process_small))
# output_pro_evrys.loc[ind, 'eff'] = 0.45 + 0.15 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (param["year"] - 1960)
# output_pro_evrys.loc[ind, 'effmin'] = 0.82 * output_pro_evrys.loc[ind, 'eff']
# output_pro_evrys.loc[ind, 'act-lo'] = 0.45
# output_pro_evrys.loc[ind, 'start-cost'] = 40
# output_pro_evrys.loc[ind, 'ru'] = 0.05
# output_pro_evrys.loc[ind, 'cotwo'] = 0.20
# output_pro_evrys.loc[ind, 'detail'] = 5
# output_pro_evrys.loc[ind, 'lambda'] = 0.1
# output_pro_evrys.loc[ind, 'heatmax'] = 0.2
# output_pro_evrys.loc[ind, 'heatupcost'] = 60
# output_pro_evrys.loc[ind, 'su'] = 0.5

# ind = (output_pro_evrys['CoIn'] == 'Gas') & ((output_pro_evrys['inst-cap'] <= 100) | (
# output_pro_evrys.index >= len(process_compact) - len(process_small)))
# output_pro_evrys.loc[ind, 'eff'] = 0.3 + 0.15 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (param["year"] - 1960)
# output_pro_evrys.loc[ind, 'effmin'] = 0.65 * output_pro_evrys.loc[ind, 'eff']
# output_pro_evrys.loc[ind, 'act-lo'] = 0.3
# output_pro_evrys.loc[ind, 'start-cost'] = 40
# output_pro_evrys.loc[ind, 'ru'] = 0.01
# output_pro_evrys.loc[ind, 'cotwo'] = 0.20
# output_pro_evrys.loc[ind, 'detail'] = 5
# output_pro_evrys.loc[ind, 'lambda'] = 0.3
# output_pro_evrys.loc[ind, 'heatupcost'] = 20
# output_pro_evrys.loc[ind, 'su'] = 0.9

# ind = output_pro_evrys['CoIn'] == 'Oil'
# output_pro_evrys.loc[ind, 'eff'] = 0.25 + 0.15 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (param["year"] - 1960)
# output_pro_evrys.loc[ind, 'effmin'] = 0.65 * output_pro_evrys.loc[ind, 'eff']
# output_pro_evrys.loc[ind, 'act-lo'] = 0.4
# output_pro_evrys.loc[ind, 'start-cost'] = 40
# output_pro_evrys.loc[ind, 'ru'] = 0.05
# output_pro_evrys.loc[ind, 'cotwo'] = 0.30
# output_pro_evrys.loc[ind, 'detail'] = 5
# output_pro_evrys.loc[ind, 'lambda'] = 0.3
# output_pro_evrys.loc[ind, 'heatupcost'] = 20
# output_pro_evrys.loc[ind, 'su'] = 0.7

# ind = output_pro_evrys['CoIn'] == 'Nuclear'
# output_pro_evrys.loc[ind, 'eff'] = 0.3 + 0.05 * (output_pro_evrys.loc[ind, 'year'] - 1960) / (param["year"] - 1960)
# output_pro_evrys.loc[ind, 'effmin'] = 0.95 * output_pro_evrys.loc[ind, 'eff']
# output_pro_evrys.loc[ind, 'act-lo'] = 0.45
# output_pro_evrys.loc[ind, 'start-cost'] = 150
# output_pro_evrys.loc[ind, 'ru'] = 0.04
# output_pro_evrys.loc[ind, 'detail'] = 5
# output_pro_evrys.loc[ind, 'lambda'] = 0.03
# output_pro_evrys.loc[ind, 'heatmax'] = 0.1
# output_pro_evrys.loc[ind, 'heatupcost'] = 100
# output_pro_evrys.loc[ind, 'su'] = 0.45

# ind = output_pro_evrys['CoIn'].isin(['Biomass', 'Waste'])
# output_pro_evrys.loc[ind, 'eff'] = 0.3
# output_pro_evrys.loc[ind, 'effmin'] = 0.3
# output_pro_evrys.loc[ind, 'ru'] = 0.05

# ind = output_pro_evrys['CoIn'].isin(['Solar', 'WindOn', 'WindOff', 'Hydro_large', 'Hydro_Small'])
# output_pro_evrys.loc[ind, 'on-off'] = 0

# output_pro_evrys['rd'] = output_pro_evrys['ru']
# output_pro_evrys['rumax'] = np.minimum(output_pro_evrys['ru'] * 60, 1)
# output_pro_evrys['rdmax'] = output_pro_evrys['rumax']
# output_pro_evrys['sd'] = output_pro_evrys['su']
# output_pro_evrys['prepow'] = output_pro_evrys['inst-cap'] * output_pro_evrys['act-lo']
# output_pro_evrys['precaponline'] = output_pro_evrys['prepow']
# # Change the order of the columns
# output_pro_evrys = output_pro_evrys[
# ['Site', 'Pro', 'CoIn', 'CoOut', 'inst-cap', 'eff', 'effmin', 'act-lo', 'act-up',
# 'on-off', 'start-cost', 'reserve-cost', 'ru', 'rd', 'rumax', 'rdmax', 'cotwo',
# 'detail', 'lambda', 'heatmax', 'maxdeltaT', 'heatupcost', 'su', 'sd', 'pdt',
# 'hotstart', 'pot', 'prepow', 'pretemp', 'preheat', 'prestate', 'precaponline', 'year']]

# # urbs
# # Take excerpt from the evrys table and group by tuple of sites and commodity
# process_grouped = output_pro_evrys[['Site', 'CoIn', 'inst-cap', 'act-lo', 'start-cost', 'ru']].groupby(
# ['Site', 'CoIn'])

# inst_cap0 = process_grouped['inst-cap'].sum().rename('inst-cap')
# max_grad0 = process_grouped['ru'].mean().rename('max-grad') * 60
# max_grad0[max_grad0 == 60] = float('Inf')
# min_fraction0 = process_grouped['act-lo'].mean().rename('min-fraction')
# startup_cost0 = process_grouped['start-cost'].mean().rename('startup-cost')

# # Combine the list of series into a dataframe
# process_existant = pd.DataFrame([inst_cap0, max_grad0, min_fraction0, startup_cost0]).transpose()

# # Get the possible commodities and add Slacks
# commodity = list(output_pro_evrys.CoIn.unique())
# commodity.append('Slack')

# # Create a dataframe to store all the possible combinations of sites and commodities
# df = pd.DataFrame(index=pd.MultiIndex.from_product([output_pro_evrys.Site.unique(), commodity],
# names=['Site', 'CoIn']))
# # Get the capacities of existing processes
# df_joined = df.join(process_existant, how='outer')

# # Set the capacity of inexistant processes to zero
# df_joined.loc[np.isnan(df_joined['inst-cap']), 'inst-cap'] = 0

# output_pro_urbs = df_joined.reset_index(drop=False)
# output_pro_urbs = output_pro_urbs.join(pd.DataFrame([], columns=['cap-lo', 'cap-up', 'inv-cost', 'fix-cost',
# 'var-cost', 'wacc', 'depreciation',
# 'area-per-cap']), how='outer')

# for c in output_pro_urbs.CoIn:
# output_pro_urbs.loc[
# output_pro_urbs.CoIn == c, ['cap-lo', 'cap-up', 'max-grad', 'min-fraction', 'inv-cost', 'fix-cost',
# 'var-cost']] = [param["pro_sto_Cal"]["Cal_urbs"]["cap_lo"][c],
# param["pro_sto_Cal"]["Cal_urbs"]["cap_up"][c],
# param["pro_sto_Cal"]["Cal_urbs"]["max_grad"][c],
# param["pro_sto_Cal"]["Cal_urbs"]["min_fraction"][c],
# param["pro_sto_Cal"]["Cal_urbs"]["inv_cost"][c],
# param["pro_sto_Cal"]["Cal_urbs"]["fix_cost"][c],
# param["pro_sto_Cal"]["Cal_urbs"]["var_cost"][c]]
# output_pro_urbs.loc[output_pro_urbs.CoIn == c, 'startup-cost'] = \
# param["pro_sto_Cal"]["Cal_urbs"]["startup_cost"][c]
# output_pro_urbs.loc[output_pro_urbs.CoIn == c, 'wacc'] = \
# param["pro_sto_Cal"]["Cal_urbs"]["wacc"]
# output_pro_urbs.loc[output_pro_urbs.CoIn == c, ['depreciation', 'area-per-cap']] = \
# [param["pro_sto_Cal"]["Cal_urbs"]["depreciation"][c],
# param["pro_sto_Cal"]["Cal_urbs"]["area_per_cap"][c]]

# # Cap-up must be greater than inst-cap
# output_pro_urbs.loc[output_pro_urbs['cap-up'] < output_pro_urbs['inst-cap'], 'cap-up'] = output_pro_urbs.loc[
# output_pro_urbs['cap-up'] < output_pro_urbs['inst-cap'], 'inst-cap']

# # inst-cap must be greater than cap-lo
# output_pro_urbs.loc[output_pro_urbs['inst-cap'] < output_pro_urbs['cap-lo'], 'inst-cap'] = output_pro_urbs.loc[
# output_pro_urbs['inst-cap'] < output_pro_urbs['cap-lo'], 'cap-lo']

# # Cap-up must be of type float
# output_pro_urbs[['cap-up']] = output_pro_urbs[['cap-up']].astype(float)

# # Delete rows where cap-up is zero
# output_pro_urbs = output_pro_urbs[output_pro_urbs['cap-up'] != 0]

# # Change the order of the columns
# output_pro_urbs = output_pro_urbs[
# ['Site', 'CoIn', 'inst-cap', 'cap-lo', 'cap-up', 'max-grad', 'min-fraction', 'inv-cost',
# 'fix-cost', 'var-cost', 'startup-cost', 'wacc', 'depreciation', 'area-per-cap']]

# return output_pro_evrys, output_pro_urbs


# def format_storage_model_California(storage_raw, param):
# # evrys
# # Take the raw storage table and group by tuple of sites and storage type
# sto_evrys = storage_raw[['Site', 'CoIn', 'CoOut', 'inst-cap']].rename(columns={'CoIn': 'Sto', 'CoOut': 'Co'})
# sto_grouped = sto_evrys.groupby(['Site', 'Sto'])

# inst_cap0 = sto_grouped['inst-cap'].sum().rename('inst-cap-pi')
# co0 = sto_grouped['Co'].first()

# # Combine the list of series into a dataframe
# sto_existant = pd.DataFrame([inst_cap0, co0]).transpose()
# output_sto_evrys = sto_existant.reset_index()
# output_sto_evrys['inst-cap-po'] = output_sto_evrys['inst-cap-pi']
# output_sto_evrys['var-cost-pi'] = 0.05
# output_sto_evrys['var-cost-po'] = 0.05
# output_sto_evrys['var-cost-c'] = -0.01
# output_sto_evrys['act-lo-pi'] = 0
# output_sto_evrys['act-up-pi'] = 1
# output_sto_evrys['act-lo-po'] = 0
# output_sto_evrys['act-up-po'] = 1
# output_sto_evrys['act-lo-c'] = 0
# output_sto_evrys['act-up-c'] = 1
# output_sto_evrys['prepowin'] = 0
# output_sto_evrys['prepowout'] = 0
# output_sto_evrys['ru'] = 0.1
# output_sto_evrys['rd'] = 0.1
# output_sto_evrys['rumax'] = 1
# output_sto_evrys['rdmax'] = 1
# output_sto_evrys['seasonal'] = 0
# output_sto_evrys['ctr'] = 1

# ind = (output_sto_evrys['Sto'] == 'PumSt')
# output_sto_evrys.loc[ind, 'inst-cap-c'] = 8 * output_sto_evrys.loc[ind, 'inst-cap-pi']
# output_sto_evrys.loc[ind, 'eff-in'] = 0.92
# output_sto_evrys.loc[ind, 'eff-out'] = 0.92

# ind = (output_sto_evrys['Sto'] == 'Battery')
# output_sto_evrys.loc[ind, 'inst-cap-c'] = 2 * output_sto_evrys.loc[ind, 'inst-cap-pi']
# output_sto_evrys.loc[ind, 'eff-in'] = 0.94
# output_sto_evrys.loc[ind, 'eff-out'] = 0.94
# output_sto_evrys['precont'] = output_sto_evrys['inst-cap-c'] / 2

# # Change the order of the columns
# output_sto_evrys = output_sto_evrys[
# ['Site', 'Sto', 'Co', 'inst-cap-pi', 'inst-cap-po', 'inst-cap-c', 'eff-in', 'eff-out',
# 'var-cost-pi', 'var-cost-po', 'var-cost-c', 'act-lo-pi', 'act-up-pi', 'act-lo-po',
# 'act-up-po', 'act-lo-c', 'act-up-c', 'precont', 'prepowin', 'prepowout', 'ru', 'rd',
# 'rumax', 'rdmax', 'seasonal', 'ctr']]

# # urbs
# # Create a dataframe to store all the possible combinations of sites and commodities
# df = pd.DataFrame(index=pd.MultiIndex.from_product([param["sites_evrys_unique"], output_sto_evrys.Sto.unique()],
# names=['Site', 'Storage']))
# # Take excerpt from the evrys table and group by tuple of sites and commodity
# storage_existant = output_sto_evrys[['Site', 'Sto', 'Co', 'inst-cap-c', 'inst-cap-pi', 'precont']].rename(
# columns={'Sto': 'Storage', 'Co': 'Commodity', 'inst-cap-pi': 'inst-cap-p'})

# # Get the capacities of existing processes
# df_joined = df.join(storage_existant.set_index(['Site', 'Storage']), how='outer')

# # Set the capacity of inexistant processes to zero
# df_joined['Commodity'].fillna('Elec', inplace=True)
# df_joined.fillna(0, inplace=True)
# out_sto_urbs = df_joined.reset_index()
# out_sto_urbs['cap-lo-c'] = 0
# out_sto_urbs['cap-up-c'] = out_sto_urbs['inst-cap-c']
# out_sto_urbs['cap-lo-p'] = 0
# out_sto_urbs['cap-up-p'] = out_sto_urbs['inst-cap-p']
# out_sto_urbs['var-cost-p'] = 0
# out_sto_urbs['var-cost-c'] = 0
# out_sto_urbs['wacc'] = 0
# out_sto_urbs['init'] = 0.5

# ind = out_sto_urbs['Storage'] == 'PumSt'
# out_sto_urbs.loc[ind, 'eff-in'] = 0.92
# out_sto_urbs.loc[ind, 'eff-out'] = 0.92
# out_sto_urbs.loc[ind, 'inv-cost-p'] = 275000
# out_sto_urbs.loc[ind, 'inv-cost-c'] = 0
# out_sto_urbs.loc[ind, 'fix-cost-p'] = 4125
# out_sto_urbs.loc[ind, 'fix-cost-c'] = 0
# out_sto_urbs.loc[ind, 'depreciation'] = 50
# ind = out_sto_urbs['Storage'] == 'Battery'
# out_sto_urbs.loc[ind, 'cap-up-c'] = np.inf
# out_sto_urbs.loc[ind, 'cap-up-p'] = np.inf
# out_sto_urbs.loc[ind, 'eff-in'] = 0.94
# out_sto_urbs.loc[ind, 'eff-out'] = 0.94
# out_sto_urbs.loc[ind, 'inv-cost-p'] = 75000
# out_sto_urbs.loc[ind, 'inv-cost-c'] = 200000
# out_sto_urbs.loc[ind, 'fix-cost-p'] = 3750
# out_sto_urbs.loc[ind, 'fix-cost-c'] = 10000
# out_sto_urbs.loc[ind, 'depreciation'] = 10

# # Change the order of the columns
# out_sto_urbs = out_sto_urbs[
# ['Site', 'Storage', 'Commodity', 'inst-cap-c', 'cap-lo-c', 'cap-up-c', 'inst-cap-p', 'cap-lo-p',
# 'cap-up-p', 'eff-in', 'eff-out', 'inv-cost-p', 'inv-cost-c', 'fix-cost-p', 'fix-cost-c',
# 'var-cost-p', 'var-cost-c', 'depreciation', 'wacc', 'init']]

# return output_sto_evrys, out_sto_urbs


# def format_transmission_model(icl_final, paths, param):
# # evrys
# output_evrys = pd.DataFrame(icl_final,
# columns=['SitIn', 'SitOut', 'Co', 'var-cost', 'inst-cap', 'act-lo', 'act-up',
# 'reactance',
# 'cap-up-therm', 'angle-up', 'length', 'tr_type', 'PSTmax', 'idx'])

# output_evrys['SitIn'] = icl_final['Region_start']
# output_evrys['SitOut'] = icl_final['Region_end']
# output_evrys['Co'] = 'Elec'
# output_evrys['var-cost'] = 0
# output_evrys['inst-cap'] = output_evrys['cap-up-therm'] = icl_final['Capacity_MVA']
# output_evrys['act-lo'] = 0
# output_evrys['act-up'] = 1
# output_evrys['reactance'] = icl_final['X_ohm'].astype(float)
# output_evrys['angle-up'] = 45
# output_evrys['PSTmax'] = 0
# output_evrys['idx'] = np.arange(1, len(output_evrys) + 1)

# # Length of lines based on distance between centroids
# coord = pd.read_csv(paths["sites"], sep=';', decimal=',').set_index('Site')
# coord = coord[coord['Population'] > 0]
# output_evrys = output_evrys.join(coord[['Longitude', 'Latitude']], on='SitIn', rsuffix='_1', how='inner')
# output_evrys = output_evrys.join(coord[['Longitude', 'Latitude']], on='SitOut', rsuffix='_2', how='inner')
# output_evrys.reset_index(inplace=True)
# output_evrys['length'] = [distance.distance(tuple(output_evrys.loc[i, ['Latitude', 'Longitude']].astype(float)),
# tuple(output_evrys.loc[i, ['Latitude_2', 'Longitude_2']].astype(
# float))).km
# for i in output_evrys.index]
# output_evrys.drop(['Longitude', 'Latitude', 'Longitude_2', 'Latitude_2', 'index'], axis=1, inplace=True)
# output_evrys = output_evrys.set_index(['SitIn', 'SitOut'])

# # Create a dataframe to store all the possible combinations of pairs of 1st order neighbors
# df = pd.DataFrame(columns=['SitIn', 'SitOut'])
# zones = param["zones"]
# weights = param["weights"]
# for z in range(len(zones)):
# for n in weights.neighbors[z]:
# if (zones[z] < zones[n]) & ~(zones[z].endswith('_off') | zones[n].endswith('_off')):
# df = df.append(pd.DataFrame([[zones[z], zones[n]]], columns=['SitIn', 'SitOut']), ignore_index=True)

# # urbs

# # Set SitIn and SitOut as index
# df.set_index(['SitIn', 'SitOut'], inplace=True)

# # Get the capacities of existing lines
# df_joined = df.join(output_evrys, how='outer')

# # Set the capacity of inexistant lines to zero
# df_joined['inst-cap'].fillna(0, inplace=True)

# # Reset the index
# df_joined.reset_index(drop=False, inplace=True)

# # Length of lines based on distance between centroids
# coord = pd.read_csv(paths["sites"], sep=';', decimal=',').set_index('Site')
# coord = coord[coord['Population'] > 0]
# df_joined = df_joined.join(coord[['Longitude', 'Latitude']], on='SitIn', rsuffix='_1', how='inner')
# df_joined = df_joined.join(coord[['Longitude', 'Latitude']], on='SitOut', rsuffix='_2', how='inner')
# df_joined['length'] = [distance.distance(tuple(df_joined.loc[i, ['Latitude', 'Longitude']].astype(float)),
# tuple(df_joined.loc[i, ['Latitude_2', 'Longitude_2']].astype(
# float))).km for i in df_joined.index]
# df_joined.drop(['Longitude', 'Latitude', 'Longitude_2', 'Latitude_2'], axis=1, inplace=True)

# output_urbs = df_joined.rename(columns={'SitIn': 'Site In', 'SitOut': 'Site Out', 'Co': 'Commodity'})
# output_urbs['tr_type'].fillna('AC_OHL', inplace=True)
# output_urbs.loc[output_urbs['tr_type'] == 'AC_OHL', 'Transmission'] = 'AC_OHL'
# output_urbs.loc[~(output_urbs['tr_type'] == 'AC_OHL'), 'Transmission'] = 'DC_CAB'
# output_urbs['Commodity'].fillna('Elec', inplace=True)

# # Use the length between the centroids [in km]
# output_urbs.loc[output_urbs['tr_type'] == 'AC_OHL', 'eff'] = 0.92 ** (
# output_urbs.loc[output_urbs['tr_type'] == 'AC_OHL', 'length'] / 1000)
# output_urbs.loc[output_urbs['tr_type'] == 'AC_CAB', 'eff'] = 0.9 ** (
# output_urbs.loc[output_urbs['tr_type'] == 'AC_CAB', 'length'] / 1000)
# output_urbs.loc[output_urbs['tr_type'] == 'DC_OHL', 'eff'] = 0.95 ** (
# output_urbs.loc[output_urbs['tr_type'] == 'DC_OHL', 'length'] / 1000)
# output_urbs.loc[output_urbs['tr_type'] == 'DC_CAB', 'eff'] = 0.95 ** (
# output_urbs.loc[output_urbs['tr_type'] == 'DC_CAB', 'length'] / 1000)

# output_urbs.loc[(output_urbs['tr_type'] == 'AC_OHL')
# & (output_urbs['length'] < 150), 'inv-cost'] = \
# 300 * output_urbs.loc[(output_urbs['tr_type'] == 'AC_OHL')
# & (output_urbs['length'] < 150), 'length']

# output_urbs.loc[(output_urbs['tr_type'] == 'AC_OHL')
# & (output_urbs['length'] >= 150), 'inv-cost'] = \
# 770 * output_urbs.loc[(output_urbs['tr_type'] == 'AC_OHL')
# & (output_urbs['length'] >= 150), 'length'] - 70000

# output_urbs.loc[(output_urbs['tr_type'] == 'AC_CAB')
# & (output_urbs['length'] < 150), 'inv-cost'] = \
# 1200 * output_urbs.loc[(output_urbs['tr_type'] == 'AC_CAB')
# & (output_urbs['length'] < 150), 'length']

# output_urbs.loc[(output_urbs['tr_type'] == 'AC_CAB')
# & (output_urbs['length'] >= 150), 'inv-cost'] = \
# 3080 * output_urbs.loc[(output_urbs['tr_type'] == 'AC_CAB')
# & (output_urbs['length'] >= 150), 'length'] - 280000

# output_urbs.loc[output_urbs['tr_type'] == 'DC_OHL', 'inv-cost'] = 288 * output_urbs.loc[
# output_urbs['tr_type'] == 'DC_OHL', 'length'] + 160000
# output_urbs.loc[output_urbs['tr_type'] == 'DC_CAB', 'inv-cost'] = 1152 * output_urbs.loc[
# output_urbs['tr_type'] == 'DC_CAB', 'length'] + 160000

# output_urbs.loc[(output_urbs['tr_type'] == 'AC_OHL')
# | (output_urbs['tr_type'] == 'DC_OHL'), 'fix-cost'] = \
# 42 * output_urbs.loc[(output_urbs['tr_type'] == 'AC_OHL')
# | (output_urbs['tr_type'] == 'DC_OHL'), 'length']

# output_urbs.loc[(output_urbs['tr_type'] == 'AC_CAB')
# | (output_urbs['tr_type'] == 'DC_CAB'), 'fix-cost'] = \
# 21 * output_urbs.loc[(output_urbs['tr_type'] == 'AC_CAB')
# | (output_urbs['tr_type'] == 'DC_CAB'), 'length']

# output_urbs['var-cost'].fillna(0, inplace=True)
# output_urbs['cap-lo'] = param["dist_ren"]["cap_lo"]
# output_urbs['cap-up'] = output_urbs['inst-cap']
# output_urbs['cap-up'].fillna(0, inplace=True)
# output_urbs['wacc'] = param["pro_sto"]["wacc"]
# output_urbs['depreciation'] = param["grid"]["depreciation"]

# # Change the order of the columns
# output_urbs = output_urbs[
# ['Site In', 'Site Out', 'Transmission', 'Commodity', 'eff', 'inv-cost', 'fix-cost', 'var-cost',
# 'inst-cap', 'cap-lo', 'cap-up', 'wacc', 'depreciation']]
# output_urbs.iloc[:, 4:] = output_urbs.iloc[:, 4:].astype('float64')
# return output_evrys, output_urbs
