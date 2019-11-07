# from config import ts_paths
from lib.util import *


def generate_urbs_model(paths, param):
    """
    Read model's .csv files, and create relevant dataframes.
    Writes dataframes to urbs input excel file.
    """
    timecheck("Start")

    urbs_model = {}

    # Read sites
    if os.path.exists(paths["sites_sub"]):
        sites = pd.read_csv(paths["sites_sub"], sep=";", decimal=",")
        sites = sites[["Name", "Area_m2"]].rename(columns={"Area_m2": "area"})
        urbs_model["Site"] = sites

    # Read electricity demand
    if os.path.exists(paths["load_regions"]):
        demand = pd.read_csv(paths["load_regions"], sep=";", decimal=",", index_col=0)
        demand.columns = demand.columns + ".Elec"
        demand.index.name = "t"
        demand.index = range(1, 8761)
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

    # TRANSMISSION
    # # Change the order of the columns
    # output_urbs = output_urbs[
    # ['Site In', 'Site Out', 'Transmission', 'Commodity', 'eff', 'inv-cost', 'fix-cost', 'var-cost',
    # 'inst-cap', 'cap-lo', 'cap-up', 'wacc', 'depreciation']]

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
    with ExcelWriter(paths["urbs_model"], mode="w") as writer:
        # populate excel file with available sheets
        status = 0
        for sheet in urbs_model.keys():
            urbs_model[sheet].to_excel(writer, sheet_name=sheet, index=False, header=True)
            status += 1
            display_progress("Writing to excel file in progress: ", (len(urbs_model.keys()), status))

    print("File saved: " + paths["urbs_model"])

    timecheck("End")

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
    timecheck("Start")

    evrys_model = {}

    # Read sites
    if os.path.exists(paths["sites_sub"]):
        sites = pd.read_csv(paths["sites_sub"], sep=";", decimal=",")
        sites = sites[
            ["Name", "slacknode", "syncharea", "Latitude", "Longitude", "ctrarea", "primpos", "primneg", "secpos", "secneg", "terpos", "terneg"]
        ].rename(columns={"Name": "Site", "Latitude": "lat", "Longitude": "long"})
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

    # TRANSMISSION
    # # evrys
    # output_evrys = pd.DataFrame(icl_final,
    # columns=['SitIn', 'SitOut', 'Co', 'var-cost', 'inst-cap', 'act-lo', 'act-up',
    # 'reactance',
    # 'cap-up-therm', 'angle-up', 'length', 'tr_type', 'PSTmax', 'idx'])

    # Create ExcelWriter
    with ExcelWriter(paths["evrys_model"], mode="w") as writer:
        # populate excel file with available sheets
        status = 0
        for sheet in evrys_model.keys():
            evrys_model[sheet].to_excel(writer, sheet_name=sheet, index=False)
            status += 1
            display_progress("Writing to excel file in progress: ", (len(evrys_model.keys()), status))

    print("File saved: " + paths["evrys_model"])

    timecheck("End")
