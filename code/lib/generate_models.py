from .util import *


def generate_urbs_model(paths, param):
    """
    This function reads all the intermediate CSV files, adapts the formatting to the structure of the urbs Excel input file,
    and combines the datasets into one dataframe. It writes the dataframe into an urbs input Excel file.
    The function would still run even if some files have not been generated. They will simply be skipped.
    
    :param paths: Dictionary including the paths to the intermediate files *sites_sub*, *commodities_regions*, *process_regions*,
      *assumptions_flows*, *grid_completed*, *storage_regions*, *load_regions*, *potential_ren*, and to the output *urbs_model*.
    :type paths: dict
    :param param: Dictionary of user preferences, including *model_year* and *technology*.
    :type param: dict
    
    :return: The XLSX model input file is saved directly in the desired path.
    :rtype: None
    """
    timecheck("Start")

    urbs_model = {}

    # Read Global

    # Read sites
    if os.path.exists(paths["sites_sub"]):
        sites = pd.read_csv(paths["sites_sub"], sep=";", decimal=",")
        sites = sites[["Name", "Area_m2"]].rename(columns={"Area_m2": "area"})
        urbs_model["Site"] = sites
        del sites

    # Read commodities
    if os.path.exists(paths["commodities_regions"]):
        com = pd.read_csv(paths["commodities_regions"], sep=";", decimal=",")
        com.rename(columns={"Type_urbs": "Type"}, inplace=True)
        com = com[["Site", "Commodity", "Type", "price", "max", "maxperhour"]]
        for col in range(3, com.shape[1]):
            try:
                com.iloc[:, col] = com.iloc[:, col].str.replace(".", "").astype("f")
            except:
                continue
        urbs_model["Commodity"] = com
        del com

    # Read Processes
    if os.path.exists(paths["process_regions"]):
        proc = pd.read_csv(paths["process_regions"], sep=";", decimal=",")
        proc.rename(columns={"Name": "Process", "min-fraction": "min-frac", "start-cost": "startup-cost"}, inplace=True)
        proc = proc[
            [
                "Site",
                "Process",
                "inst-cap",
                "cap-lo",
                "cap-up",
                "max-grad",
                "min-frac",
                "inv-cost",
                "fix-cost",
                "var-cost",
                "startup-cost",
                "wacc",
                "depreciation",
                "area-per-cap",
            ]
        ]
        for col in range(2, proc.shape[1]):
            try:
                proc.iloc[:, col] = proc.iloc[:, col].str.replace(".", "").astype("f")
            except:
                continue
        urbs_model["Process"] = proc
        del proc

    # Read Process-Commodity
    procom = pd.read_csv(paths["assumptions_flows"], sep=";", decimal=",")
    procom = procom.loc[procom["year"] == param["model_year"]]
    procom = procom.loc[procom["Process/Storage"].isin(param["technology"]["Process"])]
    procom.rename(columns={"Process/Storage": "Process"}, inplace=True)
    procom = procom[["Process", "Commodity", "Direction", "ratio", "ratio-min"]]
    for col in range(3, procom.shape[1]):
        try:
            procom.iloc[:, col] = procom.iloc[:, col].str.replace(".", "").astype("f")
        except:
            continue
    urbs_model["Process-Commodity"] = procom

    # Read transmission
    if os.path.exists(paths["grid_completed"]):
        grid = pd.read_csv(paths["grid_completed"], sep=";", decimal=",")
        grid.rename(columns={"tr_type": "Transmission"}, inplace=True)
        grid = grid[
            [
                "Site In",
                "Site Out",
                "Transmission",
                "Commodity",
                "eff",
                "inv-cost",
                "fix-cost",
                "var-cost",
                "inst-cap",
                "cap-lo",
                "cap-up",
                "wacc",
                "depreciation",
            ]
        ]
        for col in range(4, grid.shape[1]):
            try:
                grid.iloc[:, col] = grid.iloc[:, col].str.replace(".", "").astype("f")
            except:
                continue
        urbs_model["Transmission"] = grid
        del grid

    # Read storage
    if os.path.exists(paths["storage_regions"]):
        sto = pd.read_csv(paths["storage_regions"], sep=";", decimal=",")
        sto.rename(columns={"Type": "Storage", "inst-cap": "inst-cap-p"}, inplace=True)

        sto = sto[
            [
                "Site",
                "Storage",
                "Commodity",
                "inst-cap-c",
                "cap-lo-c",
                "cap-up-c",
                "inst-cap-p",
                "cap-lo-p",
                "cap-up-p",
                "eff-in",
                "eff-out",
                "inv-cost-p",
                "inv-cost-c",
                "fix-cost-p",
                "fix-cost-c",
                "var-cost-p",
                "var-cost-c",
                "wacc",
                "depreciation",
                "init",
                "discharge",
                "ep-ratio",
            ]
        ]
        for col in range(3, sto.shape[1]):
            try:
                sto.iloc[:, col] = sto.iloc[:, col].str.replace(".", "").astype("f")
            except:
                continue
        urbs_model["Storage"] = sto
        del sto

    # Read DSM

    # Read electricity demand
    if os.path.exists(paths["load_regions"]):
        demand = pd.read_csv(paths["load_regions"], sep=";", decimal=",", index_col=0)
        demand.columns = demand.columns + ".Elec"
        demand.index = range(1, 8761)
        demand.loc[0] = 0
        demand.sort_index(inplace=True)
        demand.insert(0, "t", demand.index)
        urbs_model["Demand"] = demand.astype("f")
        del demand

    # Read intermittent supply time series
    if os.path.exists(paths["potential_ren"]):
        supim = pd.read_csv(paths["potential_ren"], sep=";", decimal=",", index_col=0)
        supim.index = range(1, 8761)
        supim.insert(0, "t", supim.index)
        urbs_model["SupIm"] = supim.astype("f")
        del supim

    # # Add global parameters
    # urbs_model["Global"] = pd.read_excel(paths["assumptions"], sheet_name='Global')

    # # Add DSM and Buy-Sell-Price
    # DSM_header = ['Site', 'Commodity', 'delay', 'eff', 'recov', 'cap-max-do', 'cap-max-up']
    # urbs_model["DSM"] = pd.DataFrame(columns=DSM_header)
    # urbs_model["Buy-Sell-Price"] = pd.DataFrame(np.arange(0, 8761), columns=['t'])

    # Create ExcelWriter
    with pd.ExcelWriter(paths["urbs_model"], mode="w") as writer:
        # populate excel file with available sheets
        status = 0
        display_progress("Writing to excel file in progress: ", (len(urbs_model.keys()), status))
        for sheet in urbs_model.keys():
            urbs_model[sheet].to_excel(writer, sheet_name=sheet, index=False, header=True)

            # Display Progress bar
            status += 1
            display_progress("Writing to excel file in progress: ", (len(urbs_model.keys()), status))
    print("File saved: " + paths["urbs_model"])

    timecheck("End")


def generate_evrys_model(paths, param):
    """
    This function reads all the intermediate CSV files, adapts the formatting to the structure of the evrys Excel input file,
    and combines the datasets into one dataframe. It writes the dataframe into an evrys input Excel file.
    The function would still run even if some files have not been generated. They will simply be skipped.
    
    :param paths: Dictionary including the paths to the intermediate files *sites_sub*, *commodities_regions*, *process_regions*,
      *grid_completed*, *storage_regions*, *load_regions*, *potential_ren*, and to the output *evrys_model*.
    :type paths: dict
    :param param: Dictionary of user preferences, including *model_year* and *technology*.
    :type param: dict
    
    :return: The XLSX model input file is saved directly in the desired path.
    :rtype: None
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

    # Read commodities
    if os.path.exists(paths["commodities_regions"]):
        com = pd.read_csv(paths["commodities_regions"], sep=";", decimal=",")
        com.rename(columns={"Commodity": "Co", "Type_evrys": "type"}, inplace=True)
        com = com[["Site", "Co", "price", "annual", "losses", "type"]]
        evrys_model["Commodity"] = com
        del com

    # Read Processes
    if os.path.exists(paths["process_regions"]):
        proc = pd.read_csv(paths["process_regions"], sep=";", decimal=",")
        proc.rename(columns={"Name": "Pro", "Type": "CoIn", "Year": "year"}, inplace=True)
        proc["CoOut"] = "Elec"
        proc = proc[
            [
                "Site",
                "Pro",
                "CoIn",
                "CoOut",
                "inst-cap",
                "eff",
                "effmin",
                "act-lo",
                "act-up",
                "on-off",
                "start-cost",
                "reserve-cost",
                "ru",
                "rd",
                "rumax",
                "rdmax",
                "cotwo",
                "detail",
                "lambda",
                "heatmax",
                "maxdeltaT",
                "heatupcost",
                "su",
                "sd",
                "pdt",
                "hotstart",
                "pot",
                "prepow",
                "pretemp",
                "preheat",
                "prestate",
                "precaponline",
                "year",
            ]
        ]
        evrys_model["Process"] = proc
        del proc

    # Read transmission
    if os.path.exists(paths["grid_completed"]):
        grid = pd.read_csv(paths["grid_completed"], sep=";", decimal=",")
        grid.rename(columns={"Site In": "SitIn", "Site Out": "SitOut", "Commodity": "Co", "impedance": "reactance"}, inplace=True)
        grid = grid[
            [
                "SitIn",
                "SitOut",
                "Co",
                "var-cost",
                "inst-cap",
                "act-lo",
                "act-up",
                "reactance",
                "cap-up-therm",
                "angle-up",
                "length",
                "tr_type",
                "PSTmax",
                "idx",
            ]
        ]
        evrys_model["Transmission"] = grid
        del grid

    # Read storage
    if os.path.exists(paths["storage_regions"]):
        sto = pd.read_csv(paths["storage_regions"], sep=";", decimal=",")
        sto.rename(columns={"Type": "Sto", "Commodity": "Co"}, inplace=True)
        sto["inst-cap-pi"] = sto["inst-cap"]
        sto["inst-cap-po"] = sto["inst-cap"]
        sto["inst-cap-c"] = sto["inst-cap"] * sto["ep-ratio"]
        sto = sto[
            [
                "Site",
                "Sto",
                "Co",
                "inst-cap-pi",
                "inst-cap-po",
                "inst-cap-c",
                "eff-in",
                "eff-out",
                "var-cost-pi",
                "var-cost-po",
                "var-cost-c",
                "act-lo-pi",
                "act-up-pi",
                "act-lo-po",
                "act-up-po",
                "act-lo-c",
                "act-up-c",
                "precont",
                "prepowin",
                "prepowout",
                "ru",
                "rd",
                "rumax",
                "rdmax",
                "seasonal",
                "ctr",
            ]
        ]
        evrys_model["Storage"] = sto
        del sto

    # Read DSM

    # Read intermittent supply time series
    if os.path.exists(paths["potential_ren"]):
        # Format to evrys input format
        raw_data = pd.read_csv(paths["potential_ren"], sep=";", decimal=",", index_col=0)
        sites = []
        com = []
        for col in list(raw_data.columns):
            sit, co = str(col).split(".")
            sites = sites + [sit]
            com = com + [co]
        sites = sorted(set(sites))
        com = sorted(set(com))
        site_com_ind = pd.MultiIndex.from_product([sites, com], names=["Sites", "Commodities"])
        suplm = pd.DataFrame(data=None, index=raw_data.index, columns=site_com_ind)
        for col in list(raw_data.columns):
            sit, co = str(col).split(".")
            suplm[sit, co] = raw_data[col]
        suplm = suplm.transpose().stack().reset_index()
        suplm = suplm.rename(columns={"Sites": "sit", "Commodities": "co", "level_2": "t", 0: "value"})
        suplm = suplm[["t", "sit", "co", "value"]]
        evrys_model["suplm"] = suplm
        del suplm

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
    with pd.ExcelWriter(paths["evrys_model"], mode="w") as writer:
        # populate excel file with available sheets
        status = 0
        display_progress("Writing to excel file in progress: ", (len(evrys_model.keys()), status))
        for sheet in evrys_model.keys():
            evrys_model[sheet].to_excel(writer, sheet_name=sheet, index=False)

            # Display Progress Bar
            status += 1
            display_progress("Writing to excel file in progress: ", (len(evrys_model.keys()), status))

    print("File saved: " + paths["evrys_model"])

    timecheck("End")
