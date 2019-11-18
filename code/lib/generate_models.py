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
        del sites

    # Read electricity demand
    if os.path.exists(paths["load_regions"]):
        demand = pd.read_csv(paths["load_regions"], sep=";", decimal=",", index_col=0)
        demand.columns = demand.columns + ".Elec"
        demand.index = range(1, 8761)
        demand.loc[0] = 0
        demand.sort_index(inplace=True)
        demand.insert(0, "t", demand.index)
        urbs_model["Demand"] = demand
        del demand

    # Read intermittent supply time series
    if os.path.exists(paths["potential_ren"]):
        suplm = pd.read_csv(paths["potential_ren"], sep=";", decimal=",", index_col=0)
        suplm.index = range(1, 8761)
        suplm.insert(0, "t", suplm.index)
        urbs_model["Suplm"] = suplm
        del suplm


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

    # Read intermittent supply time series
    if os.path.exists(paths["potential_ren"]):
        # Format to evrys input format
        raw_data = pd.read_csv(paths["potential_ren"], sep=";", decimal=",", index_col=0)

        # Prepare data for pivot
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

# function to


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
