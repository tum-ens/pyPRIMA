from lib.initialization import initialization
from lib.generate_intermediate_files import *
from lib.correction_functions import *

if __name__ == '__main__':
    paths, param = initialization()
    
    ## Clean raw data
    # clean_GridKit_Europe(paths, param)
    # clean_sector_shares_Eurostat(paths, param)
    # clean_load_data_ENTSOE(paths, param)
    
    # distribute_renewable_capacities(paths, param)
    # clean_processes_and_storage_data(paths, param)
    # clean_processes_and_storage_data_FRESNA(paths, param)
    
    ## Generate intermediate files
    # generate_sites_from_shapefile(paths, param)
    # generate_load_timeseries(paths, param)
    generate_transmission(paths, param)
    
    # generate_intermittent_supply_timeseries(paths, param)
    # generate_commodity(paths, param)
    # generate_processes_and_storage_california(paths, param)
    # generate_processes(paths, param)
    # generate_storage(paths, param)
    
    ## Generate model files
    # generate_urbs_model(paths, param)
    # generate_evrys_model(paths, param)