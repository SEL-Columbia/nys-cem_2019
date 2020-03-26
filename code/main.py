import numpy as np
from model import create_model
from utils import *
import shutil

if __name__ == '__main__':
    args    = get_args()
    raw_export_columns, ts_export_columns = get_raw_columns()
    processed_export_columns = get_processed_columns()


    # Define model config and set of heating, EV loads, and/or GHG reduction target appropriately
    # model_config = 1
    #
    # # 0: LCT + Elec. specified, GHG returned
    # if model_config == 0:
    #
    #     lowc_targets   = [0.7]
    #     perc_elec_load = [0.4]
    #
    #
    #     ghg_targets    = [np.nan]*len(perc_elec_load) # indeterminate
    #
    # # 1: LCT + GHG specified, Elec. returned
    # elif model_config == 1:
    #     lowc_targets = [0.7]
    #     ghg_targets  = [0.4]
    #     perc_elec_load = [np.nan]*len(lowc_targets) # indeterminate
    #
    # # 2: Elec. + GHG specified, LCT returned.
    # else: # model_config  == 2:
    #     ghg_targets    = [0.85]
    #     perc_elec_load = [1]
    #     lowc_targets   = [np.nan]*len(ghg_targets) # indeterminate

    model_configs  = [1, 0, 1, 1, 1, 2]
    lowc_targets   = [0.7, 0.995, 0.995, 0.7, 0.995, np.nan]
    perc_elec_load = [np.nan, 0.37645, np.nan, np.nan, np.nan, 1]
    ghg_targets    = [0.4, np.nan, 0.625, 0.625, 0.85, 0.85]
    #
    # model_configs = [ 2]
    # lowc_targets = [np.nan]
    # perc_elec_load = [ 1]
    # ghg_targets = [ 0.7]



    # Establish lists to store results
    results    = []
    results_ts = []

    # Set up ts_results_dir
    ts_results_dir = os.path.join(args.results_dir, 'ts_results_dir')

    if args.solve_model:
        for i in range(len(perc_elec_load)):
            # Initialize scenario parameters
            lct  = lowc_targets[i]
            elec = perc_elec_load[i]
            ghg  = ghg_targets[i]
            model_config = model_configs[i]

            # Create the model
            m = create_model(args, model_config, lct, elec, ghg)

            # Set model solver parameters
            m.setParam("FeasibilityTol", args.feasibility_tol)
            m.setParam("Method", 2)
            m.setParam("BarConvTol", 0)
            m.setParam("BarOrder", 0)
            m.setParam("Crossover", 0)

            # Solve the model
            m.optimize()

            # Retrieve the model solution
            allvars = m.getVars()

            # Process the model solution
            tx_tuple_list = get_tx_tuples(args)
            single_scen_results, single_scen_results_ts = raw_results_retrieval(args, model_config, m, tx_tuple_list)

            # Append single set of results to full results lists
            results.append(single_scen_results)
            results_ts.append(single_scen_results_ts)

        ## Save raw results
        df_results_raw = pd.DataFrame(np.array(results), columns=raw_export_columns)
        df_results_raw.to_excel(os.path.join(args.results_dir, 'raw_results_export.xlsx'))

        if os.path.exists(ts_results_dir):
            shutil.rmtree(ts_results_dir)
        os.mkdir(ts_results_dir)


        # ts_writer = pd.ExcelWriter(os.path.join(args.results_dir, 'ts_results_export.xlsx'), engine= 'xlsxwriter')
        for i in range(len(results_ts)):
            df_results_ts = pd.DataFrame(np.array(results_ts[i]), columns=ts_export_columns)
            df_results_ts.to_csv(os.path.join(ts_results_dir,
                                'ts_results_export_sheet_{}_lct_{}_elec_{}_ghg_{}.csv').format(i, lowc_targets[i],
                                 perc_elec_load[i], ghg_targets[i]))

        df_results_processed = full_results_processing(args, np.array(results), np.array(results_ts))
        df_results_processed.to_excel(os.path.join(args.results_dir, 'processed_results_export.xlsx'))

    else:
        ## Save processed results

        file_dir = '/Users/terenceconlon/Documents/Columbia - Fall 2019/NYS/model_results/clcpa_results/supplementary_text_results/wind_lowcost'
        ts_results_dir = os.path.join(file_dir, 'ts_results_dir')
        ts_results_list = sorted(glob.glob(ts_results_dir + '/*.csv'))  # Return sorted list of the csvs
        results = np.array(pd.read_excel(os.path.join(file_dir,
                                            'raw_results_export.xlsx'), index_col=0, header=0))
        df_results_processed = pd.DataFrame(index= range(len(results)), columns=processed_export_columns)

        for i in range(len(results)):
            ts_results_sheet = [j for j in ts_results_list if 'sheet_{}_'.format(i) in j][0]
            print(ts_results_sheet)

            ts_results = np.array(pd.read_csv(ts_results_sheet,
                                            index_col=0, header = 0))

            df_results_processed.iloc[i] = full_results_processing(args, np.expand_dims(results[i], 0),
                                                       np.expand_dims(ts_results, 0)).iloc[0]


        df_results_processed.to_excel(os.path.join(file_dir, 'processed_results_export.xlsx'))
