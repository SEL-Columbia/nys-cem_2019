import os
import argparse
import yaml
import numpy as np
import pandas as pd
import glob

def annualization_rate(i, years):
    return (i*(1+i)**years)/((1+i)**years-1)

def get_args():
    # Store all parameters for easy retrieval
    parser = argparse.ArgumentParser(
        description = 'nys-cem')
    parser.add_argument('--params_filename',
                        type=str,
                        default='params.yaml',
                        help = 'Loads model parameters')
    args = parser.parse_args()
    config = yaml.load(open(args.params_filename), Loader=yaml.FullLoader)
    for k,v in config.items():
        args.__dict__[k] = v

    return args

def load_timeseries(args):
    T = args.num_years*8760 + ((args.num_years+2)//4)*24 ## Account for leap years starting in 2008

    # Load all potential generation and actual hydro generation time-series
    onshore_pot_hourly        = np.array(pd.read_csv(os.path.join(args.data_dir, 'onshore_power_hourly_norm.csv'),
                                                     index_col=0))[0:T]
    offshore_pot_hourly       = np.array(pd.read_csv(os.path.join(args.data_dir, 'offshore_power_hourly_norm.csv'),
                                                     index_col=0))[0:T]
    solar_pot_hourly          = np.array(pd.read_csv(os.path.join(args.data_dir, 'solar_power_hourly_norm.csv'),
                                                     index_col=0))[0:T]
    flex_hydro_daily_mwh      = np.array(pd.read_csv(os.path.join(args.data_dir, 'flex_hydro_daily_mwh.csv'),
                                                     index_col=0))[0:T]
    fixed_hydro_hourly_mw     = np.array(pd.read_csv(os.path.join(args.data_dir, 'fixed_hydro_hourly_mw.csv'),
                                                     index_col=0))[0:T]

    # Load baseline and full heating demand time series
    baseline_demand_hourly_mw = np.array(pd.read_csv(os.path.join(args.data_dir, 'baseline_demand_hourly_mw.csv'),
                                                     index_col=0))[0:T]
    heating_hourly            = np.array(pd.read_csv(os.path.join(args.data_dir, 'elec_heating_hourly_mw.csv'),
                                                     index_col=0))[0:T]

    # print(np.mean(offshore_pot_hourly, axis=0))
    # print(np.mean(onshore_pot_hourly, axis=0))
    # print(np.mean(solar_pot_hourly, axis = 0))



    return baseline_demand_hourly_mw, heating_hourly, onshore_pot_hourly, offshore_pot_hourly, \
           solar_pot_hourly, fixed_hydro_hourly_mw, flex_hydro_daily_mwh



def get_raw_columns():

    # Define columns for raw results export
    columns = ['lct', 'nuclear_binary', 'h2_binary', 'hq-ch_cap',
               'add_heating_load', 'add_ev_load', 'total_onshore', 'total_offshore', 'total_solar',
               'total_new_gt_cap','total_battery_cap', 'total_battery_power', 'total_h2_cap', 'total_h2_power',
               'total_new_trans','total_hq_import', 'onshore_1', 'onshore_2', 'offshore_3', 'offshore_4',
               'solar_1', 'solar_2', 'solar_3', 'solar_4', 'new_gt_cap_1', 'new_gt_cap_2', 'new_gt_cap_3',
               'new_gt_cap_4', 'battery_cap_1', 'battery_cap_2', 'battery_cap_3', 'battery_cap_4', 'battery_power_1',
               'battery_power_2', 'battery_power_3', 'battery_power_4', 'battery_discharge_1', 'battery_discharge_2',
               'battery_discharge_3','battery_discharge_4', 'h2_cap_1', 'h2_cap_2', 'h2_cap_3', 'h2_cap_4',
               'h2_power_1', 'h2_power_2','h2_power_3', 'h2_power_4', 'h2_discharge_1', 'h2_discharge_2',
               'h2_discharge_3', 'h2_discharge_4', 'hq_import_1', 'hq_import_2', 'hq_import_3', 'hq_import_4',
               'total_trans_12', 'total_trans_23', 'total_trans_34', 'total_trans_21', 'total_trans_32',
               'total_trans_43', 'ghg_reduction']


    ts_columns = ['uncurtail_wind_gen_1', 'uncurtail_wind_gen_2', 'uncurtail_wind_gen_3', 'uncurtail_wind_gen_4',
                  'uncurtail_solar_gen_1', 'uncurtail_solar_gen_2', 'uncurtail_solar_gen_3', 'uncurtail_solar_gen_4',
                  'base_demand_1', 'base_demand_2', 'base_demand_3', 'base_demand_4',
                  'heating_demand_1', 'heating_demand_2', 'heating_demand_3', 'heating_demand_4',
                  'ev_charging_1', 'ev_charging_2', 'ev_charging_3', 'ev_charging_4',
                  'existing_gt_gen_1', 'existing_gt_gen_2', 'existing_gt_gen_3', 'existing_gt_gen_4',
                  'new_gt_gen_1', 'new_gt_gen_2', 'new_gt_gen_3', 'new_gt_gen_4',
                  'battery_level_1', 'battery_level_2', 'battery_level_3', 'battery_level_4',
                  'h2_level_1','h2_level_2','h2_level_3','h2_level_4',
                  'hq_import_1', 'hq_import_2', 'hq_import_3', 'hq_import_4',
                  'flex_hydro_1','flex_hydro_2','flex_hydro_3','flex_hydro_4',
                  'trans_12', 'trans_23', 'trans_34', '', 'trans_21', 'trans_32', 'trans_43', '',
                  'curtailed_gen_1', 'curtailed_gen_2', 'curtailed_gen_3', 'curtailed_gen_4']

    return  columns, ts_columns



def get_processed_columns():

    # Define columns for processed results export

    columns = ['RGT/LCT', 'RGT Binary', 'Nuclear Binary', 'H2 Binary', 'HQ-CH Addl. Cap.', 'Heating Load [MW]',
               'EV Load [MW]',
               'EV Fixed Charging', 'Charging Hours', 'Onshore [MW]', 'Offshore [MW]', 'Solar [MW]', 'New GT [MW]',
               'Battery Energy [MWh]', 'Battery Power [MW]', 'H2 Energy [MWh]', 'H2 Power [MW]', 'New Trans. [MW]',
               'New Trans. [GW-Mi]', 'Avg. Existing HQ Imports [MW]', 'Avg. New HQ Imports [MW]', 'Peak Demand [MW]',
               'Uncurtailed Avg. Wind and Solar Gen. [MW]', 'Uncurtailed Renewable Gen. [MW]',
               'Wind and Solar Curtailment', 'Avg. Battery Discharge [MW]', 'Avg. H2 Discharge [MW]',
               'Exist. GT Cap', 'Avg. Exist. GT CF [MW]',  'Avg. New GT CF',
               'Wind Gen. Cost [$/MWh]', 'Solar Gen. Cost [$/MWh]', 'Battery Cost [$/MWh]', 'New GT Cost [$/MWh]',
               'New Trans. Cost [$/MWh]',
               'Gas Fuel Cost [$/MWh]', 'Gas Ramping Cost [$/MWh]', 'Exist. Trans. + Cap. Cost  [$/MWh]',
               'Exist. Hydro Gen Cost [$/MWh]', 'Import Cost [$/MWh]', 'Nuc. Gen. Cost [$/MWh]', 'Total LCOE [$/MWh]',
               'GHG Reduction']

    return  columns

def get_tx_tuples(args):
    cap_ann = annualization_rate(args.i_rate, args.annualize_years_cap)

    tx_matrix_limits = pd.read_excel(os.path.join(args.data_dir, 'transmission_matrix_limits.xlsx'),
                                     header=0, index_col=0)
    tx_matrix_cap_costs = pd.read_excel(os.path.join(args.data_dir, 'transmission_matrix_capacity_costs.xlsx'),
                                    header=0, index_col=0)
    tx_matrix_om_costs = pd.read_excel(os.path.join(args.data_dir, 'transmission_matrix_o&m_costs.xlsx'),
                                    header=0, index_col=0)

    tx_tuple_list = []

    for i in range(len(tx_matrix_limits)):
        for j in range(len(tx_matrix_limits.columns)):
            if tx_matrix_limits.iloc[i, j] > 0:
                tx_tuple_list.append(((i + 1, j + 1), tx_matrix_limits.iloc[i, j],
                                      args.num_years * cap_ann * tx_matrix_cap_costs.iloc[i, j] +
                                      tx_matrix_om_costs.iloc[i, j]))

    return tx_tuple_list


def load_gt_ramping_costs(args, results, results_ts):
    ramping_cost_mwh = args.gt_startup_cost_mw/2

    net_load_ramping_total_cost = np.zeros(results.shape[0])
    net_load_fuel_cost          = np.zeros(results.shape[0])
    net_load_om_cost            = np.zeros(results.shape[0])

    for i in range(results.shape[0]):
        existing_gt_gen = results_ts[i, :, args.num_regions * 5: args.num_regions * 6]
        new_gt_gen      = results_ts[i, :, args.num_regions * 6: args.num_regions * 7]

        net_load = existing_gt_gen + new_gt_gen


        for l in range(net_load.shape[0] - 1):
            for m in range(4):
                net_load_ramping_total_cost[i] += abs(net_load[l + 1, m] - net_load[l, m]) * ramping_cost_mwh
                net_load_fuel_cost[i] += args.natgas_cost_mmbtu[m] * args.mmbtu_per_mwh * \
                                                    (existing_gt_gen[l, m]/args.existing_gt_eff +
                                                     new_gt_gen[l, m] / args.new_gt_eff)
                net_load_om_cost[i]   +=  new_gt_gen[l, m] * args.new_gt_var_om_cost_mwh

    net_load_total_cost = net_load_fuel_cost + net_load_om_cost


    return net_load_total_cost, net_load_ramping_total_cost

def calculate_ghg_contributions():
    # All emissions are given in therms of MMtCO2e

    baseline_1990_emissions = 206.15
    existing_industrial_emissions = 10.8
    non_diesel_non_gas_transport_emissions = 13.51

    nat_gas_emissions_rate = 53.1148 # kg CO2e/MMBTU

    total_heating_emissions = 56.5 # MMtCO2e
    total_transport_emissions = 61.17 # MMtCO2e

    return nat_gas_emissions_rate, total_heating_emissions, total_transport_emissions, \
            baseline_1990_emissions, existing_industrial_emissions, non_diesel_non_gas_transport_emissions



def raw_results_retrieval(args, model_config, m, tx_tuple_list):
    T = args.num_years * 8760 + ((args.num_years + 2) // 4) * 24  ## Account for leap years starting in 2008

    # Model run parameters


    nuclear_boolean = args.nuclear_boolean
    h2_boolean = args.h2_boolean

    baseline_demand_hourly_mw, heating_demand, onshore_pot_hourly, offshore_pot_hourly, \
    solar_pot_hourly, fixed_hydro_hourly_mw, flex_hydro_daily_mwh = load_timeseries(args)

    heating_elec_ratio = m.getVarByName('total_heating_ratio').X
    ev_elec_ratio      = m.getVarByName('total_ev_ratio').X

    total_heating_cap = heating_elec_ratio * np.sum(np.mean(heating_demand[0:T], axis = 0))
    total_ev_cap      = ev_elec_ratio * args.ev_max_cap

    gen_batt_capacity_results = np.zeros((8, args.num_regions))
    for i in range(args.num_regions):
        gen_batt_capacity_results[0,i] = m.getVarByName('onshore_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[1,i] = m.getVarByName('offshore_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[2,i] = m.getVarByName('solar_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[3,i] = m.getVarByName('new_gt_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[4,i] = m.getVarByName('batt_energy_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[5,i] = m.getVarByName('batt_power_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[6,i] = m.getVarByName('h2_energy_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[7,i] = m.getVarByName('h2_power_cap_region_{}'.format(i + 1)).X


    timeseries_results = np.zeros((15, T, args.num_regions))
    for i in range(args.num_regions):
        for j in range(T):
            # timeseries_results[0, j, i]  = m.getVarByName('onshore_wind_util_region_{}[{}]'.format(i + 1, j)).X
            # timeseries_results[1, j, i]  = m.getVarByName('offshore_wind_util_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[2, j, i]  = -m.getConstrByName('energy_balance_constraint_region_{}[{}]'
                                                             .format(i + 1, j)).Slack
            timeseries_results[3, j, i]  = m.getVarByName('flex_hydro_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[4, j, i]  = m.getVarByName('batt_charge_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[5, j, i]  = m.getVarByName('batt_discharge_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[6, j, i]  = m.getVarByName('h2_charge_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[7, j, i]  = m.getVarByName('h2_discharge_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[8, j, i]  = m.getVarByName('batt_level_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[9, j, i]  = m.getVarByName('h2_level_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[10, j, i] = m.getVarByName('hq_import_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[11, j, i] = m.getVarByName('existing_gt_gen_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[12, j, i] = m.getVarByName('new_gt_gen_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[13, j, i] = m.getVarByName('existing_gt_abs_region_{}[{}]'.format(i + 1, j)).X + \
                                           m.getVarByName('new_gt_abs_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[14, j, i] = m.getVarByName('ev_charging_region_{}[{}]'.format(i + 1, j)).X


    # Transmission result processing
    tx_cap_base_string = 'new_export_limits_{}_{}'
    tx_ts_base_string  = 'net_exports_ts_{}_to_{}[{}]'

    tx_new_cap_results   = np.zeros((len(tx_tuple_list)))
    tx_total_cap_results = np.zeros((len(tx_tuple_list)))
    tx_ts_results_WE        = np.zeros((T, int(len(tx_tuple_list)/2)))
    tx_ts_results_EW        = np.zeros((T, int(len(tx_tuple_list)/2)))
    export_WE_ts_count     = 0
    export_EW_ts_count     = 0

    for i, txt in enumerate(tx_tuple_list):

        tx_new_cap_results[i]   = m.getVarByName(tx_cap_base_string.format(txt[0][0], txt[0][1])).X
        tx_total_cap_results[i] = m.getVarByName(tx_cap_base_string.format(txt[0][0], txt[0][1])).X + txt[1]

        if txt[0][1] > txt[0][0]:
            for j in range(T):
                tx_ts_results_WE[j, export_WE_ts_count] = m.getVarByName(tx_ts_base_string.format(txt[0][0], txt[0][1], j)).X
            export_WE_ts_count += 1

        elif txt[0][0] > txt[0][1] :
            for j in range(T):
                tx_ts_results_EW[j, export_EW_ts_count] = m.getVarByName(tx_ts_base_string.format(txt[0][0], txt[0][1], j)).X
            export_EW_ts_count += 1

    ## Export raw results
    results = np.zeros(63)
    results_ts = np.zeros((T, args.num_regions * 14))

    # Time series results
    results_ts[:, 0:2] = onshore_pot_hourly[0:T, 0:2] * gen_batt_capacity_results[0, 0:2]  # Uncurtailed onshore gen
    results_ts[:, 2:4] = offshore_pot_hourly[0:T, 2:4] * gen_batt_capacity_results[1, 2:4]  # Uncurtailed offshore gen
    results_ts[:, args.num_regions * 1: args.num_regions * 2] = \
        solar_pot_hourly[0:T] * gen_batt_capacity_results[2, :]  # Uncurtailed solar gen
    results_ts[:, args.num_regions * 2: args.num_regions * 3] = baseline_demand_hourly_mw[0:T]  # baseline demand
    results_ts[:, args.num_regions * 3: args.num_regions * 4] = heating_elec_ratio * heating_demand[0:T]  # heating
    results_ts[:, args.num_regions * 4: args.num_regions * 5] = timeseries_results[14]  # ev charging
    results_ts[:, args.num_regions * 5: args.num_regions * 6] = timeseries_results[11]  # existing GT generation
    results_ts[:, args.num_regions * 6: args.num_regions * 7] = timeseries_results[12]  # new GT generation
    results_ts[:, args.num_regions * 7: args.num_regions * 8] = timeseries_results[8]  # battery level
    results_ts[:, args.num_regions * 8: args.num_regions * 9] = timeseries_results[9]  # h2 level
    results_ts[:, args.num_regions * 9: args.num_regions * 10] = timeseries_results[10]  # hq import
    results_ts[:, args.num_regions * 10: args.num_regions * 11] = timeseries_results[3]  # flex hydro
    results_ts[:, args.num_regions * 11: args.num_regions * 11 + 3] = tx_ts_results_WE  # WE transmission flow
    results_ts[:, args.num_regions * 12: args.num_regions * 12 + 3] = tx_ts_results_EW  # EW transmission flow
    results_ts[:, args.num_regions * 13: args.num_regions * 14] = timeseries_results[2]  # Uncurtailed energy

    # Determine LCT
    if model_config == 0 or model_config == 1:
        lct = m.getVarByName('lowc_target').X
    else:
        lct = np.round(1 - (np.sum(timeseries_results[11]) / (np.sum(baseline_demand_hourly_mw[0:T]) +
                                                           (total_heating_cap + total_ev_cap) * T
                                                           - np.sum(timeseries_results[10]))), 4)


    results[0] = lct
    results[1] = int(nuclear_boolean)
    results[2] = int(h2_boolean)
    results[3] = int(args.hq_limit_mw[2])

    # Additional load parameters
    results[4] = total_heating_cap
    results[5] = total_ev_cap

    # Total new capacities + avg. hydro import
    results[6] = np.around(np.sum(gen_batt_capacity_results[0,:])) # total_onshore
    results[7] = np.around(np.sum(gen_batt_capacity_results[1,:])) # total_offshore
    results[8] = np.around(np.sum(gen_batt_capacity_results[2,:])) # total_solar
    results[9]  = np.around(np.sum(gen_batt_capacity_results[3,:] * args.reserve_req)) # total_new_gt_cap
    results[10] = np.around(np.sum(gen_batt_capacity_results[4,:])) # total_battery_cap
    results[11] = np.around(np.sum(gen_batt_capacity_results[5,:])) # total_battery_power
    results[12] = np.around(np.sum(gen_batt_capacity_results[6,:])) # total_h2_cap
    results[13] = np.around(np.sum(gen_batt_capacity_results[7,:])) # total_h2_power
    results[14] = np.around(np.sum(tx_new_cap_results)) # total_new_trans
    results[15] = np.around(np.mean(timeseries_results[10, :, 0]) + np.mean(timeseries_results[10, :, 1]) +
                            np.mean(timeseries_results[10, :, 2]) + np.mean(timeseries_results[10, :, 3]))
                    # total_hq_import

    # Wind
    results[16] = np.around(gen_batt_capacity_results[0, 0]) # onshore_1
    results[17] = np.around(gen_batt_capacity_results[0, 1]) # onshore_2
    results[18] = np.around(gen_batt_capacity_results[1, 2]) # offshore_3
    results[19] = np.around(gen_batt_capacity_results[1, 3]) # offshore_4

    # Solar
    results[20] = np.around(gen_batt_capacity_results[2, 0]) # solar_1
    results[21] = np.around(gen_batt_capacity_results[2, 1]) # solar_2
    results[22] = np.around(gen_batt_capacity_results[2, 2]) # solar_3
    results[23] = np.around(gen_batt_capacity_results[2, 3]) # solar_4

    # GT
    results[24] = np.around(gen_batt_capacity_results[3, 0] * args.reserve_req) # new_gt_cap_1
    results[25] = np.around(gen_batt_capacity_results[3, 1] * args.reserve_req) # new_gt_cap_2
    results[26] = np.around(gen_batt_capacity_results[3, 2] * args.reserve_req) # new_gt_cap_3
    results[27] = np.around(gen_batt_capacity_results[3, 3] * args.reserve_req) # new_gt_cap_4

    # Battery energy, power, average discharge
    results[28] = np.around(gen_batt_capacity_results[4, 0]) # battery_cap_1
    results[29] = np.around(gen_batt_capacity_results[4, 1]) # battery_cap_2
    results[30] = np.around(gen_batt_capacity_results[4, 2]) # battery_cap_3
    results[31] = np.around(gen_batt_capacity_results[4, 3]) # battery_cap_4
    results[32] = np.around(gen_batt_capacity_results[5, 0]) # battery_power_1
    results[33] = np.around(gen_batt_capacity_results[5, 1]) # battery_power_2
    results[34] = np.around(gen_batt_capacity_results[5, 2]) # battery_power_3
    results[35] = np.around(gen_batt_capacity_results[5, 3]) # battery_power_4
    results[36] = np.around(np.mean(timeseries_results[5, :, 0])) # battery_discharge_1
    results[37] = np.around(np.mean(timeseries_results[5, :, 1])) # battery_discharge_2
    results[38] = np.around(np.mean(timeseries_results[5, :, 2])) # battery_discharge_3
    results[39] = np.around(np.mean(timeseries_results[5, :, 3])) # battery_discharge_4

    # H2 energy, power, average discharge
    results[40] = np.around(gen_batt_capacity_results[6, 0]) # h2_cap_1
    results[41] = np.around(gen_batt_capacity_results[6, 1]) # h2_cap_2
    results[42] = np.around(gen_batt_capacity_results[6, 2]) # h2_cap_3
    results[43] = np.around(gen_batt_capacity_results[6, 3]) # h2_cap_4
    results[44] = np.around(gen_batt_capacity_results[7, 0]) # h2_power_1
    results[45] = np.around(gen_batt_capacity_results[7, 1]) # h2_power_2
    results[46] = np.around(gen_batt_capacity_results[7, 2]) # h2_power_3
    results[47] = np.around(gen_batt_capacity_results[7, 3]) # h2_power_4
    results[48] = np.around(np.mean(timeseries_results[7, :, 0])) # h2_discharge_1
    results[49] = np.around(np.mean(timeseries_results[7, :, 1])) # h2_discharge_2
    results[50] = np.around(np.mean(timeseries_results[7, :, 2])) # h2_discharge_3
    results[51] = np.around(np.mean(timeseries_results[7, :, 3])) # h2_discharge_4

    # Avg. Imports from HQ
    results[52] = np.around(np.mean(timeseries_results[10, :, 0])) # hq_import_1
    results[53] = np.around(np.mean(timeseries_results[10, :, 1])) # hq_import_2
    results[54] = np.around(np.mean(timeseries_results[10, :, 2])) # hq_import_3
    results[55] = np.around(np.mean(timeseries_results[10, :, 3])) # hq_import_4

    # Total transmission capacity: WE results presented first, EW results follow
    results[56] = np.around(tx_total_cap_results[0]) # new_trans_12
    results[57] = np.around(tx_total_cap_results[2]) # new_trans_23
    results[58] = np.around(tx_total_cap_results[4]) # new_trans_34
    results[59] = np.around(tx_total_cap_results[1]) # new_trans_21
    results[60] = np.around(tx_total_cap_results[3]) # new_trans_32
    results[61] = np.around(tx_total_cap_results[5]) # new_trans_43
    results[62] = m.getVarByName('ghg_emission_reduction').X

    return results, results_ts

def full_results_processing(args, results, results_ts):

    # Retrieve necessary model parameters
    export_columns = get_processed_columns()
    T = args.num_years * 8760 + ((args.num_years + 2) // 4) * 24  ## Account for leap years starting in 2008
    tx_tuple_list = get_tx_tuples(args)
    cap_ann = annualization_rate(args.i_rate, args.annualize_years_cap)
    cap_battery = annualization_rate(args.i_rate, args.annualize_years_storage)

    # Retrieve LCT values from results file
    lct = results[:, 0]

    # Potential generation time-series for curtailment calcs

    baseline_demand_hourly_mw, heating_hourly, onshore_pot_hourly, offshore_pot_hourly, \
    solar_pot_hourly, fixed_hydro_hourly_mw, flex_hydro_daily_mwh = load_timeseries(args)


    # Create arrays to store costs -- All costs are annual
    total_new_wind_cost                 = np.zeros(results.shape[0])
    total_new_solar_cost                = np.zeros(results.shape[0])
    total_new_battery_cost              = np.zeros(results.shape[0])
    total_new_gt_cost                   = np.zeros(results.shape[0])
    total_new_tx_cost                   = np.zeros(results.shape[0])
    total_gas_cost                      = np.zeros(results.shape[0])
    total_ramping_cost                  = np.zeros(results.shape[0])
    total_annual_import_cost            = np.zeros(results.shape[0])
    total_cost_per_mwh                  = np.zeros(results.shape[0])
    gen_uncurtailed_windsolar_energy    = np.zeros(results.shape[0])
    total_renewable_gen                 = np.zeros(results.shape[0])
    peak_demand                         = np.zeros(results.shape[0])
    total_curtailed_energy              = np.zeros(results.shape[0])
    total_wind_solar_curtailment        = np.zeros(results.shape[0])
    average_existing_gt_cf              = np.zeros(results.shape[0])
    average_new_gt_cf                   = np.zeros(results.shape[0])
    total_ancillary_cost                = np.zeros(results.shape[0])
    total_wind_collection_cost          = np.zeros(results.shape[0])

    data_for_export = np.zeros((results.shape[0], len(export_columns)))

    # Find additional load scenarios
    additional_load_domain = np.zeros(results.shape[0])
    for i in range(results.shape[0]):
        additional_load_domain[i] = results[i, 4] + results[i, 5]

    # Calculate demand for all scenario runs
    avg_baseline_demand = np.sum(np.mean(baseline_demand_hourly_mw[0:T], axis=0))
    avg_total_demand = [avg_baseline_demand + i for i in additional_load_domain]

    # Find uncurtailed capacity factors
    wind_uncurtailed_cf  = np.array((np.mean(onshore_pot_hourly[0:T, 0]),  np.mean(onshore_pot_hourly[0:T, 1]),
                                     np.mean(offshore_pot_hourly[0:T, 2]), np.mean(offshore_pot_hourly[0:T, 3])))

    solar_uncurtailed_cf = np.mean(solar_pot_hourly, axis = 0)

    # Hydro, nuclear, and netload costs
    total_annual_hydro_cost = np.sum([args.hydro_gen_mw[i] * args.instate_hydro_cost_mwh[i] for i in range(4)]) * 8760
    total_annual_nuclear_cost = int(args.nuclear_boolean) * np.sum([args.nuc_gen_mw[i] * args.instate_nuc_cost_mwh[i]
                                                                   for i in range(4)]) * 8760

    net_load_cost, net_load_ramping_cost = load_gt_ramping_costs(args, results, results_ts)

    # Calculate existing capacity and transmission cost
    total_cap_market_cost = np.sum([args.cap_market_cost_mw_yr[k] * (args.existing_gt_cap_mw[k] +
                                                                int(args.nuclear_boolean) * args.nuc_gen_mw[k] +
                                                               args.hydro_gen_mw[k]) for k in range(4)])
    total_existing_trans_cost = np.sum([float(args.existing_load_mwh[k]) * args.existing_trans_cost_mwh[k]
                                        for k in range(3)])
    total_annual_supp_cost = total_existing_trans_cost + total_cap_market_cost

    if args.ev_charging_method == 'flexible':
        fixed_charging_binary = 0
    else:
        fixed_charging_binary = 1


    # Calculate costs
    for i in range(results.shape[0]):

        total_new_wind_cost[i] = (
            results[i, 6]  * (cap_ann * float(args.onshore_cost_mw)   + float(args.onshore_fixed_om_cost_mwyr)) +
            results[i, 7]  * (cap_ann * float(args.offshore_cost_mw)  + float(args.offshore_fixed_om_cost_mwyr)))


        total_new_solar_cost[i] = (
            results[i, 20] * (cap_ann * float(args.solar_cost_mw[0])  + float(args.solar_fixed_om_cost_mwyr)) +
            results[i, 21] * (cap_ann * float(args.solar_cost_mw[1])  + float(args.solar_fixed_om_cost_mwyr)) +
            results[i, 22] * (cap_ann * float(args.solar_cost_mw[2])  + float(args.solar_fixed_om_cost_mwyr)) +
            results[i, 23] * (cap_ann * float(args.solar_cost_mw[3])  + float(args.solar_fixed_om_cost_mwyr)))


        total_new_battery_cost[i] = (
            results[i, 10] *  cap_battery * float(args.battery_cost_mwh) +
            results[i, 11] *  cap_battery * float(args.battery_cost_mw) +
            results[i, 40] *  cap_battery * float(args.h2_cost_mwh[0]) +
            results[i, 41] *  cap_battery * float(args.h2_cost_mwh[1]) +
            results[i, 42] *  cap_battery * float(args.h2_cost_mwh[2]) +
            results[i, 43] *  cap_battery * float(args.h2_cost_mwh[3]) +
            results[i, 44] * (cap_battery * float(args.h2_cost_mw[0]) + float(args.h2_fixed_om_cost_mwyr)) +
            results[i, 45] * (cap_battery * float(args.h2_cost_mw[1]) + float(args.h2_fixed_om_cost_mwyr)) +
            results[i, 46] * (cap_battery * float(args.h2_cost_mw[2]) + float(args.h2_fixed_om_cost_mwyr)) +
            results[i, 47] * (cap_battery * float(args.h2_cost_mw[3]) + float(args.h2_fixed_om_cost_mwyr)))

        total_new_gt_cost[i] = args.reserve_req * (
            results[i, 24] * (cap_ann * float(args.new_gt_cost_mw[0]) + float(args.new_gt_fixed_om_cost_mwyr)) +
            results[i, 25] * (cap_ann * float(args.new_gt_cost_mw[1]) + float(args.new_gt_fixed_om_cost_mwyr)) +
            results[i, 26] * (cap_ann * float(args.new_gt_cost_mw[2]) + float(args.new_gt_fixed_om_cost_mwyr)) +
            results[i, 27] * (cap_ann * float(args.new_gt_cost_mw[3]) + float(args.new_gt_fixed_om_cost_mwyr)))

        total_new_tx_cost[i] = ((results[i, 56] - tx_tuple_list[0][1]) * tx_tuple_list[0][2] +
                                (results[i, 57] - tx_tuple_list[2][1]) * tx_tuple_list[2][2] +
                                (results[i, 58] - tx_tuple_list[4][1]) * tx_tuple_list[4][2] +
                                (results[i, 59] - tx_tuple_list[1][1]) * tx_tuple_list[1][2] +
                                (results[i, 60] - tx_tuple_list[3][1]) * tx_tuple_list[3][2] +
                                (results[i, 61] - tx_tuple_list[5][1]) * tx_tuple_list[5][2])
                                # Already annualized in tx_tuple_list

        total_gas_cost[i]         = net_load_cost[i] / args.num_years
        total_ramping_cost[i]     = net_load_ramping_cost[i] / args.num_years

        total_annual_import_cost[i]  = (results[i, 52] * args.hq_cost_mwh[0] +
                                        results[i, 53] * args.hq_cost_mwh[1] +
                                        results[i, 54] * args.hq_cost_mwh[2] +
                                        results[i, 55] * args.hq_cost_mwh[3]) * 8760

        total_imports = results[i, 15]

        # Find Peak Demand
        total_demand_ts = results_ts[i, :, args.num_regions * 2: args.num_regions * 3] + \
                          results_ts[i, :, args.num_regions * 3: args.num_regions * 4] + \
                          results_ts[i, :, args.num_regions * 4: args.num_regions * 5] # baseline + heating + evs

        peak_demand[i] = np.max(np.sum(total_demand_ts, axis = 1))

        total_curtailed_energy[i] = np.mean(np.sum(results_ts[i, :, args.num_regions * 13: args.num_regions * 14],
                                                    axis = 1))

        gen_uncurtailed_windsolar_energy[i] = np.round(results[i, 16] * wind_uncurtailed_cf[0] +
                                          results[i, 17] * wind_uncurtailed_cf[1] +
                                          results[i, 18] * wind_uncurtailed_cf[2] +
                                          results[i, 19] * wind_uncurtailed_cf[3] +
                                          results[i, 20] * solar_uncurtailed_cf[0] +
                                          results[i, 21] * solar_uncurtailed_cf[1] +
                                          results[i, 22] * solar_uncurtailed_cf[2] +
                                          results[i, 23] * solar_uncurtailed_cf[3])


        total_renewable_gen[i]          = gen_uncurtailed_windsolar_energy[i] + np.sum(args.hydro_gen_mw)
        total_wind_solar_curtailment[i] = total_curtailed_energy[i] / gen_uncurtailed_windsolar_energy[i]

        average_existing_gt_cf[i] = np.mean(np.sum(results_ts[i, :, args.num_regions * 5: args.num_regions * 6],
                                                   axis = 1)) / np.sum(args.existing_gt_cap_mw)

        new_gt_cap = results[i, 9]
        if new_gt_cap > 0:
            average_new_gt_cf[i] = np.mean(np.sum(results_ts[i, :, args.num_regions * 6: args.num_regions * 7],
                                                   axis=1)) / new_gt_cap
        else:
            average_new_gt_cf[i] = 0

        # Calculate ancillary service costs
        if args.ancillary_boolean:
            ancillary_reserve_req = (0.05 * (results[i, 6]  + results[i, 7]) +
                                     0.01 * results[i, 8] +
                                     0.03 * avg_total_demand[i])
            ancillary_reserve_cost = 14.46557349  # /MW-h

            total_ancillary_cost[i] = ancillary_reserve_cost * ancillary_reserve_req * 8760

        # Calculate wind collection costs
        if args.wind_collection_boolean:
            wind_collection_distances = [112.45, 20.84, 56.22, 22.554]  # miles
            collection_costs = [2400, 4800, 12000, 12000]  # $/MW-mi
            total_wind_collection_cost[i] = cap_ann * \
                                         (wind_collection_distances[0] * collection_costs[0] * results[i, 16] +
                                          wind_collection_distances[1] * collection_costs[1] * results[i, 17] +
                                          wind_collection_distances[2] * collection_costs[2] * results[i, 18] +
                                          wind_collection_distances[3] * collection_costs[3] * results[i, 19])


        total_cost_per_mwh[i] = (total_new_wind_cost[i] + total_new_solar_cost[i]+
                                 total_new_battery_cost[i] + total_new_gt_cost[i] +
                                 total_new_tx_cost[i] + total_gas_cost[i] + total_ramping_cost[i] +
                                 total_annual_supp_cost + total_annual_hydro_cost + total_annual_import_cost[i] +
                                 total_annual_nuclear_cost + total_ancillary_cost[i] +
                                 total_wind_collection_cost[i]) /  (avg_total_demand[i] * 8760)

    ## Populate data_for_export
    data_for_export[:, 0] = np.multiply(lct, 100) # RGT/LCT
    data_for_export[:, 1] = int(args.rgt_boolean) # RGT Binary
    data_for_export[:, 2] = int(args.nuclear_boolean) # Nuclear Binary
    data_for_export[:, 3] = int(args.h2_boolean) # H2 Binary
    data_for_export[:, 4] = int(args.hq_limit_mw[2]) # HQ-CH Binary
    data_for_export[:, 5] = results[:, 4] # Heating Load
    data_for_export[:, 6] = results[:, 5] # EV Load
    data_for_export[:, 7] = fixed_charging_binary # EV fixed charging
    data_for_export[:, 8] = args.ev_charging_hours # EV charging hours

    data_for_export[:, 9]  = results[:, 6] # Onshore [MW]
    data_for_export[:, 10] = results[:, 7] # Offshore [MW]
    data_for_export[:, 11] = results[:, 8] # Solar [MW]
    data_for_export[:, 12] = results[:, 9] # New GT [MW]
    data_for_export[:, 13] = results[:, 10] # Battery Energy [MWh]
    data_for_export[:, 14] = results[:, 11] # Battery Power [MW]
    data_for_export[:, 15] = results[:, 12] # H2 Energy [MWh]
    data_for_export[:, 16] = results[:, 13] # H2 Power [MW]
    data_for_export[:, 17] = results[:, 14] # New Trans. [MW]
    data_for_export[:, 18] = \
        np.round(((results[:, 56] + results[:, 59] - tx_tuple_list[0][1] - tx_tuple_list[1][1]) * 300 / 1000 +
                  (results[:, 57] + results[:, 60] - tx_tuple_list[2][1] - tx_tuple_list[3][1]) * 150 / 1000 +
                  (results[:, 58] + results[:, 61] - tx_tuple_list[4][1] - tx_tuple_list[5][1]) * 60 / 1000))
                # New Trans. [GW-Mi]
    data_for_export[:, 19] = results[:, 52] # Avg. Existing HQ Imports [MW]
    data_for_export[:, 20] = results[:, 54] # Avg. New HQ Imports [MW]

    data_for_export[:, 21] = peak_demand # Peak load
    data_for_export[:, 22] = gen_uncurtailed_windsolar_energy  # Average uncurtailed wind + solar generation
    data_for_export[:, 23] = total_renewable_gen  # Average uncurtailed renewable gen
    data_for_export[:, 24] = total_wind_solar_curtailment  # wind solar generation curtailment
    data_for_export[:, 25] = np.sum(results[:, 36:40]) # Total average battery discharge
    data_for_export[:, 26] = np.sum(results[:, 48:52]) # Total average H2 discharge
    data_for_export[:, 27] = np.sum(args.existing_gt_cap_mw) # Existing GT cap
    data_for_export[:, 28] = average_existing_gt_cf  # Existing GT CF
    data_for_export[:, 29] = average_new_gt_cf  # New GT CF

    data_for_export[:, 30] = [total_new_wind_cost[i] / (avg_total_demand[i] * 8760)
                              for i in range(results.shape[0])] # Renewable generation capacity cost
    data_for_export[:, 31] = [total_new_solar_cost[i] / (avg_total_demand[i] * 8760)
                              for i in range(results.shape[0])]  # Renewable generation capacity cost
    data_for_export[:, 32] = [total_new_battery_cost[i] / (avg_total_demand[i] * 8760)
                              for i in range(results.shape[0])] # Battery capacity cost
    data_for_export[:, 33] = [total_new_gt_cost[i] / (avg_total_demand[i] * 8760)
                              for i in range(results.shape[0])] # New gas turbine capacity cost
    data_for_export[:, 34] = [total_new_tx_cost[i] / (avg_total_demand[i] * 8760)
                              for i in range(results.shape[0])] # New transmission cost
    data_for_export[:, 35] = [total_gas_cost[i] / (avg_total_demand[i] * 8760)
                              for i in range(results.shape[0])]  # Natural gas cost
    data_for_export[:, 36] = [total_ramping_cost[i] / (avg_total_demand[i] * 8760)
                              for i in range(results.shape[0])] # Ramping cost
    data_for_export[:, 37] = [total_annual_supp_cost / (avg_total_demand[i] * 8760)
                              for i in range(results.shape[0])] # Cost of existing capacity and transmission
    data_for_export[:, 38] = [total_annual_hydro_cost / (avg_total_demand[i] * 8760)
                              for i in range(results.shape[0])]  # Cost of existing hydro
    data_for_export[:, 39] = [total_annual_import_cost[i] / (avg_total_demand[i] * 8760)
                              for i in range(results.shape[0])] # Cost of imported hydro
    data_for_export[:, 40] = [total_annual_nuclear_cost / (avg_total_demand[i] * 8760)
                              for i in range(results.shape[0])] # Cost of nuclear

    data_for_export[:, 41] = total_cost_per_mwh # Total LCOE [$/MWh]
    data_for_export[:, 42] = results[:, 62] # GHG reductions


    results_df = pd.DataFrame(data_for_export, columns=export_columns)
    return results_df