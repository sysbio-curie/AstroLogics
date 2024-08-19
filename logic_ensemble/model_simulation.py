import maboss
from tqdm import tqdm
import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns


def simulate_model(input_path, output_path, 
                   project_name,
                   parameter_set):
    # Simulation results object
    ensemble_results = {}
    path = input_path
    model_list = os.listdir(path)

    for model in tqdm(model_list):
        # Load model
        simulations = maboss.loadBNet(path + model)

        # Setup the model initial condition
        ## Put all inputs at 0
        for i in model_nodes:
            simulations.network.set_istate(i,[1,0])
        ## Put all miR at 1
        for i in mir_list:
            simulations.network.set_istate(i,[0,1])

        # Modify the parameter of the model
        simulations.update_parameters(sample_count = 10000,
                                    thread_count = 15,
                                    max_time = 20,
                                    time_tick = 1)
        simulations.network.set_output(simulations.network.names)

        # Perform simulations
        result = simulations.run()

        # Get matrix
        model_mtx = result.get_nodes_probtraj().copy()

        # Setup cell matrix
        ## Cells
        model_mtx['model_id'] = model.replace('.bnet','')
        ## Timepoint
        model_mtx['timepoint'] = model_mtx.index
        ## Change index
        model_mtx.index = model_mtx.index.map(str)
        model_mtx.index = model + '_' + model_mtx.index

        # Concatenate model results in dictionary
        ensemble_results[model] = model_mtx

    # Save the simulation to /tmp folder
    simulation_df = pd.concat(ensemble_results.values(), ignore_index = True)
    simulation_df.to_csv('/home/spankaew/Git/BooleanBenchmark/tmp/Invasion_simulation_miR.csv')