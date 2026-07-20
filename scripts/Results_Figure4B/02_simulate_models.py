import pandas as pd 
import numpy as np
import os
from tqdm import tqdm
import maboss
import collections
import time
import sys
sys.path.append('../../src')

## SETTING UP THE PATHS AND IMPORTS ##
# Default parameter
param = collections.OrderedDict([
    ('time_tick', 1),
    ('max_time', 100),
    ('sample_count', 1000),
    ('discrete_time', 0),
    ('use_physrandgen', 1),
    ('seed_pseudorandom', 0),
    ('display_traj', 0),
    ('statdist_traj_count', 0),
    ('statdist_cluster_threshold', 1),
    ('thread_count', 1),
    ('statdist_similarity_cache_max_size', 20000)
])

# Set the working directory
os.chdir('../../')

# Set the path to the models
path_to_model = 'models/Invasion/'

# Set the path to simulations
path_to_simulations = 'data/simulation_files/'

## MODELS SIMULATIONS ##

# Simulation results object
ensemble_results = {}
model_list = os.listdir(path_to_model)
initial_state = None
mutation = None
output_nodes = None

print('Start simulation')

# For loop to define the sample counts
n = [50,100,250,500,750]

for i in n:
    simulation_time_df = []
    print (f'Simulation Sample count: {i}')

    # For loop to run the simulation
    for model in tqdm(model_list):
        # Load model
        simulations = maboss.loadBNet(path_to_model + model)

        # Adjust the simulation parameter
        simulations.param['max_time'] = 20
        simulations.param['time_tick'] = 1
        simulations.param['thread_count'] = 15
        simulations.param['sample_count'] = i

        # Set the initial condition
        node_names = simulations.network.names
        if initial_state is None:
            for node in node_names:
                simulations.network.set_istate(node, [0.5, 0.5])
                
        if initial_state is not None:
            assigned_node = list(initial_state.keys())
            unassigned_node = list(set(node_names) - set(assigned_node))

            # Set the initial condition - assigned node
            for node in assigned_node:
                simulations.network.set_istate(node, [1 - initial_state[node], initial_state[node]])

            # Set the initial condition - unassigned node
            for node in unassigned_node:
                simulations.network.set_istate(node, [0.5, 0.5])
        
        # Set the mutation condition
        if mutation is not None:
            condition = mutations[mutation]
            # Set the condition
            simulations.mutate(condition[0],condition[1])

        # Set the output of the simulation
        if output_nodes is not None:
            simulations.network.set_output(output_nodes)
        else:
            simulations.network.set_output(simulations.network.names)

        # Perform simulations
        start_time = time.time()
        result = simulations.run()
        end_time = time.time()
        
        # Record simulation time
        simulation_time_df.append(end_time - start_time)
        
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

    # Save the simulation to the object
    simulation_df.to_csv(path_to_simulations + 'Invasion_simulation_' + str(i) + '.csv', index=False)

    # Save the simulate time
    simulation_time_df = pd.Series(simulation_time_df, index=[m.replace('.bnet', '') for m in model_list[:len(simulation_time_df)]])
    simulation_time_df.to_csv(path_to_simulations + '/' + 'Invasion' + '_simulation_time' + str(i) + '.csv')

print('Simulation completed')