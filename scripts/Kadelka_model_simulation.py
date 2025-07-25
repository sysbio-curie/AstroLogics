import pandas as pd 
import numpy as np
import os

import sys
sys.path.append('/home/spankaew/Git/astrologics/')
import astrologics as le

## SETTING UP THE PATHS AND IMPORTS ##

# Set the working directory
os.chdir('/home/spankaew/Git/astrologics')

# Set the path to the models
path_to_model = '/home/spankaew/Git/astrologics/inferred_model/'
model_list = os.listdir(path_to_model)

# Set the path to simulations
path_to_simulations = '/home/spankaew/Git/astrologics/data/simulation_files/'

## MODELS SIMULATIONS ##

# For loop to run the simulation
for model_file in model_list:

    print(f"Processing {model_file}...")
    
    # Check if the project_name folder already exists in path_to_results
    if os.path.exists(path_to_simulations + model_file + '_simulation.csv'):
        print(f"Skipping {model_file}, calculation already exists.")
        continue
    else:
        # Load the model
        model_path = path_to_model + model_file + '/'
        model = le.LogicEnsemble(model_path, project_name = model_file)
        model.create_simulation()

        # Setting up the parameters
        test = pd.read_csv(model_path + '/bn_0.bnet', sep = ',', header = None)
        test[1] = 0.5
        test_dict = dict(zip(test[0], test[1]))
        test_dict

        # Run the simulations
        model.simulation.update_parameters(max_time = 30, sample_count = 2000)
        model.simulation.run_simulation(initial_state=test_dict)

        # Calculate the distance matrix
        simulation_df = model.simulation.simulation_df
        
        # Save the simulation DataFrame
        model.simulation.simulation_df.to_csv(path_to_simulations + model_file +'_simulation.csv')