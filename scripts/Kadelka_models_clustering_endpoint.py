import pandas as pd 
import numpy as np
import os

import sys
sys.path.append('/home/spankaew/Git/astrologics/')
import astrologics as le
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

# Define a function to calculate the distance matrix at the end timepoint
def calculate_endtimepoint_distancematrix(simulation_df):
    end_timepoint = simulation_df.timepoint.unique().max()
    simulation_df = simulation_df[simulation_df.timepoint == end_timepoint]

    # Make the model_id the index
    simulation_df = simulation_df.set_index('model_id')
    model_name = simulation_df.index
    simulation_df = simulation_df.drop(columns=['timepoint'])

    # Convert simulation_df to numpy.array
    simulation_array = simulation_df.to_numpy()

    # Create the distance matrix from the simulation
    distance_matrix = squareform(pdist(simulation_array, metric='euclidean'))
    distance_matrix = pd.DataFrame(distance_matrix, columns=model_name, index=model_name)

    return(distance_matrix)

## SETTING UP THE PATHS AND IMPORTS ##

# Set the working directory
os.chdir('/home/spankaew/Git/astrologics')

# Set the path to the models
path_to_model = '/home/spankaew/Git/astrologics/models/dev/'
model_list = os.listdir(path_to_model)

# Set the path to simulations
path_to_simulations = '/home/spankaew/Git/astrologics/data/simulation_files/'

# Set the path to attractor group
path_to_attractor = '/home/spankaew/Git/astrologics/data/attractor_group/'

# Results path
path_to_results = '/home/spankaew/Git/astrologics/data/rand_endpoint/'


## MODELS SIMULATIONS ##

# For loop to run the simulation
# for model_file in model_list:
    
#     # Load the model
#     model_path = path_to_model + model_file + '/'
#     model = le.LogicEnsemble(model_path, project_name = model_file)
#     model.create_simulation()

#     # Setting up the parameters
#     test = pd.read_csv(model_path + '/bn_0.bnet', sep = ',', header = None)
#     test[1] = 0.5
#     test_dict = dict(zip(test[0], test[1]))
#     test_dict

#     # Run the simulations
#     model.simulation.update_parameters(max_time = 30, sample_count = 2000)
#     model.simulation.run_simulation(initial_state=test_dict)

#     # Calculate the distance matrix
#     simulation_df = model.simulation.simulation_df
#     distance_matrix = calculate_endtimepoint_distancematrix(simulation_df)

#     # Save the simulation DataFrame
#     model.simulation.simulation_df.to_csv(path_to_simulations + model_file +'_simulation.csv')


## COMPARING THE CLUSTERING WITH THE ATTRACTORS ##

# For loop to calculate clustering and rand index
total_index_score = pd.DataFrame()

for model_file in model_list:
    
    # Load the attractor group
    attractor_list = pd.read_csv(path_to_attractor + model_file + '_attractor_group.csv', index_col = 0)
    attractor_group = attractor_list.attractor_group
    cluster_number = attractor_group.max() + 1 
    
    # Load the simulation DataFrame
    simulation_df = pd.read_csv(path_to_simulations + model_file + '_simulation.csv', index_col=0)

    # Calculate the distance matrix for the current model
    distance_matrix = calculate_endtimepoint_distancematrix(simulation_df)


    # Initialize a DataFrame to store the Rand index scores
    rand_index_score = pd.Series(dtype=float)

    # Loop through different random_state values for k-means clustering
    for x in tqdm(range(0, 100)):

        # Euclidean k-means
        kmeans = KMeans(n_clusters= cluster_number, random_state=x)
        kmeans.fit(distance_matrix)
        clusters = kmeans.labels_
        cluster_dict = dict(zip(list(distance_matrix.index),list(clusters)))

        # Create a DataFrame from the cluster dictionary
        cluster_df = pd.DataFrame.from_dict(cluster_dict, orient='index', columns=['Cluster'])
        cluster_df = pd.concat([cluster_df, attractor_group], axis=1).rename(columns={'index': 'model_id'})

        # Calculate the adjusted Rand index
        ari = adjusted_rand_score(cluster_df.Cluster, cluster_df.attractor_group)
        rand_index_score = pd.concat([rand_index_score, pd.Series([ari])], axis=0)
        rand_index_score.index = range(len(rand_index_score))

    # Compile the Rand index scores for each model
    total_index_score = pd.concat([total_index_score, rand_index_score], axis=1)

# Rename the columns of the total_index_score DataFrame
total_index_score.columns = model_list
    
# Save the results
total_index_score.to_csv(path_to_results + "rand_index_endpointclustering.csv")

