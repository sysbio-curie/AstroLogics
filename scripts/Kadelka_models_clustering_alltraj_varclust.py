import pandas as pd 
import numpy as np
import os

import sys
sys.path.append('/home/spankaew/Git/astrologics/')
import astrologics as le
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from tslearn.metrics import dtw

# Define a function to calculate the distance matrix at the end timepoint
def calculate_total_distancematrix(simulation_df):
    # Make the model_id the index
    simulation_df.model_id = simulation_df.model_id.astype('category')
    node_list = simulation_df.columns.drop(['timepoint','model_id'])
    model_name = simulation_df.model_id.unique()
    
    # Convert simulation_df to numpy.array
    model_original_all = {}
    for i in model_name:
        model_original = simulation_df.loc[simulation_df.model_id == i,node_list].values
        model_original_all[i] = np.array(model_original)
    pca_all_trajectory = np.array(list(model_original_all.values()))

    # Initialize an empty distance matrix
    num_trajectories = len(pca_all_trajectory)
    distance_matrix = np.zeros((num_trajectories, num_trajectories))

    # Calculate DTW distance for each pair of trajectories
    for i in tqdm(range(num_trajectories)):
        for j in range(num_trajectories):
            distance_matrix[i, j] = dtw(pca_all_trajectory[i], pca_all_trajectory[j])

    # Display the distance matrix
    distance_matrix = pd.DataFrame(distance_matrix, index=model_name, columns=model_name)

    return(distance_matrix)

## SETTING UP THE PATHS AND IMPORTS ##

# Set the working directory
os.chdir('/home/spankaew/Git/astrologics')

# Set the path to the models
path_to_model = '/home/spankaew/Git/astrologics/inferred_model/'
model_list = os.listdir(path_to_model)

# Set the path to simulations
path_to_simulations = '/home/spankaew/Git/astrologics/data/simulation_files/'

# Set the path to attractor group
path_to_attractor = '/home/spankaew/Git/astrologics/data/attractor_group/attractor_group/'

# Results path
path_to_results = '/home/spankaew/Git/astrologics/data/rand_experiment/'

## COMPARING THE CLUSTERING WITH THE ATTRACTORS ##

# For loop to calculate clustering and rand index
total_index_score = pd.DataFrame()

for model_file in model_list:
    print(f"Processing model: {model_file}")
    # Load the attractor group
    attractor_list = pd.read_csv(path_to_attractor + model_file + '_attractor_group.csv', index_col = 0)
    attractor_group = attractor_list.attractor_group
    cluster_number = attractor_group.max() + 1 
    
    # Load the simulation DataFrame
    simulation_df = pd.read_csv(path_to_simulations + model_file + '_simulation.csv', index_col=0)
    simulation_df.fillna(0, inplace=True)

    # Calculate the distance matrix for the current model
    distance_matrix = calculate_total_distancematrix(simulation_df)


    # Initialize a DataFrame to store the Rand index scores
    rand_index_score = pd.Series(dtype=float)

    # Loop through different random_state values for k-means clustering
    for x in tqdm(range(1, cluster_number+1)):

        # Euclidean k-means
        kmeans = KMeans(n_clusters= x, random_state=12345)
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
total_index_score.to_csv(path_to_results + "rand_index_alltrajclustering_v3.csv")