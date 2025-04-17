import pandas as pd 
import numpy as np
import os
os.chdir('/home/spankaew/Git/astrologics')
import astrologics as le
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans


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

# Load the model
model_path = "/home/spankaew/Git/astrologics/models/dev/BN_TUTO_Fx/"
model_1 = le.LogicEnsemble(model_path, project_name = 'BN_TUTO_Fx')
model_1.create_simulation()

# Identify attractors
model_1.create_attractor()
model_1.attractor.get_attractors()
concatenated_columns = model_1.attractor.attractors_df.apply(lambda col: ''.join(col.astype(str)), axis=0)
vis_bar = concatenated_columns.value_counts().sort_values(ascending=False)
attractor_combo = vis_bar.index
unique_labels = {label: idx + 1 for idx, label in enumerate(concatenated_columns.unique())}

# Load the simulation data
save_path = "/home/spankaew/Git/astrologics/tmp/"
simulation_df = pd.read_csv(save_path + 'BN_TUTO_Fx_simulation.csv')

# Calculate distance_matrix
attractor_combo_list = vis_bar.index
concatenated_columns = model_1.attractor.attractors_df.apply(lambda col: ''.join(col.astype(str)), axis=0)
distance_matrix = calculate_endtimepoint_distancematrix(simulation_df)

# For loop to calculate clustering and rand index
rand_index_score = []
total_index_score = pd.DataFrame()
# For loop to calculate clustering and rand index
for x in tqdm(range(0, 100)):
    for i in range(0, len(attractor_combo_list)-1):
        if i == 0:
            attractor_combo = attractor_combo_list
        else:
            attractor_combo = attractor_combo_list[:-i]
        model_list = concatenated_columns.index[concatenated_columns.isin(attractor_combo)]
        
        # Calculate the k-mean_clustering
        distance_matrix_sub = distance_matrix.loc[distance_matrix.index.isin(model_list), distance_matrix.index.isin(model_list)]
        model_name = list(distance_matrix_sub.index.unique())
        
        # Euclidean k-means
        kmeans = KMeans(n_clusters=75-i, random_state=x)
        kmeans.fit(distance_matrix_sub)
        clusters = kmeans.labels_
        cluster_dict = dict(zip(list(model_name),list(clusters)))

        # Get the cluster labels
        concatenated_columns_sub = concatenated_columns[concatenated_columns.isin(attractor_combo)]
        unique_labels = {label: idx + 1 for idx, label in enumerate(concatenated_columns_sub.unique())}
        concatenated_columns_sub = concatenated_columns_sub.map(unique_labels)

        # Create a DataFrame from the cluster dictionary
        cluster_df = pd.DataFrame.from_dict(cluster_dict, orient='index', columns=['Cluster'])
        cluster_df = pd.concat([cluster_df, concatenated_columns_sub], axis=1).rename(columns={0: 'Attractor'}).rename(columns={'index': 'model_id'})

        # Calculate the adjusted Rand index
        ari = adjusted_rand_score(cluster_df.Cluster, cluster_df.Attractor)
        rand_index_score = rand_index_score + [ari]
        rand_index_score = pd.DataFrame(rand_index_score)
    total_index_score = pd.concat([total_index_score, rand_index_score], axis=1)

# Save the results
total_index_score.to_csv("/home/spankaew/Git/astrologics/tmp/rand_index_endpointclustering.csv", index = False)