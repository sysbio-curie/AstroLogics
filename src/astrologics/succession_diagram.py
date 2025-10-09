import os
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import MDS

import networkx as nx

class SuccessionDiagram:
    def __init__(self, model_path, max_simulation_size = 30):
        """
        Initialize the SuccessionDiagram object.

        :param model_path: path to the .bnet files
        :param dict kwargs: parameters of the simulation
        """
        self.path = model_path
        self.max_simulation_size = max_simulation_size
        
    def calculate_succession_diagram(self):
        """
        Calculate the Succession Diagram of a Boolean network model from a given file.
        """
        try:
            from pyboolnet.file_exchange import bnet2primes
            import pystablemotifs as sm
            import pystablemotifs.export as ex

            model_path = self.path
            max_simulate_size = self.max_simulation_size
            model_files = os.listdir(self.path)
            
            # Create a dictionary object to store models
            models_net = {}
            # Loop through the models
            print('Calculating Succession Diagrams')
            for i in tqdm(model_files): 
                # Adjust model name
                model_name = i.replace('.bnet','')
                primes = bnet2primes(model_path + i)
                ar = sm.AttractorRepertoire.from_primes(primes, max_simulate_size=max_simulate_size)
                models_net[model_name]=ex.networkx_succession_diagram(ar,include_attractors_in_diagram=True)
            
            # Return the SD networks object
            self.models_net = models_net
            self.node_name = list(primes.keys())
            print('Succession Diagrams calculated')
        except ImportError:
            print("pyboolnet and pystablemotifs libraries are not installed. Please install them to calculate the succession diagram.")
            
    def calculate_sd_states(self):
        """
        Calculate the state distribution (SD) states for the given models.
        This method processes the models stored in `self.models_net` to extract and calculate
        the state distributions for each node in the models. It generates two matrices:
        `states_mtx` which contains the state values for each node, and `states_bin` which 
        contains the binary count of occurrences of each state.
        Attributes:
            states_bin (pd.DataFrame): A DataFrame containing the binary count of occurrences 
                           of each state.
            all_states (list): A list of all unique states found across the models.
        """
        # Load the model
        models_net = self.models_net
        node_name = self.node_name

        states_mtx = pd.DataFrame()
        states_bin = pd.DataFrame()
        for i in tqdm(list(models_net.keys())):
            # Extract states from GM nodes
            indexes = [data['index'] for _, data in models_net[i].nodes(data=True) if 'index' in data]
            indexes = ['A' if isinstance(i, str) else i for i in indexes]
            
            labels = [data['label'] for _, data in models_net[i].nodes(data=True) if 'label' in data]
            states = [data['states'] for _, data in models_net[i].nodes(data=True) if 'states' in data]

            # Create a new label index
            new_index = []
            for idx, index in enumerate(indexes):
                if isinstance(index, int):
                    new_index.append(labels[idx])
                elif isinstance(index, str):
                    new_index.append(index)
            new_index

            # Convert to DataFrame
            states_df = pd.DataFrame(states)
            # Convert values to int or logical value
            states_df = states_df.map(lambda x: int(x) if pd.notnull(x) else '*')

            # Set index to the matrix 
            states_df = states_df[node_name]
            states_df_strings = states_df.apply(lambda row: ''.join(row.astype(str)), axis=1)
            new_index = [new_index[j]+'_'+states_df_strings[j] for j in range(len(states_df_strings))]

            states_df.index = new_index

            # Fill NA with *
            states_df.fillna('*',inplace=True)

            # Concatenate the matrix
            states_mtx = pd.concat([states_mtx, states_df], axis=0, ignore_index = False)
            states_mtx = states_mtx[~states_mtx.index.duplicated(keep='first')]

            # Count the index from each of the models
            states_bin = pd.concat([states_bin, states_df.index.value_counts()], axis=1, ignore_index = False)
            states_bin.fillna(0, inplace=True)

        states_bin.columns = list(models_net.keys())

        # Save the states matrix and binary matrix
        self.states_bin = states_bin
        self.all_states = list(states_bin.index)

    def create_sd_networks(self):
        # Load the model
        models_net = self.models_net
        node_name = self.node_name
        all_states = self.all_states
        
        # Create network adjacency matrix
        model_adj = {}

        # For loop to create the adjacency matrix
        print('Creating SD networks')
        for i in tqdm(list(models_net.keys())):

            # Extract states from GM nodes
            indexes = [data['index'] for _, data in models_net[i].nodes(data=True) if 'index' in data]
            indexes = ['A' if isinstance(i, str) else i for i in indexes]

            labels = [data['label'] for _, data in models_net[i].nodes(data=True) if 'label' in data]
            states = [data['states'] for _, data in models_net[i].nodes(data=True) if 'states' in data]


            # Create a new label index
            new_index = []
            for idx, index in enumerate(indexes):
                if isinstance(index, int):
                    new_index.append(labels[idx])
                elif isinstance(index, str):
                    new_index.append(index)
            new_index

            # Convert to DataFrame
            states_df = pd.DataFrame(states)
            # Convert values to int or logical value
            states_df = states_df.map(lambda x: int(x) if pd.notnull(x) else '*')

            # Set index to the matrix 
            states_df = states_df[node_name]
            states_df_strings = states_df.apply(lambda row: ''.join(row.astype(str)), axis=1)
            new_index = [new_index[j]+'_'+states_df_strings[j] for j in range(len(states_df_strings))]

            # Create a matrix of the states
            model_adj[i] = nx.to_pandas_adjacency(models_net[i])
            model_adj[i].index = new_index
            model_adj[i].columns = new_index

            # Reindex the matrix according to the states_bin
            model_adj[i] = model_adj[i].reindex(all_states, axis=0).reindex(all_states, axis=1)
            model_adj[i].fillna(0, inplace=True)

        # Create a networks objects
        N = len(model_adj.keys())
        model_list = list(model_adj.keys())

        networks = []
        # Compute the distance matrix
        for i in model_list:
            G = nx.from_numpy_array(model_adj[i].to_numpy(), create_using=nx.DiGraph)
            networks.append(G)

        # Save the networks object
        self.networks = networks
        print('SD networks created')

    def calculate_sdnet_distance(self):
        
        try:
            from netrd.distance import DeltaCon
            
            networks = self.networks
            model_list = list(self.models_net.keys())
            
            # Compute DeltaCon distance matrix
            N = len(networks)
            distance_matrix = np.zeros((N, N))
            deltacon = DeltaCon()

            for i in tqdm(range(N)):
                for j in range(i+1, N):
                    dist = deltacon.dist(networks[i], networks[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
            distance_matrix = pd.DataFrame(distance_matrix, index=model_list, columns=model_list)

            # Save the distance matrix
            self.distance_matrix = distance_matrix
            print('SD networks distance calculated')
        except ImportError:
            print("netrd library is not installed. Please install it to calculate the distance matrix.")
            self.distance_matrix = None

    def cluster_sdnet(self, n_cluster):
        distance_matrix = self.distance_matrix
        model_list = list(self.models_net.keys())
        
        # Compute hierarchical clustering
        Z = linkage(distance_matrix, method='ward')
        clusters = fcluster(Z, t=n_cluster, criterion='maxclust')

        # Save the clusters
        self.clusters = clusters
        print('SD networks clustered')
    
    def plot_sdnet_cluster(self):
        distance_matrix = self.distance_matrix
        clusters = self.clusters
        networks = self.networks

        # Visualize clusters with MDS
        mds = MDS(dissimilarity='precomputed', random_state=12345)
        coords = mds.fit_transform(distance_matrix)

        plt.figure(figsize=(6, 5))
        for cluster in np.unique(clusters):
            idx = np.where(clusters == cluster)
            plt.scatter(coords[idx, 0], coords[idx, 1], label=f'Cluster {cluster}')
        plt.legend()
        plt.title('Clusters Visualized via MDS')
        plt.show()

        # Plot example graphs from each cluster
        for cluster in np.unique(clusters):
            idx = np.where(clusters == cluster)[0][0]
            G = networks[idx]
            plt.figure()
            pos = nx.circular_layout(G)
            nx.draw(G, pos, with_labels=True, node_color=f"C{cluster}")
            plt.title(f'Example Network from Cluster {cluster}')
            plt.show()