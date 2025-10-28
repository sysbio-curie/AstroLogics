# For General purpose functions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm

# For PCA
from sklearn.decomposition import PCA

# For distance matrix calculation and clustering
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from tslearn.metrics import dtw

# For latent space visualization
from sklearn.manifold import MDS


# Define key parameters
seed = 0
np.random.seed(seed)

class trajectory:

    def __init__(self, simulation_df, states_df=None):        
        """
        Initializes the TrajectoryClustering object.

        Parameters:
        simulation_df (pd.DataFrame): DataFrame containing the simulation data.
        states_df (pd.DataFrame, optional): DataFrame containing the states data. Default is None.
        """
        self.simulation_df = simulation_df
        self.node_list = list(simulation_df.columns.drop(['model_id','timepoint']))

        # Only assign states_df if it is not None
        if states_df is not None:
            self.states_df = states_df

    ##### In this part of the script, we focus on visualizing the whole simulation trajectory by performing PCA on the simulation data. ######
    def pca_trajectory(self, n_components = 10):
        """
        Perform PCA on the given simulation DataFrame and return the transformed DataFrame.

        Parameters:
        simulation_df (pd.DataFrame): The input DataFrame containing simulation data. 
                                        It must include 'model_id' and 'timepoint' columns.
        n_components (int): The number of principal components to keep. Default is 10.

        Returns:
        pd.DataFrame: A DataFrame containing the principal components, along with the 
                        'model_id' and 'timepoint' columns from the original DataFrame.
        """
        simulation_df = self.simulation_df

        # Initialize PCA (let's reduce to 2 principal components for this example)
        pca = PCA(n_components=n_components)

        # Fit and transform the data
        df_pca = simulation_df.drop(['model_id','timepoint'], axis = 1) 
        pca_result = pca.fit_transform(df_pca)

        # Convert the result back to a DataFrame for easier interpretation
        pca_df = pd.DataFrame(data=pca_result, index=df_pca.index)

        # number pca column
        number_list = list(range(pca_result.shape[1]))
        str_list = [str(i+1) for i in number_list]
        pca_df.columns = ['pc' + s for s in str_list]

        # Add model_id and timepoint backinto dataframe
        pca_df['model_id'] = simulation_df['model_id']
        pca_df['timepoint'] = simulation_df['timepoint'].astype('float')

        # Store the PCA DataFrame
        self.pca_df = pca_df

    ##### In this part of the script, we calculate the distance matrix, based on either endpoint simulation or whole trajectory ######
    def calculate_distancematrix(self, 
                                 mode = ['endpoint', 'trajectory'],
                                 timepoint = None):
        """
        Calculate the distance matrix for the simulation data.
        Args:
            mode (str): The mode of distance calculation. Options are 'endpoint' or 'trajectory'.
                        'endpoint' calculates distance based on the last timepoint.
                        'trajectory' calculates distance based on the entire trajectory using DTW.
            timepoint (int, optional): The specific timepoint to consider for 'endpoint' mode.
                                       If None, the maximum timepoint will be used.
        Returns:
            pd.DataFrame: A DataFrame representing the distance matrix between models.

        Raises:
            ValueError: If the mode is not 'endpoint' or 'trajectory'.
        Note:
            - For 'endpoint' mode, the distance is calculated using Euclidean distance at the specified timepoint.
            - For 'trajectory' mode, the distance is calculated using Dynamic Time Warping (DTW) over the entire trajectory.
        """
        # Extract the simulation DataFrame
        simulation_df = self.simulation_df

        # Check if the mode is valid
        if mode not in ['endpoint', 'trajectory']:
            raise ValueError("Mode must be either 'endpoint' or 'trajectory'.")
        
        # Calculate the distance matrix using endpoint
        if mode == 'endpoint':
            print("Calculating distance matrix for endpoint simulation...")
            # If mode is 'endpoint', we only consider the last timepoint
            if timepoint is None:
                timepoint = simulation_df['timepoint'].max()
            simulation_df = simulation_df.loc[simulation_df['timepoint'] == timepoint].set_index('model_id')
            model_name = simulation_df.index
            simulation_df = simulation_df.drop(columns=['timepoint'])
            simulation_array = simulation_df.to_numpy()
            distance_matrix = squareform(pdist(simulation_array, metric='euclidean'))
            distance_matrix = pd.DataFrame(distance_matrix, columns=model_name, index=model_name)

        # Calculate the distance matrix using whole trajectory
        elif mode == 'trajectory':
            print("Calculating distance matrix for whole trajectory...")
            # Preparing the calculating distance
            simulation_df.model_id = simulation_df.model_id.astype('category')
            node_list = simulation_df.columns.drop(['timepoint','model_id'])
            model_name = simulation_df.model_id.unique()
            model_original_all = {}
            
            # Loop through each model and extract the trajectory data
            for i in model_name:
                model_original = simulation_df.loc[simulation_df.model_id == i, node_list].values
                model_original_all[i] = np.array(model_original)
            pca_all_trajectory = np.array(list(model_original_all.values()))
            
            # Calculate the distance matrix using DTW
            num_trajectories = len(pca_all_trajectory)
            distance_matrix = np.zeros((num_trajectories, num_trajectories))
            for i in tqdm(range(num_trajectories)):
                for j in range(num_trajectories):
                    distance_matrix[i, j] = dtw(pca_all_trajectory[i], pca_all_trajectory[j])
            
            # Convert the distance matrix to a DataFrame
            distance_matrix = pd.DataFrame(distance_matrix, index=model_name, columns=model_name)
        
        # Return the distance matrix
        self.distance_matrix = distance_matrix
        print("Distance matrix calculated successfully.")
        
    # This script needs to be optimized more....    
    # def optimize_cluster(self,data = 'pca', n_cluster = 15, method = 'euclidean'):
    #     # Setup the variables
    #     pca_df = self.pca_df
    #     pca_df.model_id = pca_df.model_id.astype('category')
    #     model_name = pca_df.model_id.cat.categories

    #     simulation_df = self.simulation_df
    #     simulation_df.model_id = simulation_df.model_id.astype('category')
        
    #     model_name = pca_df.model_id.cat.categories
    #     model_pca_all = {}

    #     node_list = self.node_list

    #     if data == 'pca':
    #         model_pca_all = {}
    #         for i in model_name:
    #             model_pca = pca_df.loc[pca_df.model_id == i,['pc1','pc2']].values
    #             model_pca_all[i] = np.array(model_pca)
    #         pca_all_trajectory = np.array(list(model_pca_all.values()))

    #     elif data == 'original':
    #         model_original_all = {}
    #         for i in model_name:
    #             model_original = simulation_df.loc[simulation_df.model_id == i,node_list].values
    #             model_original_all[i] = np.array(model_original)
    #         pca_all_trajectory = np.array(list(model_original_all.values()))   

    #     # Calculate the optimal number of clusters
    #     distortions = []
    #     K = range(1, n_cluster)

    #     # For loop to calculate inertia for each k
    #     for k in tqdm(K):
    #         tsmodel = TimeSeriesKMeans(n_clusters=k, metric= method, random_state=0, verbose = False)
    #         tsmodel.fit(pca_all_trajectory)
    #         distortions.append(tsmodel.inertia_)

    #     plt.plot(K, distortions, 'bx-')
    #     plt.xlabel('k')
    #     plt.ylabel('Inertia')
    #     plt.title('Elbow Method For Optimal k')
    #     plt.show()
    
    def calculate_kmean_cluster(self, 
                                n_cluster, 
                                random_state = 12345):
        """
        Calculate k-means clustering on the distance matrix.
        Args:
            n_cluster (int): The number of clusters to form.
            random_state (int): Random seed for reproducibility. Default is 12345.
        Returns:
            None
        Raises:
            ValueError: If the distance matrix is not calculated yet.
        Note:
            - This method uses the pre-calculated distance matrix stored in self.distance_matrix.
        """
        # Get the distance matrix
        distance_matrix = self.distance_matrix

        # Euclidean k-means
        kmeans = KMeans(n_clusters= n_cluster, random_state=random_state)
        kmeans.fit(distance_matrix)
        clusters = kmeans.labels_
        cluster_dict = dict(zip(list(distance_matrix.index),list(clusters)))

        # Store the updated DataFrame and cluster dictionary
        self.cluster_dict = cluster_dict

        print(f"Calculated k-means clustering with {n_cluster} clusters.")
    
    def calculate_MDS(self, random_state = 12345):
        """
        Calculate MDS (Multidimensional Scaling) on the distance matrix.

        Args:
            random_state (int): Random seed for reproducibility. Default is 12345.
        Returns:
            None
        Raises:
            ValueError: If the distance matrix is not calculated yet.
        Note:
            - This method uses the pre-calculated distance matrix stored in self.distance_matrix.
        """
        # Calculate the MDS coordinates
        distance_matrix = self.distance_matrix
        mds = MDS(dissimilarity='precomputed', random_state=random_state, n_init=4)
        coords = mds.fit_transform(distance_matrix)

        # Store the MDS coordinates in a DataFrame
        coords_vis = pd.DataFrame(coords, columns = ['x', 'y'], index = distance_matrix.index)
        self.mds_coords = coords_vis

    # All the plotting functions are here 
    def plot_pca_trajectory(self, fig_size=(8,6), 
                    color='model_id', size=10,
                    plot_cluster = False, save_fig = False):
        """
        Plot PCA trajectory.
        Args:
            fig_size (tuple): Size of the figure.
            color (str): Column name to color the points. Default is 'model_id'.
            size (int): Size of the points. Default is 10.
            plot_cluster (bool): Whether to plot clusters if k-means clustering has been calculated. Default is False.
            save_fig (bool): Whether to save the figure as PNG and PDF files. Default is False.
        Returns:
            None
        Raises:
            None
        Note:
            - If plot_cluster is True, it will color the points based on the k-means clusters.
            - Ensure that k-means clustering has been calculated before setting plot_cluster to True.
        """
        
        pca_df = self.pca_df
        
        # Adjust figure size
        plt.figure(figsize = fig_size)
        
        # Line plot using seaborn
        plot = sns.lineplot(data = pca_df, 
                    x = 'pc1',y='pc2',
                    units = 'model_id', estimator = None, lw=2, alpha = .3,
                    sort = False, legend = False)

        if plot_cluster == False:
            # Line plot using seaborn
            plot2 = sns.lineplot(data = pca_df,
                            x = 'pc1', y = 'pc2',
                            units = 'model_id', estimator = None, 
                            hue = color, sort = False,
                            marker = 'o', markersize = size, 
                            linewidth = 2,
                            alpha = .5, legend = False 
                            )
        else:
            # Check if kmean_cluster is already calculated
            if not hasattr(self, 'cluster_dict'):
                print("k-mean cluster not calculated")
                return
            pca_df['kmean_cluster'] = pca_df['model_id'].map(self.cluster_dict)
            kmean_cluster = pca_df.groupby(['timepoint','kmean_cluster'])[['pc1','pc2']].mean()
            plot2 = sns.lineplot(data = kmean_cluster, 
                            x = 'pc1',y='pc2',
                            hue = 'kmean_cluster',
                            sort = False, marker = 'o', linewidth = 5, markersize = 10)
            
        # Show the plot
        plot.grid(True)
        if save_fig == True:
            plt.savefig('pca_trajectory.png', dpi=600, bbox_inches='tight')
            plt.savefig('pca_trajectory.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_model_distance_space(self, save_fig = False):

        """
        Plot the distance matrix as a clustermap.
        Args:
            save_fig (bool): Whether to save the figure as PNG and PDF files. Default is False.
        Returns:
            None
        Raises:
            None
        Note:
            - This method uses the pre-calculated distance matrix stored in self.distance_matrix.
        """
        distance_matrix = self.distance_matrix
        
        # Plot Euclidean distance using clustermap
        g = sns.clustermap(distance_matrix, figsize = (20,20))

        # Hiding the row dendrogram
        g.ax_row_dendrogram.set_visible(False)

        # Hiding the column dendrogram
        g.ax_col_dendrogram.set_visible(False)
        if save_fig == True:
            plt.savefig('distance_matrix.png', dpi=600, bbox_inches='tight')
            plt.savefig('distance_matrix.pdf', bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_MDS(self, plot_cluster = False,
                 fig_size=(8, 6), alpha=0.5, s=50,
                 save_fig = False):
        """
        Plot the MDS projection.
        Args:
            plot_cluster (bool): Whether to plot clusters if k-means clustering has been calculated. Default is False.
            fig_size (tuple): Size of the figure.
            alpha (float): Transparency level of the points. Default is 0.5.
            s (int): Size of the points. Default is 50.
            save_fig (bool): Whether to save the figure as PNG and PDF files. Default is False.
        Returns:
            None
        Raises:
            None
        Note:
            - If plot_cluster is True, it will color the points based on the k-means clusters.
            - Ensure that k-means clustering has been calculated before setting plot_cluster to True.
        """
        coords = self.mds_coords
        plt.figure(figsize=fig_size)
        if plot_cluster:
            # Add the cluster labels to the coordinates
            coords['kmean_cluster'] = coords.index.map(self.cluster_dict)
            sns.scatterplot(x=coords['x'], y=coords['y'],
                            hue=coords['kmean_cluster'],
                            palette='tab10',
                            alpha=alpha, s=s)
            plt.legend(title="Clusters")
        else:
            # Plot without cluster labels
            sns.scatterplot(x=coords['x'], y=coords['y'],
                            alpha=alpha, s=s)
        if save_fig == True:
            plt.savefig('mds_projection.png', dpi=600, bbox_inches='tight')
            plt.savefig('mds_projection.pdf', bbox_inches='tight')
        plt.grid(False)
        plt.show()

    def plot_trajectory_variance(self, fig_size=(15, 7), save_fig = False):
        """
        Plot the variance of gene expression over time.
        Args:
            fig_size (tuple): Size of the figure.
            save_fig (bool): Whether to save the figure as PNG and PDF files. Default is False.
        Returns:
            None
        Raises:
            None
        Note:
            - This method calculates the variance of each gene at each timepoint and visualizes it using a clustermap.
        """
        # Assuming `model_mtx` is your DataFrame with genes as columns and 'timepoint' as one of the columns
        model_mtx = self.simulation_df

        # Group by 'timepoint' and calculate variance for each gene
        variance_results = model_mtx.groupby('timepoint')[model_mtx.columns[:-2]].var()
        variance_results['avg'] = variance_results.mean(axis = 1)
        sns.clustermap(variance_results.drop(columns = ['avg']), 
                       row_cluster=False, cmap = 'viridis', 
                       figsize = fig_size)
        if save_fig == True:
            plt.savefig('trajectory_variance.png', dpi=600, bbox_inches='tight')
            plt.savefig('trajectory_variance.pdf', bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_node_trajectory(self, 
                             node, 
                             fig_width = 4,
                             fig_height = 4,
                             n_timesteps = 20,
                             save_fig = False):
        """
        Plot the gene expression trajectories for a specific node.
        Args:
            node (list): List of gene names to plot.
            fig_width (float): Width of each subplot. Default is 4.
            fig_height (float): Height of each subplot. Default is 4.
            n_timesteps (int): Number of timepoints to consider. Default is 20.
            save_fig (bool): Whether to save the figure as PNG and PDF files. Default is
        Returns:
            None
        Raises:
            ValueError: If the cluster_dict is not found when plot_cluster is True.
        Note:
            - This method visualizes the trajectories of specified genes over time, grouped by k-means clusters if available.

        """
        # Setup the gene list
        selected_genes = node
        num_timesteps = n_timesteps
        # Check if the cluster_dict exists
        if not hasattr(self, 'cluster_dict'):
            print("Error: cluster_dict not found. Please run calculate_kmean_cluster() first.")
            return
        model_mtx = self.simulation_df.assign(type=self.simulation_df['model_id'].map(self.cluster_dict))
        cluster_type = list(model_mtx.type.unique())

        # Function to create matrix for each condition
        def create_vis_matrix(cluster_type):
            vis = model_mtx.loc[model_mtx['type'] == cluster_type]
            vis = vis[selected_genes + ['timepoint', 'model_id']]
            vis = vis[vis.timepoint.isin(range(0, num_timesteps))]
            vis['model_id'] = vis['model_id'].astype('str')
            vis['type'] = cluster_type
            return vis

        # Create matrices for each condition
        vis_matrices = [create_vis_matrix(i) for i in cluster_type]

        # Plot with Seaborn
        plt.figure(figsize=(len(cluster_type) * fig_width, len(selected_genes) * fig_height))
        n_genes = len(selected_genes)

        for i, gene in enumerate(selected_genes):
            for j, vis in enumerate(vis_matrices):
                plt.subplot(n_genes, len(cluster_type), len(cluster_type) * i + j + 1)
                plot = sns.lineplot(data=vis, x='timepoint', y=gene, lw=2, 
                                    units = 'model_id', estimator = None, alpha = 0.4)
                plot.set_ylim(0, 1.1)
                plot.set_ylabel(gene)
                plot.set_xlabel(None)
                plot.grid(True)
                if i == 0:
                    plot.set_title(f'Cluster_{cluster_type[j]}')
                    
        for i, gene in enumerate(selected_genes):
            for j, vis in enumerate(vis_matrices):
                plt.subplot(n_genes, len(cluster_type), len(cluster_type) * i + j + 1)
                plot = sns.lineplot(data=vis, x='timepoint', y=gene, lw=2, hue = 'type',palette='Set1')
                plot.set_ylim(0, 1.1)
                plot.set_ylabel(gene)
                plot.grid(True)
                plot.set_xlabel(None)
                plot.legend().remove()

        plt.tight_layout()
        if save_fig == True:
            plt.savefig('logic_trajectory.png', dpi=600, bbox_inches='tight')
            plt.savefig('logic_trajectory.pdf', bbox_inches='tight')
        plt.show()
        plt.close()