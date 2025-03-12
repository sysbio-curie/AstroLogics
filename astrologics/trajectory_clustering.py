# For General purpose functions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# For PCA
from sklearn.decomposition import PCA
# For Trajectory clustering
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.preprocessing import TimeSeriesResampler
# Define key parameters
seed = 0
np.random.seed(seed)


class trajectory:

    def __init__(self, simulation_df):        
        """
        Initializes the TrajectoryClustering object.

        Parameters:
        simulation_df (pd.DataFrame): DataFrame containing the simulation data.
        """
        self.simulation_df = simulation_df

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

    def plot_trajectory(self, fig_size=(8,6), 
                        color='model_id', size=10,
                        show_legend = False):
        """
        Plot trajectory from PCA of simulated models.
        Parameters:
        pca_df (DataFrame): DataFrame containing PCA results with columns 'pc1' and 'pc2'.
        fig_size (tuple): Size of the figure (width, height). Default is (8, 6).
        color (str): Column name in pca_df to use for coloring the lines. Default is 'model_id'.
        size (int): Size of the markers. Default is 10.
        Returns:
        None: Displays a line plot of the PCA trajectories.
        """
        pca_df = self.pca_df
        
        # Adjust figure size
        plt.figure(figsize = fig_size)
        
        # Line plot using seaborn
        plot = sns.lineplot(data = pca_df, 
                    x = 'pc1',y='pc2',
                    units = 'model_id', estimator = None, lw=2, alpha = .5,
                    sort = False)
        
        # Line plot using seaborn
        plot = sns.lineplot(data = pca_df,
                            x = 'pc1', y = 'pc2',
                            units = 'model_id', estimator = None, 
                            hue = color, sort = False,
                            marker = 'o', markersize = size, 
                            linewidth = 2,
                            alpha = .5 
                            )
        # Show legend
        if show_legend == False:
            plot.get_legend().remove()

        # Show the plot
        plot.grid(True)
        plt.show()

    # Calculate k-means clusters for PCA-transformed data
    def calculate_kmean_cluster(self, n_cluster, data='pca', metric='euclidean'):
        """
        Perform k-means clustering on PCA-transformed data using specified metric.

        Parameters:
        -----------
        n_cluster : int
            The number of clusters to form.
        data : str, optional
            The type of data to use for clustering. Default is 'pca'.
        metric : str, optional
            The distance metric to use for clustering. Options are 'euclidean', 'dtw', and 'softdtw'. Default is 'euclidean'.

        Returns:
        --------
        None
            The function updates the `pca_df` DataFrame with cluster labels and stores the cluster dictionary in `self.cluster_dict`.

        Notes:
        ------
        - The function assumes that `self.pca_df` is a DataFrame containing PCA-transformed data with a 'model_id' column.
        - The function supports three types of k-means clustering based on the specified metric: Euclidean, DTW, and Soft-DTW.
        - The clustering results are stored in the 'kmean_cluster' column of `self.pca_df`.
        """
        # Setup the 
        pca_df = self.pca_df
        pca_df.model_id = pca_df.model_id.astype('category')
        model_name = pca_df.model_id.cat.categories

        # Create timeseries array
        if data == 'pca':
            model_pca_all = {}
            for i in model_name:
                model_pca = pca_df.loc[pca_df.model_id == i,['pc1','pc2']].values
                model_pca_all[i] = np.array(model_pca)
            pca_all_trajectory = np.array(list(model_pca_all.values()))

        # Perform clustering based on the metric
        if metric == 'euclidean':
            # Euclidean k-means
            print("Euclidean k-means")
            km = TimeSeriesKMeans(n_clusters=n_cluster, metric = 'euclidean', verbose=True, random_state=seed)
            y_pred = km.fit_predict(pca_all_trajectory)
        elif metric == 'dtw':
            # DTW k-means
            print("DTW k-means")
            km = TimeSeriesKMeans(n_clusters=n_cluster, metric="dtw", verbose=True, random_state=seed)
            y_pred = km.fit_predict(pca_all_trajectory)
        elif metric == 'softdtw':
            # Soft-DTW k-means
            print("Soft-DTW k-means")
            km = TimeSeriesKMeans(n_clusters=n_cluster, metric="softdtw", verbose=True, random_state=seed)
            y_pred = km.fit_predict(pca_all_trajectory)
        # Attach cluster information
        cluster_dict = dict(zip(list(model_name),list(y_pred)))
        pca_df['kmean_cluster'] = pca_df['model_id']
        pca_df['kmean_cluster'] = pca_df['kmean_cluster'].replace(cluster_dict)

        # Store the updated DataFrame and cluster dictionary
        self.pca_df = pca_df
        self.cluster_dict = cluster_dict

    def plot_trajectory_cluster(self, 
                                fig_size = (8,6)):
        """
        Plot the calculated clusters onto the trajectory
        """
        pca_df = self.pca_df

        ## Calculate the mean position
        kmean_cluster = pca_df.groupby(['timepoint','kmean_cluster'])[['pc1','pc2']].mean()

        # Adjust figure size
        plt.figure(figsize = fig_size)

        ## Plot with Seaborn
        plot = sns.lineplot(data = pca_df, 
                    x = 'pc1', y='pc2',
                    hue = 'kmean_cluster', units = 'model_id', estimator = None, lw=2, alpha = .1,
                    sort = False, legend=False)
        plot2 = sns.lineplot(data = kmean_cluster, 
                            x = 'pc1',y='pc2',
                            hue = 'kmean_cluster',
                            sort = False, marker = 'o', linewidth = 5, markersize = 10)
        plt.show()

#### I'm not sure if this function is necessary
    def calculate_model_distance(self):
        """
        Calculate and plot the Euclidean distance between clusters of PCA-transformed data.

        This function takes a DataFrame containing PCA-transformed data with model IDs and timepoints,
        calculates the mean PCA values for each model and timepoint, computes the Euclidean distances
        between these clusters, and then plots the distance matrix using a clustermap.

        Parameters:
        pca_df (pd.DataFrame): A DataFrame containing the PCA-transformed data with columns 'model_id',
                            'timepoint', 'pca1', and 'pca2'.

        Returns:
        None
        """
        pca_df = self.pca_df
        # Get model_id
        model_id = list(pca_df['model_id'])

        # Compact data into the right format
        kmean_cluster = pca_df.groupby(['model_id','timepoint'])[['pc1','pc2']].mean()

        # Calculate Euclidean distances between clusters
        distance_matrix = pd.DataFrame()
        for i in model_id:
            time_series1 = np.array(kmean_cluster.loc[i])
            distance_cluster = []
            for j in model_id:
                time_series2 = np.array(kmean_cluster.loc[j])
                distance = np.sqrt(np.sum((time_series1 - time_series2)**2))
                distance_cluster.append(distance)
            distance_matrix = pd.concat([distance_matrix,pd.DataFrame(distance_cluster)],axis = 1)
        distance_matrix.columns = model_id
        distance_matrix.index = model_id

        # Plot Euclidean distance using clustermap
        g = sns.clustermap(distance_matrix, figsize = (20,20))

        # Hiding the row dendrogram
        g.ax_row_dendrogram.set_visible(False)

        # Hiding the column dendrogram
        g.ax_col_dendrogram.set_visible(False)

        plt.show()

    def plot_model_distance_space(self):
        """
        Plots the distance space of models in a 2D PCA-transformed space using Euclidean k-means clustering.

        This method performs the following steps:
        1. Extracts the PCA-transformed coordinates (pc1, pc2) for each model.
        2. Aggregates the PCA coordinates for all models.
        3. Applies Euclidean k-means clustering to the aggregated PCA coordinates.
        4. Transforms the PCA coordinates into a distance space using the k-means model.
        5. Plots a clustermap of the distance space for each model.

        The clustermap visualizes the distance of each model to each cluster centroid.

        Attributes:
        -----------
        pca_df : pandas.DataFrame
            DataFrame containing PCA-transformed coordinates and model identifiers.
        
        model_pca_all : dict
            Dictionary storing PCA coordinates for each model.
        
        pca_all_trajectory : numpy.ndarray
            Array of PCA coordinates for all models.
        
        distance_space : pandas.DataFrame
            DataFrame containing the distance space of each model to each cluster centroid.
        
        km : TimeSeriesKMeans
            K-means clustering model from the `tslearn` library.
        
        seed : int
            Random seed for reproducibility.
        
        Returns:
        --------
        None
        """
        pca_df = self.pca_df
        model_pca_all = {}
        model_name = pca_df.model_id.cat.categories
        for i in model_name:
            model_pca = pca_df.loc[pca_df.model_id == i,['pc1','pc2']].values
            model_pca_all[i] = np.array(model_pca)

        pca_all_trajectory = np.array(list(model_pca_all.values()))

        # Euclidean k-mean distance
        km = TimeSeriesKMeans(n_clusters=6, verbose=True, random_state=seed)

        # Obtain the distance space from `tslearn`
        distance_space = km.transform(pca_all_trajectory)
        distance_space = pd.DataFrame(distance_space)
        distance_space.index = list(model_pca_all.keys())

        # Plot the distance space to each cluster for each model
        sns.clustermap(distance_space.transpose(), figsize = (15,4), xticklabels=False)
        plt.show()


    def plot_cluster_distance_space(self, fig_size = (8,8)):
        """
        Plots the cluster distance space using PCA-transformed data and k-means clustering.
        Parameters:
        fig_size (tuple): A tuple specifying the size of the figure (default is (8,8)).
        This function performs the following steps:
        1. Extracts PCA-transformed data for each model.
        2. Computes the Euclidean k-means distance using TimeSeriesKMeans.
        3. Transforms the PCA data into a distance space.
        4. Attaches cluster information to the distance space.
        5. Groups the distance space by cluster and computes the mean.
        6. Plots a clustered heatmap of the distance space by group.
        The resulting plot shows the mean distance space for each cluster, with annotations.
        """
        
        pca_df = self.pca_df
        model_pca_all = {}
        model_name = pca_df.model_id.cat.categories
        for i in model_name:
            model_pca = pca_df.loc[pca_df.model_id == i,['pc1','pc2']].values
            model_pca_all[i] = np.array(model_pca)
        pca_all_trajectory = np.array(list(model_pca_all.values()))

        # Euclidean k-mean distance
        km = TimeSeriesKMeans(n_clusters=6, verbose=True, random_state=seed)
        
        # Obtain the distance space from `tslearn`
        distance_space = km.transform(pca_all_trajectory)
        distance_space = pd.DataFrame(distance_space)
        distance_space.index = list(model_pca_all.keys())

        # Attach cluster information
        distance_space['kmean_cluster'] = list(distance_space.index)
        distance_space['kmean_cluster'] = distance_space['kmean_cluster'].replace(self.cluster_dict)
        distance_space_group = distance_space.groupby(['kmean_cluster']).mean()

        # Plot the distance space by group
        sns.clustermap(distance_space_group, figsize = fig_size, annot = True)
        plt.show()