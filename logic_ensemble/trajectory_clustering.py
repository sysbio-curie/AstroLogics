# For General purpose
import maboss
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# For PCA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# For Trajectory clustering
import tslearn
from tslearn.datasets import CachedDatasets
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.preprocessing import TimeSeriesResampler
# Define key parameters
seed = 0
np.random.seed(seed)

### Calculation ###

def pca_trajectory(simulation_df, n_components = 10):
    # Initialize PCA (let's reduce to 2 principal components for this example)
    pca = PCA(n_components=10)

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

    return pca_df

def calculate_kmean_cluster(pca_df, n_cluster):
    pca_df.model_id = pca_df.model_id.astype('category')
    model_name = pca_df.model_id.cat.categories

    model_pca_all = {}
    for i in model_name:
        model_pca = pca_df.loc[pca_df.model_id == i,['pc1','pc2']].values
        model_pca_all[i] = np.array(model_pca)

    pca_all_trajectory = np.array(list(model_pca_all.values()))

    # Euclidean k-means
    print("Euclidean k-means")
    km = TimeSeriesKMeans(n_clusters=6, verbose=True, random_state=seed)
    y_pred = km.fit_predict(pca_all_trajectory)


    # Plot the trajectory
    cluster_dict = dict(zip(list(model_name),list(y_pred)))
    pca_df['kmean_cluster'] = pca_df['model_id']
    pca_df['kmean_cluster'] = pca_df['kmean_cluster'].replace(cluster_dict)

    return pca_df

def calculate_model_distance(pca_df):
    # Get model_id
    model_id = list(pca_df['model_id'])

    # Compact data into the right format
    kmean_cluster = pca_df.groupby(['model_id','timepoint'])[['pca1','pca2']].mean()

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

### Plotting ###

def plot_pca(pca_df, 
             fig_size=(8,6), 
             color = 'timepoint',
             size = 10):
    """
    Plot PCA from the simulated models
    """

    plt.figure(figsize = fig_size)
    
    # Scatter plot using seaborn
    sns.scatterplot(data = pca_df,
                    x = 'pc1', y = 'pc2', s=size, 
                    hue = pca_df[color])
    
    # Add title and labels 
    plt.title('PCA')
    plt.xlabel('Princical Components 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')

    # Display the plot
    plt.grid(True)
    plt.show()

def plot_pca_trajectory(pca_df,
                        fig_size = (8,6),
                        color = 'model_id',
                        size = 10):
    """
    Plot trajectory from PCA of simulated models
    """
    # Adjust figure size
    plt.figure(figsize = fig_size)
    
    # Line plot using seaborn
    plot = sns.lineplot(data = pca_df,
                        x = 'pc1', y = 'pc2',
                        hue = pca_df[color], sort = False,
                        marker = 'o', markersize = size, 
                        linewidth = 2,
                        alpha = .5 
                        )
    plot.get_legend().remove()
    
    plot.grid(True)
    plt.show()

def plot_trajectory_cluster(pca_df,
                            fig_size = (8,6)):
    """
    Plot the calculated clusters onto the trajectory
    """
    ## Calculate the mean position
    kmean_cluster = pca_df.groupby(['timepoint','kmean_cluster'])[['pc1','pc2']].mean()

    ## Plot with Seaborn
    plot = sns.lineplot(data = pca_df, 
                        x = 'pc1',y='pc2',
                        hue = 'kmean_cluster', units = 'model_id', estimator = None, lw=2, alpha = .1,
                        sort = False)
    plot.get_legend().remove()

    plot2 = sns.lineplot(data = kmean_cluster, 
                        x = 'pc1',y='pc2',
                        hue = 'kmean_cluster',
                        sort = False, marker = 'o', linewidth = 5, markersize = 10)
    plt.show()

def plot_model_distance_space(pca_df):
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

def plot_cluster_distance_space(pca_df, fig_size = (8,8)):
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
    distance_space['kmean_cluster'] = distance_space['kmean_cluster'].replace(cluster_dict)
    distance_space_group = distance_space.groupby(['kmean_cluster']).mean()

    # Plot the distance space by group
    sns.clustermap(distance_space_group, figsize = fig_size, annot = True)
    plt.show()