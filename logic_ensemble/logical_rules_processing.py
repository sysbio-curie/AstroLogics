# General packages
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
# General logic package
import mpbn
# Packages for processing logical rules
from sklearn.preprocessing import OrdinalEncoder
from statsmodels.stats.proportion import proportions_ztest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# Model collections
from collections import Counter
# Define key parameters
seed = 0
np.random.seed(seed)

def clause_string(logic):
    logic_rules = []
    if logic == False:
        logic_rules = ['0']
    elif logic == True:
        logic_rules = ['1']
    for i in range(len(logic)):
        logic_clause = logic[i].copy()
        formatted_strings = [f"{t[0]}" if t[1] else f"!{t[0]}" for t in logic_clause]
        single_line_string = ' & '.join(formatted_strings)
        logic_rules.append(single_line_string)
    return(logic_rules)

def dataframe_model_dnf(model_dnf):
    node_names = list(model_dnf.keys())
    logic_mtx = pd.DataFrame()
    for i in node_names:
        logic_mtx = pd.concat([logic_mtx,pd.DataFrame(clause_string(model_dnf[i]))], 
                        axis = 1, ignore_index = False,)
    logic_mtx = logic_mtx.applymap(lambda x: f"({x})" if isinstance(x, str) else x)    
    logic_mtx=logic_mtx.transpose()
    logic_mtx = logic_mtx.fillna('')
    logic_mtx.index = node_names
    logic_full = logic_mtx.agg(' | '.join, axis=1).str.strip(' | ')
    return(logic_full)

def load_model_logic(path):
    # Define models path
    model_files = os.listdir(path)
    model_logic = pd.DataFrame()

    # For loop to load the model
    for i in tqdm(model_files):
        # Load file
        model = mpbn.load(path + i)
        # Convert to dnf and dnf string
        model = model.as_dnf()
        model = dataframe_model_dnf(model)
        model.name = i.split('.')[0]
        # Concatenate to matrix
        model_logic = pd.concat([model_logic, model], axis = 1, ignore_index = False)

    return(model_logic)

# Function to count strings in a row
def count_strings_in_row(row):
    return Counter(row)

def split_gene_clauses(model_logic_mtx, gene): 
    split_data = [item.split(" | ") for item in model_logic_mtx.loc[gene]]
    df = pd.DataFrame(split_data, index = model_logic_mtx.loc[gene].index)
    return(df)

def logic_clause_frequency(model_logic_mtx):
    logic_clause = pd.DataFrame()
    for i in model_logic_mtx.index:
        df = split_gene_clauses(model_logic_mtx, i)
        df = df.fillna(0)

        from collections import Counter
        # Function to count strings in a row
        def count_strings_in_row(row):
            return Counter(row)

        # Apply the function to each row and convert the result to a DataFrame
        row_counts = df.apply(lambda row: pd.Series(count_strings_in_row(row)), 
                                    axis=1).fillna(0)
        clauses = row_counts.sum()/len(df)
        logic_clause = pd.concat([logic_clause,clauses], ignore_index = True, axis = 1)
    logic_clause= logic_clause.fillna(0)
    logic_clause.columns = model_logic_mtx.index
    return(logic_clause)

def create_flattend_logic_clause(model_logic_mtx):
    model_name = list(model_logic_mtx.columns)
    logic_clause_flattend = pd.DataFrame()

    for i in tqdm(model_name):
        clause = logic_clause_frequency(model_logic_mtx[[i]]).transpose()
        clause = clause.stack()
        clause.index = clause.index.map('_'.join)
        clause = clause.loc[~(clause == 0)]
        logic_clause_flattend= pd.concat([logic_clause_flattend,clause], ignore_index = False, axis = 1)

    # Fill NA to 0
    logic_clause_flattend = logic_clause_flattend.fillna(0)

    # Add the column name
    logic_clause_flattend.columns = model_name

    return(logic_clause_flattend)

def calculate_logic_pca(logic_clause_flattend, num_components = 10):
    # Initialize PCA (let's reduce to 2 principal components for this example)
    pca = PCA(n_components=num_components)

    # Fit and transform the data
    df_transposed = logic_clause_flattend.transpose()
    pca_result = pca.fit_transform(df_transposed)

    # Convert the result back to a DataFrame for easier interpretation
    pca_df = pd.DataFrame(data=pca_result, index=df_transposed.index)

    # number pca column
    number_list = list(range(pca_result.shape[1]))
    str_list = [str(i+1) for i in number_list]
    pca_df.columns = ['pc' + s for s in str_list]

    return(pca_df)

def elbow_plot(num_components):
    # Define PCA
    pca = PCA(n_components=num_components)
    explained_variance_ratio = pca.explained_variance_ratio_

    # Create an array with the number of components (1, 2, ..., n)
    components = np.arange(1, len(explained_variance_ratio) + 1)

    # Plot the explained variance ratio
    plt.figure(figsize=(8, 6))
    plt.plot(components, explained_variance_ratio, marker='o', linestyle='--')

    # Add titles and labels
    plt.title('Elbow Plot of Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')

    # Add a grid
    plt.grid(True)

    # Display the plot
    plt.show()

def plot_logic_pca(pca_df, pca_dim = ['pc1','pc2'], fig_size = (8,6), color = 'cluster'):
    
    #Define figure_size
    plt.figure(figsize=(8, 6))
    
    # Scatter plot using Seaborn
    sns.scatterplot(x=pca_dim[0], y=pca_dim[1], 
                    data=pca_df, s=50, hue = color)

    # Add title and labels
    plt.title('PCA Scatter Plot')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')

    # Display the plot
    plt.grid(True)
    plt.show()

def calculate_kmean_cluster(pca_df, num_cluster, plot = True):
    # Assume k=2 (you can choose a different number based on the Elbow Method)
    kmeans = KMeans(n_clusters=num_cluster)

    # Fit the k-means model and predict cluster labels
    clusters = kmeans.fit_predict(pca_df)

    # Add the cluster labels to the PCA DataFrame
    pca_df['Kmean_Cluster'] = clusters

    if plot == True :
        plt.figure(figsize=(8, 6))
        # Scatter plot colored by cluster
        sns.scatterplot(x='pc1', y='pc2', hue='Kmean_Cluster', data=pca_df, palette='viridis', s=100)
        # Add cluster centers to the plot
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
        # Add title and labels
        plt.title('K-Means Clustering based on PCA')
        plt.xlabel('Principal Component 1 (PC1)')
        plt.ylabel('Principal Component 2 (PC2)')
        # Display the plot
        plt.grid(True)
        plt.show()
