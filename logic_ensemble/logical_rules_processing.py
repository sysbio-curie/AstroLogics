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
    """
    Converts a list of logical clauses into a list of string representations.

    Args:
        logic (list or bool): A list of logical clauses where each clause is a list of tuples.
                            Each tuple contains a variable and a boolean indicating its negation.
                            If the input is a boolean, it will be converted to '0' or '1'.

    Returns:
        list: A list of strings representing the logical clauses. Each clause is represented as a 
            string with variables joined by ' & '. Negated variables are prefixed with '!'.
    """
    logic_rules = []
    if logic == False:
        logic_rules = ['0']
    elif logic == True:
        logic_rules = ['1']
    else:
        for i in range(len(logic)):
            logic_clause = logic[i].copy()
            formatted_strings = [f"{t[0]}" if t[1] else f"!{t[0]}" for t in logic_clause]
            single_line_string = ' & '.join(formatted_strings)
            logic_rules.append(single_line_string)
    return(logic_rules)

def dataframe_model_dnf(model_dnf):
    """
    Converts a dictionary of Disjunctive Normal Form (DNF) logical rules into a DataFrame and 
    returns a Series with the logical expressions for each node.

    Args:
        model_dnf (dict): A dictionary where keys are node names and values are lists of clauses 
                        representing the DNF logical rules for each node.

    Returns:
        pd.Series: A Series where the index is the node names and the values are the combined 
                logical expressions in DNF format for each node.
    """
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

def split_gene_clauses(model_logic_mtx, gene): 
    """
    Splits the logical clauses for a given gene in the model logic matrix.

    Args:
        model_logic_mtx (pd.DataFrame): A DataFrame containing the logical rules for various genes.
        gene (str): The gene for which the logical clauses need to be split.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a split clause of the original logical rules for the given gene.
    """
    split_data = [item.split(" | ") for item in model_logic_mtx.loc[gene]]
    df = pd.DataFrame(split_data, index = model_logic_mtx.loc[gene].index)
    return(df)

def logic_clause_frequency(model_logic_mtx):
    """
    Calculate the frequency of logic clauses in a given model logic matrix.

    This function processes a matrix of logical rules, splits the gene clauses,
    counts the occurrences of each clause, and computes the frequency of each
    clause across all rows. The result is a DataFrame where each column 
    represents the frequency of clauses for a corresponding index in the input matrix.

    Parameters:
    model_logic_mtx (pd.DataFrame): A DataFrame where each row represents a set of 
                                    logical rules for a model.

    Returns:
    pd.DataFrame: A DataFrame containing the frequency of each logic clause. 
                Columns correspond to the indices of the input matrix.
    """
    logic_clause = pd.DataFrame()
    for i in model_logic_mtx.index:
        df = split_gene_clauses(model_logic_mtx, i)
        df = df.fillna(0)

        from collections import Counter
        # Function to count strings in a row
        def count_strings_in_row(row):
            return Counter(row)

        # Apply the function to each row and convert the result to a DataFrame
        row_counts = df.apply(lambda row: pd.Series(Counter(row)), 
                                    axis=1).fillna(0)
        clauses = row_counts.sum()/len(df)
        logic_clause = pd.concat([logic_clause,clauses], ignore_index = True, axis = 1)
    logic_clause= logic_clause.fillna(0)
    logic_clause.columns = model_logic_mtx.index
    return(logic_clause)

class logic:

    def __init__(self, path):
        """
        Loads and processes logical models from a specified directory.

        Args:
            path (str): The directory path containing the model files.

        Returns:
            pd.DataFrame: A DataFrame containing the concatenated logical models in DNF format.

        The function performs the following steps:
        1. Lists all files in the specified directory.
        2. Initializes an empty DataFrame to store the model logic.
        3. Iterates over each file in the directory:
            a. Loads the model file.
            b. Converts the model to Disjunctive Normal Form (DNF).
            c. Converts the DNF model to a DataFrame.
            d. Sets the model name based on the file name (excluding the extension).
            e. Concatenates the model DataFrame to the main DataFrame.
        4. Attaches the model logic DataFrame to the class.
        """
        # Define models path
        model_files = os.listdir(path)
        model_logic = pd.DataFrame()

# It would be interesting to make it multiprocessing - to speed up the process
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

        # Attach the model logic to the class
        self.model_logic = model_logic
        print('Model logic loaded')

    def count_logic_function(self):
        model_logic = self.model_logic
        model_logic_t= model_logic.transpose()

        # Ordinal encoding
        encoder = OrdinalEncoder()
        encoded_data = encoder.fit_transform(model_logic_t)
        encoded_df = pd.DataFrame(encoded_data, columns=model_logic_t.columns)

        # 
        sort_index = encoded_df.max().sort_values().index

    def create_flattend_logic_clause(self):
        """
        Flattens the logical clauses from a given model logic matrix.

        This function processes a DataFrame where each column represents a model's logical clauses.
        It computes the frequency of each clause, stacks them, and concatenates them into a single
        DataFrame. The resulting DataFrame has the same columns as the input, with rows representing
        the flattened logical clauses and their frequencies.

        Parameters:
        model_logic_mtx (pd.DataFrame): A DataFrame where each column represents a model's logical clauses.

        Returns:
        pd.DataFrame: A DataFrame with flattened logical clauses as rows and models as columns, 
                    filled with the frequency of each clause.
        """
        model_logic_mtx = self.model_logic
        model_name = list(model_logic_mtx.columns)
        logic_clause_flattend = pd.DataFrame()

    # This too can perhaps be made multiprocessing --- to speed up the process
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

        # Attach the model logic to the class
        self.logic_clause_flattend = logic_clause_flattend
        print('Flattend logic clause created')

    def calculate_logic_pca(self, num_components = 10):
        """
        Perform Principal Component Analysis (PCA) on the provided logical clause data.

        Parameters:
        logic_clause_flattend (pd.DataFrame): A DataFrame containing the flattened logical clauses.
        num_components (int, optional): The number of principal components to compute. Default is 10.

        Returns:
        pd.DataFrame: A DataFrame containing the principal components, with each column representing a principal component.
        """

        logic_clause_flattend = self.logic_clause_flattend

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

        # Attach the PCA DataFrame to the class
        self.pca_df = pca_df
        print('PCA calculated')

    def elbow_plot(self, num_components = 10):
        """
        Generates an elbow plot to visualize the explained variance ratio for a given number of principal components.

        Parameters:
        num_components (int): The number of principal components to consider for PCA.

        Returns:
        None: This function displays a plot and does not return any value.
        """
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

    def plot_logic_pca(self, pca_dim=['pc1', 'pc2'], fig_size=(8, 6), color='cluster'):
        """
        Plots a PCA scatter plot using the provided DataFrame.
        Parameters:
        pca_df (pd.DataFrame): DataFrame containing the PCA results.
        pca_dim (list of str, optional): List containing the names of the principal components to plot. Defaults to ['pc1', 'pc2'].
        fig_size (tuple, optional): Size of the figure. Defaults to (8, 6).
        color (str, optional): Column name in pca_df to use for coloring the points. Defaults to 'cluster'.
        Returns:
        None
        """
        pca_df = self.pca_df
        
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

    def calculate_kmean_cluster(self, num_cluster, plot = True):
        """
        Perform K-Means clustering on a PCA-transformed DataFrame and optionally plot the results.

        Parameters:
        pca_df (pd.DataFrame): DataFrame containing PCA-transformed data with at least two principal components.
        num_cluster (int): Number of clusters to form.
        plot (bool, optional): If True, generate a scatter plot of the clusters. Default is True.

        Returns:
        None: The function modifies the input DataFrame by adding a 'Kmean_Cluster' column with cluster labels.

        Notes:
        - The function assumes that the PCA DataFrame has columns named 'pc1' and 'pc2' for plotting purposes.
        - The plot will display the clusters with different colors and mark the cluster centers with red 'X' markers.
        """
        # Assume k=2 (you can choose a different number based on the Elbow Method)
        pca_df = self.pca_df
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
