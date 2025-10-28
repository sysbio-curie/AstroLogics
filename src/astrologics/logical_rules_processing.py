# General packages
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# General logic package
import mpbn

# Packages for processing logical rules
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import linkage, leaves_list

# Model collections
from collections import Counter
# multiprocessing
from multiprocessing import Pool

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
    logic_mtx = logic_mtx.map(lambda x: f"({x})" if isinstance(x, str) else x)    
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
        df = df.fillna("0")

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

# Define function to process each model file
def process_model_file(file):
    model = mpbn.load(file)
    model = model.as_dnf()
    model = dataframe_model_dnf(model)
    model.name = os.path.splitext(os.path.basename(file))[0]
    return model

# Define function to process each model's clauses
def process_model_name(i, model_logic_mtx):
    clause = logic_clause_frequency(model_logic_mtx[[i]]).transpose()
    clause = clause.stack()
    clause.index = clause.index.map('_'.join)
    clause = clause.loc[~(clause == 0)]
    clause.name = i
    return clause

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
        # os.chdir(path)
        model_files = [os.path.join(path, model_file) for model_file in os.listdir(path)]
        model_logic = pd.DataFrame()

        print('Loading models logics')
        # Process all model files in parallel
        with Pool() as pool:
            results = list(tqdm(pool.imap(process_model_file, model_files), total=len(model_files)))

        # Concatenate results into a single DataFrame
        print('Concatenate results into matrix')
        for model in tqdm(results):
            model_logic = pd.concat([model_logic, model], axis=1, ignore_index=False)

        # Attach the model logic to the class
        self.model_logic = model_logic
        self.path = path

    def create_hash(self):
        """
        Creates a hash representation for each model file in the specified directory and stores it in the class instance.
        This method performs the following steps:
        1. Defines the model paths.
        2. Lists all model files in the specified directory.
        3. Converts each model to a hash representation using a for loop.
        4. Creates model names by removing the '.bnet' extension from each file name.
        5. Converts the list of hash models to a pandas Series.
        6. Attaches the hash models to the class instance.
        Attributes:
        -----------
        self.path : str
            The directory path where model files are located.
        self.hash_models : pandas.Series
            A Series containing the hash representations of the models, indexed by model names.
        Prints:
        -------
        'Convert model logics to hash' : str
            Indicates the start of the hash conversion process.
        'Hash of models created' : str
            Indicates the completion of the hash conversion process.
        """
        
        # Define model paths
        path = self.path

        # Define models
        model_files = os.listdir(path)

        # For loop to convert models to hash
        hash_models = []
        print('Convert model logics to hash')
        for i in tqdm(model_files):
            model = mpbn.load(path + i)
            model = model.make_hash()
            hash_models.append(model)
        
        # Create model names
        model_names = [file.replace('.bnet', '') for file in model_files]

        # Convert to numpy array
        hash_models = pd.Series(hash_models)
        hash_models.index = model_names

        # Attach the hash models to the class
        self.hash_models = hash_models
        print('Hash of models created')
    
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

        print('Flatten models logic clauses')
        # Use multiprocessing to speed up the process
        with Pool() as pool:
            results = list(tqdm(pool.starmap(process_model_name, [(i, model_logic_mtx) for i in model_name]), total=len(model_name)))
        print('Concatenate results into matrix')
        # Concatenate the results
        for clause in tqdm(results):
            logic_clause_flattend = pd.concat([logic_clause_flattend, clause], ignore_index=False, axis=1)

        # Fill NA to 0
        logic_clause_flattend = logic_clause_flattend.fillna(0)

        # Attach the model logic to the class
        self.logic_clause_flattend = logic_clause_flattend
        print('Flattend logic clause created')

    def map_model_clusters(self, cluster):
        """
        Maps model names to their respective clusters and computes the logic length for each cluster.
        """
        # Map model names to their clusters
        self.cluster_dict = cluster
        model_cluster = pd.Series(cluster, name='cluster')
        model_logic = self.model_logic

        # Create a DataFrame to store the logic length for each cluster
        var_logic_clust = pd.DataFrame()
        for i in list(model_cluster.unique()):
            model_logic_sub = model_logic.transpose().loc[list(model_cluster.index[model_cluster == i])]
            #model_logic_sub = model_logic_sub.drop(['logic_cluster'], axis = 1)
            var_logic_length = []
            for j in model_logic_sub.columns:
                num_logic = model_logic_sub[j].value_counts().size
                var_logic_length.append(num_logic)
            var_logic_length = pd.DataFrame(var_logic_length, columns = ['logic_length'], index = model_logic_sub.columns)
            var_logic_clust = pd.concat([var_logic_clust, var_logic_length], axis = 1, ignore_index = False)
        var_logic_clust.columns = list(model_cluster.unique())

        # Ordinal encoding
        encoder = OrdinalEncoder()
        model_logic_t= model_logic.transpose()
        encoded_data = encoder.fit_transform(model_logic_t)
        encoded_df = pd.DataFrame(encoded_data, columns=model_logic_t.columns)

        # Sort the index based on the maximum value in each column
        self.encoded_df = encoded_df
        self.sort_index = encoded_df.max().sort_values().index
        self.var_logic_clust = var_logic_clust.loc[self.sort_index]

        print('Model clusters mapped to logic clauses')

    def calculate_logic_statistic(self, pval_threshold = 0.0001):
        """
        Calculates the statistics of logical clauses in the model logic matrix.

        This function processes the logical clauses in the model logic matrix, performs a chi-square test
        to identify marker and varied features, and creates a DataFrame summarizing the results.

        Returns:
            pd.DataFrame: A DataFrame containing the chi-square test results, feature groups, and additional
                        information about each feature.
        """
        logic_df = self.logic_clause_flattend.transpose()
        model_cluster = logic_df.index.map(self.cluster_dict)

        # Step 1: Identify constant features.
        constant_features = [col for col in logic_df.columns if logic_df[col].nunique() == 1 and col != 'group']

        # Step 2 & 3: For non-constant features, perform chi-square test.
        marker_features = []
        varied_features = []
        chi2_results = []

        for col in logic_df.columns:
            if col in constant_features or col == 'group':
                continue
            contingency_table = pd.crosstab(logic_df[col], model_cluster)
            chi2, p, _, _ = chi2_contingency(contingency_table)
            chi2_results.append({'Feature': col, 'chi2': chi2, 'p_value': p})
            if p < pval_threshold:
                marker_features.append(col)
            else:
                varied_features.append(col)
        chi2_df = pd.DataFrame(chi2_results).set_index('Feature')

        # Create a dictionary to store features and their groups
        feature_groups = {
            'Feature': constant_features + varied_features + marker_features,
            'Group': (['Constant'] * len(constant_features)) +
                    (['Varied'] * len(varied_features)) +
                    (['Marker'] * len(marker_features))
        }
        features_df = pd.DataFrame(feature_groups).set_index('Feature')

        # Combine two DataFrames
        stat_logic_df = pd.concat([chi2_df, features_df], axis=1, ignore_index=False)
        stat_logic_df['chi2'] = stat_logic_df['chi2'].replace({np.nan: 0})
        stat_logic_df['p_value'] = stat_logic_df['p_value'].replace({np.nan: 1})

        # Extract Node and Regulation from the index
        stat_logic_df['Node'] = [col.split('_')[0] if '_' in col else col for col in stat_logic_df.index]
        stat_logic_df['Regulation'] = [col.split('_')[1] if '_' in col else col for col in stat_logic_df.index]

        # Add back to the model.logic object
        self.stat_logic_df = stat_logic_df
        self.pval_threshold = pval_threshold

    def plot_manhattan(self, fig_size=(10, 5), show_label = False, save_fig = False):
        """
        Plot a Manhattan plot of the chi-square test results.
        Args:
            fig_size (tuple): Size of the figure.
            show_label (bool): Whether to show feature labels on the plot.
            save_fig (bool): Whether to save the figure as PNG and PDF files.
        Returns:
            None
        Raises:
            None  
        """
        plt.figure(figsize=fig_size)
        stat_logic_df = self.stat_logic_df
        stat_logic_df['-log10_p'] = -np.log10(self.stat_logic_df['p_value'])

        # Add jitter to x positions
        nodes = stat_logic_df['Node'].astype('category')
        x = nodes.cat.codes + np.random.uniform(-0.2, 0.2, size=len(stat_logic_df))

        group_palette = {'Constant': '#bdbdbd', 'Varied': '#1f77b4', 'Marker': '#d62728'}
        colors = stat_logic_df['Group'].map(group_palette)
        ax = plt.scatter(x, stat_logic_df['-log10_p'], 
                        s=200, alpha=0.7, 
                        linewidths=1, edgecolor='black', c=colors)

        if show_label == True:
            # Annotate points with feature names
            for i, row in stat_logic_df.iterrows():
                plt.annotate(row.name, (x[i], row['-log10_p']),
                            textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color='black')
        
        plt.axhline(-np.log10(self.pval_threshold), color='red', linestyle='--', label=f'p={self.pval_threshold}')
        plt.ylabel('-log10(p-value)')
        plt.xlabel('Node')
        plt.title('Manhattan Plot of Chi-square p-values by Node')
        plt.xticks(ticks=range(len(nodes.cat.categories)), labels=nodes.cat.categories, rotation=90)
        plt.legend()
        plt.tight_layout()
        if save_fig == True:
            plt.savefig('manhattan_plot.png', dpi=600, bbox_inches='tight')
            plt.savefig('manhattan_plot.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_logicstat_summary(self, fig_size=(12, 7), save_fig = False):
        """
        Plot a summary of logic statistics with two subplots: number of possible logics and number of clauses.
        Args:
            fig_size (tuple): Size of the figure.
            save_fig (bool): Whether to save the figure as PNG and PDF files.
        Returns:
            None
        Raises:
            None
        """
        encoded_df = self.encoded_df
        sorted_var_logic_length = encoded_df.max().sort_values() + 1
        # Create a figure with two subplots sharing the x-axis
        fig, axes = plt.subplots(2, 1, figsize=fig_size, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Plot 1
        ## Plot the first barplot
        sns.barplot(
            x=sorted_var_logic_length.index, 
            y=sorted_var_logic_length.values, 
            hue=sorted_var_logic_length.values,
            palette=sns.color_palette("viridis", as_cmap=True), 
            edgecolor='black', linewidth=0.8, ax=axes[0], width=0.8
        )
        ## Annotate the total number of logics
        for index, value in enumerate(sorted_var_logic_length.astype('int')):
            axes[0].text(index, value + 0.1, str(value), ha='center', va='bottom', fontsize=15)
        ## Customize the first plot
        axes[0].axhline(0, color="k", clip_on=False)
        axes[0].set_ylabel('Number of Logics', fontsize=15)
        axes[0].set_title('Number of Possible Logics and Clauses', fontsize=16)
        axes[0].tick_params(axis='x', rotation=90, labelsize=15)
        axes[0].tick_params(axis='y', labelsize=15)
        axes[0].set_ylim(0, sorted_var_logic_length.max() + 1)
        axes[0].grid(linestyle='--', linewidth=0.5)
        axes[0].set_axisbelow(True)

        # Plot 2
        ## Plot the second stacked barplot
        features_df_grouped = self.stat_logic_df.groupby(['Node', 'Group']).size().unstack(fill_value=0)
        features_df_grouped = features_df_grouped.loc[sorted_var_logic_length.index]
        features_df_grouped.plot(
            kind='bar', stacked=True, colormap='tab10', ax=axes[1], edgecolor='black', linewidth=0.8, width=0.8
        )
        ## Annotate the total number of clauses
        total_clauses = features_df_grouped.sum(axis=1)
        for index, value in enumerate(total_clauses.astype('int')):
            axes[1].text(index, value + 0.1, str(value), ha='center', va='bottom', fontsize=15)
        ## Customize the second plot
        axes[1].set_ylabel('Number of Clauses', fontsize=15)
        axes[1].set_xlabel(None)
        axes[1].legend(title='Group', bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=15, title_fontsize=15)
        axes[1].tick_params(axis='x', rotation=90, labelsize=15)
        axes[1].tick_params(axis='y', labelsize=15)
        axes[1].set_ylim(0, total_clauses.values.max() + 1)
        axes[1].grid(linestyle='--', linewidth=0.5)
        axes[1].set_axisbelow(True)

        # Ensure both subplots have the same x-ticks and labels
        xticks = range(len(sorted_var_logic_length.index))
        axes[1].set_xticks(xticks)
        axes[1].set_xticklabels(sorted_var_logic_length.index, rotation=90, fontsize=15)
        axes[0].set_xticks(xticks)
        axes[0].set_xticklabels(sorted_var_logic_length.index, rotation=90, fontsize=15)

        plt.tight_layout()
        if save_fig == True:
            plt.savefig('logicstat_summary.png', dpi=600, bbox_inches='tight')
            plt.savefig('logicstat_summary.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_node_logic_heatmap(self, 
                                node, 
                                fig_size=(10, 8), save_fig = False):       
        """
        Plot a heatmap of logic clauses for a specific node.

        Args:
            node (str): The node to plot.
            fig_size (tuple): Size of the figure.
            save_fig (bool): Whether to save the figure as PNG and PDF files.

        Returns:
            None
        Raises:
            None
        """
        # Load the logic statistics DataFrame 
        stat_logic_df = self.stat_logic_df
        selected_features = stat_logic_df.loc[(stat_logic_df['Node'].isin(node))].index

        # Create a color palette for the groups
        test = self.logic_clause_flattend
        test = test.transpose()
        test['group'] = test.index.map(self.cluster_dict)

        unique_groups = test['group'].unique()
        palette = sns.color_palette("tab10", n_colors=len(unique_groups))
        group_colors = test['group'].map(dict(zip(unique_groups, palette)))

        # Create the clustermap with row colors and separated cluster groups
        # First, sort the DataFrame by cluster group to visually separate them
        sorted_idx = test.sort_values('group').index

        # Get the split point between the two clusters
        split_point = (test.loc[sorted_idx, 'group'] == unique_groups[1]).idxmax()

        # Perform row clustering only within each cluster group
        for i in range(len(unique_groups)):
            group_idx = test.loc[sorted_idx, 'group'] == unique_groups[i]
            linkage_matrix = linkage(test.loc[sorted_idx[group_idx], selected_features], method='average', metric='euclidean')
            leaves = sorted_idx[group_idx][leaves_list(linkage_matrix)]
            if i == 0:
                clustered_idx = leaves
            else:
                clustered_idx = clustered_idx.append(leaves)

        # Create the clustermap
        plt.figure(figsize=fig_size)
        sns.clustermap(
            test.loc[clustered_idx, selected_features].transpose(),
            col_cluster=False,
            row_cluster=False,
            cmap='viridis',
            figsize=(10, 5),
            col_colors=group_colors.loc[clustered_idx]
        )

        if save_fig == True:
            plt.savefig('node_logic_heatmap.png', dpi=600, bbox_inches='tight')
            plt.savefig('node_logic_heatmap.pdf', bbox_inches='tight')

        plt.show()
        plt.close()