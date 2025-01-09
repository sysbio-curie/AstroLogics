import boolsim
from colomoto.minibn import BooleanNetwork
from colomoto_jupyter import tabulate # for display
import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

"""
This script is used to generate sequence of attractors from the model.
"""

def calculate_attractors(file):
        """
        Calculate the attractors of a Boolean network model from a given file.
        Args:
            file (str): The path to the file containing the Boolean network model.
        Returns:
            pandas.DataFrame: A DataFrame where each column represents an attractor 
                              in binary string format, and the index is the name of 
                              the file (without the '.bnet' extension).
        """

        # Load the model
        bn = BooleanNetwork.load(file)
        a = boolsim.attractors(bn, update_mode='asynchronous')
        
        # Make sure a has all the columns based on the model
        for i in range(len(a)):
            for key in bn.keys():
                if key not in a[i]:
                    a[i][key] = 0
            a[i] = {k: a[i][k] for k in bn.keys()}

        # Convert each dictionary in the list to a string of binary values
        binary_strings = [''.join(str(value) for value in attractor.values()) for attractor in a]
        attractors = pd.DataFrame([binary_strings],columns=binary_strings)
        attractors[:]=1
        attractors.index = [file.replace('.bnet', '')]

        # Return attractors list
        return attractors.T

class attractors:
    def __init__(self, model_path):
        """
        Initializes the ModelAttractors class with the given model path.

        Args:
            model_path (str): The file path to the Boolean network model.

        Attributes:
            path (str): The file path to the Boolean network model.
            attractors_df (pandas.DataFrame or None): DataFrame to store attractors, initialized as None.
        """
        self.path = model_path
        self.attractors_df = None
    
    def get_attractors(self, num_cores = 10):
        """
        Calculate and retrieve the attractors for the models in the specified path.
        This method processes all model files in parallel using a specified number of CPU cores.
        The results are concatenated into a single DataFrame and saved to the object's `attractors_df` attribute.
        Parameters:
        num_cores (int): The number of CPU cores to use for parallel processing. Default is 10.
        Returns:
        pd.DataFrame: A DataFrame containing the concatenated results of the attractors for all models.
        """
        # Define models path
        os.chdir(self.path)
        model_files = os.listdir(self.path)
        model_logic = pd.DataFrame()
        
        
        # Process all model files in parallel with limited number of cores
        with Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.imap(calculate_attractors, model_files), total=len(model_files)))

        # Concatenate results into a single DataFrame
        print('Concatenate results into matrix')
        for model in tqdm(results):
            model_logic = pd.concat([model_logic, model], axis=1, ignore_index=False)
    
        # Save the attractors to the object
        self.attractors_df = model_logic.fillna(0)
        
        print('Attractors calculation completed')
