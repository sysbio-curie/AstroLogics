import boolsim
from colomoto.minibn import BooleanNetwork
from colomoto_jupyter import tabulate # for display
import os
import pandas as pd
from multiprocessing import Pool
from tqdm.auto import tqdm
import gc

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
    
    def get_attractors(self, num_cores=10, chunksize=10):
        """
        Improved multiprocessing version with better memory management.
        """
        # os.chdir(self.path)
        model_files = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith('.bnet')]
        model_logic = pd.DataFrame()
        
        # Use imap with chunking for better memory efficiency
        with Pool(processes=num_cores) as pool:
            # Process in chunks to avoid memory buildup
            for result in tqdm(
                pool.imap(calculate_attractors, model_files, chunksize=chunksize), 
                total=len(model_files),
                desc="Processing models"
            ):
                model_logic = pd.concat([model_logic, result], axis=1, ignore_index=False)
                # Periodic garbage collection
                if len(model_logic.columns) % 100 == 0:
                    gc.collect()
        
        self.attractors_df = model_logic.fillna(0)
        print('Attractors calculation completed')
