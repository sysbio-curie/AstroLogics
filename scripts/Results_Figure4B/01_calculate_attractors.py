from xml.parsers.expat import model

import pandas as pd 
import os
import sys

from tqdm import tqdm
sys.path.append('/home/spankaew/Git/Git_Curie/AstroLogics/src')
import astrologics as ast
import numpy as np
import time
import boolsim
from colomoto.minibn import BooleanNetwork

os.chdir('/home/spankaew/Git/Git_Curie/AstroLogics')
path_to_model = '/home/spankaew/Git/Git_Curie/AstroLogics/models/Invasion/'
path_to_file = '/home/spankaew/Git/Git_Curie/AstroLogics/data/attractor_group/'
model_list = os.listdir(path_to_model)

def gini(array):
    """Calculate Gini coefficient of array of values."""
    array = np.sort(np.array(array))
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return (2 * np.sum(index * array) / (n * np.sum(array))) - (n + 1) / n

# Define resulting objects
print('Processing models...')
attractor_time_df = []
model_logic = pd.DataFrame()

# For loop to identify attractor matrix
for project_name in tqdm(model_list):

    # Load the model
    bn = BooleanNetwork.load(path_to_model + project_name)
    # Get the attractors
    start_time = time.time()
    a = boolsim.attractors(bn, update_mode='asynchronous')
    end_time = time.time()

    # Record simulation time
    attractor_time_df.append(end_time - start_time)

    # Process attractors: handle both plain dicts and HypercubeCollections (cyclic attractors)
    binary_strings = []
    for attractor in a:
        if isinstance(attractor, dict):
            # Single steady state: fill missing keys with 0 and order by bn.keys()
            for key in bn.keys():
                if key not in attractor:
                    attractor[key] = 0
            attractor = {k: attractor[k] for k in bn.keys()}
            binary_strings.append(''.join(str(attractor[k]) for k in bn.keys()))
        else:
            # HypercubeCollection (cyclic attractor): join each state string with ' + '
            parts = []
            for sub_dict in attractor:
                sub_dict = dict(sub_dict)
                for key in bn.keys():
                    if key not in sub_dict:
                        sub_dict[key] = 0
                sub_dict = {k: sub_dict[k] for k in bn.keys()}
                parts.append(''.join(str(sub_dict[k]) for k in bn.keys()))
            binary_strings.append(' + '.join(parts))

    # Build per-model attractor DataFrame (columns = attractor strings, value = 1)
    attractors = pd.DataFrame([binary_strings], columns=binary_strings)
    attractors[:] = 1
    attractors.index = [project_name.replace('.bnet', '')]

    # Transpose so attractors become rows
    attractors = attractors.transpose()

    # Append to the master model_logic DataFrame
    model_logic = pd.concat([model_logic, attractors], axis=1, ignore_index=False)
    
# Fill the NA
model_logic = model_logic.where(model_logic.notna(), 0)

# Save the resulting object
model_logic.to_csv(path_to_file + '/calculated_attractor/' + 'Invasion' + '_attractors.csv')
attractor_time_df = pd.Series(attractor_time_df, index=[m.replace('.bnet', '') for m in model_list[:len(attractor_time_df)]])
attractor_time_df.to_csv(path_to_file + '/' + 'Invasion' + '_attractor_time.csv')

# Identify attractor groups for each network
attractor_counts = pd.DataFrame(columns=['project_name', 'num_model', 'num_attractors', 'gini_score'])
attractor_df = pd.read_csv(path_to_file + '/calculated_attractor/' + 'Invasion_attractors.csv', index_col=0)
concatenated_columns = attractor_df.apply(lambda col: ''.join(col.astype(str)), axis=0)
vis_bar = pd.DataFrame(concatenated_columns.value_counts().sort_values(ascending=False))
vis_bar['attractor_group'] = [i for i in range(len(vis_bar))]
vis_bar = vis_bar.reset_index()

# Create dictionary for mapping
model_path = path_to_file + 'Invasion' + '/'
attractor_dict = dict(zip(vis_bar['index'], vis_bar['attractor_group']))
concatenated_columns = pd.DataFrame(concatenated_columns.map(attractor_dict), columns=['attractor_group'])
concatenated_columns.to_csv(path_to_file + '/attractor_group/' + 'Invasion' + '_attractor_group.csv', index = True)

# Calculate cluster distribution score
gini_score = gini(vis_bar['count'].values)

# Calculate the number of attractors
num_attractors = len(vis_bar)
num_model = len(concatenated_columns)
attractor_counts = pd.concat([attractor_counts, pd.DataFrame({'project_name': ['Invasion'], 
                                                                'num_model': [num_model],
                                                                'num_attractors': [num_attractors],
                                                                'gini_score': [gini_score]})])

# Save the dataframe to CSV if needed
attractor_counts.to_csv(path_to_file + 'Invasion_attractor_counts.csv', index=True)