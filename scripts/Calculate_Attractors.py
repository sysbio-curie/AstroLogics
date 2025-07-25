import pandas as pd 
import os
os.chdir('/home/spankaew/Git/astrologics')
import sys
sys.path.append('/home/spankaew/Git/astrologics/')
import astrologics as le
import numpy as np

path_to_model = '/home/spankaew/Git/astrologics/inferred_model/'
path_to_file = '/home/spankaew/Git/astrologics/data/attractor_group/'
model_list = os.listdir(path_to_model)
#model_list = ['Mammalian Cell Cycle_19118495']

def gini(array):
    """Calculate Gini coefficient of array of values."""
    array = np.sort(np.array(array))
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return (2 * np.sum(index * array) / (n * np.sum(array))) - (n + 1) / n

# For loop to identify attractor matrix
for project_name in model_list:
    print(f'Processing project: {project_name}')
    # Load the model
    model_path = path_to_model + project_name +'/'
    model = le.LogicEnsemble(model_path, project_name = project_name)
    model.create_simulation()

    # Check if the project_name folder already exists in path_to_results
    if os.path.exists(path_to_file + 'calculated_attractor/' + project_name + '_attractors.csv'):
        print(f"Skipping {project_name}, calculation already exists.")
        continue
    else:
        # Identify attractors
        model.create_attractor()
        model.attractor.get_attractors(num_cores = 15)
        model.attractor.attractors_df.to_csv(path_to_file + '/calculated_attractor/' + project_name + '_attractors.csv')

# For loop to identify attractor groups for each network
attractor_counts = pd.DataFrame(columns=['project_name', 'num_model', 'num_attractors', 'gini_score'])
for project_name in model_list:
    attractor_df = pd.read_csv(path_to_file + '/calculated_attractor/' + project_name + '_attractors.csv', index_col=0)
    concatenated_columns = attractor_df.apply(lambda col: ''.join(col.astype(str)), axis=0)
    vis_bar = pd.DataFrame(concatenated_columns.value_counts().sort_values(ascending=False))
    vis_bar['attractor_group'] = [i for i in range(len(vis_bar))]
    vis_bar = vis_bar.reset_index()

    # Create dictionary for mapping
    model_path = path_to_file + project_name +'/'
    attractor_dict = dict(zip(vis_bar['index'], vis_bar['attractor_group']))
    concatenated_columns = pd.DataFrame(concatenated_columns.map(attractor_dict), columns=['attractor_group'])
    concatenated_columns.to_csv(path_to_file + '/attractor_group/' + project_name + '_attractor_group.csv', index = True)

    # Calculate cluster distribution score
    gini_score = gini(vis_bar['count'].values)

    # Calculate the number of attractors
    num_attractors = len(vis_bar)
    num_model = len(concatenated_columns)
    attractor_counts = pd.concat([attractor_counts, pd.DataFrame({'project_name': [project_name], 
                                                                  'num_model': [num_model],
                                                                  'num_attractors': [num_attractors],
                                                                  'gini_score': [gini_score]})])

# Save the dataframe to CSV if needed
attractor_counts.to_csv(path_to_file + 'attractor_counts.csv', index=True)

