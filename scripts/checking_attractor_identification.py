from colomoto.minibn import *
import pandas as pd
import boolsim
import os
import signal
import time

##### SET UP ######
# Set up the paths
path_to_models = "/home/spankaew/Git/astrologics/models/selected_BNs/"
path_to_data = "/home/spankaew/Git/astrologics/data/model_inference/"

# Load the model
project_list = os.listdir(path_to_models)
project_list = [p.replace(".bnet", "") for p in project_list]

results = []

for model_file in project_list:
    os.chdir(path_to_models)
    print('Processing model:', model_file)
    bn = BooleanNetwork.load(model_file + '.bnet')
    start_time = time.time()
    error = None
    a = None
    try:
        a = boolsim.attractors(bn, update_mode='asynchronous')
    except Exception as e:
        error = str(e)                              
 
    elapsed = time.time() - start_time
    results.append({
        "model": model_file,
        "time_used": elapsed,
        "error": error
    })

df = pd.DataFrame(results).set_index("model")
df.to_csv(path_to_data + "attractor_calculation_results.csv")
