from colomoto.minibn import *
import pandas as pd
import bonesis
import multiprocessing as mp
import time
from tqdm import tqdm
import os
import signal

##### SET UP ######
# Set up the paths
path_to_models = "/home/spankaew/Git/astrologics/models/selected_BNs/"
path_to_results = "/home/spankaew/Git/astrologics/inferred_models/"
path_to_data = "/home/spankaew/Git/astrologics/data/model_inference/"

# Define function to write individual solution files
def write_solution_file(index, solution, previous):
    filename = f"bn_{index}.bnet"
    with open(filename, "w") as file:
        file.write(solution.source())
    previous.append(index)

def write_bn_files(solutions, num_processes=15, project = "default_project"):
    # Multiprocessing arg
    manager=mp.Manager()
    previous=manager.list()
    processes=[]

    # Define path
    project_name = project
    os.mkdir(project_name)
    os.chdir(project_name)

    # For loop to write bnet files
    for i in tqdm(range(len(solutions))):
        solution = solutions[i]
        while len(previous)<i-(num_processes-1):
            time.sleep(1)
        p = mp.Process(target = write_solution_file, 
                        args = (i,solution,previous))
        p.start()
        processes.append(p)
    for process in processes:
        process.join()

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

##### DEFINE MODEL PATH ######
project_list = os.listdir(path_to_models)
project_list = [p.replace(".bnet", "") for p in project_list]
#project_list = project_list[:2]

# For loop to load models
for project_name in project_list:
    
    # Declare network name for troubleshooting
    print(f"Processing {project_name}")
    
    # Check if the project_name folder already exists in path_to_results
    if os.path.exists(path_to_results + project_name):
        print(f"Skipping {project_name}, folder already exists.")
        continue
    else:
        ##### LOAD THE NETWORK ######
        # Load the model as network
        os.chdir(path_to_models)
        f = BooleanNetwork.load(project_name + ".bnet")
        f_ig = f.influence_graph()

        # Create influence graph with Bonesis format
        influences = [
            (u, v, dict(sign=d.get("sign", 0)))
            for u, v, d in f_ig.edges(data=True)
        ]

        # Create influence graph with Bonesis format
        net = bonesis.InfluenceGraph(influences, exact=True)

        ##### SET UP BONESIS ######
        # Assigned random initial state at zero
        initial_state = f.zero()

        # Set parameter for model inference via Bonesis
        data = {"init": initial_state}
        bo = bonesis.BoNesis(net, data)
        bo.settings["parallel"] = 15

        ##### RUN INFERENCE VIA BONESIS ######
        bo = bonesis.BoNesis(net,data)
        
        # Set a timeout for the inference process
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(10)  # Set timeout to 10 seconds
        try:
            solutions = list(bo.boolean_networks(limit=1000))
        except TimeoutException:
            print(f"Skipping {project_name}, inference took too long.")
            continue
        finally:
            signal.alarm(0)  # Disable the alarm

        ##### ANALYZE LOGIC LENGTH #####
        var_logic = pd.DataFrame(solutions)
        var_logic_length = []
        for i in var_logic.columns:
            num_logic = var_logic[i].value_counts().size
            var_logic_length.append(num_logic)
        var_logic_length = pd.DataFrame(var_logic_length, columns=['logic_length'], index=var_logic.columns)
        var_logic_length.to_csv(path_to_data + project_name + "_var_logic_length.csv")

        ##### WRITES SOLUTIONS TO FILES #####
        os.chdir(path_to_results)
        write_bn_files(solutions, num_processes=15, project = project_name)