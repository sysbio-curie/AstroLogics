from colomoto.minibn import *
import pandas as pd
import bonesis
import multiprocessing as mp
import time
from tqdm import tqdm
import os

###################################################################
# Define function necessary for generating the model
class PriorityDynamics(UpdateModeDynamics):
    def __call__(self, x):
        y = self.model(x)   # f(x)
        C = {a for a in self.nodes if y[a] != x[a]}
        genes = list(sorted(a for a in C if a.startswith("G")))
        if genes:
            z = x.copy()
            z[genes[0]] = y[genes[0]]
            yield z
            return
        for a in C:
            z = x.copy()
            z[a] = y[a]
            yield z
def node_to_dict(dyn, n):
    return {a: int(v) for (a,v) in zip(dyn.nodes, n)}
def make_labels(G):
    assert nx.is_tree(G)
    root = next(n for (n,ind) in G.in_degree() if ind == 0)
    G.nodes()[root]["label"] = "init" if G.out_degree(root) > 0 else "steady"
    ids = {"switch": 0, "steady": 0}
    def label_tree(root, begin):
        for n in G.successors(root):
            label_branch(n, begin)
    def label_branch(n, begin):
        global switch_id
        global steady_id
        if G.out_degree(root) != 1:
            assert NotImplementedError
        final_branch = len(list(nx.attracting_components(G.subgraph(nx.descendants(G, n))))) == 1
        final_key = "steady" if final_branch else "switch"
        ids[final_key] += 1
        dest = f"{final_key}{ids[final_key]}"
        i = 1
        while G.out_degree(n) == 1:
            label = f"{begin}_to_{dest}_{i}"
            G.nodes()[n]["label"] = label
            n = list(G.successors(n))[0]
            i += 1
        G.nodes()[n]["label"] = dest
        label_tree(n, dest)
    label_tree(root, "init")
    return G
def make_traj_df(f, initial_state):
    dyn = PriorityDynamics(f)
    stg = dyn.partial_dynamics(initial_state)
    make_labels(stg)
    return pd.DataFrame.from_dict({d["label"]: node_to_dict(dyn, n) for n, d in stg.nodes(data=True)}).T
def write_solution_file(index, solution, previous):
    filename = f"bn_{index}.bnet"
    with open(filename, "w") as file:
        file.write(solution.source())
    previous.append(index)
def write_bn_files(solutions, project_name, num_processes = 15):
    # Multiprocessing arg
    manager=mp.Manager()
    previous=manager.list()
    processes=[]

    # Define path
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
###################################################################

# Define path and model list
os.chdir(path = '/home/spankaew/Git/astrologics/models/selected_BNs/')
# Create a list of models based on all .bnet files in the folder (without '.bnet')
model_files = [f for f in os.listdir() if f.endswith('.bnet')]
model_list = [os.path.splitext(f)[0] for f in model_files]

# Define the path to save the generated models
path_to_results = "/home/spankaew/Git/astrologics/inferred_model/"

# Create a for loop to iterate through each model in the model list
for i in model_list:
    os.chdir(path = '/home/spankaew/Git/astrologics/models/selected_BNs/')
    project_name = i
    model_path = project_name + '.bnet'
    
    # Skip if folder already exists
    if os.path.exists(os.path.join(path_to_results, project_name)):
        print(f"Skipping {project_name}: folder already exists.")
    
    # Follow through the process of generating models
    else:
        print("Generating model for:", i)
        
        # Load the Boolean network
        f = BooleanNetwork.load(model_path)
        f_ig = f.influence_graph()

        # Create influence graph with Bonesis format
        influences = [
            (u, v, dict(sign=d.get("sign", 0)))
            for u, v, d in f_ig.edges(data=True)
        ]

        # Create influence graph with Bonesis format
        net = bonesis.InfluenceGraph(influences, exact=True)

        # Set the initial states as zero
        initial_state = f.zero()
        data = {"init": initial_state}
        bo = bonesis.BoNesis(net, data)
        bo.settings["parallel"] = 15

        # Generate Bonesis object
        bo = bonesis.BoNesis(net,data)

        # Generate the model with limited number of solutions to 1000
        solutions = list(bo.boolean_networks(limit = 1000))
        
        # Writes the generated models into .bnet files
        os.chdir(path_to_results)
        write_bn_files(solutions, num_processes=15, project_name = i)
