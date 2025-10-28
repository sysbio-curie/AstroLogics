import maboss
from tqdm.auto import tqdm
import collections
import os
import pandas as pd

"""
This script is used to simulate the model with the given parameter set.
"""
_default_parameter_list = collections.OrderedDict([
    ('time_tick', 1),
    ('max_time', 100),
    ('sample_count', 1000),
    ('discrete_time', 0),
    ('use_physrandgen', 1),
    ('seed_pseudorandom', 0),
    ('display_traj', 0),
    ('statdist_traj_count', 0),
    ('statdist_cluster_threshold', 1),
    ('thread_count', 1),
    ('statdist_similarity_cache_max_size', 20000)
])

class simulation:
    """
    Attributes:
        network (Network): A Network object, that will be translated in a bnd file.
        mutations (list): A list of nodes for which mutation can be triggered by modifying the cfg file.
        palette (dict): A mapping of nodes to color for plotting the results of the simulation.
        param (dict): A dictionary that contains global variables (keys starting with a '$'), and simulation parameters (keys not starting with a '$').
    Methods:
        __init__(model_path, parameters=collections.OrderedDict({}), **kwargs):
                model_path (str): Path to the .bnet files.
                parameters (dict, optional): Parameters of the simulation. Defaults to an empty OrderedDict.
                kwargs (dict): Additional parameters of the simulation.
        update_parameters(**kwargs):
            Add elements to `self.param`.
                **kwargs: Arbitrary keyword arguments to be added to `self.param`.
        mutate(node, value):
                node (str): The name of the node to be mutated.
                value (int): The value of the mutation (0 or 1).
        run_simulation(output_nodes=None):
    """
    def __init__(self, model_path, parameters=collections.OrderedDict({}), **kwargs):
        """
        Initialize the Simulation object.

        :param model_path: path to the .bnet files
        :param dict kwargs: parameters of the simulation
        """
        self.path = model_path
        self.param = _default_parameter_list.copy()
        self.palette = {}
        self.mutations = {}
        self.mutationTypes = {}
        self.refstate = {}
        
    def update_parameters(self, **kwargs):
        """
        Add elements to ``self.param``.

        This method updates the parameters stored in the `self.param` dictionary.
        It accepts keyword arguments and adds them to `self.param` if they are 
        present in the `_default_parameter_list` or if their key starts with a '$'.
        If a parameter is not recognized, a warning message is printed.

        Parameters:
        **kwargs: Arbitrary keyword arguments.
            The keyword arguments to be added to `self.param`.

        Example:
        >>> obj.update_parameters(param1=value1, param2=value2)
        """

        for p in kwargs:
            if p in _default_parameter_list or p[0] == '$':
                self.param[p] = kwargs[p]
            else:
                print("Warning: unused parameter %s" % p)

    def mutate(self, condition ,node, value):
        """
        Add a mutation to the simulation.

        This method adds a mutation to the simulation. It accepts a node name and a value
        (0 or 1) as arguments. The mutation is stored in the `self.mutations` list.

        Parameters:
        node: str
            The name of the node to be mutated.
        value: int
            The value of the mutation (0 or 1).

        Example:
        >>> obj.mutate("node1", 0)
        """

        self.mutations[condition] = (node,value)

    def run_simulation(self, output_nodes = None, initial_state = None, mutation = None):
        """
        Run simulations for a list of models and store the results.
        Parameters:
        output_nodes (list, optional): List of nodes to set as output for the simulation. 
                                       If None, all nodes will be set as output. Default is None.
        Returns:
        None: The results of the simulations are stored in the `self.simulation_df` attribute.
        The function performs the following steps:
        1. Initializes an empty dictionary to store the results of each model.
        2. Retrieves the list of models from the specified model path.
        3. Iterates over each model in the list:
           a. Loads the model using `maboss.loadBNet`.
           b. Updates the model parameters with `self.param`.
           c. Sets the output nodes for the simulation.
           d. Runs the simulation.
           e. Retrieves the probability trajectory matrix of the nodes.
           f. Adds model ID and timepoint information to the matrix.
           g. Stores the matrix in the results dictionary.
        4. Concatenates the results from all models into a single DataFrame.
        5. Saves the concatenated DataFrame to the `self.simulation_df` attribute.
        """
        
        # Simulation results object
        ensemble_results = {}
        path = self.path
        model_list = os.listdir(path)
        print('Start simulation')
        
        # For loop to run the simulation
        for model in tqdm(model_list):
            # Load model
            simulations = maboss.loadBNet(path + model)

            # Setup the model initial condition
            simulations.param = self.param

            # Set the initial condition
            if initial_state is not None:
                node_names = simulations.network.names
                assigned_node = list(initial_state.keys())
                unassigned_node = list(set(node_names) - set(assigned_node))

                # Set the initial condition - assigned node
                for i in assigned_node:
                    simulations.network.set_istate(i, [1 - initial_state[i], initial_state[i]])

                # Set the initial condition - unassigned node
                for i in unassigned_node:
                    simulations.network.set_istate(i, [0.5, 0.5])
            
            # Set the mutation condition
            if mutation is not None:
                condition = self.mutations[mutation]
                # Set the condition
                simulations.mutate(condition[0],condition[1])

            # Set the output of the simulation
            if output_nodes is not None:
                simulations.network.set_output(output_nodes)
            else:
                simulations.network.set_output(simulations.network.names)

            # Perform simulations
            result = simulations.run()
            # Get matrix
            model_mtx = result.get_nodes_probtraj().copy()

            # Setup cell matrix
            ## Cells
            model_mtx['model_id'] = model.replace('.bnet','')
            ## Timepoint
            model_mtx['timepoint'] = model_mtx.index
            ## Change index
            model_mtx.index = model_mtx.index.map(str)
            model_mtx.index = model + '_' + model_mtx.index

            # Concatenate model results in dictionary
            ensemble_results[model] = model_mtx

        # Save the simulation to /tmp folder
        simulation_df = pd.concat(ensemble_results.values(), ignore_index = True)
        
        # Save the simulation to the object
        self.simulation_df = simulation_df
        print('Simulation completed')

    def run_states_simulation(self, output_nodes = None, initial_state = None, mutation = None):
        """
        Run MaBoSS simulations for a list of models and store the state results.
        
        Parameters:
        output_nodes (list, optional): List of nodes to set as output for the simulation. 
                                       If None, all nodes will be set as output. Default is None.
        
        Returns:
        None: The results of the simulations are stored in the `self.states_df` attribute.
        
        The function performs the following steps:
        1. Simulation results object
        2. Path to the model files
        3. List of models to simulate
        4. Start simulation
        
        """
        # Simulation results object
        ensemble_results = {}
        path = self.path
        model_list = os.listdir(path)
        print('Start simulation')
        
        # For loop to run the simulation
        for model in tqdm(model_list):
            # Load model
            simulations = maboss.loadBNet(path + model)
            
            # Setup the model initial condition
            simulations.param = self.param

            # Set the initial condition
            if initial_state is not None:
                node_names = simulations.network.names
                assigned_node = list(initial_state.keys())
                unassigned_node = list(set(node_names) - set(assigned_node))

                # Set the initial condition - assigned node
                for i in assigned_node:
                    simulations.network.set_istate(i, [1 - initial_state[i], initial_state[i]])

                # Set the initial condition - unassigned node
                for i in unassigned_node:
                    simulations.network.set_istate(i, [0.5, 0.5])
            
            # Set the mutation condition
            if mutation is not None:
                condition = self.mutations[mutation]
                # Set the condition
                simulations.mutate(condition[0],condition[1])

            # Set the output of the simulation
            if output_nodes is not None:
                simulations.network.set_output(output_nodes)
            else:
                simulations.network.set_output(simulations.network.names)

            # Perform simulations
            result = simulations.run()
            # Get matrix
            model_mtx = result.get_last_states_probtraj().copy()

            # Setup cell matrix
            ## Cells
            model_mtx['model_id'] = model.replace('.bnet','')
            ## Timepoint
            model_mtx['timepoint'] = model_mtx.index
            ## Change index
            model_mtx.index = model_mtx.index.map(str)
            model_mtx.index = model + '_' + model_mtx.index

            # Concatenate model results in dictionary
            ensemble_results[model] = model_mtx

        # Save the simulation to /tmp folder
        states_df = pd.concat(ensemble_results.values(), ignore_index = True)
        states_df.fillna(0, inplace=True
                         )
        # Save the simulation to the object
        self.states_df = states_df
        print('Simulation completed : object states_df has been created')