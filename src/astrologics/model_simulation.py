import maboss
from tqdm.auto import tqdm
import collections
import os
import pandas as pd
import numpy as np

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

    def _run_single_model_probtraj(self, model_file, output_nodes=None, initial_state=None, mutation=None, sample_count=None, seed=None):
        """
        Run one MaBoSS simulation and return node probability trajectories.

        Parameters:
        model_file (str): Name of the model file.
        output_nodes (list, optional): List of output nodes.
        initial_state (dict, optional): Initial condition for selected nodes.
        mutation (str, optional): Mutation condition name stored in ``self.mutations``.
        sample_count (int, optional): Override value for ``sample_count``.
        seed (int, optional): Override value for ``seed_pseudorandom``.

        Returns:
        pandas.DataFrame: Node probability trajectories.
        """

        model_path = os.path.join(self.path, model_file)
        simulations = maboss.loadBNet(model_path)

        # Setup simulation parameters
        simulations.param = self.param.copy()
        if sample_count is not None:
            simulations.param['sample_count'] = int(sample_count)
        if seed is not None:
            simulations.param['seed_pseudorandom'] = int(seed)

        # Set the initial condition
        node_names = simulations.network.names
        if initial_state is None:
            for i in node_names:
                simulations.network.set_istate(i, [0.5, 0.5])
        else:
            assigned_node = list(initial_state.keys())
            unassigned_node = list(set(node_names) - set(assigned_node))

            for i in assigned_node:
                simulations.network.set_istate(i, [1 - initial_state[i], initial_state[i]])

            for i in unassigned_node:
                simulations.network.set_istate(i, [0.5, 0.5])

        # Set the mutation condition
        if mutation is not None:
            condition = self.mutations[mutation]
            simulations.mutate(condition[0], condition[1])

        # Set simulation outputs
        if output_nodes is not None:
            simulations.network.set_output(output_nodes)
        else:
            simulations.network.set_output(simulations.network.names)

        result = simulations.run()
        return result.get_nodes_probtraj().copy()
        
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
            node_names = simulations.network.names
            if initial_state is None:
                for i in node_names:
                    simulations.network.set_istate(i, [0.5, 0.5])
                    
            if initial_state is not None:
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

    def analyze_sample_count_convergence(
        self,
        sample_counts=None,
        n_replicates=3,
        tolerance=0.01,
        output_nodes=None,
        initial_state=None,
        mutation=None,
        model_subset=15,
        metric='mean_std',
        progress=True
    ):
        """
        Analyze convergence of MaBoSS simulations as a function of ``sample_count``.

        This method runs repeated simulations for each tested ``sample_count`` and
        quantifies variability of final node probabilities across replicates.
        The smallest ``sample_count`` satisfying ``metric <= tolerance`` is selected
        as the recommended value.

        Parameters:
        sample_counts (list, optional): Values of ``sample_count`` to test.
            Defaults to ``[100, 250, 500, 1000, 2500, 5000]``.
        n_replicates (int, optional): Number of repeated simulations per
            ``sample_count``. Defaults to 5.
        tolerance (float, optional): Threshold used to recommend an optimal
            ``sample_count``. Defaults to 0.01.
        output_nodes (list, optional): Nodes used as simulation outputs.
            If ``None``, all nodes are used.
        initial_state (dict, optional): Initial condition for selected nodes.
        mutation (str, optional): Name of a mutation condition already stored in
            ``self.mutations``.
        model_subset (int or list, optional):
            - ``None``: use all models in ``self.path``.
            - ``int``: use the first N models.
            - ``list``: use the provided list of model filenames.
        metric (str, optional): Summary metric used for recommendation.
            Supported: ``'mean_std'``, ``'median_std'``, ``'max_std'``, ``'p95_std'``.
        progress (bool, optional): Display progress bars. Defaults to True.

        Returns:
        dict: Dictionary with keys:
            - ``recommended_sample_count``
            - ``summary`` (DataFrame, per-sample_count convergence metrics)
            - ``node_statistics`` (DataFrame, per-node variability metrics)
            - ``raw`` (DataFrame, replicate-level probabilities)

        Notes:
        - The method stores results in:
            ``self.convergence_raw_df``, ``self.convergence_node_stats_df``,
            ``self.convergence_summary_df``, and ``self.recommended_sample_count``.
        - If no tested value satisfies ``tolerance``, the largest tested
          ``sample_count`` is returned as recommendation.
        """

        if sample_counts is None:
            sample_counts = [100, 250, 500, 1000, 2500, 5000]

        sample_counts = sorted({int(x) for x in sample_counts if int(x) > 0})
        if len(sample_counts) == 0:
            raise ValueError("`sample_counts` must contain at least one positive integer.")

        if n_replicates < 2:
            raise ValueError("`n_replicates` must be >= 2 to estimate convergence variability.")

        model_list = sorted([m for m in os.listdir(self.path) if m.endswith('.bnet')])
        if len(model_list) == 0:
            raise ValueError(f"No .bnet model files found in '{self.path}'.")

        if isinstance(model_subset, int):
            if model_subset <= 0:
                raise ValueError("`model_subset` as int must be > 0.")
            model_list = model_list[:model_subset]
        elif model_subset is not None:
            model_subset = set(model_subset)
            model_list = [m for m in model_list if m in model_subset]
            if len(model_list) == 0:
                raise ValueError("`model_subset` does not match any .bnet file in model path.")

        supported_metrics = {'mean_std', 'median_std', 'max_std', 'p95_std'}
        if metric not in supported_metrics:
            raise ValueError(f"Unsupported metric '{metric}'. Supported metrics: {sorted(supported_metrics)}")

        raw_records = []
        outer_iterator = sample_counts
        if progress:
            outer_iterator = tqdm(sample_counts, desc='Convergence analysis', leave=True)

        original_sample_count = self.param.get('sample_count', None)
        original_seed = self.param.get('seed_pseudorandom', None)

        try:
            for sc in outer_iterator:
                for rep in range(n_replicates):
                    seed = rep + 1
                    inner_iterator = model_list
                    if progress:
                        inner_iterator = tqdm(model_list, desc=f'sample_count={sc}, rep={rep + 1}/{n_replicates}', leave=False)

                    for model_file in inner_iterator:
                        model_id = model_file.replace('.bnet', '')
                        probtraj = self._run_single_model_probtraj(
                            model_file=model_file,
                            output_nodes=output_nodes,
                            initial_state=initial_state,
                            mutation=mutation,
                            sample_count=sc,
                            seed=seed
                        )

                        final_state = probtraj.iloc[-1]
                        for node, value in final_state.items():
                            if pd.notna(value):
                                raw_records.append({
                                    'sample_count': int(sc),
                                    'replicate': int(rep + 1),
                                    'model_id': model_id,
                                    'node': node,
                                    'probability': float(value)
                                })
        finally:
            if original_sample_count is not None:
                self.param['sample_count'] = original_sample_count
            if original_seed is not None:
                self.param['seed_pseudorandom'] = original_seed

        if len(raw_records) == 0:
            raise RuntimeError('No simulation results were collected during convergence analysis.')

        raw_df = pd.DataFrame(raw_records)

        node_stats_df = (
            raw_df
            .groupby(['sample_count', 'model_id', 'node'], as_index=False)['probability']
            .agg(
                mean_probability='mean',
                std_probability='std',
                min_probability='min',
                max_probability='max'
            )
            .fillna(0.0)
        )

        summary_df = (
            node_stats_df
            .groupby('sample_count', as_index=False)['std_probability']
            .agg(
                mean_std='mean',
                median_std='median',
                max_std='max',
                p95_std=lambda x: float(np.percentile(x, 95))
            )
            .sort_values('sample_count')
            .reset_index(drop=True)
        )

        recommended_sample_count = int(summary_df['sample_count'].max())
        if tolerance is not None:
            valid = summary_df[summary_df[metric] <= tolerance]
            if len(valid) > 0:
                recommended_sample_count = int(valid['sample_count'].iloc[0])

        self.convergence_raw_df = raw_df
        self.convergence_node_stats_df = node_stats_df
        self.convergence_summary_df = summary_df
        self.recommended_sample_count = recommended_sample_count

        return {
            'recommended_sample_count': recommended_sample_count,
            'summary': summary_df,
            'node_statistics': node_stats_df,
            'raw': raw_df
        }