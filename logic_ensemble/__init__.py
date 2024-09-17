"""
{Test script for the Boolean Benchmark project}
"""
__author__ = 'Saran PANKAEW'
__version__ = '0.1'
__maintainer__ = 'Saran PANKAEW'
__email__ = 'saran.pankeaw@curie.fr'
__status__ = 'development'
__date__ = '16/09/2024'

from .model_simulation import simulation
from .trajectory_clustering import trajectory
from .logical_rules_processing import logic
# from .logic_analysis import analysis

class LogicEnsemble:
    def __init__(self, path, project_name):
        self.path = path
        self.project = project_name

    def create_simulation(self):
        self.simulation = simulation(self.path)
        print('Simulation object created')

    def create_trajectory(self, parameters = None):
        # Check if the simulation object is created
        if self.simulation is None:
            print('Simulation object is not created yet')
            return
        else:
            self.trajectory = trajectory(self.simulation.simulation_df)
            print('Trajectory object created')

    def create_logic(self):
        self.logic = logic(self.path)
        print('Logic object created')

    def __repr__(self):
        print(f'LogicEnsemble object for the project {self.project}')
        print(f'Path: {self.path}')
        print(f'Simulation object: {self.simulation}')

    