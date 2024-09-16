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

class LogicEnsemble:
    def __init__(self, path, project_name):
        self.path = path
        self.project = project_name

    def create_simulation(self, parameters = None):
        self.simulation = simulation(self.path, parameters)
        print('Simulation object created')

    def create_trajectory(self, parameters = None):
        # Check if the simulation object is created
        if self.simulation is None:
            print('Simulation object is not created yet')
            return
        else:
            self.trajectory = trajectory(self.path, self.simulation.simulation_df)
            print('Trajectory object created')

    def create_logic(self, parameters = None):
        self.logic = logic(self.path, parameters)
        print('Logic object created')
