"""
{Test script for the Boolean Benchmark project}
"""
__author__ = 'Saran PANKAEW'
__version__ = '0.4'
__maintainer__ = 'Saran PANKAEW'
__email__ = 'saran.pankeaw@curie.fr'
__status__ = 'development'
__date__ = '05/08/2025'

from .model_attractors import attractors
from .model_simulation import simulation
from .trajectory_clustering import trajectory
from .logical_rules_processing import logic
from .succession_diagram import SuccessionDiagram
# from .logic_analysis import analysis

class LogicEnsemble:
    def __init__(self, path, project_name):
        self.path = path
        self.project = project_name

    def create_attractor(self):
        self.attractor = attractors(self.path)
        print('Attractor object created')    

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

    def create_succession_diagram(self):
        self.succession_diagram = SuccessionDiagram(self.path)
        print('Succession diagram object created')
    
    def __repr__(self):
        print(f'LogicEnsemble object for the project {self.project}')
        print(f'Path: {self.path}')
        print(f'Simulation object: {self.simulation}')

    
    