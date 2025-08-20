import numpy as np
import pandas as pd
from tqdm.auto import tqdm

class logic_analysis:
    def __init__(self, logic_clause_flattend, ):
        self.logic = logic_clause_flattend
        

    def logic_analysis(self):
        # Get the logic rules
        logic_rules = self.logic.get_logic_rules
