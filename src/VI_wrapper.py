
import numpy as np
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer
import pickle as pickle

class VI(dynamicprogramming.DynamicProgrammingWithLookUpTable):

    def __init__(self, sys, state_dim, action_dim, cost_function, policy=None):
        self.env = sys
        self.env_dim = np.array([[state_dim],[action_dim]])
        grid_sys = discretizer.GridDynamicSystem( sys , [state_dim,state_dim] , [action_dim] )
        grid_sys.dt = 0.05
        super().__init__(grid_sys, cost_function)
        self.policy = policy

    def learn(self, tol = 0.1):
        super().solve_bellman_equation(tol=tol)
        super().clean_infeasible_set()
        ctl = super().get_lookup_table_controller()
        self.policy = ctl

    def predict(self, obs, *args, **kwargs):
        obs = np.array(obs).squeeze()
        states = np.array([np.arctan2(obs[1],obs[0]), obs[2]])
        action = self.policy.lookup_table_selection(states)
        return action, states
    
    def save(self, path):
        with open(path + '_VI.pkl', 'wb') as f:
            pickle.dump(self, f)

    def load(path):
        with open(path + '_VI.pkl', 'rb') as f:
            return pickle.load(f)


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
