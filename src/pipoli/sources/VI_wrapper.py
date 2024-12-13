
import numpy as np
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer
import pickle as pickle

class VI(dynamicprogramming.DynamicProgrammingWithLookUpTable):

    def __init__(self, sys, state_dim, action_dim, cost_function, policy=None):
        self.env = sys
        self.env_dim = np.array([[state_dim],[action_dim]])
        print(self.env_dim)
        grid_sys = discretizer.GridDynamicSystem( sys , [state_dim,state_dim] , [action_dim] )
        super().__init__(grid_sys, cost_function)
        self.policy = policy

    def learn(self, tol = 0.1):
        super().solve_bellman_equation(tol=tol)
        super().clean_infeasible_set()
        ctl = super().get_lookup_table_controller()
        print(ctl)
        self.policy = ctl

    def predict(self, obs, *args, **kwargs):
        obs = np.array(obs).squeeze()
        states = np.array([angle_normalize(np.arctan2(obs[1],obs[0]) + np.pi), obs[2]])
        return self.policy.lookup_table_selection(states), states
    
    def save(self, path):
        with open(path + '_VI.pkl', 'wb') as f:
            pickle.dump(self, f)
        # np.save(path + '_VI.npy', self.policy)
        # np.save(path + '_VI_dim.npy', self.env_dim)

    def load(path, env):
        with open(path + '_VI.pkl', 'rb') as f:
            return pickle.load(f)
        # sys = sys
        # policy = np.load(path + '_VI.npy', allow_pickle=True)
        # env_dim = np.load(path + '_VI_dim.npy', allow_pickle=True)
        # state_dim = env_dim[0][0]
        # action_dim = env_dim[1][0]

        # return VI(sys, state_dim, action_dim, cost_function, policy)


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
