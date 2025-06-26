import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from typing import Callable

from VI_wrapper import VI


class Context:

    def __init__(
        self,
        base_vars: np.ndarray,
        obs_transform: Callable[[np.ndarray],np.ndarray],
        act_transform: Callable[[np.ndarray], np.ndarray]
    ):
        """Initialize a context.

        Parameters:
        -----------
        base_vars: array[n]
            the system's variable used to cancel the units
        obs_transform: fn(array[n]) -> array[m]
            a function that computes the factors to non dimensionalize an observation
        act_transform: fn(array[n]) -> array[p]
            a function that computes the factors to non dimensionalize an action
        """
        self.base_vars = base_vars
        self.obs_transform = obs_transform
        self.act_transform = act_transform

    def from_adim_obs(self, obs_s: np.ndarray) -> np.ndarray:
        """Converts a non dimensional observation back to a dimensional one."""
        return obs_s / self.obs_transform(self.base_vars)

    def to_adim_obs(self, obs: np.ndarray) -> np.ndarray:
        """Converts a dimensional observation to a non dimensional one."""
        return obs * self.obs_transform(self.base_vars)

    def from_adim_act(self, act_s: np.ndarray) -> np.ndarray:
        """Converts a non dimensional action back to a dimensional one."""
        return act_s / self.act_transform(self.base_vars)

    def to_adim_act(self, act: np.ndarray) -> np.ndarray:
        """Converts a dimensional action to a non dimensional one."""
        return act * self.act_transform(self.base_vars)
    
    def save(self, path: str):
        np.save(path + '.npy', self.base_vars)

    def compare(self, other: "Context") -> bool:
        a = self.base_vars - other.base_vars
        b = np.linalg.norm(a)
        out = np.linalg.norm(self.base_vars - other.base_vars) #* np.sign(other.base_vars[0] - self.base_vars[0])
        return out
    
    def adim_compare(self, other: "Context") -> bool:
        # norm_self = self.base_vars / np.linalg.norm(self.base_vars)
        # norm_other = other.base_vars / np.linalg.norm(other.base_vars)
        # return np.linalg.norm(norm_self - norm_other) * np.sign(other.base_vars[0] - self.base_vars[0])
        return self.base_vars[2]/(self.base_vars[0]*self.base_vars[1]) - other.base_vars[2]/(other.base_vars[0]*other.base_vars[1])


Policy = A2C | DDPG | PPO | SAC | TD3 | VI

class DimensionalPolicy:

    def __init__(self, policy: Policy, context: Context, obs_space: Box, act_space: Box):
        """Initialize a dimensional policy from a Stable-Baselines3 policy.

        Parameters:
        -----------
        policy: A2C | DDPG | PPO | SAC | TD3 | VI
            the trained Stable-Baselines3 policy
        context: Context
            the context of this policy
        obs_space: Box
            the observation space on which this policy was trained
        act_space: Box
            the action space this policy produced in training
        """
        self.context = context
        self.policy = policy
        self.obs_space = obs_space
        self.act_space = act_space

    def __rshift__(self, context: Context) -> "ScaledPolicy":
        return self.to_scaled(context)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalizes the observation in [-1, 1]."""
        low = self.obs_space.low
        high = self.obs_space.high
        return -1 + 2 / (high-low) * (obs-low)

    def _denormalize_act(self, norm_act: np.ndarray) -> np.ndarray:
        """Denormalizes the action from [-1, 1] to `act_space`."""
        low = self.act_space.low
        high = self.act_space.high
        return low + (high-low) / 2 * (norm_act+1)

    def predict(self, obs: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predicts the dimensional action based on a dimensional observation.

        Should work 1:1 with Stable-Baselines3's predict. It does not support
        recurrent policies.
        """
        # obs = self._normalize_obs(obs)
        act, norm_sta = self.policy.predict(obs, *args, **kwargs)
        act = self._denormalize_act(act)
        return act, norm_sta

    def adim_predict(self, obs_s: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predicts the adimensional action based on an adimensional observation.

        Should work with Stable-Baselines3's predict arguments. It does not
        support recurrent policies.
        """
        obs = self.context.to_adim_obs(obs_s)
        act, state = self.predict(obs, *args, **kwargs)
        # act_s = self.context.from_adim_act(act)
        return act, state
        return act_s, state

    def to_scaled(self, context: Context) -> "ScaledPolicy":
        """Returns a `ScaledPolicy` with new `context`."""
        context.base_vars = context.base_vars / self.context.base_vars
        scaled_pol = ScaledPolicy(self, context)
        return scaled_pol

    def save_policy(self, path: str):
        self.policy.save(path + "model")
        self.context.save(path + "context")


class ScaledPolicy:

    def __init__(self, dim_pol: DimensionalPolicy, context: Context):
        """Initialize a scaled dimensional policy from a dimensional policy.

        Parameters:
        -----------
        dim_pol: DimensionalPolicy
            the base dimensional policy
        context: Context
            the context of the scaled policy
        """
        self.dim_pol = dim_pol
        self.context = context

    def predict(self, obs: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predicts the dimensional action based on a dimensional observation.

        Should work 1:1 with Stable-Baselines3's predict. It does not support
        recurrent policies.
        """
        obs_s = self.context.to_adim_obs(obs)
        act_s, state = self.dim_pol.predict(obs_s, *args, **kwargs)
        act = self.context.from_adim_act(act_s)
        # return act_s, state
        return act, state


class PolicyTrainer:

    def __init__(self, algo, context, env, algo_params):
        self.model = algo(**algo_params)
        self.context = context
        self.env = env


    def train(self, save=False, *args, **kwargs) -> DimensionalPolicy:
        self.model.learn(*args, **kwargs)
        if save:
            self.save_policy("trained_policy")
        return DimensionalPolicy(self.model, self.context, self.env.observation_space, self.env.action_space)

    def retrainer(self, policy: DimensionalPolicy, env: Env) -> "PolicyTrainer":
        pass

class PolicyEvaluator:

    def __init__(self, policy: DimensionalPolicy, env):
        self.policy = policy
        self.policy.obs_space = env.observation_space
        self.policy.act_space = env.action_space
        self.env = env
        self.J = 0

    def compute_score(self):
        """Computes the score of the current timestamp (state and action)."""
        dJ = 1.0
        self.J += dJ

    def evaluate(self, n_steps: int, render=False, adim=False, **kwargs) -> float:
        """Evaluates the policy for `n_steps` steps."""
        obs = self.env.reset()[0]
        self.J = 0
        for ind in range(n_steps):
            action, states = self.policy.predict(obs, **kwargs)
            out = self.env.step(action)
            if len(out) == 4:
                obs, reward, done, info = out
            elif len(out) == 5:
                obs, reward, done, info, _ = out

            if adim:
                obs2 = self.policy.context.to_adim_obs(obs)
                action2 = self.policy.context.to_adim_act(action)
            else:
                obs2 = obs
                action2 = action
                
            # print(action2)
            self.compute_score(obs2, action2)
            if render:
                self.env.render_mode = 'human'
            else:
                self.env.render_mode = 'rgb_array'

        return self.J/n_steps