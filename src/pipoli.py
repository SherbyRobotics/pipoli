import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines import A2C, DDPG, PPO, SAC, TD3

from typing import Callable


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


Policy = A2C | DDPG | PPO | SAC | TD3

class DimensionalPolicy:

    def __init__(self, policy: Policy, context: Context, obs_space: Box, act_space: Box):
        """Initialize a dimensional policy from a Stable-Baselines3 policy.

        Parameters:
        -----------
        policy: A2C | DDPG | PPO | SAC | TD3
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
        norm_obs = self._normalize_obs(obs)
        norm_act = self.policy.predict(norm_obs, *args, **kwargs)
        act = self._denormalize_act(norm_act)
        return act

    def adim_predict(self, obs_s: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predicts the adimensional action based on an adimensional observation.

        Should work with Stable-Baselines3's predict arguments. It does not
        support recurrent policies.
        """
        obs = self.context.from_adim_obs(obs_s)
        act = self.predict(obs, *args, **kwargs)
        act_s = self.context.to_adim_act(act)
        return act_s

    def to_scaled(self, context: Context) -> "ScaledPolicy":
        """Returns a `ScaledPolicy` with new `context`."""
        scaled_pol = ScaledPolicy(self, context)
        return scaled_pol


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
        act_s = self.dim_pol.adim_predict(obs_s, *args, **kwargs)
        act = self.context.from_adim_act(act_s)
        return act


class PolicyTrainer:

    def __init__(self, algo, context, env, algo_params={}):
        pass

    def train(self, *args, **kwargs) -> DimensionalPolicy:
        pass

    def retrainer(self, policy: DimensionalPolicy, env: Env) -> "PolicyTrainer":
        pass
