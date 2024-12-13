from abc import abstractmethod, ABC
import numpy as np

from typing import Callable, override


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
        return np.linalg.norm(self.base_vars - other.base_vars)
    
    def adim_compare(self, other: "Context") -> bool:
        # return the angle between the two vectors
        return np.arccos(np.dot(self.base_vars, other.base_vars) / (np.linalg.norm(self.base_vars) * np.linalg.norm(other.base_vars)))


class Policy(ABC):
    """A generic policy interface."""

    @abstractmethod
    def action(self, obs: np.ndarray) -> np.ndarray:
        """Returns an action given an observation."""
        ...


class DimensionalPolicy(Policy):

    def __init__(self, policy: Policy, context: Context):
        """Initialize a dimensional policy from a source policy.

        Parameters:
        -----------
        policy: Policy
            a source policy
        context: Context
            the context of this policy
        """
        self.context = context
        self.policy = policy

    def __rshift__(self, context: Context) -> "ScaledPolicy":
        return self.to_scaled(context)
    
    @override
    def action(self, obs: np.ndarray) -> np.ndarray:
        """Returns the dimensional action corresponding to the dimensional observation."""
        act = self.policy.action(obs)
        return act

    def adim_action(self, obs_s: np.ndarray) -> np.ndarray:
        """Returns the adimensional action corresponding to the adimensional observation."""
        obs = self.context.to_adim_obs(obs_s)
        act = self.predict(obs, *args, **kwargs)
        act_s = self.context.from_adim_act(act)
        return act_s

    def to_scaled(self, context: Context) -> "ScaledPolicy":
        """Returns a `ScaledPolicy` with new `context`."""
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

    @override
    def action(self, obs: np.ndarray) -> np.ndarray:
        """Returns the dimensional action corresponding to the dimensional observation."""
        obs_s = self.context.to_adim_obs(obs)
        act_s = self.dim_pol.adim_predict(obs_s, *args, **kwargs)
        act = self.context.from_adim_act(act_s)
        return act
