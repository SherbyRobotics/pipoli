from abc import abstractmethod, ABC
import numpy as np

from typing import Callable, Union#, override


class Dimension:
    """Represents the units of quantity in terms of their dimensions.
    
    Example:
    --------
    >>> M = Dimension([1, 0, 0])  # a kilogram (kg)
    >>> L = Dimension([0, 1, 0])  # a meter (m)
    >>> T = Dimension([0, 0, 1])  # a second (s)
    >>> F = M * L / T**2  # a Newton (N)
    >>> km = 1000 * L  # a kilometer
    """

    def __init__(self, powers: np.ndarray, factors: np.ndarray | float=1):
        """Initialize a dimension.

        Parameters:
        -----------
        powers: np.ndarray
            the power of each base dimension
        """
        self.powers = np.array(powers)
    
    def __str__(self):
        return f"Dimension({self.powers})"
    
    def __repr__(self):
        return str(self)
    
    def __mul__(self, other: Union["Dimension", float, np.ndarray]) -> "Dimension":
        if isinstance(other, Dimension):
            return Dimension(self.powers + other.powers)
        elif isinstance(other, (float, np.ndarray)):
            return Dimension(self.powers + np.log10(other))
        else:
            return NotImplemented

    def __truediv__(self, other: Union["Dimension", float, np.ndarray]) -> "Dimension":
        if isinstance(other, Dimension):
            return Dimension(self.powers - other.powers)
        elif isinstance(other, (float, np.ndarray)):
            return Dimension(self.powers - np.log10(other))
        else:
            return NotImplemented
    
    def __pow__(self, power: float | np.ndarray) -> "Dimension":
        return Dimension(self.powers * power)


class NewContext:

    def __init__(self,
        base_dimensions: list[Dimension],
        symbols: list[str],
        dimensions: dict[str, Dimension],
        values: dict[str, float]
    ):
        """Initialize a context.

        Parameters:
        -----------
        base_dimensions: list[Dimension]
            the base dimensions of the units of the system
        symbols: list[str]
            the symbols of the quantities of the system
        dimensions: dict[str, Dimension]
            the dimensions of each quantity of the system
        values: dict[str, float]
            the value of each quantity of the system

        """
        self.base_dimensions = base_dimensions
        self.symbols = symbols
        self.dimensions = dimensions
        self.values = values
    
    def __str__(self):
        return f"Context({self.base_dimensions}, {self.symbols}, {self.dimensions}, {self.values})"

    def __repr__(self):
        return str(self)

    def _compute_factor(self, dimension: Dimension, Binv: np.ndarray, values: np.ndarray) -> float:
        mi = -Binv @ dimension.powers
        factor = np.prod(values**mi)

        print(dimension, mi, factor)
        
        return factor
    
    def make_transforms(self, dimensions: list[Dimension], base: list[str]) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        if not all(b in self.symbols for b in base):
            raise ValueError(f"all symbols in `base` ({base}) must be in the context ({self.symbols})")
        
        values = np.array([self.values[b] for b in base])

        dims_base = [self.dimensions[b].powers for b in base]
        B = np.vstack(dims_base).T

        if np.linalg.matrix_rank(B) != len(self.base_dimensions):
            return ValueError("the base doesn't span the dimension space")
        
        Binv = np.linalg.inv(B)

        factors = np.zeros(len(dimensions))
        for i, dim in enumerate(dimensions):

            if len(dim.powers) != len(self.base_dimensions):
                raise ValueError(f"the number of component of the dimension should be the same as the number of base dimensions")

            factors[i] = self._compute_factor(dim, Binv, values)

        def to_adim(x: np.ndarray) -> np.ndarray:
            return x * factors

        def from_adim(x: np.ndarray) -> np.ndarray:
            return x / factors

        return to_adim, from_adim


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
        self.base_vars = np.array(base_vars)
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
    
    #@override
    def action(self, obs: np.ndarray) -> np.ndarray:
        """Returns the dimensional action corresponding to the dimensional observation."""
        act = self.policy.action(obs)
        return act

    def adim_action(self, obs_s: np.ndarray) -> np.ndarray:
        """Returns the adimensional action corresponding to the adimensional observation."""
        obs = self.context.to_adim_obs(obs_s)
        act = self.action(obs)
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

    #@override
    def action(self, obs: np.ndarray) -> np.ndarray:
        """Returns the dimensional action corresponding to the dimensional observation."""
        obs_s = self.context.to_adim_obs(obs)
        act_s = self.dim_pol.adim_action(obs_s)
        act = self.context.from_adim_act(act_s)
        return act
