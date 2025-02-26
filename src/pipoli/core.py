from abc import abstractmethod, ABC
from bisect import bisect_left

import numpy as np

from typing import Callable, Union


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

    def __init__(self, powers: Union[np.ndarray, list[float]]):
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
        elif isinstance(other, (int, float, np.ndarray)):
            return Dimension(self.powers + np.log10(other))
        else:
            return NotImplemented
    
    def __rmul__(self, other: Union["Dimension", float, np.ndarray]) -> "Dimension":
        return self * other

    def __truediv__(self, other: Union["Dimension", float, np.ndarray]) -> "Dimension":
        if isinstance(other, Dimension):
            return Dimension(self.powers - other.powers)
        elif isinstance(other, (int, float, np.ndarray)):
            return Dimension(self.powers - np.log10(other))
        else:
            return NotImplemented
    
    def __rtruediv__(self, other: Union["Dimension", float, np.ndarray]) -> "Dimension":
        return self**-1 * other
    
    def __pow__(self, power: Union[float, np.ndarray]) -> "Dimension":
        return Dimension(self.powers * power)


class Context:
    """Represents the context of a system.
    
    A context contains all the quantities that parametrizes a system.
    A quantity is described by its symbol, its dimension (read units) and its
    value.
    
    The context allows to keep track of how the quantities change when they are
    seen in a similar system with different parameters.

    Example:
    --------
    >>> context = Context(
    ...     base_dimension,
    ...     *zip(
    ...         ("sym1", D, 42),
    ...         ("sym2", D, 37),
    ...     )
    ... )
    """

    def __init__(self,
        base_dimensions: list[Dimension],
        symbols: list[str],
        dimensions: list[Dimension],
        values: list[float]
    ):
        """Initialize a context.

        Parameters:
        -----------
        base_dimensions: list[Dimension]
            the base dimensions of the units of the system
        symbols: list[str]
            the symbols of the quantities of the system
        dimensions: list[Dimension]
            the dimensions of each quantity of the system
        values: list[float]
            the value of each quantity of the system

        """
        self._assert_well_formed(symbols, dimensions, values)
        
        # reorder the symbols
        all_sorted = zip(
            *sorted(
                zip(symbols, dimensions, values),
                key=lambda t: t[0]
            )
        )
        symbols = next(all_sorted)
        dimensions = next(all_sorted)
        values = np.array(next(all_sorted))


        # formal definition of a Context
        self.base_dimensions = base_dimensions
        self.symbols = symbols
        self.dimensions = dimensions
        self.values = values
    
    @staticmethod
    def _assert_well_formed(symbols, dimensions, values):
        symbols_set = set(symbols)

        if len(symbols) != len(symbols_set):
            raise ValueError("a symbol is repeated")

        if len(dimensions) != len(symbols):
            raise ValueError(
                f"the number of dimensions ({len(dimensions)}) doesn't match the number of symbols ({len(symbols)})"
            )
        
        if len(values) != len(symbols):
            raise ValueError(
                f"the number of values ({len(values)}) doesn't match the number of symbols ({len(symbols)})"
            )
    
    def __str__(self):
        dimensions = dict(zip(self.symbols, self.dimensions))
        return f"Context({self.base_dimensions}, {self.symbols}, {self.dimensions}, {self.values})"

    def __repr__(self):
        return str(self)
    
    def value(self, symbol):
        index = bisect_left(self.symbols, symbol)
        return self.values[index]
    
    def dimension(self, symbol):
        index = bisect_left(self.symbols, symbol)
        return self.dimensions[index]

    @staticmethod
    def _compute_factor(dimension: Dimension, Binv: np.ndarray, values: np.ndarray) -> float:
        mi = -Binv @ dimension.powers
        factor = np.prod(values**mi)
        
        return factor
    
    def _compute_Binv(self, base):
        dims_base = [self.dimension(b).powers for b in base]
        B = np.vstack(dims_base).T

        if np.linalg.matrix_rank(B) != len(self.base_dimensions):
            return ValueError("the base doesn't span the dimension space")

        return np.linalg.inv(B)

    def make_transforms(
        self,
        dimensions: list[Dimension],
        base: list[str]
    ) -> tuple[
        Callable[[np.ndarray], np.ndarray],
        Callable[[np.ndarray], np.ndarray]
    ]:
        """Returns two function to adimensionalize and re-dimensionalize the dimensions using the base."""
        if not all(b in self.symbols for b in base):
            raise ValueError(f"all symbols in `base` ({base}) must be in the context ({self.symbols})")
        
        values = np.array([self.value(b) for b in base])

        Binv = self._compute_Binv(base)

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
    
    def scale_to(self, base: list[str], values: list[float]) -> "Context":
        """Returns a new context with its values scaled to `values` according to the `base`."""
        og_values = np.array([self.value(b) for b in base])
        Binv = self._compute_Binv(base)

        factors = np.zeros(self.values.shape)
        for i, dim in enumerate(self.dimensions):
            to_adim = self._compute_factor(dim, Binv, og_values)
            to_new = self._compute_factor(dim, Binv, values)
            factors[i] = to_adim / to_new
        
        new_values = self.values * factors

        return Context(self.base_dimensions, self.symbols, self.dimensions, new_values)
    
    def _assert_compatible(self, other):
        same_symbols = self.symbols == other.symbols
        # TODO complete checks
        same_base_dimensions = ...
        sames_dimensions = ...

        if not (same_symbols and same_base_dimensions and sames_dimensions):
            raise ValueError(f"contexts {self} and {other} are incompatible")

    def cosine_similarity(self, other: "Context") -> float:
        self._assert_compatible(other)

        return self.values @ other.values / np.linalg.norm(self.values) / np.linalg.norm(other.values)
    
    def adimensional_distance(self, other: "Context", base: list[str]) -> float:
        self._assert_compatible(other)

        self_to_adim, _ = self.make_transforms(self.dimensions, base)
        other_to_adim, _ = other.make_transforms(other.dimensions, base)

        self_adim_values = self_to_adim(self.values)
        other_adim_values = other_to_adim(other.values)

        return np.linalg.norm(self_adim_values - other_adim_values)

    def euclidian_distance(self, other: "Context") -> float:
        self._assert_compatible(other)

        return np.linalg.norm(self.values - other.values)

class Policy(ABC):
    """A generic policy interface."""

    @abstractmethod
    def action(self, obs: np.ndarray) -> np.ndarray:
        """Returns an action given an observation."""
        ...


class DimensionalPolicy(Policy):

    def __init__(self,
        policy: Policy,
        context: Context,
        obs_dims: list[Dimension],
        act_dims: list[Dimension]
    ):
        """Initialize a dimensional policy from a source policy.

        Parameters:
        -----------
        policy: Policy
            a source policy
        context: Context
            the context of this policy
        obs_dims: list[Dimension]
            the dimension of each input of the policy
        act_dims: list[Dimension]
            the dimension of each output of the policy
        """
        self.context = context
        self.policy = policy
        self.obs_dims = obs_dims
        self.act_dims = act_dims

    def action(self, obs: np.ndarray) -> np.ndarray:
        """Returns the dimensional action corresponding to the dimensional observation."""
        act = self.policy.action(obs)
        return act

    def to_scaled(self, context: Context, base: list[str]) -> "ScaledPolicy":
        """Returns a `ScaledPolicy` with new `context`."""
        scaled_pol = ScaledPolicy(self, context, base)
        return scaled_pol

    def save_policy(self, path: str):
        # self.policy.save(path + "model")
        # self.context.save(path + "context")
        raise NotImplementedError("todo")


class ScaledPolicy:

    def __init__(self,
        dim_pol: DimensionalPolicy,
        context: Context,
        base: list[str]
    ):
        """Initialize a scaled dimensional policy from a dimensional policy.

        Parameters:
        -----------
        dim_pol: DimensionalPolicy
            the base dimensional policy
        context: Context
            the context of the scaled policy
        base: list[str]
            the symbols used to scale the policy
        """
        self.dim_pol = dim_pol
        self.context = context
        self.base = base
        
        obs_dims = dim_pol.obs_dims
        act_dims = dim_pol.act_dims

        _                  , dim_pol_obs_from_adim = dim_pol.context.make_transforms(obs_dims, base)
        dim_pol_act_to_adim, _                     = dim_pol.context.make_transforms(act_dims, base)
        self_obs_to_adim   , _                     = context.make_transforms(obs_dims, base)
        _                  , self_act_from_adim    = context.make_transforms(act_dims, base)

        def scale_obs(obs):
            obs_s = self_obs_to_adim(obs)
            return dim_pol_obs_from_adim(obs_s)
        
        def unscale_act(act):
            act_s = dim_pol_act_to_adim(act)
            return self_act_from_adim(act_s)
        
        self._scale_obs = scale_obs
        self._unscale_act = unscale_act

    def action(self, obs: np.ndarray) -> np.ndarray:
        """Returns the dimensional action corresponding to the dimensional observation."""
        obs_scaled = self._scale_obs(obs)
        act_scaled = self.dim_pol.action(obs_scaled)
        act = self._unscale_act(act_scaled)
        return act


if __name__ == "__main__":

    M = Dimension([1, 0, 0])
    L = Dimension([0, 1, 0])
    T = Dimension([0, 0, 1])

    dim_tau = M * L**2 * 1/T**2

    context = Context(
        [M, L, T],
        *zip(
            ("m", M, 2),
            ("g", L/T**2, 7),
            ("l", L, 3),
            ("taumax", M*L**2/T**2, 19)
        )
    )

    basis = ["m", "g", "l"]
    to_adim, to_dim = context.make_transforms([dim_tau], basis)

    assert to_adim(1) == 1/42
    assert to_dim(1) == 42

    print("dimension and context ok")

    class DummyPolicy(Policy):

        def action(self, obs):
            return obs * 11
    
    dim_pol = DimensionalPolicy(DummyPolicy(), context, [L], [1/M/L])

    assert dim_pol.action(np.array([1])) == np.array([11])

    print("dimensional policy ok")

    import copy

    new_context = Context(
        [M, L, T],
        *zip(
            ("m", M, 2),
            ("g", L/T**2, 7),
            ("l", L, 5),
            ("taumax", M*L**2/T**2, 19)
        )
    )

    scale_pol = dim_pol.to_scaled(new_context, basis)

    np.testing.assert_almost_equal(scale_pol.action(np.array([1])), np.array([1/5 * 3 * 11 * 1/10 * 6]))
   
    print("scaled policy ok")

    np.testing.assert_almost_equal(context.cosine_similarity(context), 1)
    np.testing.assert_almost_equal(
        context.cosine_similarity(new_context),
        (2*2 + 7*7 + 3*5 + 19*19) / np.sqrt(2*2 + 7*7 + 3*3 + 19*19) / np.sqrt(2*2 + 7*7 + 5*5 + 19*19)
    )

    print("cosine similarity ok")

    assert context.adimensional_distance(context, basis) == 0
    assert context.adimensional_distance(new_context, basis) == abs(19 / 2 / 7 / 3 - 19 / 2 / 7 / 5)

    print("adimensional distance ok")

    scaled_context = new_context.scale_to(["m","g","l"], [23, 29, 31])

    assert scaled_context.value("taumax") == 19 / 2 / 7 / 5 * 23 * 29 * 31

    print("context scaling ok")

