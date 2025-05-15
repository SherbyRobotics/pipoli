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


class Transform:
    """Represents the tranformation to go to the adimensional space and back."""

    def __init__(self, dims: list[Dimension], base_dims: list[Dimension]):
        """Initialize a Transform.

        Parameters:
        -----------
        dims: list[Dimension]
            the dimensions to simplify
        base_dims: list[Dimension]
            the dimensions used to simplify the `dims`
        """
        self._assert_all_same_number_of_base_dimension(dims, base_dims)

        self.dims = dims
        self.base_dims = base_dims
        
        B = np.vstack([bd.powers for bd in base_dims]).T

        self._assert_base_dims_span_all_dimension_space(B)

        Binv = np.linalg.inv(B)
        dims_mat = np.vstack([dim.powers for dim in dims]).T

        # the powers that simplifies the input dimensions with the base dimensions
        self.mi = -Binv @ dims_mat

    
    @staticmethod
    def _assert_all_same_number_of_base_dimension(dims, base_dims):
        if len(dims) < 1:
            raise ValueError("`dims` must be non empty")

        nb_base_dim = dims[0].powers.size
        for dim in dims:
            if dim.powers.size != nb_base_dim:
                raise ValueError("all the dimensions in `dims` must have the same number of base dimension")
        
        for dim in base_dims:
            if dim.powers.size != nb_base_dim:
                raise ValueError("all the dimensions in `base_dims` must have the same number of base dimension")
    
    @staticmethod
    def _assert_base_dims_span_all_dimension_space(B):
        if np.linalg.matrix_rank(B) != B.shape[0]:
            raise ValueError("the base dimensions must span the dimension space")
    
    def _compute_factors(self, base_values: np.ndarray) -> float:
        b = base_values.reshape(base_values.size, 1)
        factors = np.prod(b**self.mi, axis=0)
        
        return factors

    def to_adim(self, nat: np.ndarray, base: np.ndarray) -> np.ndarray:
        factors = self._compute_factors(base)
        return factors * nat

    def from_adim(self, adim: np.ndarray, base: np.ndarray) -> np.ndarray:
        factors = self._compute_factors(base)
        return adim / factors

    def make_to_adim(self, base: np.ndarray) -> np.ndarray:
        factors = self._compute_factors(base)

        def to_adim(nat):
            return nat * factors
        
        return to_adim

    def make_from_adim(self, base: np.ndarray) -> np.ndarray:
        factors = self._compute_factors(base)

        def from_adim(adim):
            return adim / factors
        
        return from_adim

    def translate_to(self, transform: "Transform"):
        raise NotImplementedError("todo maybe")
        return ...


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
        
        # sort the symbols for faster indexing
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
                f"the number of dimensions ({len(dimensions)}) must match the number of symbols ({len(symbols)})"
            )
        
        if len(values) != len(symbols):
            raise ValueError(
                f"the number of values ({len(values)}) must match the number of symbols ({len(symbols)})"
            )
    
    def __str__(self):
        quantities = ",\n    ".join((f"{s} = {v} {d}" for s, d, v in zip(self.symbols, self.dimensions, self.values)))
        return f"Context(\n    {self.base_dimensions},\n    {quantities}\n)"

    def __repr__(self):
        return str(self)
    
    def __iter__(self):
        return zip(self.symbols, self.dimensions, self.values)

    def _index_of(self, symbol):
        index = bisect_left(self.symbols, symbol)
        if self.symbols[index] != symbol:
            raise KeyError(f"symbol '{symbol}' not in context")
        return index
    
    def value(self, symbol):
        return self.values[self._index_of(symbol)]
    
    def dimension(self, symbol):
        return self.dimensions[self._index_of(symbol)]
    
    def _assert_base_in_symbols(self, base):
        if not all(b in self.symbols for b in base):
            raise ValueError(f"all symbols in `base` ({base}) must be in the context ({self.symbols})")

    def make_transforms(
        self,
        dimensions: list[Dimension],
        base: list[str]
    ) -> tuple[
        Callable[[np.ndarray], np.ndarray],
        Callable[[np.ndarray], np.ndarray]
    ]:
        """Returns two function to adimensionalize and re-dimensionalize the dimensions using the base."""
        self._assert_base_in_symbols(base)
        
        base_values = np.array([self.value(b) for b in base])
        base_dims = [self.dimension(b) for b in base]

        transform = Transform(dimensions, base_dims)

        to_adim = transform.make_to_adim(base_values)
        from_adim = transform.make_from_adim(base_values)

        return to_adim, from_adim

    def scale_to(self, base: list[str], values: list[float]) -> "Context":
        """Returns a new context with its values scaled to `values` according to the `base`."""
        self._assert_base_in_symbols(base)

        base_dims = [self.dimension(b) for b in base]
        transform = Transform(self.dimensions, base_dims)

        original_base_values = np.array([self.value(b) for b in base])
        adim_values = transform.to_adim(self.values, original_base_values)

        new_base_values = np.array(values)
        new_values = transform.from_adim(adim_values, new_base_values)

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
    
    def sample_around(self, scale_min, scale_max) -> "Context":
        scale = np.random.uniform(scale_min, scale_max)
        values = self.values * scale

        return Context(self.base_dimensions, self.symbols, self.dimensions, values)


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

    print("dimension operations ok")

    transform = Transform([dim_tau], [M, L/T**2, L])
    base_values = np.array([2, 7, 3])

    assert transform.to_adim(np.array([1]), base_values) == np.array([1/42])
    assert transform.from_adim(np.array([1]), base_values) == np.array([42])

    print("transform ok")

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

    np.testing.assert_almost_equal(scaled_context.value("taumax"), 19 / 2 / 7 / 5 * 23 * 29 * 31)

    print("context scaling ok")

    try:
        scaled_context.value("not in context")
    except KeyError as e:
        assert str(e) == '"symbol \'not in context\' not in context"'

        print("context wrong symbol value ok")
    
    try:
        scaled_context.dimension("not in context")
    except KeyError as e:
        assert str(e) == '"symbol \'not in context\' not in context"'

        print("context wrong symbol dimension ok")

