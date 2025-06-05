"""Usefull functions used to evaluate a policy."""

from itertools import product

import numpy as np
from gymnasium import Env

from pipoli.core import Context, Dimension, Policy

from typing import Callable, Any, Optional


def linsweep_scale_contexts(
    original_context: Context,
    sweep_desc: dict[str, tuple[float, float, int]]
) -> list[Context]:
    """Returns a list of linearly scaled contexts from `original_context` in
    the space described by `sweep_desc`.

    The symbols in the sweep description must form a valid base to scale the context.

    Example sweep description for a pendulum with base `["m", "l", "g"]`:
    ```python
    sweep_desc = dict(
        m = (0.5, 1.5, 10),  # 10 masses in the interval [0.5, 1.5]
        l = (1, 10, 100),    # 100 lengths in the interval [1, 10]
        g = (9.81, 9.81, 1), # set gravity to 9.81
    )
    ```
    """
    base = tuple(sweep_desc.keys())
    ranges = [np.linspace(low, up, num) for low, up, num in sweep_desc.values()]

    return [original_context.scale_to(base, base_values) for base_values in product(*ranges)]


def linsweep_change_contexts(
    original_context: Context,
    sweep_desc: dict[str, tuple[float, float, int]]
) -> list[Context]:
    """Returns a list of linearly changed contexts from `original_context` in
    the space described by `sweep_desc`.

    There can be any symbol from the original context in the sweep description.
    The only the symbols in the sweep description will be changed, all the others
    will be equal to the ones in the original context.

    Example sweep description for a pendulum:
    ```python
    sweep_desc = dict(
        m = (0.5, 1.5, 10),  # 10 masses in the interval [0.5, 1.5]
        l = (1, 10, 100),    # 100 lengths in the interval [1, 10]
    )
    ```
    """
    symbols = tuple(sweep_desc.keys())
    ranges = [np.linspace(low, up, num) for low, up, num in sweep_desc.values()]
    
    return [original_context.change(**dict(zip(symbols, values))) for values in product(*ranges)]


def record_episode(
    policy: Policy,
    env: Env,
    fn: Optional[Callable[[Any, tuple[Any, float, bool, bool, dict[str, Any]], int], Any]] = None,
    fn0: Optional[Callable[[Any, dict[str, Any]], Any]] = None,
    reduce_fn: Optional[Callable[[list[Any]], Any]] = None,
    max_steps: Optional[int] = None,
) -> tuple[Any, list[Any]]:
    """Records every step of the `policy` in the `env` in the episode.

    `env.close()` is **not** called at the end.
    
    `fn(act, (obs, reward, term, trunc, info), step)` can be supplied to modify what is recorded after each `env.step(act)` call.
    The `act` argument is the action that caused the `(obs, reward, term, trunc, info)` on step `step`.
    If `None`, `fn` returns `act, (obs, reward, term, trunc, info)`.

    `fn0(obs, info)` can be supplied to modify what is recorded after `env.reset()`.
    If `None`, `fn0` returns `obs, info`.

    `reduce_fn(datas)` can be supplied to reduce all the recorded steps (summing the reward for example).
    If `None`, `reduce_fn` returns `None`.

    If supplied, `max_step` will stop the recording once reached.
    """
    if max_steps is None:
        max_steps = np.inf

    if fn is None:
        fn = lambda act, env_step, _step: (act, env_step)

    if fn0 is None:
        fn0 = lambda obs, info: (obs, info)
    
    if reduce_fn is None:
        reduce_fn = lambda _datas: None

    obs, info = env.reset()
    data0 = fn0(obs, info)

    term = trunc = False
    step = 0
    datas = []

    while not (term or trunc) and step < max_steps:
        act = policy.action(obs)
        obs, reward, term, trunc, info = env.step(act)

        datas.append(fn(act, (obs, reward, term, trunc, info), step))

        step += 1

    reduced = reduce_fn(datas)
    return data0, datas, reduced


def record_sweep(
    all_contexts: list[Context],
    make_policy: Callable[[Context], Policy],
    make_env: Callable[[Context], Env],
    context_fn: Optional[Callable[[Context], Any]] = None,
    fuse_fn: Optional[Callable[[Any, tuple[Any, list[Any], Any]], Any]] = None,
    ep_fn: Optional[Callable[[Any, tuple[Any, float, bool, bool, dict[str, Any]], int], Any]] = None,
    ep_fn0: Optional[Callable[[Any, dict[str, Any]], Any]] = None,
    ep_reduce_fn: Optional[Callable[[list[Any]], Any]] = None,
    ep_max_steps: Optional[int] = None,
) -> list[tuple[Any, Any]]:
    """Records every step of each policy in their environment in function of their context.
    
    `make_policy(context)` returns the policy that will be recorded.
    
    `make_env(context)` returns the environment in which the policy will be recorded.
    
    `context_fn(context)` can be supplied to modify what is recorded about the context.
    If `None`, it returns the `context`.
    
    `fuse_fn(context_record, episode_record)` can be supplied to modify how information
    recorded the context will be concatenated with information recorded during the episode.
    If `None`, it returns `context_record, episode_record`.
    
    All `ep_*` arguments are passed directly to `record_episode()` internally.
    See its documentation for details.
    """
    if context_fn is None:
        context_fn = lambda context: context
    
    if fuse_fn is None:
        fuse_fn = lambda context_record, episode_record: (context_record, episode_record)

    records = [None] * len(all_contexts)

    for i, context in enumerate(all_contexts):
        policy = make_policy(context)
        env = make_env(context)
        
        context_record = context_fn(context)
        episode_record = record_episode(
            policy,
            env,
            ep_fn,
            ep_fn0,
            ep_reduce_fn,
            ep_max_steps
        )
        env.close()

        records[i] = fuse_fn(context_record, episode_record)

    return records