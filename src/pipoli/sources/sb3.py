from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.wrappers import RescaleAction, RescaleObservation
import numpy as np
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env

from pipoli.core import Policy

from typing import Callable, Optional, Any, Type


Model = A2C | DDPG | DQN | PPO | SAC | TD3
Algo = Type[Model]


class SB3Policy(Policy):

    def __init__(self,
        model: Model,
        model_obs_space: Box,
        model_act_space: Box,
        env_obs_space: Box,
        env_act_space: Box,
        predict_kwargs: Optional[dict[str, Any]] = None
    ):
        """Initializes a policy backed by a Stable Baselines 3 model.

        It does not support recurrent policies.
        
        Parameters:
        -----------
        model: Model
            a trained Stable Baselines 3 model
        model_obs_space: Box
            the model's observation space
        model_act_space: Box
            the model's action space
        env_obs_space: Box
            the environment's observation space
        env_act_space: Box
            the environment's action space
        predict_kwargs: Optional[dict[str, Any]]
            arguments to be passed to the model's `predict()` calls
        """
        self.model = model
        self.model_obs_space = model_obs_space
        self.model_act_space = model_act_space
        self.env_obs_space = env_obs_space
        self.env_act_space = env_act_space

        self.predict_kwargs = {} if predict_kwargs is None else predict_kwargs

    def _env_to_model_obs(self, env_obs):
        model_low = self.model_obs_space.low
        model_high = self.model_obs_space.high
        env_low = self.env_obs_space.low
        env_high = self.env_obs_space.high

        return model_low + (model_high - model_low) / (env_high - env_low) * (env_obs - env_low)

    def _model_to_env_act(self, model_act):
        model_low = self.model_act_space.low
        model_high = self.model_act_space.high
        env_low = self.env_act_space.low
        env_high = self.env_act_space.high

        return env_low + (env_high - env_low) / (model_high - model_low) * (model_act - model_low)

    def action(self, obs: np.ndarray) -> np.ndarray:
        """Returns the action corresponding to the observation."""
        model_obs = self._env_to_model_obs(obs)
        model_act, _ = self.model.predict(model_obs, **self.predict_kwargs)
        act = self._model_to_env_act(model_act)

        return act


class SB3PolicyTrainer:

    def __init__(self,
        algo: Algo,
        env_fn: Callable[[Any, ...], Env],
        model_obs_space: Optional[Box] = None,
        model_act_space: Optional[Box] = None,
        algo_params: Optional[dict[str, Any]] = None,
        env_params: Optional[dict[str, Any]] = None
    ):
        self.algo = algo
        self.env_fn = env_fn
        self.model_obs_space = model_obs_space
        self.model_act_space = model_act_space
        self.algo_params = {} if algo_params is None else algo_params
        self.env_params = {} if env_params is None else env_params

    def _wrap_env(self, env):
        if self.model_obs_space is not None:
            env = RescaleObservation(env, self.model_obs_space.low, self.model_obs_space.high)

        if self.model_act_space is not None:
            env = RescaleAction(env, self.model_act_space.low, self.model_act_space.high)

        return env

    def optimize_hyperparams(self, *args, **kwargs):
        raise NotImplementedError()

    def train(self, *args, save=False, **kwargs) -> SB3Policy:
        vec_env = make_vec_env(
            self.env_fn,
            n_envs=1, 
            env_kwargs=self.env_params,
            wrapper_class=self._wrap_env,
            # TODO add all keywords
        )
        model = self.algo("MlpPolicy", vec_env, **self.algo_params).learn(*args, **kwargs)

        # if save:
        #     self.save_policy("trained_policy")

        env_obs_space = vec_env.unwrapped.observation_space
        env_act_space = vec_env.unwrapped.action_space

        return SB3Policy(
            model, 
            model_obs_space=self.model_obs_space or env_obs_space,
            model_act_space=self.model_act_space or env_act_space,
            env_obs_space=env_obs_space,
            env_act_space=env_act_space,
            predict_kwargs=dict(deterministic=True)  # TODO more keywords
        )
    
    def save(self, *args, **kwargs):
        raise NotImplementedError()

    def continue_training(self, sb3_policy: SB3Policy, timesteps) -> SB3Policy:
        # TODO handle info of more training?
        raise NotImplementedError()
