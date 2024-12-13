from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.wrappers import RescaleAction, RescaleObservation
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from pipoli.core import Policy

from typing import Optional, Any


Model = A2C | DDPG | DQN | PPO | SAC | TD3

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

        self.predict_kwargs = {} if kwargs is None else kwargs

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

    def __init__(self, algo, task, algo_params=None):
        self.algo = algo
        self.task = task
        self.algo_params = algo_params

    def _wrap_env(self, env):
        pass

    def train(self, save=False, *args, **kwargs) -> SB3Policy:
        self.model.learn(*args, **kwargs)
        if save:
            self.save_policy("trained_policy")
        return SB3Policy(self.model, self.context, self.env.observation_space, self.env.action_space)

    def retrainer(self, policy: SB3Policy, env: Env) -> "SB3PolicyTrainer":
        pass
