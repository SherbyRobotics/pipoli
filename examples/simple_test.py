import sys
sys.path.append("/home/fp/Dev/pipoli/src")
print(sys.path)

import numpy as np
import gymnasium as gym
import pygame


class Pendulum(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 100}

    def __init__(self, m, g, l, taumax, tf, dt=0.01, render_mode="rgb_array"):
        self.m = m
        self.g = g
        self.l = l
        self.taumax = taumax
        self.tf = tf
        self.dt = dt
    
        self.observation_space = gym.spaces.Box(low=np.array([-2*np.pi, -10]), high=np.array([2*np.pi, 10]), shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-taumax, high=taumax, shape=(1,), dtype=np.float32)

        self.t = 0
        self.theta = 0
        self.thetadot = 0

        self.render_mode = render_mode
        self.clock = None
        self.display = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.t = 0
        self.theta = 0
        self.thetadot = 0

        if self.render_mode == "human":
            self.render()

        return np.array([self.theta, self.thetadot]), {}
    
    def step(self, action):
        tau, = action
        self.t += self.dt

        # theta = theta0 + thetadot0 * dt + thetaddot * dt**2 / 2 = theta0 + delta_theta
        # thetadot = thetadot0 + thetaddot * dt = thetadot0 + delta_thetadot
        thetaddot = -self.g / self.l * np.sin(self.theta) + tau / self.m / self.l**2
        delta_thetadot = thetaddot * self.dt
        delta_theta = self.thetadot * self.dt + 0.5 * thetaddot * self.dt**2

        self.theta += delta_theta
        self.thetadot += delta_thetadot

        position_cost = 1.0 * (np.pi - self.theta)**2
        velocity_cost = 0.1 * (0 - self.thetadot)**2
        tau_cost = tau**2 * 0.001

        cost = position_cost + velocity_cost + tau_cost
        cost *= 0.1

        info = dict(position_cost=position_cost, velocity_cost=velocity_cost, tau_cost=tau_cost, cost=cost, tau=tau)

        # if np.isclose(self.theta, np.pi) and np.isclose(self.thetadot, 0):
        #     trunc = True
        # else:
        #     trunc = False
        
        trunc = self.t > self.tf

        if self.render_mode == "human":
            self.render()

        return np.array([self.theta, self.thetadot], dtype=np.float32), -cost, False, trunc, info

    def render(self):
        size = pygame.Vector2(500)
    
        if self.display is None and self.render_mode == "human":
            pygame.display.init()
            self.display = pygame.display.set_mode(size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        surface = pygame.Surface(size)

        center = size / 2

        l = min(self.l * 50, 450)
        m = min(self.m * 10, 50)
        theta = np.degrees(self.theta) + 90
        pendulum = pygame.Vector2.from_polar((l, theta))

        pygame.draw.line(surface, "white", center, center + pendulum, 3)
        pygame.draw.circle(surface, "red", center + pendulum, m)

        if self.render_mode == "human":
            self.display.blit(surface, surface.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])

        return pygame.surfarray.pixels2d(surface)
    
    def close(self):
        pygame.display.quit()
        self.display = None
        self.clock = None


from stable_baselines3 import PPO, SAC
from pipoli.sources.sb3 import SB3Policy, SB3PolicyTrainer

m = 1
g = 9.8
l = 2
taumax = 200
tf = 10

env = Pendulum(m, g, l, taumax, tf)
train_env = gym.wrappers.RescaleAction(env, -1, 1)
train_env = gym.wrappers.RescaleObservation(train_env, -1, 1)

# model = SAC("MlpPolicy", train_env, verbose=1).learn(10000)
# model.save("model")
# model = SAC.load("sandbox/model")

# sb3_pol = SB3Policy(
#     model,
#     train_env.observation_space,
#     train_env.action_space,
#     env.observation_space,
#     env.action_space,
#     dict(deterministic=True)
# )

trainer = SB3PolicyTrainer(
    PPO,
    Pendulum,
    model_obs_space=None,#train_env.observation_space,
    model_act_space=None,#train_env.action_space,
    env_params=dict(m=m, g=g, l=l, taumax=taumax, tf=tf)
)

sb3_pol = trainer.train(1000)

from pipoli.core import DimensionalPolicy, Context

obs_transform = lambda base: np.array([1, np.sqrt(base[0]*base[1]/base[2])])
act_transform = lambda base: np.array([1/base[2]])

context1 = Context([m, l, taumax], obs_transform, act_transform)

dim_pol = DimensionalPolicy(sb3_pol, context1)  # first pendulum policy

og_env = Pendulum(m, g, l, taumax, tf, render_mode="human")

trunc = False
obs, _ = og_env.reset()
# breakpoint()
while not trunc:
    act = dim_pol.action(obs)
    obs, _, _, trunc, info = og_env.step(act)
    print(act, obs, trunc, info)

# New pendulum
m2 = 2
l2 = 1
taumax2 = 100

context2 = Context([m2, l2, taumax2], obs_transform, act_transform)

scaled_pol = dim_pol >> context2

eval_env = Pendulum(m2, g, l2, taumax2, tf, render_mode="human")

# print(evaluate_policy(model, eval_env))

trunc = False
obs, _ = eval_env.reset()
while not trunc:
    act = scaled_pol.action(obs)
    obs, _, _, trunc, info = eval_env.step(act)
    print(act, obs, trunc, info)
