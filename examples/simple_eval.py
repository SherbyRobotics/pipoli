from pathlib import Path
import sys
sys.path.append(f"{str(Path('.').resolve())}/src")
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


from pipoli.core import Policy

class PendulumPID(Policy):

    def action(self, obs):
        theta, thetadot = obs
        tau = 55 * (np.pi - theta) - 30 * thetadot
        return np.array([tau])


from pipoli.core import Dimension, Context, DimensionalPolicy
from pipoli.evaluation import record_episode

M = Dimension([1, 0, 0])
L = Dimension([0, 1, 0])
T = Dimension([0, 0, 1])
Unit = Dimension([0, 0, 0])

original_context = Context.from_quantities(
    m = 1.5         | M,
    l = 3.5         | L,
    g = 9.81        | L / T**2,
    taumax = 100    | M * L**2 / T**2,
    tf = 3          | T,
)

original_policy = DimensionalPolicy(PendulumPID(), original_context, [Unit, 1 / T], [M * L**2 / T**2])

def make_env(context):
    m = context.value("m")
    l = context.value("l")
    g = context.value("g")
    taumax = context.value("taumax")
    tf = context.value("tf")

    return Pendulum(m, g, l, taumax, tf, render_mode="human")


env = make_env(original_context)
data, datas, reduction = record_episode(original_policy, env)
env.close()

# print(datas)

from pipoli.evaluation import linsweep_scale_contexts, linsweep_change_contexts, record_sweep

sweep_desc = dict(
    m = (1, 5, 2),
    l = (2, 4, 2),
    g = (0.981, 98.1, 2),
)
base = [*sweep_desc.keys()]

all_contexts = linsweep_scale_contexts(original_context, sweep_desc)
all_contexts = linsweep_change_contexts(original_context, sweep_desc)


def make_policy(context):
    return original_policy.to_scaled(context, base)

def ep_fn(_act, step_res, _step):
    _, rew, _, _, info = step_res
    # print(info)
    return rew

def ep_reduce_fn(rews):
    return sum(rews)


sweep = record_sweep(all_contexts, make_policy, make_env, ep_fn=ep_fn, ep_reduce_fn=ep_reduce_fn)

print(len(all_contexts))
print(sweep)

