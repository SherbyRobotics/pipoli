from gymnasium import Env
from pipoli.core import DimensionalPolicy

class PolicyEvaluator:

    def __init__(self, policy: DimensionalPolicy, env: Env):
        self.policy = policy
        self.env = env
        self.J = 0

    def compute_score(self):
        """Computes the score of the current timestamp (state and action)."""
        dJ = 1.0
        self.J += dJ

    def evaluate(self, n_steps: int, render=False, **kwargs) -> float:
        """Evaluates the policy for `n_steps` steps."""
        obs = self.env.reset()[0]
        self.J = 0
        for _ in range(n_steps):
            action, states = self.policy.predict(obs, **kwargs)
            out = self.env.step(action)
            if len(out) == 4:
                obs, reward, done, info = out
            elif len(out) == 5:
                obs, reward, done, info, _ = out
            self.compute_score(obs, action)
            if render:
                self.env.render("human")

        return self.J/n_steps
