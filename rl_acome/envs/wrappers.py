"""Define environment wrappers."""

import numpy as np
from rlberry.envs import Wrapper


class EvaluatingEnv(Wrapper):
    """
    Environment recording data from interaction to provide evaluation measures.

    Results of steps are stored, in order to compute perforcmance measures such
    as regret, sample efficiency, error... In addition to, provide methods to
    evaluate specific policies.

    Parameters:
    -----------
    env : gym.Env
        The environment to wrap.
    opti_policy : S or (S, A)  numpy.array or None, default None
        Optimal policy for environment, if provided.
    opti_Q : (S, A) numpy.array or None, default None
        Optimal Q-value function for environment, if provided.
    opti_V : S numpy.array or None, default None
        Optimal value function for environment, if provided. If opti_Q is
        provided, opti_V is computed (and overwritten) from opti_Q.
    opti_reward : S numpy.array or None, default None
        Maximum expected reward at each state, if provided.
    """

    def __init__(self, env, opti_policy=None, opti_Q=None,
                 opti_V=None, opti_reward=None):
        Wrapper.__init__(self, env)

        self.opti_pi = opti_policy
        self.opti_Q = opti_Q
        if opti_Q:
            self.opti_V = np.max(opti_Q, axis=-1)
        else:
            self.opti_V = opti_V
        self.opti_r = opti_reward

        self.S, self.A = env.observation_space.n, env.action_space.n

        self._init_values()

    def _init_values(self):
        """Set counters to 0."""
        self.sa = np.zeros((self.S, self.A), dtype=int)
        self.r = np.zeros((self.S, self.A))
        self.episodes_steps = [0]  # number of states per episode

    def step(self, action):
        """Execute action, and update statistics accordingly to results."""
        state, reward, done, info = self.env.step(action)

        self.sa[state, action] += 1
        self.r[state, action] += reward
        self.episodes_steps[-1] += 1

        if done:
            self.episodes_steps.append(0)

        return state, reward, done, info

    def estimate_optimal_reward(self, n_iter=20):
        """
        Estimate the expected reward for optimal policy.

        For each state-action pair, average reward over n_iter iterations.
        Overwrite self.opti_r. If an optimal policy wasn't provided, use the
        greedy policy from the optimal Q-value function. If not provided
        neither, raise an error.
        """
        if self.opti_pi:
            pi = self.opti_pi
        else:
            assert self.opti_Q, "No optimal policy nor Q function provided!"
            pi = np.zeros((self.S, self.A))
            pi[range(self.S), np.argmax(self.opti_Q, axis=-1)] = 1.

        env = self.env
        for s in range(self.S):
            a = np.random.choice(self.A, p=pi[s, :])
            R = 0
            for _ in range(n_iter):
                env.state = s
                _, r, _, _ = env.step(a)
                R += r
            self.opti_r[s] = R / n_iter

        return self.opti_r

    def value_error(self, value):
        """Return error of value w.r.t. optimal value."""
        assert self.opti_V, "Optimal value wasn't given!"

        return ((value - self.opti_V)**2).mean()
