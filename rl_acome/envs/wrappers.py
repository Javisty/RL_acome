"""Define environment wrappers."""

import numpy as np
from rlberry.envs import Wrapper

from .utils import greedy_policy


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
    opti_gain : float or None, default None
        Optimal gain, if provided.
    """

    def __init__(self, env, opti_policy=None, opti_Q=None,
                 opti_V=None, opti_reward=None, opti_gain=None):
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
        Estimate the expected reward in each state for optimal policy.

        For each state-action pair, average reward over n_iter iterations.
        Overwrite self.opti_r. If an optimal policy wasn't provided, use the
        greedy policy from the optimal Q-value function. If not provided
        neither, raise an error.
        """
        if self.opti_pi is not None:
            pi = self.opti_pi
        else:
            assert self.opti_Q, "No optimal policy nor Q function provided!"
            pi = greedy_policy(self.opti_Q)

        env = self.env
        self.opti_r = np.zeros(self.S)
        for s in range(self.S):
            R = 0
            for _ in range(n_iter):
                env.state = s
                a = np.random.choice(self.A, p=pi[s, :])
                _, r, _, _ = env.step(a)
                R += r
            self.opti_r[s] = R / n_iter

        return self.opti_r

    def estimate_state_gain(self, policy, s0, traj_length=10000, burn_in=1000):
        """
        Estimate the gain of policy, starting from state s0.

        Run policy over traj_length steps, and return the average reward
        collected after a burn-in period.

        Parameters:
        -----------
        policy : (S, A) numpy.array
            Policy to use.
        s0 : 0 <= integer < self.S
            Initial state.
        traj_length : integer > 0, default 10000.
            Number of steps to execute under policy.
        burn_in : 0 <= integer < traj_length, default 1000
            Number of burn-in steps to make, before collecting rewards.

        Output:
        -------
        gain : float
            Estimated gain of policy starting from s0.
        """
        assert burn_in < traj_length, "Burn-in length must be less than total!"

        env = self.env
        env.state, s = s0, s0

        for _ in range(burn_in):
            a = np.random.choice(self.A, p=policy[s, :])
            s, _, _, _ = env.step(a)

        R = 0  # start collecting rewards
        t = traj_length - burn_in
        for _ in range(t):
            a = np.random.choice(self.A, p=policy[s, :])
            s, r, _, _ = env.step(a)
            R += r

        return R / t

    def estimate_gain(self, policy, traj_length=10000, burn_in=1000):
        """
        Estimate the gain vector of policy.

        Run policy over traj_length steps, and return the average reward
        collected after a burn-in period, for all starting state.

        See self.estimate_state_gain

        Parameters:
        -----------
        policy : (S, A) numpy.array
            Policy to use.
        traj_length : integer > 0, default 10000.
            Number of steps to execute under policy.
        burn_in : 0 <= integer < traj_length, default 1000
            Number of burn-in steps to make, before collecting rewards.

        Output:
        -------
        gain : S numpy.array
            Estimated gain vector of policy over states.
        """
        state_gain = self.estimate_state_gain
        return np.array([state_gain(policy, s, traj_length, burn_in)
                         for s in range(self.S)])

    def estimate_optimal_gain(self, traj_length=10000, burn_in=1000):
        """
        Estimate the optimal gain.

        Run policy over traj_length steps, and return the average reward
        collected after a burn-in period.
        !!! It is assumed that the environment is weakly communicating, and
        hence the optimal gain is independent of the starting state !!!

        The environment initial distribution is then used.
        Use self.opti_pi if provided, else greedy_policy(self.opti_Q).
        Overwrite self.opti_gain.

        See self.estimate_gain

        Parameters:
        -----------
        traj_length : integer > 0, default 10000.
            Number of steps to execute under policy.
        burn_in : 0 <= integer < traj_length, default 1000
            Number of burn-in steps to make, before collecting rewards.

        Output:
        -------
        gain : float
            Estimated optimal gain.
        """
        if self.opti_pi is not None:
            pi = self.opti_pi
        else:
            assert self.opti_Q, "No available optimal policy!"
            pi = greedy_policy(self.opti_Q)

        s0 = self.env.reset()
        self.opti_gain = self.estimate_state_gain(pi, s0, traj_length, burn_in)
        return self.opti_gain

    def value_error(self, value):
        """Return error of value w.r.t. optimal value."""
        assert self.opti_V, "Optimal value wasn't given!"

        return ((value - self.opti_V)**2).mean()
