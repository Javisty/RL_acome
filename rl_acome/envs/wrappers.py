"""Define environment wrappers."""

import numpy as np
import matplotlib.pyplot as plt
from rlberry.envs import Wrapper

from .utils import greedy_stochastic_policy


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
        if opti_Q is not None:
            self.opti_V = np.max(opti_Q, axis=-1)
        else:
            self.opti_V = opti_V
        self.opti_r = opti_reward
        self.opti_gain = opti_gain

        self.S, self.A = env.observation_space.n, env.action_space.n

        self._init_values()
        self.step = self._step  # set the step method to default

    def _init_values(self):
        """Set counters to 0."""
        self.t = 0  # number of time-steps
        self.sa = np.zeros((self.S, self.A), dtype=int)  # number of visits
        self.r = np.zeros((self.S, self.A))  # accumulated reward
        self.episodes_steps = [0]  # number of states per episode

    def _step(self, action):
        """Execute action, and update statistics accordingly to results."""
        state, reward, done, info = self.env.step(action)

        self.t += 1
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
            assert self.opti_Q is not None, (
                "No optimal policy nor Q function provided!")
            pi = greedy_stochastic_policy(self.opti_Q)

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
            s, _, done, _ = env.step(a)
            if done:
                s = env.reset()

        R = 0  # start collecting rewards
        t = traj_length - burn_in
        for _ in range(t):
            a = np.random.choice(self.A, p=policy[s, :])
            s, r, done, _ = env.step(a)
            R += r

            if done:
                s = env.reset()

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
        Use self.opti_pi if provided, greedy_stochastic_policy(self.opti_Q)
        otherwise.
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
            assert self.opti_Q is not None, (
                "No available optimal policy!")
            pi = greedy_stochastic_policy(self.opti_Q)

        s0 = self.env.reset()
        self.opti_gain = self.estimate_state_gain(pi, s0, traj_length, burn_in)
        return self.opti_gain

    def value_error(self, value):
        """Return error of value w.r.t. optimal value."""
        assert self.opti_V is not None, (
            "Optimal value wasn't given!")

        return ((value - self.opti_V)**2).mean()

    def start_recording_regret(self, delta=100, nb_steps=None):
        """
        Record the regret every delta steps for nb_steps.

        self.opti_gain must be not None.
        Overwrite the current step method so that the EvaluatingEnv stores
        samples of the regret. The sampling frequency is given by times, and
        the duration of the recording by nb_steps.

        Notes:
        ------
        It is more efficient to specify the total number of steps (array VS
        list).

        Parameters:
        -----------
        delta : 0 < int, default 100
            Frequency of regret recording. Store regret every delta time steps.
        nb_steps : 0 < int or None, default None
            The number of steps until which regret is stored. If None, regret
            is appended to a list. Otherwise, regret is stored in an array.

        Output:
        -------
        regret : list or (nb_steps) array
            History of regret.
        """
        assert self.opti_gain is not None, (
            "No optimal gain available to compute regret!")

        # Save current step method
        self._previous_step = self.step  # save current step method

        self._delta_times = delta

        self._R = 0  # accumulated reward since last record
        if nb_steps:  # store in arrays
            self._nb_steps = nb_steps
            self.regret = np.full((nb_steps // delta) + 1, np.nan)
            self.regret[0] = 0
            self._regret_times = np.arange(0, nb_steps + 1, delta)
            # Next time step of recording and corresponding index
            self._next_t, self._idx = self._regret_times[1], 1

            self.step = self._step_with_array_regret

        else:  # store in lists
            self.regret = [0]
            self._regret_times = [0]  # time steps of recording
            # Next time step of recording and corresponding index
            self._next_t = delta

            self.step = self._step_with_list_regret

    def stop_recording_regret(self):
        """Stop recording regret, and fall back to previous step method."""
        self.step = self._previous_step

    def _step_with_array_regret(self, action):
        """Wrap previous step method to record regret in an array."""
        state, reward, done, info = self._previous_step(action)

        self._R += reward
        t = self.t
        if t < self._nb_steps:
            if t == self._next_t:
                self.regret[self._idx] = t * self.opti_gain - self._R
                self._idx += 1
                self._next_t = self._regret_times[self._idx]

        else:  # stop regret recording
            self.stop_recording_regret()

        return state, reward, done, info

    def _step_with_list_regret(self, action):
        """Wrap previous step method to record regret in a list."""
        state, reward, done, info = self._step(action)

        self._R += reward
        t = self.t
        if t == self._next_t:
            self.regret.append(t * self.opti_gain - self._R)
            self._regret_times.append(t)
            self._next_t += self._delta_times

        return state, reward, done, info

    def plot_regret(self, with_x_fun=False, title=None):
        """Plot T x opti_gain - R along time steps."""
        fig, ax = plt.subplots()
        ax.plot(self._regret_times, self.regret)

        if with_x_fun:
            ax.plot(self._regret_times, self._regret_times, c='k')

        if title:
            ax.set_title(title)
        else:
            ax.set_title("Regret over time")

        ax.set_xlabel("Time steps")
        ax.set_ylabel("Regret")
        return fig
