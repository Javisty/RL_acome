"""Q-Learning with options from Sutton et al. (1999)."""

import numpy as np
from numpy import ma

import gym.spaces as spaces
from rlberry.agents import Agent

from .options import OptionSet
from .utils import greedy_policy


class QL_options(Agent):
    """
    Q-Learning with options from [1].

    Parameters:
    -----------
    env : gym.Env
        Environment with discrete states and actions.
    options : rl_acome.learners.options.OptionSet
        Set of options to use.
    alpha : float > 0 or callable object, default 1/8
        Learning rate. If callable, must take state index, option index and
        time-step as arguments.
    gamma : 0 < float <= 1, default 1.0
        Discount factor.
    """

    def __init__(self, env, options, alpha=1/8, gamma=1., **kwargs):
        Agent.__init__(self, env, **kwargs)

        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)
        assert isinstance(options, OptionSet)
        assert options.env == env, "The options environment doesn't match env!"

        self.options = options
        if isinstance(alpha, float):  # use alpha as a callable object
            self.alpha = lambda s, o, t: alpha
        else:
            self.alpha = alpha
        self.gamma = gamma

        self.O = options.O
        self.S, self.A = self.env.observation_space.n, self.env.action_space.n

        # Initialise value holders

        self.t = 1  # current time-step

        mask = ~self.options.mask  # indicate invalid state-option pairs

        self.V = np.zeros(self.S)
        self.Q = ma.zeros((self.S, self.O))
        self.reward = 0.

        # Current policy, with random initialisation over available options
        self.pi = np.array([np.random.choice(self.options.available_options(s))
                            for s in range(self.S)])

        # Mask array, to avoid considering non-available options
        self.Q[mask] = ma.masked

    def play_option(self, idx, env=None):
        """
        Play option until termination.

        Return next state, accumulated reward and duration.
        """
        return self.options.play_option(idx, env=env)

    def choose_option(self, state):
        """Return the index of the option to play, according to the Q-value."""
        q = self.Q[state, :]
        return np.random.choice(np.where(q == np.max(q))[0])

    def fit(self, budget, **kwargs):
        """
        Train the agent using the provided environment.

        Parameters:
        -----------
        budget : int
            Horizon, number of time steps to perform before stopping.
        **kwargs
            Extra arguments.

        Output:
        -------
        policy : S numpy.array
            The policy learned by the agent.
        reward : float
            The cumulative reward obtained during learning.
        """
        alpha, gamma = self.alpha, self.gamma
        Q = self.Q

        state = self.env.state
        while self.t <= budget:
            option = self.choose_option(state)

            print((f"Playing option {option} at time step {self.t} "
                   f"from state {state}"), end='\r')

            s_next, r, tau = self.play_option(option)

            # Update Q-value estimate
            l_rate = alpha(state, option, self.t)
            q_bar = r + gamma**tau * np.max(Q[s_next])
            Q[state, option] = (1 - l_rate) * Q[state, option] + l_rate * q_bar

            state = s_next
            self.reward += r
            self.t += tau

        return self.get_policy(), self.reward

    def get_policy(self):
        """Return greedy policy w.r.t. current Q-value estimate."""
        return greedy_policy(self.Q)

    def update_policy(self):
        """Update intern policy."""
        self.pi = self.get_policy()

    def policy(self, state):
        """Deterministic policy over options. Doesn't update it before."""
        return self.pi[state]

    def eval(self, eval_horizon=10 ** 5, n_simulations=10, **kwargs):
        """
        Monte-Carlo policy evaluation to estimate the value of initial state.

        Parameters:
        -----------
        eval_horizon : int, default 10**5
            Horizon, maximum episode length.
        n_simulations : int, default 10
            Number of Monte Carlo simulations.

        Output:
        -------
        Mean over the n simulations of the sum of rewards in each simulation.
        """
        del kwargs  # unused
        env = self.eval_env
        # Make sure we have the up-to-date policy
        self.update_policy()

        episode_rewards = np.zeros(n_simulations)
        for sim in range(n_simulations):
            s = env.reset()
            tt = 0

            while tt < eval_horizon:
                option = self.policy(s)
                s, reward, duration = self.play_option(option, env=env)
                episode_rewards[sim] += reward
                tt += duration

        return episode_rewards.mean()
