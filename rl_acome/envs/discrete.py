"""Provide the DiscreteEnv base class, for finite environments."""
import numpy as np
import numpy.random as npr

from gym import Env, spaces
from gym.utils import seeding

from .utils import assert_is_transition_matrix


class DiscreteEnv(Env):
    """
    Structure for a discrete MDP.

    DiscreteEnv is meant as a base class to construct specific discrete
    MDP environment. It provides the necessary methods to run RL experiments
    with an agent.

    Parameters:
    ----------
    P : (nS, nA, nS) np.array
        Transition matrix. P[s, a, s'] is the probability of moving to state
        s' after taking action a in state s.
    R : dict(dict(scipy.stats (frozen) random variable))
        Reward distribution for every state-action pair.
        R[s][a].rvs() samples a reward for action a in state s.
        See https://docs.scipy.org/doc//scipy/tutorial/stats.html for more
        information about scipy distributions.
    mu0 : scipy.stats random variable
        mu0.rvs() samples an initial state.
    r_lim : (float, float) tuple
        Give the support of rewards.
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
        A seed for the numpy/scipy random generator.
        See https://numpy.org/doc/stable/reference/random/generator.html for
        for more information.
    name : string
        Name of the environment.

    Methods:
    --------
    reset
    step
    seed
    sample_next_state
    sample_reward
    """

    def __init__(self, P, R, mu0, r_lim=(0, 1), seed=None, name="DiscreteMDP"):
        """Initialize self. See help(type(self)) for accurate signature."""
        self.P = P
        self.R = R
        self.mu0 = mu0

        self.nS = self.P.shape[0]
        self.nA = self.P.shape[1]

        # Check sum-to-one constraint and 1st dimension == 3rd dimension for P
        assert_is_transition_matrix(P)
        # Check matching lengths and dimensions
        assert len(R) == self.nS and all(len(r) == self.nA for r in R.values())

        self.states = range(0, self.nS)
        self.actions = range(0, self.nA)

        self.reward_range = r_lim
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        """Set up new seed for random sampling."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Reset environment to initial state."""
        # Sample initial state from intial distribution
        self.s = self.mu0.rvs()
        self.lastaction = None
        return self.s

    def step(self, a):
        """Perform one transition from current state with action a."""
        self.s = self.sample_next_state(self.s, a)
        r = self.sample_reward(self.s, a)

        self.lastaction = a
        return self.s, r, False, ""

    def sample_next_state(self, s, a):
        """Sample a next state coming from state s with action a."""
        return npr.choice(self.states, p=self.P[s, a])

    def sample_reward(self, s, a):
        """Sample a reward for taking action a in state s."""
        return np.clip(self.R[s][a].rvs(), *self.reward_range)
