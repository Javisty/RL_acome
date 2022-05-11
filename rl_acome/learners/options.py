"""Provide the Option class, cf Sutton & al. (1999)."""
import numpy as np
import numpy.random as npr
from scipy.stats import bernoulli

import gym.spaces as spaces


class Option:
    """
    Create an Option as defined in [1] as an interface between agents and envs.

    [1] Between MDPs and semi-MDPs: A framework for Temporal Abstraction in
    Reinforcement Learning.

    Options correspond to temporally extended actions: starting from some
    initiation state, a sub-policy is applied until a termination condition is
    met. Options are a form of prior knowledge over a MDP (giving optimal
    policies for sub-problems), and many algorithms have been developped to use
    it with benefits.

    Parameters:
    -----------
    env : gym.Env or tuple (constructor, kwargs)
        Environment on which the option operates.
    I : array-like of ints
        Initiation set, the states from env at which the option can be started.
    pi : (S, A) numpy.array
        The non-deterministic option policy.
    beta : S numpy.array
        Termination probability upon reaching a state.

    Methods:
    --------
    is_initiation_state
    terminate
    policy
    play
    """

    def __init__(self, env, I, pi, beta):
        assert isinstance(env.observation_space, spaces.Discrete)
        assert isinstance(env.action_space, spaces.Discrete)

        self.env = env
        self.I = np.zeros(env.observation_space.n, dtype=bool)
        self.I[I] = True  # a bit faster to check from numpy array
        self.pi = pi
        self.beta = beta

        self.A = env.action_space.n

    def is_initiation_state(self, s):
        return self.I[s]

    def terminate(self, s):
        return bernoulli.rvs(self.beta[s])

    def policy(self, s):
        return npr.choice(self.A, p=self.pi[s, :])

    def step(self):
        """
        Perform one step within the option.

        For episodic tasks, termination of an episode also terminates the
        option.

        Output:
        -------
        s_next : int
            The state reached.
        r : float
            The reward gained.
        terminate : bool
            Whether the option must terminate.
        """
        env = self.env

        s_next, r, done, _ = env.step(self.policy(env.state))
        if done:
            env.reset()

        return s_next, r, self.terminate(s_next) or done

    def play(self):
        """
        Play the option from current state until termination.

        Output:
        -------
        s : int
            Starting state.
        R : float
            Accumulated reward during the option.
        s_next : int
            Terminating state.
        """
        env = self.env
        s = env.state
        R = 0

        assert self.is_initiation_state(s), "Not an initiation state!"

        s_next, r, terminate = self.step()
        R += r
        while not terminate:
            s_next, r, terminate = self.step()
            R += r

        return s, R, s_next
