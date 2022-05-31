"""Provide options framework, cf Sutton & al. (1999)."""
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

    Notes:
    ------
    Only discrete environments are accepted so far.

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
        s_next : int
            Terminating state.
        R : float
            Accumulated reward during the option.
        t : int
            Duration of the execution.
        """
        env = self.env
        s = env.state
        R, t = 0, 0

        assert self.is_initiation_state(s), "Not an initiation state!"

        s_next, r, terminate = self.step()
        R += r
        while not terminate:
            s_next, r, terminate = self.step()
            R += r
            t += 1

        return s_next, R, t


class OptionSet:
    """
    Interface to use a set of Options altogether.

    Parameters:
    -----------
    env : gym.Env or tuple (constructor, kwargs)
        Environment on which all the options operate.
    options : list of Option
        The set of options.

    Notes:
    ------
    Only discrete environments are accepted so far.

    Methods:
    --------
    check_options_compatibility
    buil_map
    available_options
    get_option
    """

    def __init__(self, env, options):
        self.env = env
        self.options = options
        self.O = len(options)

        self.check_options_compatibility()
        self.build_map()

    def check_options_compatibility(self):
        """Check that the options share the good environment."""
        incompatible = list()
        env = self.env
        for i, o in enumerate(self.options):
            if o.env != env:
                incompatible.append(i)

        assert not incompatible, (f"Options {incompatible} aren't defined on "
                                  "the good environemnt!")

    def build_map(self):
        """Build a mapping between states and starting options in states."""
        S = self.env.observation_space.n
        state_options = np.zeros((S, self.O), dtype=bool)

        for i, o in enumerate(self.options):
            state_options[:, i] = o.I

        self.mask = state_options
        # Transform it to a dictionary of arrays of indices
        self.state_options = {s: np.where(state_options[s, :])[0]
                              for s in range(S)}

    def available_options(self, s):
        """Return the array of indices of options starting at state."""
        return self.state_options[s]

    def get_option(self, i):
        """Return the option at index i."""
        return self.options[i]

    def play_option(self, i):
        """Make option i play."""
        return self.get_option(i).play()


def create_primitive_options(env):
    """Return the list of primitive options."""
    from rlberry.envs import FiniteMDP
    assert isinstance(env, FiniteMDP)

    pi = np.zeros((env.A, env.S, env.A))
    for a in range(env.A):
        pi[a, :, a] = 1
    beta = np.ones((env.A, env.S))

    return [Option(env, list(range(env.S)), p, b) for p, b in zip(pi, beta)]
