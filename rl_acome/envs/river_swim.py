"""
Implementation of the River Swim task.

This is a modelisation of someone trying to swim up a riverflow.
The MDP consists in a chain of states, with two actions possible (go right or
go left). The reward is at the far right state, so the goal is to go go right,
however as the agent is swimming against the current there is a chance to stay
at the same state or even be deported on the left.
"""
import numpy as np

from .discrete import DiscreteEnv
from .utils import Constant


class RiverSwim(DiscreteEnv):
    """
    Swim up the river to be rewarded.

    The environment consists in a finite sequence of states, which is possible
    to navigate left or right. The only non-null rewards are at the far right
    and the far left states, but is much higher for the right one.
    An obvious optimal strategy would be to try to go right every time, however
    there is a chance that the agent stay in place or even go left.
    This is an interesting case for learning agents: a lot of exploration is
    required to realise that the reward is at the right end. It also depends on
    the actual probabilities and the rewards.

    Parameters:
    -----------
    nS : int
        Number of states.
    p_success : float between 0 and 1
        Probability to go right when trying to.
    p_failure : float between 0 and 1
        Probability to go left when trying to go right.
    r_left : float
        Reward for staying in the far left state.
    r_right : float
        Reward for staying in the far right state.
    s0: int
        Starting state.
    p_inplace : {None, float}
        The probability of staying in the current state.
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
        A seed for the numpy/scipy random generator.
        See https://numpy.org/doc/stable/reference/random/generator.html for
        for more information.
    name : string
        Name of the environment.

    Notes:
    ------
    If p_inplace is given, or p_failure + p_success > 1 then the values are
    normalised in order to obtain a proper probability distribution.
    In the end: p_success + p_failure + p_inplace = 1, with
    p_inplace=max(0, 1 - (p_failure + p_succes)) if not given.

    Action 0 is 'going to the left', and action 1 is 'going to the right'.
    """

    def __init__(self, nS: int, p_success=0.6, p_failure=0.05, r_left=0.1,
                 r_right=1, p_inplace=None, s0=0, seed=None, name="RiverSwim"):
        # Get the probabilities right
        # Make sure we have non-negative probabilities
        p_success, p_failure = max(0, p_success), max(0, p_failure)

        if p_inplace:
            p_inplace = max(0, p_inplace)
        else:
            p_inplace = max(0, 1 - (p_failure + p_success))

        # Normalise probabilities
        total = p_success + p_failure + p_inplace
        if total != 1:
            p_success /= total
            p_failure /= total
            p_inplace /= total

        # River swim transition matrix
        P = np.zeros((nS, 2, nS))

        # Going left: probability 1 to go to previous state
        P[:, 0, :] += np.eye(nS, k=-1)

        # Going right: go right, left or stay in place
        P[:, 1, :] += p_success * np.eye(nS, k=1) \
                      + p_failure * np.eye(nS, k=-1) \
                      + p_inplace * np.eye(nS, k=0)

        # Edge cases
        # Stay in place when going left from leftmost state
        P[0, 0, 0] = 1
        P[0, 1, 0] += p_failure
        # Stay in place when going right from rightmost state
        P[-1, 1, -1] += p_success

        # Rewards
        R = {i: {0: Constant(0), 1: Constant(0)} for i in range(nS)}
        R[0][0], R[nS-1][1] = Constant(r_left), Constant(r_right)

        # Starting distribution
        mu0 = Constant(s0)

        # Reward range
        r_lim = (min(r_left, r_right, 0), max(r_left, r_right, 0))

        super(RiverSwim, self).__init__(P, R, mu0, r_lim=r_lim,
                                        seed=seed, name=name)
