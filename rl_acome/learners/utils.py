"""Miscellaneous utilities for learners."""
import numpy as np


def greedy_policy(q):
    """
    Return the greedy policy for a Q-value function.

    The greedy policy selects the action maximising the value. If there are
    multiple candidates, choose randomly.

    Parameters:
    -----------
    q : (S, A) numpy.array
        Q-value vector.

    Output:
    -------
    S numpy.array, giving the action chosen for each state.
    """
    S = q.shape[0]
    policy = np.zeros(S, dtype=int)

    for s in range(S):
        policy[s] = np.random.choice(np.where(q[s, :] == np.max(q[s, :]))[0])

    return policy
