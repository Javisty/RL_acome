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
        Q-value matrix.

    Output:
    -------
    S numpy.array, giving the action chosen for each state.
    """
    S = q.shape[0]
    policy = np.zeros(S, dtype=int)

    for s in range(S):
        policy[s] = np.random.choice(np.where(q[s, :] == np.max(q[s, :]))[0])

    return policy


def eps_greedy_action(q_state, eps):
    """
    Sample action from q with eps-greedy policy.

    With probability eps, choose uniformly among the set of actions. Otherwise,
    choose of one the maximisers of q_state.

    Parameters:
    -----------
    q_state : A numpy.array
        Q-value vector for one state.
    eps : 0 <= eps <= 1
        Probability of choosing uniformly.

    Output:
    -------
    Int, the chosen action.
    """
    if np.random.random() < eps:
        return np.random.randint(0, q_state.size)
    else:
        return np.random.choice(np.where(q_state == np.max(q_state))[0])
