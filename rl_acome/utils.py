"""General utilities for user."""

import numpy as np


def deterministic_to_stochastic(policy, A):
    """Convert action to probabilities, with shape (S, A)."""
    stoch = np.zeros((policy.size, A))
    stoch[range(policy.size), policy] = 1.
    return stoch
