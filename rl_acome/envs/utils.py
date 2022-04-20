"""Bunch of useful functions."""
import numpy as np


def assert_is_transition_matrix(P):
    """
    Check that P is a valid transition matrix.

    A transition matrix should be a S x A x S np.array with non-negative
    numbers, s.t. P[s, a, :] sums to one, for every state-action pair (s, a).
    """
    assert isinstance(P, np.ndarray), "P should be a Numpy array."
    assert len(P.shape) == 3, "P should have exactly 3 dimensions."
    assert P.shape[0] == P.shape[2], ("First and last dimensions should have "
                                      "same size.")
    np.testing.assert_allclose(P.sum(axis=2), 1,
                               err_msg=("Probabilities should sum to 1 over "
                                        "the last dimension."))
