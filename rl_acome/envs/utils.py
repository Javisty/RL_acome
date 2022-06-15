"""Bunch of useful functions."""
import numpy as np
import matplotlib.pyplot as plt

from rlberry.envs import GridWorld


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


class Constant:
    """
    Constant random variable.

    Usage similar to scipy.stats random variables.

    Parameters:
    -----------
    value : float-like
        The constant value of the random variable.

    Methods:
    --------
    rvs
    mean
    """

    def __init__(self, value):
        self.v = value

    def rvs(self):
        return self.v

    def mean(self):
        return self.v


def plot_gridworld(env, colormap='Oranges', save_path=None):
    """Plot Gridworld with walls, start, terminal and reward states."""
    assert isinstance(env, GridWorld), "Only GridWorld environments!"

    state_data = np.zeros(env.observation_space.n)
    c2i = env.coord2index

    # Color start, reward and terminal states
    state_data[c2i[env.start_coord]] = 0.25
    state_data[[c2i[s] for s in env.reward_at]] = 0.5
    state_data[[c2i[s] for s in env.terminal_states]] = 1

    img = np.zeros((env.nrows+2, env.ncols+2, 3))
    img[1:-1, 1:-1, :] = env.get_layout_img(state_data=state_data,
                                            colormap_name=colormap)

    if save_path:
        plt.imsave(save_path, img)

    return plt.imshow(img)
