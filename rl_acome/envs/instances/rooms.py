"""Provide functions to generate basic environments."""
from rlberry.envs.finite import GridWorld


def four_rooms():
    """Generate an instance of the classic four rooms (Sutton1999)."""
    nrows = 11
    ncols = 11
    start_coord = (1, 1)
    terminal_states = ((nrows - 2, ncols - 2),)
    success_probability = 2/3
    reward_at = {(nrows - 2, ncols - 2): 1.}
    default_reward = 0.
    walls = ((5, 0), (5, 2), (5, 3), (5, 4), (5, 5),
             (6, 5), (6, 6), (6, 7), (6, 9), (6, 10),
             (0, 5), (1, 5), (3, 5), (4, 5), (7, 5), (8, 5), (10, 5))

    return GridWorld(nrows, ncols, start_coord, terminal_states,
                     success_probability, reward_at, walls, default_reward)


def one_room():
    """
    Generate an instance of a 3x3 room with one wall.

    The starting and goal states are in opposite corners.
    Layout:
    #####
    #001#
    #00##
    #X00#
    #####
    """
    nrows = 3
    ncols = 3
    start_coord = (2, 0)
    terminal_states = ((0, 2),)
    success_probability = 2/3
    reward_at = {(0, 2): 1.}
    default_reward = 0.
    walls = ((1, 2),)

    return GridWorld(nrows, ncols, start_coord, terminal_states,
                     success_probability, reward_at, walls, default_reward)
