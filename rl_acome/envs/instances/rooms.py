"""Provide functions to generate basic environments."""
from itertools import product
import numpy as np
from rlberry.envs.finite import GridWorld

from rl_acome.learners.options import (Option,
                                       OptionSet,
                                       create_primitive_options)


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


def four_rooms_with_options():
    """
    Generate four_rooms environment with some relevant option sets.

    The options consist of going from one hallway to another, primitive
    actions, reaching the closest hallway from a state inside a room, and going
    from the two closest hallways to goal state (9, 9).

    Output:
    -------
    env : rlberry.envs.GridWorld
        Four rooms environment.
    options1 : OptionSet
        Hallways + primitive.
    options2 : OptionSet
        Hallways + reach hallways + goal state.
    """
    env = four_rooms()

    hallways = [env.coord2index[coords]
                for coords in [(5, 1), (2, 5), (6, 8), (9, 5)]]

    init_states = [[s] for s in hallways]

    # Useful functions to get the list of states corresponding to rooms
    rooms_limits = {0: [0, 4, 0, 4], 1: [0, 5, 6, 10],
                    2: [7, 10, 6, 10], 3: [6, 10, 0, 4]}
    box_states = lambda xmin, xmax, ymin, ymax: [env.coord2index[a, b]
                                                 for a, b in product(
                                                         range(xmin, xmax+1),
                                                         range(ymin, ymax+1))]
    room_states = lambda room: box_states(*rooms_limits[room])

    ### Options to go from hallway to hallway ###
    # Termination -> terminate when out of the room
    betas1 = np.ones((8, env.Ns))
    betas1[0, room_states(0)] = 0
    betas1[1, room_states(3)] = 0
    betas1[2, room_states(1)] = 0
    betas1[3, room_states(0)] = 0
    betas1[4, room_states(2)] = 0
    betas1[5, room_states(1)] = 0
    betas1[6, room_states(3)] = 0
    betas1[7, room_states(2)] = 0

    betas1[[0, 1], hallways[0]] = 0
    betas1[[2, 3], hallways[1]] = 0
    betas1[[4, 5], hallways[2]] = 0
    betas1[[6, 7], hallways[3]] = 0

    # Policies
    # 0 = left, 1 = right, 2 = up, 3 = down
    pis1 = np.zeros((8, env.Ns, env.Na))

    pis1[0, box_states(3, 4, 0, 4), 2] = 1
    pis1[0, box_states(0, 1, 0, 4), 3] = 1
    pis1[0, box_states(2, 2, 0, 4), 1] = 1
    pis1[0, hallways[0], 2] = 1

    pis1[1, box_states(6, 8, 0, 4), 3] = 1
    pis1[1, box_states(10, 10, 0, 4), :] = [0, 0, 1, 0]
    pis1[1, box_states(9, 9, 0, 4), :] = [0, 1, 0, 0]
    pis1[1, hallways[0], 3] = 1

    pis1[2, box_states(0, 5, 6, 10), 3] = 1
    pis1[2, box_states(5, 5, 6, 7), :] = [0, 1, 0, 0]
    pis1[2, box_states(5, 5, 9, 10), :] = [1, 0, 0, 0]
    pis1[2, hallways[1], 1] = 1

    pis1[3, box_states(0, 4, 0, 4), 3] = 1
    pis1[3, box_states(4, 4, 0, 0), :] = [0, 1, 0, 0]
    pis1[3, box_states(4, 4, 2, 4), :] = [1, 0, 0, 0]
    pis1[3, hallways[1], 0] = 1

    pis1[4, box_states(7, 10, 6, 10), 0] = 1
    pis1[4, box_states(7, 8, 6, 6), :] = [0, 0, 0, 1]
    pis1[4, box_states(10, 10, 6, 6), :] = [0, 0, 1, 0]
    pis1[4, hallways[2], 3] = 1

    pis1[5, box_states(0, 5, 6, 10), 0] = 1
    pis1[5, box_states(3, 5, 6, 6), :] = [0, 0, 1, 0]
    pis1[5, box_states(0, 1, 6, 6), :] = [0, 0, 0, 1]
    pis1[5, hallways[2], 2] = 1

    pis1[6, box_states(6, 10, 0, 4), 2] = 1
    pis1[6, box_states(6, 6, 0, 0), :] = [0, 1, 0, 0]
    pis1[6, box_states(6, 6, 2, 4), :] = [1, 0, 0, 0]
    pis1[6, hallways[3], 0] = 1

    pis1[7, box_states(7, 10, 6, 10), 2] = 1
    pis1[7, box_states(7, 7, 6, 7), :] = [0, 1, 0, 0]
    pis1[7, box_states(7, 7, 9, 10), :] = [1, 0, 0, 0]
    pis1[7, hallways[3], 1] = 1

    ### Option to go from inner room states to closest hallway ###
    # Initiation states
    init2 = room_states(0) + room_states(1) + room_states(2) + room_states(3)

    # Termination
    beta2 = np.zeros(env.Ns)
    beta2[hallways] = 1

    # Policy
    pi2 = np.zeros((env.Ns, env.Na))
    pi2[box_states(7, 9, 9, 10) + box_states(9, 9, 6, 8) +
        box_states(1, 5, 9, 10) + box_states(2, 2, 6, 8) +
        box_states(4, 4, 2, 3) + box_states(6, 7, 2, 3), 0] = 1
    pi2[box_states(6, 10, 0, 0) + box_states(9, 9, 1, 4) +
        box_states(0, 4, 0, 0) + box_states(2, 2, 2, 4) +
        box_states(5, 5, 6, 7) + box_states(7, 7, 7, 7), 1] = 1
    pi2[box_states(3, 4, 4, 4) + box_states(3, 3, 3, 3) +
        box_states(10, 10, 1, 4) + box_states(10, 10, 6, 10) +
        box_states(3, 4, 6, 6) + box_states(3, 3, 7, 7) +
        box_states(7, 8, 8, 8), 2] = 1
    pi2[pi2.sum(axis=-1) == 0, 3] = 1

    ### Option to reach (9, 9) from hallways 2 and 3 ###
    # Initiation states
    init3 = hallways[2:4]

    # Termination
    beta3 = np.ones(env.Ns)
    beta3[hallways[2:4] + room_states(2)] = 0

    # Policy
    pi3 = np.zeros((env.Ns, env.Na))
    pi3[box_states(8, 9, 10, 10), 0] = 1
    pi3[box_states(8, 9, 6, 8) + [hallways[3]], 1] = 1
    pi3[box_states(10, 10, 6, 10), 2] = 1
    pi3[pi3.sum(axis=-1) == 0, 3] = 1

    options1 = list()
    for i in range(8):
        options1.append(Option(env, init_states[i//2], pis1[i], betas1[i]))

    option2 = [Option(env, init2, pi2, beta2)]

    option3 = [Option(env, init3, pi3, beta3)]

    primitives = create_primitive_options(env)

    return (env,
            OptionSet(env, primitives + options1),
            OptionSet(env, option2 + option3 + options1))
