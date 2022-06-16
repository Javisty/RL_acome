"""
Implementation of the River Swim task.

This is a modelisation of someone trying to swim up a riverflow.
The MDP consists in a chain of states, with two actions possible (go right or
go left). The reward is at the far right state, so the goal is to go go right,
however as the agent is swimming against the current there is a chance to stay
at the same state or even be deported on the left.
"""
import numpy as np

from rlberry.envs.finite import FiniteMDP
from rlberry.rendering import RenderInterface2D, Scene, GeometricPrimitive


class RiverSwim(RenderInterface2D, FiniteMDP):
    """
    Swim up the river to be rewarded.

    The environment consists in a finite sequence of states, which is possible
    to navigate left or right. The only non-null rewards are at the far right
    and the far left states, but is much higher for the right one.
    An obvious optimal strategy is to try to go right every time, however
    there is a chance that the agent stays in place or even go left.
    This is an interesting case for learning agents: a lot of exploration is
    required to realise that the reward is at the rightmost state. It also
    depends on the actual probabilities and the rewards.

    Notes:
    ------
    If p_inplace is given, or p_failure + p_success > 1 then the values are
    normalised in order to obtain a proper probability distribution.
    In the end: p_success + p_failure + p_inplace = 1, with
    p_inplace=max(0, 1 - (p_failure + p_succes)) if not given.

    Action 0 is 'going to the left', and action 1 is 'going to the right'.

    Very similar to the Chain environment, with the possibility to choose
    rewards and probability to stay in place.

    Parameters:
    -----------
    S : int
        Number of states.
    p_success : 0 <= float <= 1, default 0.6
        Probability to go right when trying to.
    p_failure : 0 <= float <= 1, default 0.05
        Probability to go left when trying to go right.
    r_left : float, default 0.05
        Reward for staying in the far left state.
    r_right : float, default 0.1
        Reward for staying in the far right state.
    s0: int < S, default 0
        Starting state.
    p_inplace : 0 <= float <= 1 or None, default None
        The probability of staying in the current state.
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
        A seed for the numpy/scipy random generator.
        See https://numpy.org/doc/stable/reference/random/generator.html for
        for more information.
    name : string
        Name of the environment.

    Methods:
    --------
    See DiscreteEnv.
    """

    def __init__(self, S: int, p_success=0.6, p_failure=0.05, r_left=0.1,
                 r_right=1, p_inplace=None, s0=0, seed=None, name="RiverSwim"):
        self.L = S

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
        P = np.zeros((S, 2, S))

        # Going left: probability 1 to go to previous state
        P[:, 0, :] += np.eye(S, k=-1)

        # Going right: go right, left or stay in place
        P[:, 1, :] += (p_success * np.eye(S, k=1)
                       + p_failure * np.eye(S, k=-1)
                       + p_inplace * np.eye(S, k=0))

        # Edge cases
        # Stay in place when going left from leftmost state
        P[0, 0, 0] = 1
        P[0, 1, 0] += p_failure
        # Stay in place when going right from rightmost state
        P[-1, 1, -1] += p_success

        # Rewards
        R = np.zeros((S, 2))
        R[S - 1, 1] = r_right
        R[0, 0] = r_left

        # Init base classes
        FiniteMDP.__init__(self, R, P, initial_state_distribution=s0)
        RenderInterface2D.__init__(self)

        self.reward_range = (min(r_left, r_right, 0), max(r_left, r_right, 0))

        # Rendering info
        self.set_clipping_area((0, S, 0, 1))
        self.set_refresh_interval(100)  # in milliseconds

    def step(self, action):
        assert action in self._actions, "Invalid action!"

        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(self.state)

        # take step
        next_state, reward, done, info = self.sample(self.state, action)

        self.state = next_state
        return next_state, reward, done, info

    #
    # Code for rendering
    #

    def get_background(self):
        """Return a scene (list of shapes) representing the background."""
        bg = Scene()
        colors = [(0.8, 0.8, 0.8), (0.9, 0.9, 0.9)]
        for ii in range(self.L):
            shape = GeometricPrimitive("QUADS")
            shape.add_vertex((ii, 0))
            shape.add_vertex((ii + 1, 0))
            shape.add_vertex((ii + 1, 1))
            shape.add_vertex((ii, 1))
            shape.set_color(colors[ii % 2])
            bg.add_shape(shape)

        flag = GeometricPrimitive("TRIANGLES")
        flag.set_color((0.0, 0.5, 0.0))
        x = self.L - 0.5
        y = 0.25
        flag.add_vertex((x, y))
        flag.add_vertex((x + 0.25, y + 0.5))
        flag.add_vertex((x - 0.25, y + 0.5))
        bg.add_shape(flag)

        return bg

    def get_scene(self, state):
        """Return scene (list of shapes) representing a given state."""
        scene = Scene()

        agent = GeometricPrimitive("QUADS")
        agent.set_color((0.75, 0.0, 0.5))

        size = 0.25
        x = state + 0.5
        y = 0.5

        agent.add_vertex((x - size / 4.0, y - size))
        agent.add_vertex((x + size / 4.0, y - size))
        agent.add_vertex((x + size / 4.0, y + size))
        agent.add_vertex((x - size / 4.0, y + size))

        agent.add_vertex((x - size, y - size / 4.0))
        agent.add_vertex((x + size, y - size / 4.0))
        agent.add_vertex((x + size, y + size / 4.0))
        agent.add_vertex((x - size, y + size / 4.0))

        scene.add_shape(agent)
        return scene
