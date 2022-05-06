"""
Check the dependencies are correctly installed, and ready for use.

DISCLAIMER: the following lines are taken from built-in tests from rlberry
v0.2.1, and can be found here:
https://github.com/rlberry-py/rlberry/tree/v0.2.1/rlberry
"""
# From rlberry/agents/tests/test_ucbvi.py

import pytest
from rlberry.agents.ucbvi import UCBVIAgent
from rlberry.envs.finite import GridWorld


@pytest.mark.parametrize("gamma, stage_dependent, real_time_dp",
                         [
                             (1.0, True, True),
                             (1.0, True, False),
                             (1.0, False, True),
                             (1.0, False, False),
                             (0.9, True, True),
                             (0.9, True, False),
                             (0.9, False, True),
                             (0.9, False, False),
                         ])
def test_ucbvi(gamma, stage_dependent, real_time_dp):
    env = GridWorld(walls=(), nrows=5, ncols=5)
    agent = UCBVIAgent(env,
                       horizon=11,
                       stage_dependent=stage_dependent,
                       gamma=gamma,
                       real_time_dp=real_time_dp,
                       bonus_scale_factor=0.1)
    agent.fit(budget=50)
    agent.policy(env.observation_space.sample())


# From rlberry/envs/test/test_instantiation.py

import numpy as np
import pytest

from rlberry.envs.finite import Chain
from rlberry.envs.finite import GridWorld
from rlberry.envs.benchmarks.grid_exploration.four_room import FourRoom
from rlberry.rendering.render_interface import RenderInterface2D

classes = [
    Chain,
    FourRoom
]


@pytest.mark.parametrize("ModelClass", classes)
def test_instantiation(ModelClass):
    env = ModelClass()

    if env.is_online():
        for _ in range(2):
            state = env.reset()
            for _ in range(50):
                assert env.observation_space.contains(state)
                action = env.action_space.sample()
                next_s, _, _, _ = env.step(action)
                state = next_s

    if env.is_generative():
        for _ in range(100):
            state = env.observation_space.sample()
            action = env.action_space.sample()
            next_s, _, _, _ = env.sample(state, action)
            assert env.observation_space.contains(next_s)


@pytest.mark.parametrize("ModelClass", classes)
def test_rendering_calls(ModelClass):
    env = ModelClass()
    if isinstance(env, RenderInterface2D):
        _ = env.get_background()
        _ = env.get_scene(env.observation_space.sample())


def test_gridworld_aux_functions():
    env = GridWorld(nrows=5, ncols=8, walls=((1, 1),),
                    reward_at={(4, 4): 1, (4, 3): -1})
    env.log()  # from FiniteMDP
    env.render_ascii()  # from GridWorld
    vals = np.arange(env.observation_space.n)
    env.display_values(vals)
    env.print_transition_at(0, 0, 'up')

    layout = env.get_layout_array(vals, fill_walls_with=np.inf)
    for rr in range(env.nrows):
        for cc in range(env.ncols):
            if (rr, cc) in env.walls:
                assert layout[rr, cc] == np.inf
            else:
                assert layout[rr, cc] == vals[env.coord2index[(rr, cc)]]


@pytest.mark.parametrize("reward_free, difficulty, array_observation",
                         [
                             (True, 0, False),
                             (False, 0, False),
                             (False, 0, True),
                             (False, 1, False),
                             (False, 1, True),
                             (False, 2, False),
                             (False, 2, True),
                         ])
def test_four_room(reward_free, difficulty, array_observation):
    env = FourRoom(reward_free=reward_free,
                   difficulty=difficulty,
                   array_observation=array_observation)

    initial_state = env.reset()
    next_state, reward, _, _ = env.step(1)

    assert env.observation_space.contains(initial_state)
    assert env.observation_space.contains(next_state)

    if reward_free:
        assert env.reward_at == {}

    if difficulty == 2:
        assert reward < 0.0

    if array_observation:
        assert isinstance(initial_state, np.ndarray)
        assert isinstance(next_state, np.ndarray)
