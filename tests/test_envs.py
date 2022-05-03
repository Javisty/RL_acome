#import pytest
from scipy import stats
import numpy as np
from numpy import random as npr

from rl_acome.envs.discrete import DiscreteEnv
from rl_acome.envs.river_swim import RiverSwim, RiverSwim_rlberry


P = np.array([[[0.25, 0.75], [0.75, 0.25]], [[0.9, 0.1], [0.1, 0.9]]])
R = {i: {j: stats.norm(loc=npr.random(), scale=npr.random())
         for j in {0, 1}}
     for i in {0, 1}}


class TestDiscreteEnv:
    def test_init(self):
        mdp = DiscreteEnv(P, R, stats.bernoulli(0.2))

    def test_reset(self):
        mdp = DiscreteEnv(P, R, stats.bernoulli(1))
        mdp.reset()
        assert mdp.s == 1 and not mdp.lastaction

    def test_sample_next_state(self):
        mdp = DiscreteEnv(P, R, stats.bernoulli(0.2))
        assert mdp.sample_next_state(1, 1) in {0, 1}

    def test_sample_reward(self):
        mdp = DiscreteEnv(P, R, stats.bernoulli(0.2))
        assert 0 <= mdp.sample_reward(1, 1) <= 1



class TestRiverSwim:
    def test_init(self):
        rs = RiverSwim(1)
        rs = RiverSwim(2)
        rs = RiverSwim(3, p_failure=0.6)
        rs = RiverSwim(3, p_failure=0.6, p_inplace=0.4)
        rs = RiverSwim(3, p_success=0.2)
        rs = RiverSwim(3, p_success=0.2, p_inplace=0.2)
        rs = RiverSwim(3, p_success=0.2, p_inplace=1)
        rs = RiverSwim(3, p_success=0.2, p_inplace=0.2)

    def test_P(self):
        rs = RiverSwim(3)
        assert np.isclose(rs.P[:, 0, :], np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])).all()
        assert np.isclose(rs.P[:, 1, :], np.array([[0.4, 0.6, 0], [0.05, 0.35, 0.6], [0, 0.05, 0.95]])).all()

    def test_R(self):
        rs = RiverSwim(3)
        assert (rs.sample_reward(0, 0) == 0.1)
        assert (rs.sample_reward(1, 1) == 0)
        assert (rs.sample_reward(2, 1) == 1)


class TestRiverSwim_rlberry:
    def test_init(self):
        rs = RiverSwim_rlberry(1)
        rs = RiverSwim_rlberry(2)
        rs = RiverSwim_rlberry(3, p_failure=0.6)
        rs = RiverSwim_rlberry(3, p_failure=0.6, p_inplace=0.4)
        rs = RiverSwim_rlberry(3, p_success=0.2)
        rs = RiverSwim_rlberry(3, p_success=0.2, p_inplace=0.2)
        rs = RiverSwim_rlberry(3, p_success=0.2, p_inplace=1)
        rs = RiverSwim_rlberry(3, p_success=0.2, p_inplace=0.2)

    def test_P(self):
        rs = RiverSwim_rlberry(3)
        assert np.isclose(rs.P[:, 0, :], np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])).all()
        assert np.isclose(rs.P[:, 1, :], np.array([[0.4, 0.6, 0], [0.05, 0.35, 0.6], [0, 0.05, 0.95]])).all()
