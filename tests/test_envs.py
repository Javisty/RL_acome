#import pytest
from scipy import stats
import numpy as np
from numpy import random as npr

from rl_acome.envs.discrete import DiscreteEnv


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
