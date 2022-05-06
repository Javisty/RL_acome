import numpy as np

from rl_acome.envs.instances.rooms import one_room
from rl_acome.learners.UCRL2 import UCRL2


grid_world = one_room()

class TestUCRL2:
    def test_init(self):
        agent = UCRL2(grid_world)

    def test_fit(self):
        agent = UCRL2(grid_world)

        agent.fit(50)
