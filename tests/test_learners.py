import numpy as np

from rl_acome.envs.instances import rooms
from rl_acome.learners import UCRL2, SUCRL, QL_options


one_room = rooms.one_room()
four_rooms, _, options = rooms.four_rooms_with_options()


class TestUCRL2:
    def test_init(self):
        agent = UCRL2.UCRL2(one_room)

    def test_fit(self):
        agent = UCRL2.UCRL2(one_room)

        agent.fit(50)


class TestSUCRL:
    def test_init(self):
        agent = SUCRL.SUCRL(four_rooms, options, 20, 1, 0, 1, 0, 17)

    def test_fit(self):
        agent = SUCRL.SUCRL(four_rooms, options, 20, 1, 0, 1, 0, 17)

        agent.fit(50)



class TestQL_options:
    def test_init(self):
        agent = QL_options.QL_options(four_rooms, options)

    def test_fit(self):
        agent = QL_options.QL_options(four_rooms, options)

        agent.fit(50)
