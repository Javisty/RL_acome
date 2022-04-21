=========================================
Reinforcement Learning tools for research
=========================================

This repository contains the code developped and used by Aymeric CÔME, to conduct experiments in RL.

Dependencies
============

The project is managed using `Poetry <https://python-poetry.org/>`_, and the dependencies are specified in ``pyproject.toml``. The development has been made in Python 3.10, and relies on the `Gym <https://github.com/openai/gym>`_ library.


Testing
=======
This module uses `Pytest <https://docs.pytest.org/en/7.1.x/>`_ as a testing framework. The unitary tests are located in ``tests/``, and can be run with the following command from root:
::
   pytest

Credits
=======

Numerous pieces of codes in this repository are largely inspired by '<https://gitlab.inria.fr/omaillar/average-reward-reinforcement-learning/>', developped by Odalric MAILLARD.
