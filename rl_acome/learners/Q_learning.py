"""Q-learning algorithm from Watkins (1989)."""
import numpy as np

import gym.spaces as spaces
from rlberry.agents import AgentWithSimplePolicy

from .utils import greedy_policy


class Qlearning(AgentWithSimplePolicy):
    """
    Q-learning algorithm from [1]. Temporal Difference on Q-values.

    [1] Learning with Delayed Rewards, Watkins (1989).

    Parameters:
    -----------
    env : gym.Env or tuple (constructor, kwargs)
        Environment used to fit the agent.
    alpha : float > 0 or callable object, default 1/8
        Learning rate. If callable, must take state index, option index and
        time-step as arguments.
    **kwargs :
        Additional keyword arguments are passed to AgentWithSimplePolicy.
    """

    def __init__(self, env, alpha=0.1, gamma=1., **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)

        if isinstance(alpha, float):  # use alpha as a callable object
            self.alpha = lambda s, a, t: alpha
        else:
            self.alpha = alpha
        self.gamma = gamma

        self.S, self.A = self.env.observation_space.n, self.env.action_space.n

        # Initialise value holders

        self.t = 1  # current time-step

        self.V = np.zeros(self.S)
        self.Q = np.zeros((self.S, self.A))
        self.reward = 0.

        # Current policy, with random initialisation over available options
        self.pi = np.random.randint(0, self.A, size=self.S)

    def fit(self, budget, **kwargs):
        """
        Train the agent using the provided environment.

        Parameters:
        -----------
        budget : int
            Horizon, number of time steps to perform before stopping.
        **kwargs
            Extra arguments.

        Output:
        -------
        policy : S numpy.array
            The policy learned by the agent.
        reward : float
            The cumulative reward obtained during learning.
        """
        alpha, gamma = self.alpha, self.gamma
        Q = self.Q

        state = self.env.state
        while self.t <= budget:
            action = self.choose_action(state)
            # print((f"Playing action {action} at time step {self.t} "
            #        f"from state {state}"))

            s_next, r, done, _ = self.env.step(action)

            l_rate = alpha(state, action, self.t)
            q_bar = r + gamma * np.max(Q[s_next])
            Q[state, action] = (1 - l_rate) * Q[state, action] + l_rate * q_bar

            if done:
                state = self.env.reset()
            else:
                state = s_next

            self.reward += r
            self.t += 1

        return self.get_policy(), self.reward

    def choose_action(self, state):
        """Return an action according to greedy policy on current state."""
        q = self.Q[state, :]
        return np.random.choice(np.where(q == np.max(q))[0])

    def get_policy(self):
        """Return greedy policy w.r.t. current Q-value estimate."""
        return greedy_policy(self.Q)

    def update_policy(self):
        """Update intern policy."""
        self.pi = self.get_policy()

    def policy(self, state):
        """Deterministic policy over options. Doesn't update it before."""
        return self.pi[state]
