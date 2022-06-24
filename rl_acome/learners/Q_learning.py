"""Q-learning algorithm from Watkins (1989)."""

import numpy as np

import gym.spaces as spaces
from rlberry.agents import AgentWithSimplePolicy

from .utils import greedy_policy, eps_greedy_action


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
    gamma : 0 <= float <= 1, default 1
        Discount factor.
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

            if self.t % 1000 == 0:
                print((f"Playing action {action} at time step {self.t} "
                       f"from state {state}"), end='\r')

            s_next, r, done, _ = self.env.step(action)

            # Update Q-value estimate
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


class UCB_QL(AgentWithSimplePolicy):
    """
    UCB-Q-learning from [1]. Optimistic Q-value estimates.

    [1] Q-learning with UCB Exploration is Sample efficient for
    Infinite-Horizon MDP, Dong, Wang, Chen, Wang (2019).

    Notes:
    ------
    This algorithm is very conservative, leading to poor convergence speed in
    practice. See UCB_QL2 for a faster variant.

    Parameters:
    -----------
    env : gym.Env or tuple (constructor, kwargs)
        Environment used to fit the agent.
    eps : 0 < float, default 0.1
        Parameter for eps-optimality, in PAC-MDP defintion.
    delta : 0 < float < 1, default 0.1
        Confidence probability parameter, in PAC-MDP definition.
    gamma : 0 <= float < 1, default 0.9
        Discount factor.
    **kwargs :
        Additional keyword arguments are passed to AgetnWithSimplePolicy.
    """

    def __init__(self, env, eps=0.1, delta=0.1, gamma=0.9, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)

        self.eps = eps
        self.delta = delta
        self.gamma = gamma

        self.S, self.A = self.env.observation_space.n, self.env.action_space.n

        # Initialise value holders

        self.t = 1  # current time-step

        self.V_hat = np.zeros(self.S)
        # Two parallel estimates for Q
        self.Q = 1 / (1 - gamma) * np.ones((self.S, self.A))
        self.Q_hat = 1 / (1 - gamma) * np.ones((self.S, self.A))
        self.N_sa = np.zeros((self.S, self.A), dtype=int)  # s-a pairs count

        # Constants from paper
        self.c2 = 4 * np.sqrt(2)
        self.R = np.ceil(np.log(3 / (eps * (1 - gamma))) / (1 - gamma))
        self.L = np.floor(np.log2(self.R))
        self.xi_L = - eps / (3 * 2**(self.L + 2) * np.log(1 - gamma))
        self.M = max(10, np.ceil(- 2 * np.log2(self.xi_L * (1 - gamma))))
        self.eps1 = - eps / (24 * self.R * self.M * np.log(1 - gamma))
        self.H = np.log((1 - gamma) * self.eps1) / np.log(gamma)

        self.reward = 0.

        # Current policy, with random initialisation over available options
        self.pi = np.random.randint(0, self.A, size=self.S)

    def iota(self, k):
        """Implement iota function from the paper."""
        return np.log(self.S * self.A * (k+1) * (k+2) / self.delta)

    def alpha(self, k):
        """Implement alpha function from the paper."""
        return (self.H + 1) / (self.H + k)

    def b(self, k):
        """Implement b function from the paper."""
        return self.c2 * np.sqrt(self.H * self.iota(k) / k) / (1 - self.gamma)

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
        N, gamma = self.N_sa, self.gamma
        Q, Q_hat, V_hat = self.Q, self.Q_hat, self.V_hat

        state = self.env.state
        while self.t <= budget:
            action = self.choose_action(state)

            if self.t % 1000 == 0:
                print((f"Playing action {action} at time step {self.t} "
                       f"from state {state}"), end='\r')

            s_next, r, done, _ = self.env.step(action)

            # Update Q-value estimates
            N[state, action] += 1
            k = N[state, action]
            l_rate = self.alpha(k)

            V_hat[s_next] = np.max(Q_hat[s_next])
            q_bar = r + self.b(k) + gamma * V_hat[s_next]
            Q[state, action] = (1 - l_rate) * Q[state, action] + l_rate * q_bar
            Q_hat[state, action] = min(Q[state, action], Q_hat[state, action])

            if done:
                state = self.env.reset()
            else:
                state = s_next

            self.reward += r
            self.t += 1

        return self.get_policy(), self.reward

    def choose_action(self, state):
        """Return an action according to greedy policy on current state."""
        q = self.Q_hat[state, :]
        return np.random.choice(np.where(q == np.max(q))[0])

    def get_policy(self):
        """Return greedy policy w.r.t. current Q-value estimate."""
        return greedy_policy(self.Q)

    def get_optimistic_policy(self):
        """Return greedy policy w.r.t. current Q_hat-value."""
        return greedy_policy(self.Q_hat)

    def update_policy(self):
        """Update intern policy."""
        self.pi = self.get_policy()

    def policy(self, state):
        """Deterministic policy over options. Doesn't update it before."""
        return self.pi[state]


class UCB_QL2(AgentWithSimplePolicy):
    """
    UCB-Q-learning from [1], revisited. Optimistic Q-value estimates.

    [1] Q-learning with UCB Exploration is Sample efficient for
    Infinite-Horizon MDP, Dong, Wang, Chen, Wang (2019).

    Notes:
    ------
    Variant from UCB_QL. Should provide faster convergence.

    Parameters:
    -----------
    env : gym.Env or tuple (constructor, kwargs)
        Environment used to fit the agent.
    eps : 0 < float, default 0.1
        Parameter for eps-greedy policy over Q-values.
    mu : 0 < float, default 0.55
        Scaling factor for bonus term.
    gamma : 0 <= float < 1, default 0.9
        Discount factor.
    **kwargs :
        Additional keyword arguments are passed to AgetnWithSimplePolicy.
    """

    def __init__(self, env, eps=0.1, mu=0.55, gamma=0.9, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)

        self.eps = eps
        self.mu = mu
        self.gamma = gamma

        self.S, self.A = self.env.observation_space.n, self.env.action_space.n

        # Initialise value holders

        self.t = 1  # current time-step

        self.H = 1 / (1 - gamma)

        self.V_hat = np.zeros(self.S)
        # Two parallel estimates for Q
        self.Q = self.H * np.ones((self.S, self.A))
        self.Q_hat = self.H * np.ones((self.S, self.A))
        self.N_sa = np.zeros((self.S, self.A), dtype=int)  # s-a pairs count

        self.reward = 0.

        # Current policy, with random initialisation over available options
        self.pi = np.random.randint(0, self.A, size=self.S)

    def alpha(self, k):
        """Give custom learning rate, for state-action count k."""
        return (self.H + 1) / (self.H + k)

    def b(self, k):
        """Give custom bonus, for state-action count k."""
        return self.mu * np.sqrt(self.H / k)

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
        N, gamma = self.N_sa, self.gamma
        Q, Q_hat, V_hat = self.Q, self.Q_hat, self.V_hat

        state = self.env.state
        while self.t <= budget:
            action = self.choose_action(state)

            if self.t % 1000 == 0:
                print((f"Playing action {action} at time step {self.t} "
                       f"from state {state}"), end='\r')

            s_next, r, done, _ = self.env.step(action)

            # Update Q-value estimates
            N[state, action] += 1
            k = N[state, action]
            l_rate = self.alpha(k)

            V_hat[s_next] = np.max(Q_hat[s_next])
            q_bar = r + self.b(k) + gamma * V_hat[s_next]
            Q[state, action] = (1 - l_rate) * Q[state, action] + l_rate * q_bar
            Q_hat[state, action] = min(Q[state, action], Q_hat[state, action])

            if done:
                state = self.env.reset()
            else:
                state = s_next

            self.reward += r
            self.t += 1

        return self.get_policy(), self.reward

    def choose_action(self, state):
        """Return an action according to eps-greedy policy on current state."""
        return eps_greedy_action(self.Q_hat[state], self.eps)

    def get_policy(self):
        """Return greedy policy w.r.t. current Q-value estimate."""
        return greedy_policy(self.Q)

    def get_optimistic_policy(self):
        """Return greedy policy w.r.t. current Q_hat-value."""
        return greedy_policy(self.Q_hat)

    def update_policy(self):
        """Update intern policy."""
        self.pi = self.get_policy()

    def policy(self, state):
        """Deterministic policy over options. Doesn't update it before."""
        return self.pi[state]
