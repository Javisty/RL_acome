"""UCRL2 learning algorithm from Jaksch et al. (2010)."""
import numpy as np

import gym.spaces as spaces
from rlberry.agents import AgentWithSimplePolicy

from .utils import greedy_policy


class UCRL2(AgentWithSimplePolicy):
    """
    UCRL2 algorithm from [1]. Extended Value Iteration on optimistic MDP.

    [1] Near-optimal Regret Bounds for Reinforcement Learning,
    Jaksch et al. (The Journal of Machine Learning Research, 2010).

    Parameters:
    -----------
    env : gym.Env or tuple (constructor, kwargs)
        Environment used to fit the agent.
    delta : positive float, default 0.1
        Confidence factor.
    r_max : float, default 1.
        Maximum value for reward in environments.
    n_EVI : int > 0, default 1000
        Maximum number of iterations for EVI, in case it doesn't converge.
    **kwargs :
        Additional keyword arguments are passed to AgentWithSimplePolicy.
    """

    def __init__(self, env, delta=0.1, r_max=1., n_EVI=1000, **kwargs):
        # Check arguments
        assert delta > 0, "The confidence factor must be positive"
        assert r_max < float('inf'), "The problem must be bounded in reward!"
        assert isinstance(env.observation_space, spaces.Discrete)
        assert isinstance(env.action_space, spaces.Discrete)

        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.n_evi = n_EVI

        self.delta = delta
        self.r_max = r_max

        self.S = env.observation_space.n
        self.A = env.action_space.n

        # Initialise value holders

        self.t = 1  # current time-step
        self.episode = 0  # current episode
        self.tk = self.t  # start time-step of current episode

        # State-action counts within the current episode
        self.episode_counts = np.zeros((self.S, self.A), dtype=int)

        # State-action counts before episode. Init with 1
        self.sa_counts = np.ones((self.S, self.A), dtype=int)

        # Accumulated rewards before episode
        self.r_accum = np.zeros((self.S, self.A))

        # Transition (s, a, s') counts before episode
        self.sas_counts = np.zeros((self.S, self.A, self.S), dtype=int)

        self.pi = np.zeros(self.S, dtype=int)  # Current policy

    def compute_estimates(self):
        """Compute and returns current estimates for reward and transitions."""
        return self.r_accum/self.sa_counts, self.sas_counts/self.sa_counts[:, :, np.newaxis]

    def reward_bound(self):
        """Return the reward optimistic bounds."""
        S, A, tk, delta = self.S, self.A, self.tk, self.delta
        Nk = self.sa_counts
        bound = np.sqrt(7 * np.log(2 * S * A * tk / delta)/(2 * Nk))
        return bound

    def transition_bound(self):
        """Return the transition probabilities optimistic bounds."""
        S, A, tk, delta = self.S, self.A, self.tk, self.delta
        Nk = self.sa_counts
        bound = np.sqrt(14 * S * np.log(2 * S * A * tk / delta)/Nk)
        return bound

    def extended_value_iteration(self, r_hat, p_hat, d_r, d_p, epsilon):
        """
        Perform Extended Value Iteration on the optimistic MDP.

        Parameters:
        -----------
        r_hat : (S, A) numpy.array
            Estimated rewards.
        p_hat : (S, A, S) numpy.array
            Estimated transition probabilities.
        d_r : (S, A) numpy.array
            The optimistic bounds on estimated reward.
        d_p : (S, A) numpy.array

            The optimistic bounds on estimated transition kernel.
        epsilon : float >= 0
            The precision of EVI.

        Output:
        -------
        The value function and the corresponding greedy policy on the
        optimistically extended MDP.
        """
        S = self.S

        u0, u = np.zeros(S), np.zeros(S)
        proba_max = np.zeros_like(p_hat)

        n_iter, n_max = 0, self.n_evi

        diff = float('inf')
        while diff > epsilon and n_iter < n_max:
            proba_max = self.inner_max_EVI(p_hat, d_p, u0)
            inner_max = proba_max @ u0
            q = r_hat + d_r + inner_max  # Q-value function
            u = np.max(q, axis=-1)  # update value

            grad = np.abs(u - u0)
            diff = np.max(grad) - np.min(grad)
            u0 = u

            n_iter += 1

        if n_iter >= n_max:
            print("EVI didn't converge")

        return u, greedy_policy(q)

    def inner_max_EVI(self, p, bound, value):
        """
        Compute the maximum scalar product between value and optimistic probas.

        The optimistic probability distributions are the set of probabilities
        distant from p by less than bound, w.r.t. l1 norm. This is a convex
        polytope, leading to an easy optimisation.

        Parameters:
        -----------
        p : (S, A, S) numpy.array
            Center probability.
        d_p : (S, A) numpy.array
            Maximal distance from p accepted.
        value : S numpy.array
            Value function over states.

        Output:
        -------
        The (S, A) maximal scalar product over the last dimension.
        """
        sorted = list(np.argsort(value))
        idx = sorted.pop()  # index of highest value
        sorted = sorted[::-1]

        p_max = np.copy(p)
        p_max[:, :, idx] = np.minimum(1, p[:, :, idx] + bound/2)

        while (np.sum(p_max, axis=-1) > 1).any():
            idx = sorted.pop()  # index of lowest value not visited yet
            p_max[:, :, idx] = np.maximum(0, 1 - np.sum(np.delete(p_max, [idx],
                                                                  axis=-1),
                                                        axis=-1))

        return p_max

    def compute_optimistic_policy(self):
        """Return the optimistic policy based on current observations."""
        r_hat, p_hat = self.compute_estimates()
        d_r, d_p = self.reward_bound(), self.transition_bound()
        eps = 1/np.sqrt(self.tk)

        return self.extended_value_iteration(r_hat, p_hat, d_r, d_p, eps)[1]

    def compute_empirical_policy(self):
        """Return the policy based on current observations."""
        r_hat, p_hat = self.compute_estimates()
        d_r, d_p = np.zeros((self.S, self.A)), np.zeros((self.S, self.A))
        tk = self.tk

        return self.extended_value_iteration(r_hat, p_hat, d_r, d_p, tk)[1]

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
        s = self.env.state  # current state and action
        print()

        while self.t < budget:
            self.episode += 1
            self.tk = self.t
            print(f"\rStarting episode {self.episode} at time-step {self.t}")

            self.episode_counts.fill(0)

            self.pi = self.compute_optimistic_policy()

            a = self.policy(s)
            while self.episode_counts[s, a] < self.sa_counts[s, a]:
                s_next, r, done, _ = self.env.step(a)

                # Update counts, except for sa_counts
                self.episode_counts[s, a] += 1
                self.r_accum[s, a] += r
                self.sas_counts[s, a, s_next] += 1

                if done:  # for episodic tasks
                    self.env.reset()
                    s = self.env.state
                else:
                    s = s_next

                a = self.policy(s)
                self.t += 1

            # Add episode counts to overall counts
            self.sa_counts += self.episode_counts

        self.episode += 1
        self.tk = self.t

        self.pi = self.compute_empirical_policy()
        return self.pi, self.r_accum.sum()

    def policy(self, state):
        """Deterministic policy."""
        return self.pi[state]
