"""SMDP-UCRL from Fruit & Lazaric (2017)."""
import numpy as np
from numpy import ma

import gym.spaces as spaces
from rlberry.agents import AgentWithSimplePolicy

from .options import OptionSet
from .utils import greedy_policy


class SUCRL(AgentWithSimplePolicy):
    """
    UCRL-SMDP algorithm from [1]. Direct adaptation of UCRL [2] on semi-MDPs.

    [1] Exploration-Exploitation in MDPs with Options, Fruit & Lazaric
    (AISTATS, 2017).
    [2] Near-optimal Regret Bounds for Reinforcement Learning,
    Jaksch et al. (The Journal of Machine Learning Research, 2010).

    Notes:
    ------
    Only MDPs with options are handled so far, and not any semi-MDPs.

    Parameters:
    -----------
    env : gym.Env or tuple (constructor, kwargs)
        Environment used to fit the agent.
    options : rl_acome.learners.options.OptionSet
        Set of options to consider onto env.
    tau_min : float > 0
        Minimum holding time.
    tau_max : float > 0
        Maximum holding time.
    b_r, sigma_r : floats
        Parameters of the sub-exponential distribution of reward.
    b_tau, sigma_tau : floats
        Parameters of the sub-exponential distribution of holding time.
    delta : positive float, default 0.1
        Confidence factor.
    r_max : float, default 1.
        Maximum value for reward in environments.
    tau_diff : float between 0 and tau_min, default 0.1
        tau_min - tau_diff gives the tau parameter for EVI.
    n_EVI : int > 0, default 1000
        Maximum number of iterations for EVI, in case it doesn't converge.
    **kwargs :
        Additional keyword arguments are passed to AgentWithSimplePolicy.
    """

    def __init__(self, env, options, tau_max, tau_min, b_r, sigma_r, b_tau,
                 sigma_tau, delta=0.1, r_max=1, tau_diff=0.1, n_EVI=1000,
                 **kwargs):
        # Check arguments
        assert delta > 0, "The confidence factor must be positive"
        assert r_max < float('inf'), "The problem must be bounded in reward!"
        assert 0 <= tau_diff < tau_min
        assert isinstance(env.observation_space, spaces.Discrete)
        assert isinstance(env.action_space, spaces.Discrete)
        assert isinstance(options, OptionSet)
        assert options.env == env, "The options environment doesn't match env!"

        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.options = options

        self.n_evi = n_EVI

        self.delta = delta
        self.r_max = r_max
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.b_r = b_r
        self.sigma_r = sigma_r
        self.b_tau = b_tau
        self.sigma_tau = sigma_tau
        self.tau = tau_min - tau_diff

        self.S = env.observation_space.n
        self.A = env.action_space.n
        self.O = options.O

        # Initialise value holders

        self.t = 1  # current time-step
        self.episode = 0  # current episode
        self.tk = self.t  # start time-step of current episode

        mask = ~self.options.mask  # indicate invalid state-option pairs
        # State-action counts within the current episode
        self.episode_counts = ma.zeros((self.S, self.O), dtype=int)

        # State-action counts before episode. Init with 1
        self.sa_counts = ma.ones((self.S, self.O), dtype=int)
        # When (s, a) is visited first, don't increment sa_counts
        self.already_visited = ma.zeros((self.S, self.O), dtype=bool)

        # Accumulated rewards before episode
        self.r_accum = ma.zeros((self.S, self.O))

        # Transition (s, a, s') counts before episode
        self.sas_counts = ma.zeros((self.S, self.O, self.S), dtype=int)

        # Accumulated duration of option before episode
        self.t_accum = ma.zeros((self.S, self.O), dtype=int)

        self.pi = np.zeros(self.S, dtype=int)  # Current policy

        # Mask array, to avoid considering non-available options
        self.episode_counts[mask] = ma.masked
        self.sa_counts[mask] = ma.masked
        self.already_visited[mask] = ma.masked
        self.r_accum[mask] = ma.masked
        self.sas_counts[mask] = ma.masked
        self.t_accum[mask] = ma.masked

    def compute_estimates(self):
        """Compute and returns current estimates for reward and transitions."""
        return (self.r_accum/self.sa_counts,
                self.sas_counts/self.sa_counts[:, :, np.newaxis],
                self.t_accum/self.sa_counts)

    def counts_condition(self):
        """
        Inequality on sa_counts, in the confidence bounds formulae.

        The formulae for the confidence bounds on reward and holding time
        depend on an inequality between sa_counts and the output of this
        method.

        Output:
        -------
        out : (S, A) numpy.array
            Mask array, True if the first bound term should be used, False
            otherwise.
        """
        return (2 * self.b_r**2 / self.sigma_r**2 *
                np.log(240 * self.S * self.O * self.tk**7 / self.delta)
                <= self.sa_counts)

    def reward_bound(self, Nk_gt_cond=None):
        """Return the reward optimistic bounds."""
        S, O, tk, delta = self.S, self.O, self.tk, self.delta
        Nk, s_r, b_r = self.sa_counts, self.sigma_r, self.b_r

        Nk_gt_cond = self.counts_condition() if Nk_gt_cond is None else Nk_gt_cond
        bound = np.zeros((S, O))

        # If sa_counts >= condition
        bound[Nk_gt_cond] = s_r * np.sqrt(14 * np.log(2 * S * O * tk / delta)
                                          / Nk[Nk_gt_cond])
        # If sa_counts < condition
        bound[~Nk_gt_cond] = (14 * b_r * np.log(2 * S * O * tk / delta)
                              / Nk[~Nk_gt_cond])

        return bound

    def transition_bounds(self):
        """Return the transition probabilities optimistic bounds."""
        S, O, tk, delta = self.S, self.O, self.tk, self.delta
        Nk = self.sa_counts

        return np.sqrt(14 * S * np.log(2 * S * O * tk / delta)/Nk)

    def holding_time_bound(self, Nk_gt_cond=None):
        """Return the holding time optimistic bounds."""
        S, O, tk, delta = self.S, self.O, self.tk, self.delta
        Nk, s_tau, b_tau = self.sa_counts, self.sigma_tau, self.b_tau

        Nk_gt_cond = self.counts_condition() if Nk_gt_cond is None else Nk_gt_cond
        bound = np.zeros((S, O))

        # If sa_counts >= condition
        bound[Nk_gt_cond] = s_tau * np.sqrt(14 * np.log(2 * S * O * tk / delta)
                                            / Nk[Nk_gt_cond])
        # If s_counts < condition
        bound[~Nk_gt_cond] = (14 * b_tau * np.log(2 * S * O * tk / delta)
                              / Nk[~Nk_gt_cond])

        return bound

    def extended_value_iteration(self, r_hat, p_hat, tau_hat,
                                 d_r, d_p, d_tau, epsilon):
        """
        Perform Extended Value Iteration on the optimistic MDP.

        Parameters:
        -----------
        r_hat : (S, A) numpy.array
            Estimated rewards.
        p_hat : (S, A, S) numpy.array
            Estimated transition probabilities.
        tau_hat : (S, A) numpy.array
            Estimated holding times.
        d_r : (S, A) numpy.array
            The optimistic bounds on estimated reward.
        d_p : (S, A) numpy.array
            The optimistic bounds on estimated transition kernel.
        d_tau : (S, A) numpy.array
            The optimistic bounds on estimated holding times.
        epsilon : float >= 0
            The precision of EVI.

        Output:
        -------
        The value function and the corresponding greedy policy on the
        optimistically extended MDP.
        """
        S = self.S
        tau_max, tau_min, tau = self.tau_max, self.tau_min, self.tau
        r_max = self.r_max

        u0, u = np.zeros(S), np.zeros(S)
        p_tilde = np.zeros_like(p_hat)

        n_iter, n_max = 0, self.n_evi

        diff = float('inf')
        while diff > epsilon and n_iter < n_max:
            p_tilde = self.inner_max_EVI(p_hat, d_p, u0)
            diff_u = ma.dot(p_tilde, u0) - u0[:, np.newaxis]
            r_tilde = np.minimum(r_hat + d_r, r_max * tau_max)
            tau_tilde = np.clip(tau_hat - np.sign(r_tilde + tau * diff_u)
                                * d_tau, tau_max, tau_min)
            q = (r_tilde + tau * diff_u) / tau_tilde  # Q-value function
            u = np.max(q, axis=-1) + u0  # update value

            grad = np.abs(u - u0)
            diff = np.max(grad) - np.min(grad)
            u0 = u

            n_iter += 1

        if n_iter >= n_max:
            print("EVI didn't converge")

        return u, greedy_policy(q)

    def inner_max_EVI(self, p, bound, value):
        """
        Compute the optimistic probas maximizing the scalar product with value.

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
        The (S, A, S) maximal optimistic probability distribution.
        """
        sorted = list(np.argsort(value))
        idx = sorted.pop()  # index of highest value
        sorted = sorted[::-1]

        p_max = p  # ma.copy(p) if needs be
        p_max[:, :, idx] = np.minimum(1, p[:, :, idx] + bound/2)

        while (np.sum(p_max, axis=-1) > 1).any():
            idx = sorted.pop()  # index of lowest value not visited yet
            p_max[:, :, idx] = np.maximum(0, 1 - np.sum(np.delete(p_max, [idx],
                                                                  axis=-1),
                                                        axis=-1))

        return p_max

    def compute_optimistic_policy(self):
        """Return the optimistic policy based on current observations."""
        r_hat, p_hat, tau_hat = self.compute_estimates()
        condition = self.counts_condition()

        d_r = self.reward_bound(Nk_gt_cond=condition)
        d_p = self.transition_bounds()
        d_tau = self.holding_time_bound(Nk_gt_cond=condition)
        eps = 1/np.sqrt(self.tk)

        return self.extended_value_iteration(r_hat, p_hat, tau_hat,
                                             d_r, d_p, d_tau, eps)[1]

    def compute_empirical_policy(self):
        """Return the policy based on current observations."""
        r_hat, p_hat, tau_hat = self.compute_estimates()

        d_r = np.zeros_like(r_hat)
        d_p = np.zeros_like(p_hat[:, :, 0])
        d_tau = np.zeros_like(tau_hat)
        eps = 1/np.sqrt(self.tk)

        return self.extended_value_iteration(r_hat, p_hat, tau_hat,
                                             d_r, d_p, d_tau, eps)[1]

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

        while self.t < budget:
            self.episode += 1
            self.tk = self.t
            print(f"Starting episode {self.episode} at time-step {self.t}")

            self.episode_counts.fill(0)

            self.pi = self.compute_optimistic_policy()

            o = self.policy(s)
            while self.episode_counts[s, o] < self.sa_counts[s, o]:
                s_next, r, tau = self.play_option(o)

                if not self.already_visited[s, o]:  # first visit of (s, o)
                    self.already_visited[s, o] = True
                    self.sa_counts[s, o] -= 1  # remove initial shift

                # Update counts, except for sa_counts
                self.episode_counts[s, o] += 1
                self.r_accum[s, o] += r
                self.t_accum[s, o] += tau
                self.sas_counts[s, o, s_next] += 1

                s = s_next
                if s not in self.options.state_options:
                    s = self.env.reset()

                o = self.policy(s)
                self.t += 1

            # Add episode counts to overall sa_counts
            self.sa_counts += self.episode_counts

        self.episode += 1
        self.tk = self.t

        self.pi = self.compute_empirical_policy()
        return self.pi, self.r_accum.sum()

    def play_option(self, idx):
        """
        Play option until termination.

        Return next state, accumulated reward and duration.
        """
        return self.options.get_option(idx).play()

    def policy(self, state):
        """Deterministic policy."""
        return self.pi[state]
