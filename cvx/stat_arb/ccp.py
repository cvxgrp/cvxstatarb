from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


# Filter out the specific UserWarning
warnings.filterwarnings(
    "ignore", message="Solution may be inaccurate.", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="Solution may be inaccurate.\
          Try another solver,\
              adjusting the solver settings,\
                  or solve with verbose=True\
                      for more information.",
)


@dataclass(frozen=True)
class Metrics:
    daily_profit: pd.Series
    entry_date: pd.DatetimeIndex
    exit_trigger: pd.DatetimeIndex
    exit_date: pd.DatetimeIndex

    # days_in_period: int = 252

    @property
    def mean_profit(self):
        """Mean profit"""
        return self.daily_profit.mean()

    @property
    def std_profit(self):
        """Standard deviation of profit"""
        return self.daily_profit.std()

    @property
    def total_profit(self):
        """Total profit"""
        return self.daily_profit.sum()

    @property
    def sr_profit(self):
        """Sharpe ratio of profit"""
        return self.mean_profit / self.std_profit


def construct_stat_arb(
    prices,
    P_max=None,
    spread_max=1,
    moving_mean=True,
    s_init=None,
    mu_init=None,
    seed=None,
    M=None,
    solver="CLARABEL",
    second_pass=False,
):
    if seed is not None:
        np.random.seed(seed)

    # Drop nan columns in prices; allows for missing data
    prices = prices.dropna(axis=1)

    # Scale prices; remember to scale back later
    P_bar = prices.mean(axis=0).values.reshape(-1, 1)
    prices = prices / P_bar.T

    state = _State(
        prices,
        P_max=P_max,
        spread_max=spread_max,
        moving_mean=moving_mean,
        solver=solver,
    )

    if s_init is None or mu_init is None:
        state.reset()
    else:
        state.s.value = s_init
        state.mu.value = mu_init

    obj_old, obj_new = 1, 10
    i = 0

    while np.linalg.norm(obj_new - obj_old) / obj_old > 1e-3 and i < 5:
        state.iterate()
        if state.prob.status == "optimal" or state.prob.status == "optimal_inaccurate":
            obj_old = obj_new
            obj_new = state.prob.value
        else:
            print("Solver failed... resetting (2)")
            # print(state.prob.status) # TODO: print?
            state.reset()
            obj_old, obj_new = 1, 10
        i += 1

    # Second pass
    # if P_max is not None:
    if not second_pass:
        s_times_P_bar = np.abs(state.s.value) * state.P_bar
        s_at_P_bar = np.abs(state.s.value).T @ state.P_bar
        non_zero_inds = np.where(np.abs(s_times_P_bar) > 0.5e-1 * s_at_P_bar)[0]

        if len(non_zero_inds) == 0:  # TODO: fix this
            return None

        # Scale back prices for second pass
        prices = prices * P_bar.T
        prices_new = prices.iloc[:, non_zero_inds]

        s_init = state.s.value[non_zero_inds]
        mu_init = state.mu.value

        return construct_stat_arb(
            prices_new,
            P_max=P_max,  # TODO: P_max=None for second pass?
            spread_max=spread_max,
            moving_mean=moving_mean,
            s_init=s_init,
            mu_init=mu_init,
            seed=None,
            second_pass=True,
        )

    #### TODO: clean up below

    # remove zero inds again
    s_times_P_bar = np.abs(state.s.value) * state.P_bar
    s_at_P_bar = np.abs(state.s.value).T @ state.P_bar
    non_zero_inds = np.where(np.abs(s_times_P_bar) > 0.5e-1 * s_at_P_bar)[0]

    if len(non_zero_inds) == len(state.s.value):
        # Scale s and return stat arb
        state.s.value = state.s.value / P_bar
        stat_arb = state.build()

        return stat_arb

    elif len(non_zero_inds) == 0:  # TODO: fix this
        return None

    else:
        # Scale back prices for second pass
        prices = prices * P_bar.T
        prices_new = prices.iloc[:, non_zero_inds]

        s_init = state.s.value[non_zero_inds]
        mu_init = state.mu.value

        return construct_stat_arb(
            prices_new,
            P_max=P_max,  # TODO: P_max=None for second pass?
            spread_max=spread_max,
            moving_mean=moving_mean,
            s_init=s_init,
            mu_init=mu_init,
            seed=None,
            second_pass=True,
        )


class _State:
    """
    Helper class for constructing stat arb using the convex-concave procedure
    """

    def __init__(
        self,
        prices,
        P_max=None,
        spread_max=1,
        solver="CLARABEL",
        mu_memory=21,
        moving_mean=1,
    ):
        self.T, self.n = prices.shape
        self.s = cp.Variable((self.n, 1), name="s")
        # self.mu = cp.Variable(name="mu", nonneg=True)
        self.mu_memory = mu_memory
        self.moving_mean = moving_mean

        if moving_mean:
            self.mu = cp.Variable((self.T, 1))
        else:
            self.mu = cp.Variable(nonneg=True)

        self.p = cp.Variable((self.T, 1), name="p")
        self.P_max = P_max  # allow for P_max=0 for second pass
        self.prices = prices
        self.spread_max = spread_max

        if self.moving_mean:
            self.P_bar = prices.iloc[mu_memory:].mean(axis=0).values.reshape(-1, 1)
        else:
            self.P_bar = prices.mean(axis=0).values.reshape(-1, 1)

        # print(self.P_bar)
        self.solver = solver
        # self.prices = self.prices / self.P_bar.T

        # Construct linearized convex-concave problem
        self.grad_g = cp.Parameter((self.T, 1), name="grad_g")

        self.obj = cp.Maximize(self.grad_g[mu_memory:].T @ self.p[mu_memory:])

        if self.moving_mean:
            self.cons = [
                cp.abs(self.p[mu_memory:] - self.mu[mu_memory:]) <= self.spread_max
            ]

            self.cons += [self.p[mu_memory:] == self.prices.values[mu_memory:] @ self.s]
        else:
            self.cons = [cp.abs(self.p - self.mu) <= self.spread_max]
            self.cons += [self.p == self.prices.values @ self.s]

        if moving_mean:
            ################################################################
            conv_arr = 1 / mu_memory * np.ones(mu_memory)
            conv_correction = np.array(
                [mu_memory / (i + 1) for i in range(mu_memory - 1)]
                + [1] * (self.T - mu_memory + 1)
            )

            conv = cp.multiply(
                cp.convolve(conv_arr, self.p.flatten())[: self.T], conv_correction
            )

            self.cons += [self.mu.flatten()[mu_memory:] == conv[mu_memory:]]
        ################################################################
        else:
            self.cons += [self.mu >= 0]
            # self.cons += [cp.sum(self.p) / self.T == self.mu]

        if self.P_max is not None:
            self.cons += [cp.abs(self.s).T @ self.P_bar <= self.P_max]

        self.prob = cp.Problem(self.obj, self.cons)

        # Solve once for speedup later
        self.reset()
        p = self.prices.values @ self.s.value
        self.grad_g.value = self._get_grad_g(p)
        try:
            self.prob.solve(solver=self.solver, verbose=False)
        except cp.SolverError:
            print("Initial solve failed...")

            print(np.max(p))

        # For debugging
        self.solve_times = []

    @property
    def assets(self):
        return list(self.prices.columns)

    @property
    def shape(self):
        return self.prices.shape

    def reset(self):
        """
        Resets to random feasible point
        """

        # Initialize s uniformly at random
        s = np.random.uniform(0, 1, (self.n, 1))
        self.s.value = s

    def iterate(self):
        """
        Performs one iteration of the convex concave procedure
        """

        # Update p_centered and grad_g
        p = self.prices.values @ self.s.value
        self.grad_g.value = self._get_grad_g(p)

        # Solve problem
        start = time.time()

        try:
            self.prob.solve(solver=self.solver, verbose=False)

        except cp.SolverError:
            print("Solver failed, resetting...")

            self.reset()

        end = time.time()
        self.solve_times.append(end - start)

        return self

    def _get_grad_g(self, pk):
        """
        param pk: Tx1 array of current portfolio evolution

        returns the gradient of g at pk
        """
        M = self.mu_memory
        (pd.Series(pk.flatten()).rolling(M, min_periods=1).mean().values.reshape(-1, 1))
        start = M

        grad_g = np.zeros(pk.shape)

        grad_g[start + 0] = pk[start + 0] - pk[start + 1]
        grad_g[-1] = pk[-1] - pk[-2]
        grad_g[start + 1 : -1] = 2 * pk[start + 1 : -1] - pk[start:-2] - pk[start + 2 :]

        return (
            2 * grad_g / np.mean(np.abs(grad_g))
        )  # 2 * grad_g is the actual gradient; we scale for numerical stability

    def build(self):
        assets_dict = dict(zip(self.assets, self.s.value))
        stat_arb = StatArb(
            assets=assets_dict,
            mu=self.mu.value,
            mu_memory=self.mu_memory,
            moving_mean=self.moving_mean,
            P_max=self.P_max,
        )

        return stat_arb


@dataclass(frozen=True)
class StatArb:
    """
    Stat arb class
    """

    assets: dict
    mu: float  #
    mu_memory: int = None
    moving_mean: bool = True
    P_max: float = None
    spread_max: float = 1

    def __setitem__(self, __name: int, __value: float) -> None:
        self.assets[__name] = __value

    def __getitem__(self, key: int) -> float:
        return self.assets.__getitem__(key)

    def items(self):
        return self.assets.items()

    def evaluate(self, prices: pd.DataFrame):
        value = 0

        for asset, position in self.items():
            value += prices[asset] * position
        return value

    def leverage(self, prices: pd.DataFrame):
        """Computes the value of absolute dollar value holdings"""
        return prices[self.asset_names] @ self.stocks.abs()

    def copy(self):
        return StatArb(
            assets=self.assets.copy(),
            mu=self.mu.copy(),
            mu_memory=self.mu_memory,
            moving_mean=self.moving_mean,
            P_max=self.P_max,
            spread_max=self.spread_max,
        )

    @property
    def stocks(self):
        """
        returns the vector of positions
        """
        return pd.Series(self.assets).astype(float)

    @property
    def asset_names(self):
        """
        returns list of assets in StatArb
        """
        return list(self.assets.keys())

    @property
    def n(self):
        """
        returns the number of assets
        """
        return len(self.assets)

    def get_q(self, prices: pd.DataFrame, mu=None, T_max=500):
        """
        param prices: DataFrame of prices
        param mu: series of stat-arb midpoints if moving-band; else float
        param T_max: the number of periods until position is exited. The
        position is exited linearly over the 21 following periods. The position
        is zero on the last period.

        returns the vector of investments in stat arb based on linear trading
        policy: q_t = mu_t - p_t
        """
        if mu is None:
            assert not self.moving_mean
            "mu must be specified for moving mean stat arb"
            mu = self.mu

        if self.moving_mean:
            assert (mu.index == prices.index).all()

        p = self.evaluate(prices)

        q = mu - p
        q.name = "q"

        exit_length = 21
        exit_index = min(T_max - 1, len(q) - 1)

        # after first breach, exit over exit_length days
        weights = [i / exit_length for i in range(exit_length)]
        weights = np.array(weights)[::-1]

        length = len(q[exit_index : exit_index + exit_length])
        q[exit_index : exit_index + exit_length] = (
            q[exit_index : exit_index + exit_length] * weights[:length]
        )

        q[exit_index + exit_length :] = 0
        q[-1] = 0

        exit_trigger = min(exit_index, len(q) - 1)
        exit_trigger = q.index[exit_trigger]

        exit_date = min(exit_index + exit_length - 1, len(q) - 1)
        exit_date = q.index[exit_date]

        return q, exit_trigger, exit_date

    def get_positions(self, prices: pd.DataFrame, mu=None, T_max=500):
        """
        param prices: DataFrame of prices
        param mu: series of stat-arb midpoints if moving-band; else float
        param T_max: the number of periods until position is exited. The
        position is exited linearly over the 21 following periods. The position
        is zero on the last period.

        returns the positions of each individual asset in the stat-arb,
        following the linear policy given by self.get_q
        """

        q, _, _ = self.get_q(prices, mu, T_max=T_max)
        q = pd.concat([q] * self.n, axis=1)
        stocks = self.stocks.values.reshape(-1, 1)
        positions = q * (stocks.T)
        positions.columns = self.asset_names

        return positions

    def validate(self, prices, mu, cutoff, profit_cutoff=None):
        """
        param prices: DataFrame of prices
        param mu: series of stat-arb midpoints if moving-band; else float
        param cutoff: the maximum deviation from the stat-arb mean
        param profit_cutoff: the minimum profit of the stat-arb

        returns True if the stat-arb is valid on the validation set (prices);
        else False
        """
        p = self.evaluate(prices)
        if (p - mu).abs().max() >= cutoff:
            return False

        m, _ = self.metrics(prices, mu, cutoff)

        if m is None:
            return False

        if profit_cutoff is not None:
            if m.total_profit < profit_cutoff:
                return False

        return True

    def metrics(
        self,
        prices: pd.DataFrame,
        mu: pd.Series = None,
        T_max=500,
    ):
        """
        param prices: DataFrame of prices
        param mu: series of stat-arb midpoints if moving-band; else float
        param T_max: the number of periods until position is exited. The
        position is exited linearly over the 21 following periods. The position
        is zero on the last period.

        returns: Metrics object
        """

        # Get price evolution of portfolio
        p = self.evaluate(prices)

        q, exit_trigger, exit_date = self.get_q(prices, mu, T_max=T_max)

        price_changes = p.ffill().diff()
        previous_position = q.shift(1)
        profits = previous_position * price_changes

        # Set first row to NaN; no profit on first day
        profits.iloc[0] = np.nan
        entry_date = q.index[0]

        return Metrics(
            daily_profit=profits.dropna(),
            entry_date=entry_date,
            exit_trigger=exit_trigger,
            exit_date=exit_date,
        )
