from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm

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
    moving_midpoint=True,
    midpoint_memory=21,
    s_init=None,
    mu_init=None,
    seed=None,
    solver="CLARABEL",
    second_pass=False,
):
    """
    NOTE: For numerical reasons, we recommend prices as the cumulative product
    of (1+adjusted returns) with the first row as 1.

    param prices: DataFrame of prices
    param P_max: maximum absolute value of portfolio (L in paper)
    param spread_max: maximum deviation from mean
    param moving_midpoint: whether to use moving midpoint (versus constant)
    midpoint_memory: number of periods
      to compute moving midpoint
        (only applicable if moving_midpoint=True)
    param s_init: initial portfolio holdings
    param mu_init: initial midpoint
    param seed: random seed
    param solver: solver to use
    param second_pass:
    whether this is second pass;
      if True, no finetuning of s and mu is done
    """
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
        midpoint_memory=midpoint_memory,
        moving_midpoint=moving_midpoint,
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
            state.reset()
            obj_old, obj_new = 1, 10
        i += 1

    # Second pass
    # if P_max is not None:
    s_times_P_bar = np.abs(state.s.value) * state.P_bar
    s_at_P_bar = np.abs(state.s.value).T @ state.P_bar
    non_zero_inds = np.where(np.abs(s_times_P_bar) > 0.5e-1 * s_at_P_bar)[0]

    if len(non_zero_inds) == 0:  # All holdings are zero
        return None

    # Scale back prices for second pass
    prices = prices * P_bar.T
    prices_new = prices.iloc[:, non_zero_inds]
    s_init = state.s.value[non_zero_inds]
    mu_init = state.mu.value

    if not second_pass or (len(non_zero_inds) != len(state.s.value)):  # polish solution
        return construct_stat_arb(
            prices_new,
            P_max=P_max,
            spread_max=spread_max,
            moving_midpoint=moving_midpoint,
            midpoint_memory=midpoint_memory,
            s_init=s_init,
            mu_init=mu_init,
            seed=None,
            second_pass=True,
        )

    else:  # if second_pass or some holdings are zero
        # Scale back s and return stat-arb
        state.s.value = state.s.value / P_bar
        stat_arb = state.build()

        return stat_arb


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
        midpoint_memory=None,
        moving_midpoint=True,
    ) -> None:
        self.T, self.n = prices.shape
        self.s = cp.Variable((self.n, 1), name="s")
        self.midpoint_memory = midpoint_memory
        self.moving_midpoint = moving_midpoint

        if moving_midpoint:
            self.mu = cp.Variable((self.T, 1))
        else:
            self.mu = cp.Variable(nonneg=True)

        self.p = cp.Variable((self.T, 1), name="p")
        self.P_max = P_max  # allow for P_max=0 for second pass
        self.prices = prices
        self.spread_max = spread_max

        if self.moving_midpoint:
            self.P_bar = (
                prices.iloc[midpoint_memory:].mean(axis=0).values.reshape(-1, 1)
            )
        else:
            self.P_bar = prices.mean(axis=0).values.reshape(-1, 1)

        self.solver = solver

        # Construct linearized convex-concave problem
        self.grad_g = cp.Parameter((self.T, 1), name="grad_g")

        self.obj = cp.Maximize(
            self.grad_g[midpoint_memory:].T @ self.p[midpoint_memory:]
        )

        if self.moving_midpoint:
            self.cons = [
                cp.abs(self.p[midpoint_memory:] - self.mu[midpoint_memory:])
                <= self.spread_max
            ]

            self.cons += [
                self.p[midpoint_memory:]
                == self.prices.values[midpoint_memory:] @ self.s
            ]
        else:
            self.cons = [cp.abs(self.p - self.mu) <= self.spread_max]
            self.cons += [self.p == self.prices.values @ self.s]

        if midpoint_memory:
            ################################################################
            conv_arr = 1 / midpoint_memory * np.ones(midpoint_memory)
            conv_correction = np.array(
                [midpoint_memory / (i + 1) for i in range(midpoint_memory - 1)]
                + [1] * (self.T - midpoint_memory + 1)
            )

            conv = cp.multiply(
                cp.convolve(conv_arr, self.p.flatten())[: self.T], conv_correction
            )

            norm_constant = 1 / conv_arr[-1] * 100  # for numerical stability
            self.cons += [
                norm_constant * self.mu.flatten()[midpoint_memory:]
                == norm_constant * conv[midpoint_memory:]
            ]  # 100 for numerical stability
        ################################################################
        else:
            self.cons += [self.mu >= 0]

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
        if self.moving_midpoint:
            start = self.midpoint_memory
        else:
            start = 0

        grad_g = np.zeros(pk.shape)

        grad_g[start + 0] = pk[start + 0] - pk[start + 1]
        grad_g[-1] = pk[-1] - pk[-2]
        grad_g[start + 1 : -1] = 2 * pk[start + 1 : -1] - pk[start:-2] - pk[start + 2 :]

        return (
            2 * grad_g / np.mean(np.abs(grad_g)) * 100  # 100 for numerical stability
        )  # 2 * grad_g is the actual gradient; we scale for numerical stability

    def build(self):
        assets_dict = dict(zip(self.assets, self.s.value.flatten()))
        stat_arb = StatArb(
            assets=assets_dict,
            mu=self.mu.value,
            midpoint_memory=self.midpoint_memory,
            moving_midpoint=self.moving_midpoint,
            P_max=self.P_max,
            spread_max=self.spread_max,
            entry_date=self.prices.index[-1],
        )

        return stat_arb


@dataclass(frozen=True)
class StatArb:
    """
    Stat arb class
    """

    assets: dict
    mu: float  #
    midpoint_memory: int = None
    moving_midpoint: bool = True
    P_max: float = None
    spread_max: float = 1
    entry_date: pd.Timestamp = None  # will be the last 'day/timestamp' of training data

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
            midpoint_memory=self.midpoint_memory,
            moving_midpoint=self.moving_midpoint,
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
            assert not self.moving_midpoint
            "mu must be specified for moving mean stat arb"
            mu = self.mu

        if self.moving_midpoint:
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
        q.iloc[-1] = 0

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


@dataclass(frozen=True)
class StatArbManager:
    sized_stat_arbs: dict = field(default_factory=dict)
    active_stat_arbs: dict = field(default_factory=dict)
    positions: dict = field(default_factory=dict)
    filtered_stat_arbs: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.active_stat_arbs)

    @property
    def times(self):
        return sorted(self.active_stat_arbs.keys())

    def update(self, times, stat_arbs_new):
        """
        Updates active stat arb tuples for time in 'times'

        param times: list of times to update
        param new_stat_arbs: a list of stat-arbs
        """

        if self.times:
            time_prev = self.times[-1]
            stat_arbs_prev = self.active_stat_arbs[time_prev]
        else:
            stat_arbs_prev = []

        active_stat_arbs = (stat_arbs_prev + stat_arbs_new).copy()

        for t in times:
            self.active_stat_arbs[t] = active_stat_arbs.copy()

    def _pad_stocks(self, stocks, asset_names):
        """
        pads stocks with zeros to include all assets
        """

        stocks_padded = pd.Series(0, index=asset_names, dtype=float)
        stocks_padded[stocks.index] = stocks

        return stocks_padded

    def _get_stat_arb_stocks(self, asset_names, stat_arbs):
        """
        param asset_names: list of asset names
        param stat_arbs: list of stat-arb tuples; a stat-arb tuple is
        a namedtuple of (stat_arb, metrics, prices_train, prices_test)
        returns the stocks of the stat-arbs as columns of a dataframe
        """
        stat_arb_stocks = pd.DataFrame(
            np.zeros((len(asset_names), len(stat_arbs))),
            index=asset_names,
            columns=range(len(stat_arbs)),
        )

        for i, stat_arb in enumerate(stat_arbs):
            stat_arb_stocks.iloc[:, i] = self._pad_stocks(stat_arb.stocks, asset_names)

        return stat_arb_stocks

    def _alpha(self, prices, stat_arbs):
        """
        param prices: DataFrame of prices
        param stat_arbs: list of stat-arbs

        returns the vector of alphas for each stat-arb
        """
        alpha = pd.Series(index=range(len(stat_arbs)), dtype=float)
        stat_arb_prices = pd.Series(index=range(len(stat_arbs)), dtype=float)

        for i, stat_arb in enumerate(stat_arbs):
            p_i = stat_arb.evaluate(prices)  # -stat_arb.SA.mu
            stat_arb_prices[i] = p_i.iloc[-1]

            if stat_arb.moving_midpoint:
                mu_i = p_i.rolling(stat_arb.midpoint_memory).mean()
                alpha[i] = mu_i.iloc[-1] - p_i.iloc[-1]
            else:
                alpha[i] = stat_arb.mu - p_i.iloc[-1]

        return alpha, stat_arb_prices

    def _all_asset_names(self, stat_arbs):
        """
        param stat_arbs: list of stat-arb tuples; a stat-arb tuple is
        a namedtuple of (stat_arb, metrics, prices_train, prices_test)
        returns the list of all combined assets in the stat-arb tuples
        """
        asset_names = []
        for stat_arb in stat_arbs:
            asset_names += stat_arb.asset_names

        asset_names = list(set(asset_names))
        return asset_names

    def _filter_stat_arbs(self, stat_arbs, tradeable_assets, time, prices):
        """
        param time: time to filter stat-arbs at
        param stat_arbs: list of stat arbs
        param prices: DataFrame of prices

        remove duplicates and stat-arb tuples that have been exited
        """
        tradeable_assets_set = set(tradeable_assets)
        stat_arb_multipliers = []

        # find number of unique stat arbs
        filtered_stat_arbs = []
        filtered_stat_arb_assets = []

        times = prices.index

        for stat_arb in stat_arbs:
            assets_temp = set(stat_arb.asset_names)

            # if any asset in assets_temp is not tradeable, skip
            if not assets_temp.issubset(tradeable_assets_set):
                continue

            # is same sata-arb already added, skip
            if set(assets_temp) in filtered_stat_arb_assets:
                continue

            # if time >= stat_arb_tup.metrics.exit_date:
            exit_date = stat_arb.entry_date + pd.Timedelta(days=365 * 2)
            if time >= exit_date:
                # exit_index is closest time index to exit_date

                today_index = times.get_loc(time)
                exit_index = abs(times - exit_date).argmin()

                days_since_exit = today_index - exit_index

                if days_since_exit >= 21:
                    continue

                multiplier = 1 - days_since_exit / 21
                stat_arb_multipliers.append(multiplier)
            else:
                stat_arb_multipliers.append(1)

            filtered_stat_arbs.append(stat_arb)
            filtered_stat_arb_assets.append(assets_temp)
        self.filtered_stat_arbs[time] = filtered_stat_arbs.copy()

        stat_arb_multipliers = pd.Series(stat_arb_multipliers)

        return filtered_stat_arbs.copy(), stat_arb_multipliers

    def size_stat_arbs(
        self,
        time,
        h_prev,
        prices,
        covariance,
        risk_limit,
        active_stat_arbs,
        tradeable_assets,
        eta,
        xi,
        kappa_spread,
        portfolio_value,
    ):
        """
        Sizes stat arbs at time 'time'

        param time: time to size stat arbs
        param prices: pandas dataframe of prices
        param horizon: horizon to compute covariance matrix over

        returns sized stat arbs: asset holdings, asset weights, and stat arb sizes
        """
        latest_prices = prices.loc[time]

        ### Find active stat-arbs and remove duplicates ###
        active_stat_arbs, stat_arb_multipliers = self._filter_stat_arbs(
            active_stat_arbs, tradeable_assets, time, prices
        )

        ### If no active stat-arbs, return None ###
        n_stat_arbs = len(active_stat_arbs)
        if n_stat_arbs == 0:
            return None, None, None, None

        ### Build optimization problem ###
        q = cp.Variable(n_stat_arbs, name="q")
        h = cp.Variable(len(tradeable_assets), name="h")
        weights = cp.multiply(h, latest_prices.values) / portfolio_value
        cash = cp.Variable(name="cash")  # cash in USD
        alpha, stat_arb_prices = self._alpha(prices, active_stat_arbs)

        # risk
        try:
            chol = np.linalg.cholesky(covariance.values)
        except np.linalg.LinAlgError:
            print("Covariance not PSD, projecting onto PSD cone")
            # project onto positive semidefinite cone
            eigvals, eigvecs = np.linalg.eigh(covariance.values)
            eigvals[eigvals < 1e-6] = 1e-6
            covariance_proj = eigvecs @ np.diag(eigvals) @ eigvecs.T
            chol = np.linalg.cholesky(covariance_proj)

        stocks_stacked = self._get_stat_arb_stocks(tradeable_assets, active_stat_arbs)

        chol_q = stocks_stacked.values.T @ chol
        risk = cp.norm2(chol_q.T @ q)

        ### objective
        trades = cp.abs(cp.multiply(h - h_prev[tradeable_assets].values, latest_prices))

        gamma_turn = 1
        # gamma_cash_neutral = 1
        gamma_arb_to_asset = 0.1

        objective = cp.Maximize(
            alpha.values @ q
            - gamma_turn * kappa_spread[tradeable_assets].values @ trades
            # - gamma_cash_neutral * cp.abs(h @ latest_prices.values)
            - gamma_arb_to_asset
            * cp.norm1(cp.multiply(latest_prices, h - stocks_stacked.values @ q))
        )

        ### constraints
        constraints = [cash == portfolio_value]
        one_period_risk_limit = risk_limit / np.sqrt(252) * cash
        constraints += [risk <= one_period_risk_limit]  # risk limit

        # constraints += [h == stocks_stacked.values @ q]  # arb-to-asset
        constraints += [q @ stat_arb_prices.values == 0]  # cash-neutrality

        if eta > 1:
            constraints += [cash >= (eta - 1) * cp.neg(h) @ latest_prices.values]
        constraints += [
            cp.multiply(cp.abs(q), cp.abs(stat_arb_prices.values))
            <= xi * cash * stat_arb_multipliers.values
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver="MOSEK")

        return (
            pd.Series(h.value, index=tradeable_assets),
            pd.Series(weights.value, index=tradeable_assets),
            pd.Series(q.value),
            cash.value,
            active_stat_arbs,
        )

    def build_from_positions(self):
        """
        Builds portfolio from positions
        """
        times = sorted([*self.positions.keys()])
        asset_names = []
        for t in times:
            asset_names += self.positions[t][1]
        asset_names = list(set(asset_names))

        stocks = pd.DataFrame(data=0, index=times, columns=asset_names, dtype=float)

        # fill out stocks_final
        for t in tqdm(times[:]):
            pos_t = self.positions[t]
            asset_names = pos_t.index
            stocks.loc[t, asset_names] = pos_t.values

        return stocks

    def build(self):
        """
        Builds portfolio from sized_stat_arbs
        """
        # build portfolio from sized_stat_arbs
        times = [*self.sized_stat_arbs.keys()]
        asset_names = []
        for t in times:
            asset_names += self._all_asset_names(self.sized_stat_arbs[t][0])
        asset_names = list(set(asset_names))

        stocks = pd.DataFrame(data=0, index=times, columns=asset_names, dtype=float)

        # fill out stocks_final
        for t in tqdm(times):
            stat_arbs, sizes = self.sized_stat_arbs[t]

            asset_names = self._all_asset_names(stat_arbs)
            stocks_stacked = self._get_stat_arb_stocks(asset_names, stat_arbs)

            stocks_temp = stocks_stacked @ sizes

            stocks.loc[t, asset_names] = stocks_temp.values

        return stocks

    def reset(self):
        """
        Resets all attributes
        """
        self.sized_stat_arbs = {}
        self.active_stat_arbs = {}
        self.positions = {}
        self.filtered_stat_arbs = {}
