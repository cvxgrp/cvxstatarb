import numpy as np
from tqdm import tqdm
import pandas as pd
from cvx.stat_arb.ccp import construct_stat_arb
from cvx.simulator.trading_costs import TradingCostModel
from cvx.simulator.portfolio import EquityPortfolio
from collections import namedtuple
import matplotlib.pyplot as plt
from typing import Any
from dataclasses import dataclass
import multiprocessing as mp

import json

PERMNO_to_COMNAM = pd.read_csv("../data/PERMNO_to_COMNAM.csv", index_col=0)
with open("../data/PERMNO_TO_SECTOR.json") as f:
    PERMNO_TO_SECTOR = json.load(f)


def stat_arb_names2(cols):
    for asset_name in cols:
        try:
            company = PERMNO_to_COMNAM.loc[int(asset_name)].COMNAM.iloc[0]
        except AttributeError:
            company = PERMNO_to_COMNAM.loc[int(asset_name)].COMNAM
        sector = PERMNO_TO_SECTOR[asset_name]
        print(f"{company}, {sector}")


def matching_sector2(cols):
    """
    returns the fraction of stocks in stat-arb that has the same sector as some
    other stock in the stat-arb
    """

    sectors = {}

    for asset in cols:
        sector = PERMNO_TO_SECTOR[asset]

        if sector in sectors:
            sectors[sector] += 1
        else:
            sectors[sector] = 1

    counts = list(sectors.values())

    num_one_counts = np.sum([1 for x in counts if x == 1])

    return 1 - num_one_counts / len(cols)


def matching_sector(stat_arb):
    """
    returns the fraction of stocks in stat-arb that has the same sector as some
    other stock in the stat-arb
    """

    asset_names = stat_arb.asset_names

    sectors = {}

    for asset in asset_names:
        sector = PERMNO_TO_SECTOR[asset]

        if sector in sectors:
            sectors[sector] += 1
        else:
            sectors[sector] = 1

    counts = list(sectors.values())

    num_one_counts = np.sum([1 for x in counts if x == 1])

    return 1 - num_one_counts / len(asset_names)


def stat_arb_names(stat_arb):
    asset_names = stat_arb.asset_names

    for asset_name in asset_names:
        try:
            company = PERMNO_to_COMNAM.loc[int(asset_name)].COMNAM.iloc[0]
        except AttributeError:
            company = PERMNO_to_COMNAM.loc[int(asset_name)].COMNAM

        sector = PERMNO_TO_SECTOR[asset_name]
        print(f"{company}, {sector}")


@dataclass(frozen=True)
class SpreadCostModel(TradingCostModel):
    spreads: pd.DataFrame

    def eval(
        self, prices: pd.DataFrame, trades: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        volume = prices * trades
        return 0.5 * self.spreads.loc[volume.index] * volume.abs()


def metrics(portfolios_after_cost, results):
    means = []
    stdevs = []
    sharpes = []
    profits = []
    min_navs = []
    min_cum_prof = []
    drawdowns = []

    NAVs = []
    short_costs = []
    cashs = []
    equities = []
    short_poss = []
    long_poss = []

    for i, portfolio in tqdm(
        enumerate(portfolios_after_cost), total=len(portfolios_after_cost)
    ):
        res = results[i]

        exit_date = res.metrics.exit_date

        profit = portfolio.profit
        profit -= portfolio.trading_costs.sum(axis=1)

        ### Shorting cost
        short_rate = 0.5 / 100 / 252 # half a percent per year

        short_cost = (
            portfolio.stocks[portfolio.stocks < 0].abs() * portfolio.prices
        ).sum(axis=1) * short_rate


        profit -= short_cost

        profits.append(profit.sum())

        nav = (portfolio.nav-short_cost.cumsum()).loc[:exit_date]
        returns = nav.pct_change().loc[:exit_date]
        means.append(returns.mean() * 250)
        stdevs.append(returns.std() * np.sqrt(250))
        sharpes.append(returns.mean() / returns.std() * np.sqrt(250))

        min_navs.append(nav.min())

        NAVs.append(nav)
        short_costs.append(short_cost.loc[:exit_date])
        cashs.append(portfolio.cash.loc[:exit_date])
        equities.append(portfolio.equity.sum(axis=1).loc[:exit_date])
        short_poss.append(portfolio.equity[portfolio.equity < 0].sum(axis=1).loc[:exit_date])
        long_poss.append(portfolio.equity[portfolio.equity > 0].sum(axis=1).loc[:exit_date])



    
        

        min_cum_prof.append(profit.cumsum().min())

        drawdowns.append(portfolio.drawdown.max())

    means = pd.Series(means)
    stdevs = pd.Series(stdevs)
    sharpes = pd.Series(sharpes)
    profits = pd.Series(profits)
    min_navs = pd.Series(min_navs)
    min_cum_prof = pd.Series(min_cum_prof)
    drawdowns = pd.Series(drawdowns)

    return pd.DataFrame(
        {
            "means": means,
            "stdevs": stdevs,
            "sharpes": sharpes,
            "profits": profits,
            "min_navs": min_navs,
            "min_cum_prof": min_cum_prof,
            "drawdowns": drawdowns,
        }
    ), NAVs, short_costs, cashs, equities, short_poss, long_poss


def simulate(res, portfolio, trading_cost_model, lev_fraction):
    stat_arb = res.stat_arb
    lev0 = stat_arb.leverage(portfolio.prices).iloc[0]

    prices_train = res.prices_train
    prices_0 = prices_train.iloc[-1]
    prices_temp = pd.concat([pd.DataFrame(prices_0).T, portfolio.prices])

    stocks_0 = portfolio.stocks.iloc[0] * 0
    stocks_0.name = prices_0.name
    stocks_temp = pd.concat([pd.DataFrame(stocks_0).T, portfolio.stocks])

    portfolio_temp = EquityPortfolio(
        prices_temp,
        stocks=stocks_temp,
        trading_cost_model=trading_cost_model,
        initial_cash=lev_fraction * lev0,
    )

    ### Shorting cost
    short_rate = 0.5 / 100 / 252 # half a percent per year
    short_cost = (portfolio.stocks[portfolio.stocks < 0].abs() * portfolio.prices).sum(
        axis=1
    ) * short_rate

    navs = portfolio_temp.nav-short_cost.cumsum()
    positions = portfolio_temp.stocks.abs() * portfolio_temp.prices
    long_positions = positions[positions > 0].sum(axis=1)
    short_positions = positions[positions < 0].sum(axis=1)
    cash_position = portfolio_temp.cash

    cash0 = cash_position.iloc[0]
    bust_times1 = navs[navs < 0.5 * cash0]
    bust_times2 = navs[long_positions + cash_position < short_positions]

    bust_sort = None

    if len(bust_times1) > 0:
        bust_time1 = bust_times1.index[0]
    else:
        bust_time1 = None
    if len(bust_times2) > 0:
        bust_time2 = bust_times2.index[0]
    else:
        bust_time2 = None

    if bust_time1 is not None and bust_time2 is not None:
        bust_time = min(bust_time1, bust_time2)

        if bust_time1 < bust_time2:
            bust_sort = 1
        else:
            bust_sort = 2
    elif bust_time1 is not None:
        bust_time = bust_time1
        bust_sort = 1
    elif bust_time2 is not None:
        bust_time = bust_time2
        bust_sort = 2
    else:
        bust_time = None

    if bust_time is not None:
        zeros = 0 * stocks_temp.loc[bust_time:].iloc[1:]
        stocks_temp = pd.concat([stocks_temp.loc[:bust_time], zeros], axis=0)
        print(f"\nPortfolio exited early at {bust_time}")
        print(f"bust_sort: {bust_sort}")

    return EquityPortfolio(
        prices_temp,
        stocks=stocks_temp,
        trading_cost_model=trading_cost_model,
        initial_cash=lev_fraction * lev0,
    )


def construct_stat_arbs(
    prices,
    K=1,
    P_max=None,
    spread_max=1,
    moving_mean=True,
    s_init=None,
    mu_init=None,
    seed=None,
    solver="ECOS",
    verbose=True,
):
    if seed is not None:
        np.random.seed(seed)
    # np.random.seed(1)

    all_seeds = list(np.random.choice(range(10 * K), K, replace=False))

    all_args = zip(
        [prices] * K,
        [P_max] * K,
        [spread_max] * K,
        [moving_mean] * K,
        [s_init] * K,
        [mu_init] * K,
        all_seeds,
        [solver] * K,
    )

    pool = mp.Pool()
    all_stat_arbs = []

    if verbose:
        iterator = tqdm(
            pool.imap_unordered(construct_stat_arb_helper, all_args), total=K
        )
    else:
        iterator = pool.imap_unordered(construct_stat_arb_helper, all_args)

    for stat_arb in iterator:
        all_stat_arbs.append(stat_arb)
    pool.close()
    pool.join()

    # remove None
    all_stat_arbs = [x for x in all_stat_arbs if x is not None]
    return all_stat_arbs


def construct_stat_arb_helper(args):
    """
    Call this when using of imap_unordered in multiprocessing

    param args: tuple of arguments to pass to construct_stat_arb
    """

    return construct_stat_arb(*args)


StatArbResult = namedtuple(
    "StatArbResult", ["stat_arb", "metrics", "prices_train", "prices_test"]
)


def run_backtest(
    prices_full,
    market_cap,
    spreads,
    P_max,
    moving_mean,
    T_max,
):
    np.random.seed(0)  # for reproducibility

    trading_cost_model = SpreadCostModel(spreads)
    n_stocks = 200
    remaining_to_stop = 125

    K = 10
    spread_max = 1

    train_len = 500 + 21
    update_freq = 21
    solver = "CLARABEL"

    t_start = 0
    t_end = len(prices_full)

    portfolios = []

    n_iters = int((t_end - (train_len + 1) - t_start - remaining_to_stop) / update_freq)

    results = []

    seeds = [np.random.randint(9999) for _ in range(2 * n_iters)]
    i = 1
    while t_start < t_end - (train_len + 1) - remaining_to_stop:
        if i % 10 == 0:
            print(f"{i/n_iters:.0%}", end=" ")
        i += 1

        time = t_start + train_len
        # get 200 largest stocks at time-1
        assets = (
            market_cap.iloc[time - 1].sort_values(ascending=False).iloc[:n_stocks].index
        )
        prices = prices_full[assets]

        ### Get data, clean over train and val
        prices_train = prices.iloc[t_start : t_start + train_len]
        prices_test = prices.iloc[time:]

        ### Find stat-arbs
        seed = seeds[i]
        stat_arbs = construct_stat_arbs(
            prices_train,
            K=K,
            P_max=P_max,
            spread_max=spread_max,
            moving_mean=moving_mean,
            seed=seed,
            solver=solver,
            verbose=False,
        )

        new_stat_arb_results = []
        asset_names_found = []
        for stat_arb in stat_arbs:
            if set(stat_arb.asset_names) in asset_names_found:
                continue
            else:
                asset_names_found.append(set(stat_arb.asset_names))

            if stat_arb is None:
                continue

            prices_train_test = pd.concat([prices_train, prices_test], axis=0)
            p = prices_train_test[stat_arb.asset_names] @ stat_arb.stocks

            if stat_arb.moving_mean:
                mu = p.rolling(stat_arb.mu_memory).mean()
                mu = mu.iloc[-len(prices_test) :]
            else:
                mu = stat_arb.mu

            # prices_test = pd.concat([prices_val, prices_test], axis=0)
            m = stat_arb.metrics(prices_test, mu, T_max=T_max)

            if m is not None:
                new_stat_arb_results.append(
                    StatArbResult(stat_arb, m, prices_train, prices_test)
                )
            else:
                pass

            ### Construct stat arb portfolio
            positions = stat_arb.get_positions(prices_test, mu, T_max=T_max)
            portfolio = EquityPortfolio(
                prices_test, stocks=positions, trading_cost_model=trading_cost_model
            )
            if m is not None:
                portfolios.append(portfolio)

        results += new_stat_arb_results

        ### Update t_start
        t_start += update_freq
    print(f"\nFinished after {i} iterations")
    return results, portfolios


def plot_stat_arb(
    stat_arb_tuple, insample_bound, outsample_bound, spreads, legend=True
):
    stat_arb = stat_arb_tuple.stat_arb

    mu_memory = stat_arb.mu_memory
    asset_names = stat_arb.asset_names
    stocks = stat_arb.stocks

    prices_train = stat_arb_tuple.prices_train.iloc[:]
    # add first value of prices_test to prices_train for plot
    prices_train = pd.concat(
        [prices_train, stat_arb_tuple.prices_test.iloc[0:1]], axis=0
    )
    prices_test = stat_arb_tuple.prices_test
    prices_train_test = pd.concat([prices_train, prices_test], axis=0)

    p_train = prices_train[asset_names] @ stocks

    if stat_arb.moving_mean:
        mu_train = p_train.rolling(mu_memory).mean().dropna()
    else:
        mu_train = stat_arb.mu

    prices_train_test = pd.concat([prices_train, prices_test], axis=0).iloc[
        -len(prices_test) - mu_memory + 1 :
    ]

    p_test = prices_test[asset_names] @ stocks
    if stat_arb.moving_mean:
        mu_test = (
            (prices_train_test[asset_names] @ stocks).rolling(mu_memory).mean().dropna()
        )
    else:
        mu_test = stat_arb.mu
    # exit date where (prices_test_temp@s-mu).abs() > cutoff

    exit_trigger = stat_arb_tuple.metrics.exit_trigger
    exit_date = stat_arb_tuple.metrics.exit_date
    entry_date = stat_arb_tuple.metrics.entry_date
    elim_end = exit_date + pd.Timedelta(days=100)

    plt.figure()
    plt.plot(p_train, color="b", label="In-sample")

    if stat_arb.moving_mean:
        plt.plot(mu_train, color="g")

    plt.plot(p_test.loc[:elim_end], color="r", label="Out-of-sample")
    if stat_arb.moving_mean:
        plt.plot(mu_test.loc[:elim_end], color="g", label="Midpoint" + r" $(\mu_t)$")
    else:
        plt.plot(
            [prices_train.index[0], prices_test.index[-1]],
            [stat_arb.mu, stat_arb.mu],
            color="g",
            label="Midpoint" + r" $(\mu)$",
        )

    plt.axvline(prices_test.index[0], color="k", linewidth=2)

    stocks_str = ""
    for i in range(stocks.shape[0]):
        if i == 0:
            if stocks.iloc[i] > 0:
                stocks_str += f"{np.round(stocks.iloc[i], 1)}" + "×" + asset_names[i]
            else:
                stocks_str += (
                    f"-{np.abs(np.round(stocks.iloc[i], 1))}" + "×" + asset_names[i]
                )
        else:
            if stocks.iloc[i] > 0:
                stocks_str += f"+{np.round(stocks.iloc[i], 1)}" + "×" + asset_names[i]
            else:
                stocks_str += (
                    f"-{np.abs(np.round(stocks.iloc[i], 1))}" + "×" + asset_names[i]
                )

    print("stat-arb: ", stocks_str)

    plt.gcf().autofmt_xdate()

    xlim_start = prices_train.index[21]

    if legend:
        plt.xlabel("Date")
        plt.ylabel("Stat-arb price")
    # only show exit_date days before and after
    plt.xlim(xlim_start, elim_end)

    # plot vertical line at exit date
    if exit_trigger is not None:
        plt.axvline(
            exit_trigger, linestyle="--", color="k", label="Exit period", linewidth=2
        )
        plt.axvline(exit_date, linestyle="--", color="k", linewidth=2)

    ## plot horizontal line at +- insample_bound over training period
    if stat_arb.moving_mean:
        band_label = r"$\mu_t\pm 1$"

        if outsample_bound is not np.inf:
            plt.plot(
                mu_test.loc[:elim_end] + outsample_bound,
                linestyle=":",
                color="k",
                linewidth=2,
            )
            plt.plot(
                mu_test.loc[:elim_end] - outsample_bound,
                linestyle=":",
                color="k",
                linewidth=2,
            )
        else:
            pd.concat([p_train, p_test], axis=0)
            mu_train_test = pd.concat([mu_train, mu_test], axis=0)

            sigma = 1 / 4.2

            mu_train_test = pd.concat([mu_train, mu_test], axis=0)
            plt.plot(
                mu_train_test.loc[:elim_end] + 4.2 * sigma,
                linestyle=":",
                color="k",
                linewidth=2,
                label=band_label,
            )
            plt.plot(
                mu_train_test.loc[:elim_end] - 4.2 * sigma,
                linestyle=":",
                color="k",
                linewidth=2,
            )
    else:
        plt.plot(
            [prices_train.index[0], prices_train.index[-1]],
            [stat_arb.mu + insample_bound, stat_arb.mu + insample_bound],
            linestyle=":",
            color="k",
            linewidth=2,
            label=r"$\mu\pm 1$",
        )
        plt.plot(
            [prices_train.index[0], prices_train.index[-1]],
            [stat_arb.mu - insample_bound, stat_arb.mu - insample_bound],
            linestyle=":",
            color="k",
            linewidth=2,
        )
        plt.plot(
            [prices_test.index[0], prices_test.index[-1]],
            [stat_arb.mu + insample_bound, stat_arb.mu + insample_bound],
            linestyle=":",
            color="k",
            linewidth=2,
        )
        plt.plot(
            [prices_test.index[0], prices_test.index[-1]],
            [stat_arb.mu - insample_bound, stat_arb.mu - insample_bound],
            linestyle=":",
            color="k",
            linewidth=2,
        )

    if legend:
        plt.legend(
            bbox_to_anchor=(0.5, 1.225), loc="upper center", ncol=3, borderaxespad=0
        )

    plt.show()

    # make another plot to the right of the first one
    plt.figure(figsize=(5, 5))
    plt.figure()

    plt.plot(p_train - mu_train, color="b")

    if stat_arb.moving_mean:
        plt.plot(p_test.loc[:elim_end] - mu_test.loc[:elim_end], color="r")

    else:
        plt.plot(p_test.loc[:elim_end] - mu_test, color="r")

    plt.axvline(prices_test.index[0], color="k", linewidth=2)
    plt.gcf().autofmt_xdate()

    plt.xlim(xlim_start, elim_end)
    plt.ylabel(r"$p_t-\mu_t$")

    plt.show()

    # evaluate stat arb metrics on prices train and test

    plt.figure()
    prices_train, prices_test = stat_arb_tuple.prices_train, stat_arb_tuple.prices_test

    prices_train_test = pd.concat([prices_train, prices_test], axis=0)
    if stat_arb.moving_mean:
        mu = (
            stat_arb.evaluate(prices_train_test)
            .rolling(mu_memory, min_periods=1)
            .mean()
        )

    else:
        stat_arb.metrics(prices_train, stat_arb.mu, T_max=np.inf)
        stat_arb.metrics(prices_test, stat_arb.mu, T_max=63)

    #### TESTING

    if stat_arb.moving_mean:
        mu = mu
    else:
        mu = stat_arb.mu

    ### Construct stat arb portfolio
    # prices_temp = pd.concat([prices_train, prices_test], axis=0)
    positions = stat_arb.get_positions(prices_train_test, mu, T_max=1e6)
    trading_cost_model = SpreadCostModel(spreads)
    portfolio = EquityPortfolio(
        prices_train_test, stocks=positions, trading_cost_model=trading_cost_model
    )
    profit = portfolio.profit
    profit -= portfolio.trading_costs.sum(axis=1)

    ### Shorting cost
    short_rate = 0.5 / 100 / 252 # half a percent per year
    short_cost = (portfolio.stocks[portfolio.stocks < 0].abs() * portfolio.prices).sum(
        axis=1
    ) * short_rate
    profit -= short_cost

    profits_train = profit.loc[: prices_train.index[-1]]
    profits_test = profit.loc[prices_test.index[0] :]
    profits_test.loc[exit_date:] = 0

    if exit_trigger is not None:
        prices_train.index[-1] - pd.Timedelta(days=(exit_trigger - entry_date).days)
    else:
        prices_train.index[-1] - pd.Timedelta(days=(exit_date - entry_date).days)

    plt.plot(profits_train.loc[xlim_start:].cumsum(), color="b", label="In-sample")

    plt.plot(profits_test.cumsum(), color="r", label="Out-of-sample")

    plt.axvline(prices_test.index[0], color="k", linewidth=2)
    plt.gcf().autofmt_xdate()

    if exit_trigger is not None:
        plt.xlim(xlim_start, elim_end)
    else:
        plt.xlim(xlim_start, elim_end)

    if legend:
        plt.ylabel("Cumulative profit")
    plt.xlabel("Date")

    plt.show()
