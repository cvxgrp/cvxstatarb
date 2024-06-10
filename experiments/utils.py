import json
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from cvx.simulator.portfolio import Portfolio
from cvx.stat_arb.ccp import construct_stat_arb

PERMNO_to_COMNAM = pd.read_csv("../data/PERMNO_to_COMNAM.csv", index_col=0)
with open("../data/PERMNO_TO_SECTOR.json") as f:
    PERMNO_TO_SECTOR = json.load(f)


def compute_drawdowns(navs):
    """
    computes drawdowns from a time series of NAVs
    """

    max_nav = navs.cummax()
    drawdowns = -(navs - max_nav) / max_nav

    return drawdowns


def compute_drawdowns_from_returns(returns):
    """
    computes drawdowns from a time series of NAVs
    """

    navs = (1 + returns).cumprod()

    return compute_drawdowns(navs)


def get_next_ewma(EWMA, y_last, t, beta, clip_at=None, min_periods=None):
    """
    param EWMA: EWMA at time t-1
    param y_last: observation at time t-1
    param t: current time step
    param beta: EWMA exponential forgetting parameter
    param clip_at: clip y_last at  +- clip_at*EWMA (optional)

    returns: EWMA estimate at time t (note that this does not depend on y_t)
    """

    old_weight = (beta - beta**t) / (1 - beta**t)
    new_weight = (1 - beta) / (1 - beta**t)

    if clip_at:
        assert min_periods, "min_periods must be specified if clip_at is specified"
        if t >= min_periods + 2:
            return old_weight * EWMA + new_weight * np.clip(
                y_last, -clip_at * EWMA, clip_at * EWMA
            )
    return old_weight * EWMA + new_weight * y_last


def ewma(y, halflife, clip_at=None, min_periods=None):
    """
    param y: array with measurements for times t=1,2,...,T=len(y)
    halflife: EWMA half life
    param clip_at: clip y_last at  +- clip_at*EWMA (optional)

    returns: list of EWMAs for times t=2,3,...,T+1 = len(y)


    Note: We define EWMA_t as a function of the
    observations up to time t-1. This means that
    y = [y_1,y_2,...,y_T] (for some T), while
    EWMA = [EWMA_2, EWMA_3, ..., EWMA_{T+1}]
    This way we don't get a "look-ahead bias" in the EWMA
    """
    times = [*y.keys()]
    beta = np.exp(-np.log(2) / halflife)
    EWMAs = {}
    EWMAs[times[0]] = y[times[0]].fillna(0)

    t = 1

    for t_prev, t_curr in zip(times[:-1], times[1:]):  # First EWMA is for t=2
        EWMAs[t_curr] = get_next_ewma(
            EWMAs[t_prev], y[t_curr].fillna(0), t + 1, beta, clip_at, min_periods
        )
        t += 1

    return EWMAs


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


def compute_trading_costs(trades, spreads):
    volume = trades.abs()

    if type(volume) == pd.Series:
        assert type(spreads) == pd.Series, "spreads and trades must be of the same type"
        assert (
            trades.index == spreads.index
        ).all(), "Index of trades and spreads must be the same"

        return 0.5 * (spreads * volume).sum()

    elif type(volume) == pd.DataFrame:
        assert (
            type(spreads) == pd.DataFrame
        ), "spreads and trades must be of the same type"
        assert (
            trades.index == spreads.index
        ).all(), "Index of trades and spreads must be the same"
        assert (
            trades.columns == spreads.columns
        ).all(), "Columns of trades and spreads must be the same"

        return 0.5 * (spreads * volume).sum(axis=1)

    else:
        raise ValueError("trades and spreads must be a pd.Series or pd.DataFrame")


def simulate(res, portfolio, spreads, lev_fraction):
    """
    Simulates the stat-arb performance

    param res: StatArbResult namedtuple (see backtest.py); includes stat_arb,
    metrics, prices_train, prices_test
    param portfolio: Portfolio object
    param spreads: pd.DataFrame with spreads
    param lev_fraction: initial cash is lev_fraction * lev0, where lev0 is the
    initial (dollar) leverage
    """
    stat_arb = res.stat_arb
    exit_date = res.metrics.exit_date
    assets = portfolio.units.columns
    times = portfolio.units.loc[:exit_date].index
    lev0 = stat_arb.leverage(portfolio.prices).iloc[0]

    ### Construct prices and stocks (units)
    # hold only cash at day before entry date
    prices_train = res.prices_train
    prices_0 = prices_train.iloc[-1]
    prices_temp = pd.concat([pd.DataFrame(prices_0).T, portfolio.prices])

    stocks_0 = portfolio.units.iloc[0] * 0
    stocks_0.name = prices_0.name
    stocks_temp = pd.concat([pd.DataFrame(stocks_0).T, portfolio.units])

    ### Construct stat arb portfolio
    initial_cash = lev_fraction * lev0
    portfolio_new = Portfolio(
        prices_temp,
        units=stocks_temp,
        aum=initial_cash,
    )

    ### Shorting cost
    short_rate = 0.5 / 100 / 252  # half a percent per year
    short_costs = (portfolio.units[portfolio.units < 0].abs() * portfolio.prices).sum(
        axis=1
    ) * short_rate

    ### Trading costs
    trading_costs = compute_trading_costs(
        portfolio.trades_currency[assets].loc[times], spreads[assets].loc[times]
    )

    ### NAVs
    navs = (portfolio_new.nav - short_costs.cumsum() - trading_costs.cumsum()).loc[
        :exit_date
    ]
    positions = portfolio_new.units
    absolute_notionals = positions.abs() * portfolio_new.prices
    long_positions = absolute_notionals[positions > 0].sum(axis=1)
    short_positions = absolute_notionals[positions < 0].sum(axis=1)
    cash = portfolio_new.nav - portfolio_new.equity.sum(axis=1)

    ### Did we exit early?
    cash0 = cash.iloc[0]
    bust_times1 = navs[navs < 0.25 * cash0]
    bust_times2 = navs[long_positions + cash < short_positions]

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
        if bust_sort == 1:
            print("NAV fell below 0.5 * cash")
        elif bust_sort == 2:
            print("Long positions + cash fell below short position")

    ### Compute metrics
    profits = (portfolio.profit - short_costs - trading_costs).loc[:exit_date]

    if bust_time is not None:  # exit stat-arb at bust_time
        navs.loc[bust_time:] = navs.loc[bust_time]
        profits.loc[bust_time:] = 0

    returns = navs.pct_change().loc[:exit_date]
    mean = returns.mean() * 250
    stdev = returns.std() * np.sqrt(250)
    sharpe = returns.mean() / returns.std() * np.sqrt(250)
    min_nav = navs.min()

    mean_profit = profits.sum()
    min_cum_prof = profits.cumsum().min()
    drawdown = compute_drawdowns(navs).max()

    went_bust = bust_time is not None

    return mean, stdev, sharpe, mean_profit, min_nav, min_cum_prof, drawdown, went_bust


def construct_stat_arbs(
    prices,
    K=1,
    P_max=None,
    spread_max=1,
    moving_midpoint=True,
    midpoint_memory=None,
    s_init=None,
    mu_init=None,
    seed=None,
    solver="CLARABEL",
    verbose=True,
):
    if seed is not None:
        np.random.seed(seed)

    all_seeds = list(np.random.choice(range(10 * K), K, replace=False))

    all_args = zip(
        [prices] * K,
        [P_max] * K,
        [spread_max] * K,
        [moving_midpoint] * K,
        [midpoint_memory] * K,
        [s_init] * K,
        [mu_init] * K,
        all_seeds,
        [solver] * K,
    )

    # pool = mp.Pool()
    # all_stat_arbs = []

    # if verbose:
    #     iterator = tqdm(
    #         pool.imap_unordered(construct_stat_arb_helper, all_args), total=K
    #     )
    # else:
    #     iterator = pool.imap_unordered(construct_stat_arb_helper, all_args)

    # for stat_arb in iterator:
    #     all_stat_arbs.append(stat_arb)
    # pool.close()
    # pool.join()

    pool = mp.Pool()
    all_stat_arbs = []

    try:
        if verbose:
            iterator = tqdm(
                pool.imap_unordered(construct_stat_arb_helper, all_args), total=K
            )
        else:
            iterator = pool.imap_unordered(construct_stat_arb_helper, all_args)

        for stat_arb in iterator:
            all_stat_arbs.append(stat_arb)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, terminating workers...")
        pool.terminate()
    finally:
        pool.close()  # Make sure no more tasks are submitted to the pool
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


def plot_stat_arb(
    stat_arb_tuple, insample_bound, outsample_bound, spreads, legend=True
):
    stat_arb = stat_arb_tuple.stat_arb

    # print stat-arb companies
    stat_arb_names(stat_arb)

    midpoint_memory = stat_arb.midpoint_memory
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

    if stat_arb.moving_midpoint:
        mu_train = p_train.rolling(midpoint_memory).mean().dropna()
    else:
        mu_train = stat_arb.mu

    p_test = prices_test[asset_names] @ stocks
    if stat_arb.moving_midpoint:
        prices_train_test = pd.concat([prices_train, prices_test], axis=0).iloc[
            -len(prices_test) - midpoint_memory + 1 :
        ]
        mu_test = (
            (prices_train_test[asset_names] @ stocks)
            .rolling(midpoint_memory)
            .mean()
            .dropna()
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

    if stat_arb.moving_midpoint:
        plt.plot(mu_train, color="g")

    plt.plot(p_test.loc[:elim_end], color="r", label="Out-of-sample")
    if stat_arb.moving_midpoint:
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
    if stat_arb.moving_midpoint:
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

    if stat_arb.moving_midpoint:
        plt.plot(p_test.loc[:elim_end] - mu_test.loc[:elim_end], color="r")

    else:
        plt.plot(p_test.loc[:elim_end] - mu_test, color="r")

    plt.axvline(prices_test.index[0], color="k", linewidth=2)
    plt.gcf().autofmt_xdate()

    plt.xlim(xlim_start, elim_end)
    plt.ylabel(r"$p_t-\mu_t$")

    plt.show()

    ### evaluate stat arb metrics on prices train and test

    plt.figure()
    prices_train, prices_test = stat_arb_tuple.prices_train, stat_arb_tuple.prices_test

    prices_train_test = pd.concat([prices_train, prices_test], axis=0)
    if stat_arb.moving_midpoint:
        mu = (
            stat_arb.evaluate(prices_train_test)
            .rolling(midpoint_memory, min_periods=1)
            .mean()
        )

    else:
        stat_arb.metrics(prices_train, stat_arb.mu, T_max=np.inf)
        stat_arb.metrics(prices_test, stat_arb.mu, T_max=63)

    #### TESTING

    if stat_arb.moving_midpoint:
        mu = mu
    else:
        mu = stat_arb.mu

    ### Construct stat arb portfolio
    positions = stat_arb.get_positions(prices_train_test, mu, T_max=1e6)
    initial_cash = 1e6
    aum = initial_cash + (prices_train_test.iloc[0] * positions).sum(axis=1)

    portfolio = Portfolio(prices_train_test, units=positions, aum=aum)
    profit = portfolio.profit
    times = portfolio.units.index
    assets = portfolio.units.columns
    trading_costs = compute_trading_costs(
        portfolio.trades_currency[assets], spreads[assets].loc[times]
    )

    ### Shorting cost
    short_rate = 0.5 / 100 / 252  # half a percent per year
    short_costs = (portfolio.units[portfolio.units < 0].abs() * portfolio.prices).sum(
        axis=1
    ) * short_rate
    profit = profit - short_costs - trading_costs

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
