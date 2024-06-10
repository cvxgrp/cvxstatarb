from collections import namedtuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import compute_trading_costs, construct_stat_arbs

from cvx.simulator.builder import Builder
from cvx.simulator.portfolio import Portfolio
from cvx.stat_arb.ccp import StatArbManager

StatArbResult = namedtuple(
    "StatArbResult", ["stat_arb", "metrics", "prices_train", "prices_test"]
)


def _find_and_filter_stat_arbs(
    prices_train,
    prices_test,
    K,
    P_max,
    T_max,
    spread_max,
    moving_midpoint,
    midpoint_memory,
    seed,
    solver,
    verbose,
    construct_portfolios=False,
):
    """
    Find stat arbs and filter out duplicates
    """
    stat_arbs = construct_stat_arbs(
        prices_train,
        K=K,
        P_max=P_max,
        spread_max=spread_max,
        moving_midpoint=moving_midpoint,
        midpoint_memory=midpoint_memory,
        seed=seed,
        solver=solver,
        verbose=False,
    )

    stat_arb_results = []
    asset_names_found = []
    portfolios = []
    for stat_arb in stat_arbs:
        if set(stat_arb.asset_names) in asset_names_found:
            continue
        else:
            asset_names_found.append(set(stat_arb.asset_names))

        if stat_arb is None:
            continue

        prices_train_test = pd.concat([prices_train, prices_test], axis=0)
        p = prices_train_test[stat_arb.asset_names] @ stat_arb.stocks

        if stat_arb.moving_midpoint:
            mu = p.rolling(stat_arb.midpoint_memory).mean()
            mu = mu.iloc[-len(prices_test) :]
        else:
            mu = stat_arb.mu

        m = stat_arb.metrics(prices_test, mu, T_max=T_max)

        if m is not None:
            stat_arb_results.append(
                StatArbResult(stat_arb, m, prices_train, prices_test)
            )
        else:
            pass

        ### Construct stat arb portfolio
        if construct_portfolios:
            positions = stat_arb.get_positions(prices_test, mu, T_max=T_max)
            portfolio = Portfolio(
                prices_test,
                units=positions,
                aum=1e6,
            )
            if m is not None:
                portfolios.append(portfolio)

    if construct_portfolios:
        return stat_arb_results, portfolios
    else:
        return stat_arb_results, None


def run_finding_backtest(
    prices_full,
    market_cap,
    P_max,
    moving_midpoint,
    midpoint_memory,
    T_max,
):
    np.random.seed(0)  # for reproducibility

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
        ### get 200 largest stocks at time-1 ###
        assets = (
            market_cap.iloc[time - 1].sort_values(ascending=False).iloc[:n_stocks].index
        )
        prices = prices_full[assets]

        ### Get data, clean over train and val ###
        prices_train = prices.iloc[t_start : t_start + train_len]
        prices_test = prices.iloc[time:]

        ### Find stat-arbs ###
        seed = seeds[i]
        new_stat_arb_results, new_portfolios = _find_and_filter_stat_arbs(
            prices_train=prices_train,
            prices_test=prices_test,
            K=K,
            P_max=P_max,
            T_max=T_max,
            spread_max=spread_max,
            moving_midpoint=moving_midpoint,
            midpoint_memory=midpoint_memory,
            seed=seed,
            solver=solver,
            verbose=False,
            construct_portfolios=True,
        )

        results += new_stat_arb_results
        portfolios += new_portfolios

        ### Update t_start
        t_start += update_freq

    print(f"\nFinished after looking for stat-arbs {i} times")
    return results, portfolios


def _update_portfolio(
    manager,
    state,
    quantities_prev,
    prices,
    time,
    risk_limit,
    eta,
    xi,
    kappa_spread,
    shorting_cost,
    covariance,
    active_stat_arbs,
    tradeable_assets,
    interest_and_fees,
    trading_costs,
):
    n_assets = quantities_prev.shape[0]

    cash = state.cash + interest_and_fees.sum() - trading_costs.sum()
    value_temp = state.value
    portfolio_value = cash + value_temp
    weights_prev = quantities_prev / portfolio_value

    latest_prices = prices.loc[time]

    (
        quantities_new,
        weights_new,
        stat_arb_quantities,
        _,  # cash equals to portfolio_value after rebalancing
        active_stat_arbs_filtered,
    ) = manager.size_stat_arbs(
        time,
        quantities_prev,
        prices.loc[:time][tradeable_assets],
        covariance.loc[tradeable_assets, tradeable_assets],
        # cash,
        risk_limit,
        active_stat_arbs,
        tradeable_assets,
        eta,
        xi,
        kappa_spread,
        portfolio_value,
    )

    if quantities_new is None:
        quantities_new = quantities_prev
        weights_new = weights_prev

    # set rest of quantities and weights to zero
    quantities_temp = pd.Series(np.zeros(n_assets), index=prices.columns)
    quantities_temp.loc[tradeable_assets] = quantities_new
    quantities_new = quantities_temp

    weights_temp = pd.Series(np.zeros(n_assets), index=prices.columns)
    weights_temp.loc[tradeable_assets] = weights_new
    weights_new = weights_temp

    # Update manager
    manager.positions[time] = quantities_new
    manager.sized_stat_arbs[time] = (active_stat_arbs_filtered, stat_arb_quantities)

    ### holding costs
    short_pos = np.clip(quantities_prev, None, 0).abs()
    latest_asset_names = quantities_prev.index
    holding_cost = (short_pos @ latest_prices.loc[latest_asset_names]) * shorting_cost

    return quantities_new, weights_new, stat_arb_quantities, holding_cost


def run_portfolio_backtest(
    prices,
    market_cap,
    spreads,
    covariances,
    P_max,
    moving_midpoint,
    midpoint_memory,
    T_max,
    shorting_cost,
):
    np.random.seed(0)  # for reproducibility

    n_assets = len(prices.columns)
    all_times = prices.index
    n_stocks = 200
    K = 10
    spread_max = 1
    train_len = 500 + 21
    update_freq = 21
    solver = "CLARABEL"

    n_iters = int(np.ceil(len(all_times) / update_freq))
    seeds = [np.random.randint(9999) for _ in range(2 * n_iters)]

    results = []

    ### Managing the portfolio
    manager = StatArbManager()

    quantities_prev = pd.Series(np.zeros(n_assets), index=prices.columns)
    pd.Series(np.zeros(n_assets), index=prices.columns)

    b = Builder(
        prices=prices,
        initial_aum=1,
    )

    ### set parameters for stat-arb portfolio optimization problem
    risk_limit = 0.1
    eta = 1
    xi = 1

    interest_and_fees = pd.Series(np.zeros(len(all_times)), index=all_times) * np.nan
    trading_costs = pd.Series(np.zeros(len(all_times)), index=all_times) * np.nan
    cashs = pd.Series(np.zeros(len(all_times)), index=all_times) * np.nan
    values = pd.Series(np.zeros(len(all_times)), index=all_times) * np.nan
    holdings = (
        pd.DataFrame(
            np.zeros((len(all_times), n_assets)),
            index=all_times,
            columns=prices.columns,
        )
        * np.nan
    )
    all_weights = (
        pd.DataFrame(
            np.zeros((len(all_times), n_assets)),
            index=all_times,
            columns=prices.columns,
        )
        * np.nan
    )
    all_stat_arb_quantities = []

    iteration = -1
    i = 1  # index for seeds
    print("Note: the countdown timer is not accurate")
    for t, state in tqdm(b, total=len(all_times)):
        date_time = t[-1]
        # get index of date_time in all_times
        time = all_times.get_loc(date_time)
        if time < train_len:  # skip if before train_len
            continue

        iteration += 1

        if iteration % update_freq == 0:  # find stat-arbs every update_freq days
            # get n_stocks largest stocks at time-1
            assets = (
                market_cap.iloc[time - 1]
                .sort_values(ascending=False)
                .iloc[:n_stocks]
                .index
            )
            prices_temp = prices[assets]
            ### Get data, clean over train and test ###
            prices_train = prices_temp.iloc[time - train_len : time]
            prices_test = prices_temp.iloc[time:]

            ### Find stat-arbs
            seed = seeds[i]
            i += 1
            new_stat_arb_results, _ = _find_and_filter_stat_arbs(
                prices_train=prices_train,
                prices_test=prices_test,
                K=K,
                P_max=P_max,
                T_max=T_max,
                spread_max=spread_max,
                moving_midpoint=moving_midpoint,
                midpoint_memory=midpoint_memory,
                seed=seed,
                solver=solver,
                verbose=False,
                construct_portfolios=False,
            )

            new_stat_arbs = [res.stat_arb for res in new_stat_arb_results]
            results += new_stat_arb_results

            ### size stat arbs ###
            date_times = prices.index[time : time + update_freq]
            manager.update(date_times, new_stat_arbs)

        ### Update portfolio ###
        kappa_spread = spreads.loc[date_time] / 2
        covariance = covariances[date_time]
        active_stat_arbs = manager.active_stat_arbs[date_time]

        asset_names = manager._all_asset_names(active_stat_arbs)
        tradeable_assets, _ = _get_tradable_assets(
            prices[asset_names].loc[:date_time].iloc[-21:], covariance
        )
        (
            quantities_new,
            weights_new,
            stat_arb_quantities,
            holding_cost,
        ) = _update_portfolio(
            manager,
            state,
            quantities_prev,
            prices,
            date_time,
            risk_limit,
            eta,
            xi,
            kappa_spread,
            shorting_cost,
            covariance,
            active_stat_arbs,
            tradeable_assets,
            interest_and_fees,
            trading_costs,
        )

        trades = quantities_new - quantities_prev
        trade_cost = compute_trading_costs(trades, spreads.loc[date_time])

        ### update quantities ###
        quantities_prev = quantities_new

        ### store results ###
        interest_and_fees.loc[date_time] = -holding_cost
        trading_costs.loc[date_time] = trade_cost

        holdings.loc[date_time] = quantities_new
        all_weights.loc[date_time] = weights_new
        cash = state.cash + interest_and_fees.sum() - trading_costs.sum()
        value = cash + quantities_new @ prices.loc[date_time]
        cashs.loc[date_time] = cash
        values.loc[date_time] = value
        all_stat_arb_quantities.append(stat_arb_quantities)

        ### update builder positions ###
        b.position = quantities_new
        b.aum = state.aum  # keep track of fees and trading costs separately

    print(f"\nFinished after looking for stat-arbs {i} times")

    ### build portfolio ###
    portfolio = b.build()

    return (
        results,
        manager,
        portfolio,
        holdings,
        all_weights,
        all_stat_arb_quantities,
        values,
        cashs,
        interest_and_fees,
        trading_costs,
    )


def run_portfolio_backtest_from_manager(
    manager,
    prices,
    spreads,
    covariances,
    shorting_cost,
):
    n_assets = len(prices.columns)
    all_times = prices.index
    train_len = 500 + 21
    update_freq = 21

    n_iters = int(np.ceil(len(all_times) / update_freq))
    [np.random.randint(9999) for _ in range(2 * n_iters)]

    results = []

    quantities_prev = pd.Series(np.zeros(n_assets), index=prices.columns)
    pd.Series(np.zeros(n_assets), index=prices.columns)

    b = Builder(
        prices=prices,
        initial_aum=1,
    )

    ### set parameters for stat-arb portfolio optimization problem
    risk_limit = 0.1
    eta = 1
    xi = 1

    interest_and_fees = pd.Series(np.zeros(len(all_times)), index=all_times) * np.nan
    trading_costs = pd.Series(np.zeros(len(all_times)), index=all_times) * np.nan
    cashs = pd.Series(np.zeros(len(all_times)), index=all_times) * np.nan
    values = pd.Series(np.zeros(len(all_times)), index=all_times) * np.nan
    holdings = (
        pd.DataFrame(
            np.zeros((len(all_times), n_assets)),
            index=all_times,
            columns=prices.columns,
        )
        * np.nan
    )
    all_weights = (
        pd.DataFrame(
            np.zeros((len(all_times), n_assets)),
            index=all_times,
            columns=prices.columns,
        )
        * np.nan
    )
    all_stat_arb_quantities = []

    iteration = -1
    print("Note: the countdown timer is not accurate")
    for t, state in tqdm(b, total=len(all_times)):
        date_time = t[-1]
        # get index of date_time in all_times
        time = all_times.get_loc(date_time)
        if time < train_len:  # skip if before train_len
            continue

        iteration += 1

        ### Update portfolio ###
        kappa_spread = spreads.loc[date_time] / 2
        covariance = covariances[date_time]
        active_stat_arbs = manager.active_stat_arbs[date_time]

        asset_names = manager._all_asset_names(active_stat_arbs)
        tradeable_assets, _ = _get_tradable_assets(
            prices[asset_names].loc[:date_time].iloc[-21:], covariance
        )
        (
            quantities_new,
            weights_new,
            stat_arb_quantities,
            holding_cost,
        ) = _update_portfolio(
            manager,
            state,
            quantities_prev,
            prices,
            date_time,
            risk_limit,
            eta,
            xi,
            kappa_spread,
            shorting_cost,
            covariance,
            active_stat_arbs,
            tradeable_assets,
            interest_and_fees,
            trading_costs,
        )

        trades = quantities_new - quantities_prev
        trade_cost = compute_trading_costs(trades, spreads.loc[date_time])

        ### update quantities ###
        quantities_prev = quantities_new

        ### store results ###
        interest_and_fees.loc[date_time] = -holding_cost
        trading_costs.loc[date_time] = trade_cost

        holdings.loc[date_time] = quantities_new
        all_weights.loc[date_time] = weights_new
        cash = state.cash + interest_and_fees.sum() - trading_costs.sum()
        value = (
            cash + quantities_new @ prices.loc[date_time]
        )  # value and cash should be same, due to cash-neutrality
        cashs.loc[date_time] = cash
        values.loc[date_time] = value
        all_stat_arb_quantities.append(stat_arb_quantities)

        ### update builder positions ###
        b.position = quantities_new
        b.aum = state.aum  # keep track of fees and trading costs separately

    ### build portfolio ###
    portfolio = b.build()

    return (
        results,
        manager,
        portfolio,
        holdings,
        all_weights,
        all_stat_arb_quantities,
        values,
        cashs,
        interest_and_fees,
        trading_costs,
    )


def _get_tradable_assets(prices_recent, covariance):
    # don't trade assets that have constant price over the past 21 days
    asset_names = prices_recent.columns
    tradeable_assets = []
    for asset in asset_names:
        if (
            (prices_recent[asset].max() != prices_recent[asset].min())
            and (not prices_recent[asset].isnull().any())
            and (asset in covariance.columns)
            and (not covariance[asset].isnull().all())
            and (not (covariance[asset] == 0).all())
        ):
            tradeable_assets.append(asset)
    not_tradeable_assets = list(set(asset_names) - set(tradeable_assets))

    return tradeable_assets, not_tradeable_assets


# def run_portfolio_backtest_old(
#     manager,
#     prices,
#     spreads,
#     covariances,
#     shorting_cost,
# ):
#     """
#     param manager: StatArbManager
#     param prices: pd.DataFrame of prices
#     param spreads: pd.DataFrame of spreads
#     """

#     rolling_spreads = spreads.rolling(5).mean().ffill().fillna(10 * (0.01**2)).abs()
#     n_assets = len(prices.columns)

#     quantities_prev = pd.Series(np.zeros(n_assets), index=prices.columns)
#     pd.Series(np.zeros(n_assets), index=prices.columns)
#     cash = 1

#     times0 = manager.times
#     times1 = list(covariances.keys())

#     times = sorted(set(times0) & set(times1))

#     portfolio = Builder(
#         prices=prices.loc[times],
#         initial_cash=cash,
#     )

#     ### parameters
#     risk_limit = 0.1
#     eta = 1
#     xi = 1
#     # three basis points

#     interest_and_fees = pd.Series(np.zeros(len(times)), index=times)
#     i = 0
#     cashs = []
#     values = []
#     holdings = []
#     all_weights = []
#     all_stat_arb_quantities = []

#     for t, state in tqdm(portfolio, total=len(times)):
#         i += 1

#         time = t[-1]
#         cash = state.cash + interest_and_fees.sum()
#         value_temp = state.value
#         portfolio_value = cash + value_temp

#         latest_prices = prices.loc[time]
#         kappa_spread = rolling_spreads.loc[time] / 2
#         covariance = covariances[time]
#         active_stat_arbs = manager.active_stat_arbs[time]

#         asset_names = manager._all_asset_names(active_stat_arbs)
#         tradeable_assets, _ = _get_tradable_assets(
#             prices[asset_names].loc[:time].iloc[-21:], covariance
#         )

#         (
#             quantities_new,
#             weights_new,
#             stat_arb_quantities,
#         ) = manager.size_stat_arbs(
#             time,
#             quantities_prev,
#             prices.loc[:time][tradeable_assets],
#             covariance.loc[tradeable_assets, tradeable_assets],
#             cash,
#             risk_limit,
#             active_stat_arbs,
#             tradeable_assets,
#             eta,
#             xi,
#             kappa_spread,
#             portfolio_value,
#         )

#         if quantities_new is None:
#             quantities_new = quantities_prev

#         # set rest of quantities and weights to zero
#         quantities_temp = pd.Series(np.zeros(n_assets), index=prices.columns)
#         quantities_temp.loc[tradeable_assets] = quantities_new
#         quantities_new = quantities_temp

#         weights_temp = pd.Series(np.zeros(n_assets), index=prices.columns)
#         weights_temp.loc[tradeable_assets] = weights_new
#         weights_new = weights_temp

#         portfolio.set_position(time, quantities_new)

#         ### holding costs
#         short_pos = np.clip(quantities_prev, None, 0).abs()
#         latest_asset_names = quantities_prev.index
#         holding_cost = (
#             short_pos @ latest_prices.loc[latest_asset_names]
#         ) * shorting_cost
#         interest = 0
#         interest_and_fees.loc[time] = interest - holding_cost

#         # update quantities
#         quantities_prev = quantities_new

#         # update risk duals

#         holdings.append(quantities_prev)
#         all_weights.append(weights_new)

#         cash = state.cash + interest_and_fees.sum()
#         value = cash + quantities_new @ prices.loc[time]
#         cashs.append(cash)

#         values.append(value)
#         all_stat_arb_quantities.append(stat_arb_quantities)

#     holdings = pd.DataFrame(holdings, index=times)
#     cashs = pd.Series(cashs, index=times)
#     values = pd.Series(values, index=times)
#     all_weights = pd.DataFrame(all_weights, index=times)

#     return (
#         portfolio,
#         holdings,
#         all_weights,
#         all_stat_arb_quantities,
#         values,
#         cashs,
#         interest_and_fees.shift(1),
#     )
