import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cvxpy as cp
import multiprocessing as mp
from tqdm import tqdm

from cvx.stat_arb.ccp import construct_stat_arb

import cvxportfolio as cvx
from cvxportfolio.policies import Policy
from cvxportfolio import DownloadedMarketData, StockMarketSimulator
from cvxportfolio.data.symbol_data import _timestamp_convert
from yfinance.data import YfData
from cvxportfolio.errors import DataError
import requests

def parallel_helper(args):
    return construct_stat_arb(*args)

def construct_stat_arb_parallel(prices,
        K=1,
        P_max=None,
        spread_max=1,
        moving_midpoint=True,
        midpoint_memory=None,
        s_init=None,
        mu_init=None,
        seed=None,
        solver="CLARABEL",
        verbose=True):

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

    pool = mp.Pool()
    all_stat_arbs = []

    try:
        if verbose:
            iterator = tqdm(
                pool.imap_unordered(parallel_helper, all_args), total=K
            )
        else:
            iterator = pool.imap_unordered(parallel_helper, all_args)

        for stat_arb in iterator:
            all_stat_arbs.append(stat_arb)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, terminating workers...")
        pool.terminate()
    finally:
        pool.close()  
        pool.join()

    all_stat_arbs = [x for x in all_stat_arbs if x is not None]
    return all_stat_arbs

def get_alpha(prices, stat_arbs):
    """
    param prices: DataFrame of prices
    param stat_arbs: list of stat-arbs

    returns the vector of alphas for each stat-arb
    """
    alpha = pd.Series(index=range(len(stat_arbs)), dtype=float)
    stat_arb_prices = pd.Series(index=range(len(stat_arbs)), dtype=float)

    for i, stat_arb in enumerate(stat_arbs):
        p_i = stat_arb.evaluate(prices) 
        stat_arb_prices[i] = p_i.iloc[-1]

        if stat_arb.moving_midpoint:
            mu_i = p_i.rolling(stat_arb.midpoint_memory).mean()
            alpha[i] = mu_i.iloc[-1] - p_i.iloc[-1]
        else:
            alpha[i] = stat_arb.mu - p_i.iloc[-1]
    return alpha, stat_arb_prices

def arb_asset_transform(asset_names, stat_arbs):
    """
    param asset_names: list of asset names
    param stat_arbs: list of stat-arb tuples; a stat-arb tuple is
    a namedtuple of (stat_arb, metrics, prices_train, prices_test)
    returns the stocks of the stat-arbs as columns of a dataframe
    """
    arb_asset_matrix = pd.DataFrame(
        np.zeros((len(asset_names), len(stat_arbs))),
        index=asset_names,
        columns=range(len(stat_arbs)),
    )

    for i, stat_arb in enumerate(stat_arbs):
        arb_asset_matrix.iloc[:, i] = pad_arb_assets(stat_arb.stocks, asset_names)

    return arb_asset_matrix

def pad_arb_assets(stocks, asset_names):
    """
    pads stocks with zeros to include all assets
    """

    stocks_padded = pd.Series(0, index=asset_names, dtype=float)
    stocks_padded[stocks.index] = stocks

    return stocks_padded

class yfin_latest(cvx.YahooFinance):
    """ Fix broken cvxportfolio datasource """
    @staticmethod
    def _get_data_yahoo(ticker, start='1900-01-01', end='2100-01-01'):
        """ The cvxportfolio get method brokes, replace with yfinance """
        base_url = 'https://query2.finance.yahoo.com'
        start = int(pd.Timestamp(start).timestamp())
        end = int(pd.Timestamp(end).timestamp())
        _data_yf: YfData = YfData(session=None)
        try:
            res = _data_yf.cache_get(
                url=f"{base_url}/v8/finance/chart/{ticker}",
                params={'interval': '1d',
                    "period1": start,
                    "period2": end},
                timeout=10) # seconds
        except requests.ConnectionError as exc:
            raise DataError(
                f"Download of {ticker} from YahooFinance failed."
                + " Are you connected to the Internet?") from exc
        if res.status_code == 404:
            raise DataError(
                f'Data for symbol {ticker} is not available.'
                + 'Json: ' + str(res.json()))
        if res.status_code != 200:
            raise DataError(
                f'Yahoo finance download of {ticker} failed. Json: ' +
                str(res.json())) # pragma: no cover
        data = res.json()['chart']['result'][0]
        try:
            index = pd.DatetimeIndex(
                [_timestamp_convert(el) for el in data['timestamp']])
            df_result = pd.DataFrame(
                data['indicators']['quote'][0], index=index)
            df_result['adjclose'] = data[
                'indicators']['adjclose'][0]['adjclose']
        except KeyError as exc: # pragma: no cover
            raise DataError(f'Yahoo finance download of {ticker} failed.'
                + ' Json: ' + str(res.json())) from exc # pragma: no cover
        this_periods_open_time = _timestamp_convert(
            data['meta']['currentTradingPeriod']['regular']['start'])
        if (df_result.index[-1].time() != this_periods_open_time.time())\
            and not (np.abs((df_result.index[-1] - this_periods_open_time
                ) % pd.Timedelta('1d')) == pd.Timedelta('3600s')):
            # set the last time to the announced open time
            index = df_result.index.to_numpy()
            dt = df_result.index[-1]
            dt = dt.replace(hour=this_periods_open_time.time().hour)
            dt = dt.replace(minute=this_periods_open_time.time().minute)
            dt = dt.replace(second=this_periods_open_time.time().second)
            index[-1] = dt
            df_result.index = pd.DatetimeIndex(index)
        return df_result[
            ['open', 'low', 'high', 'close', 'adjclose', 'volume']]

def plot_results(result):
    how_many_weights = 7
    fig, axes = plt.subplots(3, figsize=(10, 10), layout='constrained')
    fig.suptitle('Back-test result')

    """ value """
    result.v.plot(label='Portfolio value', ax=axes[0])
    axes[0].set_ylabel(result.cash_key)
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', which="both")

    """ weights """
    biggest_weights = np.abs(result.w).mean(
        ).sort_values().iloc[-how_many_weights:].index
    result.w[biggest_weights].plot(ax=axes[1])
    axes[1].set_ylabel(f'Largest {how_many_weights} weights')
    axes[1].grid(True, linestyle='--')

    """ leverage / turnover """
    result.leverage.plot(ax=axes[2], linestyle='--',
                        color='k', label='Leverage')
    result.turnover.plot(ax=axes[2], linestyle='-',
                        color='r', label='Turnover')
    axes[2].legend()
    axes[2].grid(True, linestyle='--')

    plt.show()
    return

class LinearArbitrage(Policy):
    """ Trading Moving Band with Linear Arbitrage Policy"""
    def __init__(self, cash_key):
        self.target_weights = None
        self.arb_history = 521 
        self.last_search = None
        self.cash_key = cash_key
        self.unique_stat_arbs = {}

        self.search_freq = 21
        self.exit_len = 21

        self.Tmax = 21
        self.Pmax = 100
        self.moving_midpoint = True
        self.midpoint_memory = 21
        self.parallel_K = 10

    def values_in_time_recursive( # pylint: disable=arguments-differ
            self, t, current_prices, current_weights, past_returns, past_volumes, current_portfolio_value, **kwargs):
        """ Recover the past prices with past cumulative returns"""
        windowd_returns = past_returns.iloc[-self.arb_history:].drop(columns=[self.cash_key])
        cumulative_returns = (1 + windowd_returns).cumprod()
        unit_return = current_prices/cumulative_returns.iloc[-1]
        """ the true past prices can be recovered as 
            past_prices = train_prices.shift(1).dropna() if needed in the future """
        prices_train = unit_return * cumulative_returns # index shifted by 1 vs ground truth prices
        """ Search for new stat arbs every `search_freq` days 
            past returns is behind 1 business day of [t, current_prices]"""
        if self.last_search is not None:
            search_interval = len(pd.bdate_range(self.last_search, t))
        else:
            search_interval = np.inf

        if search_interval > self.search_freq: 
            """ record last search timestamp """
            self.last_search = t
            """ rank asset by market cap to narrow down arb search space """
            market_cap = current_prices * past_volumes.iloc[-1]
            partial_assets = market_cap.sort_values(ascending=False).index
            new_stat_arbs = construct_stat_arb_parallel(
                prices_train[partial_assets],
                K=self.parallel_K,
                P_max=self.Pmax,
                spread_max=1,
                moving_midpoint=self.moving_midpoint,
                midpoint_memory=self.midpoint_memory,
                solver="CLARABEL")
            for arb_item in new_stat_arbs:
                arb_key = frozenset(set(arb_item.asset_names))
                """ filter out existing arbs"""
                if arb_key in self.unique_stat_arbs.keys():
                    continue
                self.unique_stat_arbs[arb_key] = arb_item
                """ evaluate the performance of new arb profits """

        """ Check activated arbs """
        active_arbs = []
        exit_decay = []
        remove_arbs = []
        active_assets = set()
        for arb_key, arb_item in self.unique_stat_arbs.items():
            """ compute exit decay """
            entry_days = len(pd.bdate_range(arb_item.entry_date, t)) 
            if entry_days >= (self.Tmax + self.exit_len):
                decay_item = 0
            elif entry_days <= self.Tmax:
                decay_item = 1
            else:
                decay_item = (self.Tmax+self.exit_len-entry_days)/self.exit_len
            if decay_item > 0:
                exit_decay.append(decay_item)
                active_arbs.append(arb_item)
                active_assets.update(arb_key)
            else:
                remove_arbs.append(arb_key)

        """ Aggregate arb performance when exit """
        self.unique_stat_arbs = {k: v for k, v in self.unique_stat_arbs.items() if k not in remove_arbs}

        active_assets = list(active_assets)
        new_weights = pd.Series(0., current_weights.index)
        if len(active_arbs) > 0:
            new_weights[active_assets] = opt_policy(
                stat_arbs = active_arbs, 
                prices_train = prices_train[active_assets],
                exit_decay = pd.Series(exit_decay),
                cash=current_portfolio_value)
            new_weights.iloc[-1] = 1 - new_weights[active_assets].sum()
        else:
            new_weights.iloc[-1] = 1
        assert not new_weights.isna().any(), print(new_weights)
        return new_weights

def opt_policy(stat_arbs, 
            prices_train, 
            exit_decay, 
            cash):
    """ Build optimization problem """
    n_universe = len(prices_train.columns)
    n_stat_arbs = len(stat_arbs)
    """ Quanity of Arbs """
    qty_arb = cp.Variable(n_stat_arbs, name="q")
    """ Share holdings of all assets """
    holdings = cp.Variable(n_universe, name="h")
    latest_prices = prices_train.iloc[-1]
    weights = cp.multiply(holdings, latest_prices.values) / cash 
    """ Compute alpha """
    arb_alphas, stat_arb_prices = get_alpha(prices_train, stat_arbs)
    """ the arb qty to portfolio holding transform """
    arb_asset_matrix = arb_asset_transform(prices_train.columns, stat_arbs)
    objective = cp.Maximize(arb_alphas.values @ qty_arb/cash)
    constraints = []
    constraints += [ qty_arb @ stat_arb_prices.values <= cash ]  
    constraints += [ arb_asset_matrix.values @ qty_arb == holdings ]  
    eta = 5
    constraints += [cash >= (eta - 1) * cp.neg(holdings) @ latest_prices.values]
    """ linear exit """
    constraints += [ cp.multiply(cp.abs(qty_arb), cp.abs(stat_arb_prices.values)) <= cash * exit_decay.values]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver='CLARABEL')

    if problem.status == 'optimal':
        valid_weights = weights.value
    else:
        valid_weights = np.zeros(n_universe)

    return valid_weights

def main():
    CASH_KEY = 'USDOLLAR'
    DOW = ["AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WMT"]

    market_data = DownloadedMarketData(
                DOW, 
                cash_key=CASH_KEY,
                datasource=yfin_latest)

    simulator = StockMarketSimulator(market_data=market_data)
    policy = LinearArbitrage(CASH_KEY)

    result = simulator.backtest(
        policy, 
        start_time='2024-1-1',
        end_time='2025-1-1')

    print(result)

    plot_results(result)

if __name__ == '__main__':
    main()
