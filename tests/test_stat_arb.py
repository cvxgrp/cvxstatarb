from __future__ import annotations

import pandas as pd
import pytest

from cvx.stat_arb.ccp import construct_stat_arb


@pytest.fixture()
def stat_arb(prices):
    P_max = 100
    spread_max = 1
    moving_midpoint = True
    midpoint_memory = 21
    seed = 100

    return construct_stat_arb(
        prices,
        P_max=P_max,
        spread_max=spread_max,
        moving_midpoint=moving_midpoint,
        midpoint_memory=midpoint_memory,
        seed=seed,
    )


def test_assets(stat_arb, stat_arb_assets):
    pd.testing.assert_series_equal(
        pd.Series(stat_arb.assets), pd.Series(stat_arb_assets)
    )


def test_leverage(prices, stat_arb, stat_arb_leverage):
    pd.testing.assert_series_equal(
        stat_arb.leverage(prices).dropna(), stat_arb_leverage
    )


def test_stocks(stat_arb, stat_arb_stocks):
    pd.testing.assert_series_equal(stat_arb.stocks, stat_arb_stocks)


def test_q(stat_arb, prices, stat_arb_q):
    mu = stat_arb.evaluate(prices).rolling(stat_arb.midpoint_memory).mean()
    q, _, _ = stat_arb.get_q(prices, mu)
    pd.testing.assert_series_equal(q.dropna(), stat_arb_q)


def test_positions(stat_arb, prices, stat_arb_positions):
    mu = stat_arb.evaluate(prices).rolling(stat_arb.midpoint_memory).mean()
    positions = stat_arb.get_positions(prices, mu)
    pd.testing.assert_frame_equal(positions.dropna(), stat_arb_positions)


def test_exit_trigger(stat_arb, prices, stat_arb_exit_trigger):
    mu = stat_arb.evaluate(prices).rolling(stat_arb.midpoint_memory).mean()
    q, exit_trigger, _ = stat_arb.get_q(prices, mu)
    assert exit_trigger == stat_arb_exit_trigger


def test_exit_date(stat_arb, prices, stat_arb_exit_date):
    mu = stat_arb.evaluate(prices).rolling(stat_arb.midpoint_memory).mean()
    _, _, exit_date = stat_arb.get_q(prices, mu)
    assert exit_date == stat_arb_exit_date


def test_metrics(stat_arb, prices, stat_arb_metrics_series):
    mu = stat_arb.evaluate(prices).rolling(stat_arb.midpoint_memory).mean()
    metrics = stat_arb.metrics(prices, mu)
    mean_profit = metrics.mean_profit
    std_profit = metrics.std_profit
    total_profit = metrics.total_profit
    sr_profit = metrics.sr_profit
    metrics_series = pd.Series(
        {
            "mean_profit": mean_profit,
            "std_profit": std_profit,
            "total_profit": total_profit,
            "sr_profit": sr_profit,
        }
    )
    pd.testing.assert_series_equal(metrics_series, stat_arb_metrics_series)
