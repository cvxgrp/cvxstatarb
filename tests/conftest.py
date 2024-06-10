"""global fixtures"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cvx.simulator.portfolio import Portfolio


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture()
def prices(resource_dir):
    temp = pd.read_csv(
        resource_dir / "price.csv", index_col=0, header=0, parse_dates=True
    )
    return (1 + temp.pct_change()).dropna().cumprod()


@pytest.fixture()
def portfolio(prices):
    positions = pd.DataFrame(index=prices.index, columns=prices.columns, data=1.0)
    return Portfolio(prices=prices, units=positions, aum=1e6)


@pytest.fixture()
def stat_arb_assets(resource_dir):
    with open(resource_dir / "test_data/stat_arb_assets.json") as f:
        assets = json.load(f)
    return assets


@pytest.fixture()
def stat_arb_leverage(resource_dir):
    leverage = (
        pd.read_csv(resource_dir / "test_data/stat_arb_leverage.csv", index_col=0)
        .astype(float)
        .squeeze()
    )
    leverage.name = None
    leverage.index = pd.to_datetime(leverage.index)
    return leverage


@pytest.fixture()
def stat_arb_stocks(resource_dir):
    stocks = (
        pd.read_csv(resource_dir / "test_data/stat_arb_stocks.csv", index_col=0)
        .astype(float)
        .squeeze()
    )
    stocks.name = None
    return stocks


@pytest.fixture()
def stat_arb_positions(resource_dir):
    positions = pd.read_csv(
        resource_dir / "test_data/stat_arb_positions.csv", index_col=0, parse_dates=True
    )
    positions.index = pd.to_datetime(positions.index)
    return positions


@pytest.fixture()
def stat_arb_q(resource_dir):
    q = (
        pd.read_csv(resource_dir / "test_data/stat_arb_q.csv", index_col=0)
        .astype(float)
        .squeeze()
    )
    q.name = "q"
    q.index = pd.to_datetime(q.index)
    return q


@pytest.fixture()
def stat_arb_exit_trigger(resource_dir):
    exit_trigger = pd.read_csv(
        resource_dir / "test_data/stat_arb_exit_trigger.csv",
        index_col=0,
        parse_dates=True,
    )
    exit_trigger = pd.Timestamp(exit_trigger.iloc[0, 0])
    return exit_trigger


@pytest.fixture()
def stat_arb_exit_date(resource_dir):
    exit_date = pd.read_csv(
        resource_dir / "test_data/stat_arb_exit_date.csv", index_col=0, parse_dates=True
    )
    exit_date = pd.Timestamp(exit_date.iloc[0, 0])
    return exit_date


@pytest.fixture()
def stat_arb_metrics_series(resource_dir):
    metrics = (
        pd.read_csv(resource_dir / "test_data/stat_arb_metrics.csv", index_col=0)
        .astype(float)
        .squeeze()
    )
    metrics.name = None
    return metrics
