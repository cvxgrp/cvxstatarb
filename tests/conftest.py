"""global fixtures"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cvx.simulator.portfolio import EquityPortfolio

# from cvx.simulator.portfolio import build_portfolio


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture()
def prices(resource_dir):
    return pd.read_csv(
        resource_dir / "price.csv", index_col=0, header=0, parse_dates=True
    )


@pytest.fixture()
def portfolio(prices):
    positions = pd.DataFrame(index=prices.index, columns=prices.columns, data=1.0)
    return EquityPortfolio(prices=prices, stocks=positions)
