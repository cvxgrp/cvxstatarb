import json

import pandas as pd

from cvx.stat_arb.ccp import construct_stat_arb

prices = (
    (pd.read_csv("../price.csv", index_col=0, parse_dates=True).pct_change() + 1)
    .cumprod()
    .dropna()
)

P_max = 100
spread_max = 1
moving_midpoint = True
midpoint_memory = 21
T_max = 500
seed = 100

stat_arb = construct_stat_arb(
    prices,
    P_max=P_max,
    spread_max=spread_max,
    moving_midpoint=moving_midpoint,
    midpoint_memory=midpoint_memory,
    seed=seed,
)

# creaate test data
assets = stat_arb.assets
leverage = stat_arb.leverage(prices)
stocks = stat_arb.stocks

mu = stat_arb.evaluate(prices).rolling(stat_arb.midpoint_memory).mean()
q, exit_trigger, exit_date = stat_arb.get_q(prices, mu)
positions = stat_arb.get_positions(prices, mu)
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


if False:
    # save test data
    with open("stat_arb_assets.json", "w") as f:
        json.dump(assets, f)

    leverage.to_csv("stat_arb_leverage.csv")
    stocks.to_csv("stat_arb_stocks.csv")
    mu.dropna().to_csv("stat_arb_mu.csv")
    q.dropna().to_csv("stat_arb_q.csv")
    positions.dropna().to_csv("stat_arb_positions.csv")

    # save exit_trigger and entry_trigger TimeStamps to a file
    pd.DataFrame({"exit_trigger": [exit_trigger]}).to_csv("stat_arb_exit_trigger.csv")
    pd.DataFrame({"exit_date": [exit_date]}).to_csv("stat_arb_exit_date.csv")
    metrics_series.to_csv("stat_arb_metrics.csv")
