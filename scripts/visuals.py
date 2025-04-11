# visuals.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Import portfolio simulation functions from your portfolio.py script
from portfolio import (
    simulate_equity_portfolio,
    simulate_covered_calls_portfolio,
    compute_risk_metrics,
    bootstrap_sensitivity,
    compute_sector_weights,
    dynamic_weight_allocation,
    risk_free_equity
)

# --- Configuration (should match portfolio.py) ---
Start_Date = "2019-06-01"
End_Date   = "2024-12-31"
benchmark_ticker = 'SPY'
etf_list = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV',
            'XLI', 'XLB', 'XLRE', 'XLK', 'XLU', 'XLC']

# --- Data Acquisition ---
print("Fetching historical price data...")
all_tickers = etf_list + [benchmark_ticker]
price_data = yf.download(all_tickers, start=Start_Date, end=End_Date, auto_adjust=True)['Close']
price_data.dropna(inplace=True)

# --- Run Portfolio Simulations ---
print("Running portfolio simulations...")
equity_index = simulate_equity_portfolio(etf_list, benchmark_ticker, price_data)
covered_call_index = simulate_covered_calls_portfolio(etf_list, price_data)

# Define common date range from the later of the two series' first dates.
common_start = max(equity_index.index.min(), covered_call_index.index.min())
full_dates = pd.date_range(start=common_start, end=price_data.index.max(), freq='B')

equity_index_full = equity_index.reindex(full_dates).ffill().bfill()
covered_call_index_full = covered_call_index.reindex(full_dates).ffill().bfill()

# Compute daily returns for each strategy.
equity_daily_ret = equity_index_full.pct_change().fillna(0)
option_daily_ret = covered_call_index_full.pct_change().fillna(0)

# Prepare SPY normalized series on the same date range.
spy_index = price_data[benchmark_ticker].reindex(full_dates).ffill().bfill()
spy_index = spy_index / spy_index.iloc[0]

# Compute dynamic equity allocation using SPY volatility.
spy_returns = spy_index.pct_change().dropna()
spy_vol = spy_returns.rolling(window=30).std().iloc[-1] * np.sqrt(252)
dyn_equity_weight = dynamic_weight_allocation(spy_vol)
print(f"Dynamic equity weight based on SPY volatility ({spy_vol:.2%}): {dyn_equity_weight:.2f}")

# Combine equity and covered call strategies using the dynamic equity weight.
combined_daily_ret = dyn_equity_weight * equity_daily_ret + (1 - dyn_equity_weight) * option_daily_ret
combined_index = (1 + combined_daily_ret).cumprod()

# Create a dictionary with strategy series.
strategies = {
    "Equity Portfolio": equity_index_full,
    "Covered Call Strategy": covered_call_index_full,
    "Combined Portfolio": combined_index,
    "SPY Buy & Hold": spy_index
}

# --- Visualization Functions ---

def plot_cumulative_returns(strategy_dict):
    plt.figure(figsize=(12, 7))
    for label, series in strategy_dict.items():
        plt.plot(series.index, series.values, label=label)
    plt.title("Cumulative Returns Comparison")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value (Base = 1.0)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("cumulative_returns.png")
    plt.show()

def plot_daily_return_histograms(daily_returns_dict):
    """Plot histograms for daily returns for each strategy, filtering out non-finite values."""
    plt.figure(figsize=(14, 8))
    for i, (label, daily_ret) in enumerate(daily_returns_dict.items()):
        # Filter out any non-finite values
        daily_ret = daily_ret[np.isfinite(daily_ret)]
        if daily_ret.empty:
            print(f"Warning: {label} daily returns are empty or contain no finite values.")
            continue
        plt.subplot(2, 2, i+1)
        plt.hist(daily_ret, bins=50, alpha=0.75, color="skyblue")
        plt.title(f"Daily Returns: {label}")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("daily_return_histograms.png")
    plt.show()

def plot_drawdown_curves(strategy_dict):
    plt.figure(figsize=(12, 7))
    for label, series in strategy_dict.items():
        running_max = series.cummax()
        drawdown = (series - running_max) / running_max
        plt.plot(drawdown.index, drawdown.values, label=label)
    plt.title("Drawdown Curves")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("drawdown_curves.png")
    plt.show()

def plot_rolling_metrics(series, window=60):
    daily_ret = series.pct_change().dropna()
    rolling_vol = daily_ret.rolling(window=window).std() * np.sqrt(252)
    # Compute rolling cumulative return over the window
    rolling_return = (1 + daily_ret).rolling(window=window).apply(np.prod, raw=True) - 1
    rolling_sharpe = (rolling_return - risk_free_equity) / rolling_vol
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(rolling_vol.index, rolling_vol, color="blue", label="Rolling Volatility")
    ax1.set_ylabel("Volatility", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.plot(rolling_sharpe.index, rolling_sharpe, color="red", label="Rolling Sharpe")
    ax2.set_ylabel("Sharpe Ratio", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    plt.title(f"Rolling {window}-Day Volatility & Sharpe Ratio (Combined Portfolio)")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1))
    plt.savefig("rolling_metrics.png")
    plt.show()

def plot_correlation_heatmap(daily_returns_dict):
    data = {}
    for label, ret in daily_returns_dict.items():
        data[label] = ret
    returns_df = pd.DataFrame(data)
    corr_matrix = returns_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Daily Returns")
    plt.savefig("correlation_heatmap.png")
    plt.show()

def plot_bootstrap_histograms(bootstrap_df):
    plt.figure(figsize=(14, 8))
    metrics = bootstrap_df.columns
    for i, met in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        plt.hist(bootstrap_df[met].dropna(), bins=30, alpha=0.75, color="purple")
        plt.title(f"Bootstrap Distribution: {met}")
        plt.xlabel(met)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("bootstrap_histograms.png")
    plt.show()

def plot_sector_weights(sector_weights):
    plt.figure(figsize=(10, 6))
    sector_weights.sort_values(ascending=False).plot(kind="bar", color="green")
    plt.title("Performance-based Sector Weights (Trailing 6 Months)")
    plt.xlabel("ETF")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.savefig("sector_weights.png")
    plt.show()

# --- Generate Visualizations ---
print("Generating visualizations...")

# 1. Plot Cumulative Returns
plot_cumulative_returns(strategies)

# 2. Plot Daily Return Histograms
daily_returns_strategies = {
    "Equity": equity_index_full.pct_change().dropna(),
    "Covered Call": covered_call_index_full.pct_change().dropna(),
    "Combined": combined_index.pct_change().dropna(),
    "SPY": spy_index.pct_change().dropna()
}
plot_daily_return_histograms(daily_returns_strategies)

# 3. Plot Drawdown Curves
plot_drawdown_curves(strategies)

# 4. Plot Rolling Metrics for Combined Portfolio
plot_rolling_metrics(combined_index, window=60)

# 5. Plot Correlation Heatmap of Daily Returns
plot_correlation_heatmap(daily_returns_strategies)

# 6. Plot Bootstrap Sensitivity Histograms
bootstrap_results = bootstrap_sensitivity(combined_index, n_iter=500, sample_frac=0.5)
plot_bootstrap_histograms(bootstrap_results)

# 7. Plot Sector Weights
sector_weights = compute_sector_weights(price_data, etf_list, lookback_days=126)
plot_sector_weights(sector_weights)

# 8. Compute and Export Risk Metrics
metrics_list = []
for label, series in strategies.items():
    cum_ret, ann_ret, ann_vol, sharpe, max_dd = compute_risk_metrics(series, risk_free=risk_free_equity)
    metrics_list.append({
        "Strategy": label,
        "Cumulative Return (%)": cum_ret * 100,
        "Annualized Return (%)": ann_ret * 100,
        "Annualized Volatility (%)": ann_vol * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd * 100
    })
metrics_df = pd.DataFrame(metrics_list)
print("\nRisk Metrics Comparison:")
print(metrics_df)
metrics_df.to_excel("portfolio_risk_metrics.xlsx", index=False)
print("Risk metrics have been written to portfolio_risk_metrics.xlsx")
