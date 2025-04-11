import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Import our portfolio functions from equities.py and bonds.py.
from equities import get_equity_portfolio
from bonds import get_bond_portfolio
from config import Start_Date, End_Date

# ======================================================
# 1. Set analysis parameters
# ======================================================
risk_free_rate = 0.02   # Annual risk-free rate

# Allocation weights: adjust these later (e.g., via Bayesian optimization)
equity_weight = 0.7
options_weight = 0.3  # Initially, we assume cash returns of 0% for this allocation; later, this portion can be replaced with an options strategy.

# ======================================================
# 2. Build individual sub-portfolios (Quarterly)
# ======================================================

# Equity Portfolio: quarterly rebalanced returns
eq_quarterly_returns, eq_cum_returns, equity_expected_return = get_equity_portfolio(
    Start_Date, End_Date, risk_free_rate, plot_frontier=True
)

# Ensure the indices are datetime objects and adjust them to quarter-end.
eq_quarterly_returns.index = pd.to_datetime(eq_quarterly_returns.index) - pd.Timedelta(days=1)
eq_cum_returns.index = pd.to_datetime(eq_cum_returns.index) - pd.Timedelta(days=1)

# Bond Portfolio: for plotting and reference only.
bond_expected_df, bond_expected_return = get_bond_portfolio(Start_Date, End_Date, T=0.25, plot_results=False)
bond_price = yf.download('TLT', start=Start_Date, end=End_Date, auto_adjust=True)['Close'].dropna()
# Resample TLT to quarter-end dates using 'QE' then compute quarterly returns.
bond_quarterly_returns = bond_price.resample('QE').last().pct_change().dropna()
bond_quarterly_returns.index = pd.to_datetime(bond_quarterly_returns.index)
if isinstance(bond_quarterly_returns, pd.DataFrame) and bond_quarterly_returns.shape[1] == 1:
    bond_quarterly_returns = bond_quarterly_returns.iloc[:, 0]
bond_cum_returns = (1 + bond_quarterly_returns).cumprod()

# ======================================================
# 3. Build the Combined Portfolio (Quarterly)
# ======================================================
# Here we remove bonds from the final portfolio so that we have 70% equities and 30% cash (for an options strategy overlay).
# For consistency in plotting, we align the equity returns to the common quarter-end dates.

# Use common dates from equity and bond series (useful for the plot even though bonds aren’t included in performance).
common_dates = eq_quarterly_returns.index.intersection(bond_quarterly_returns.index)
print("Common Dates (Quarter-End):")
print(common_dates)
eq_qtr = eq_quarterly_returns.loc[common_dates]

# The combined portfolio return is now computed as 70% equity return plus 30% cash return (cash assumed to be 0%).
combined_qtr_returns = equity_weight * eq_qtr + options_weight * 0
print("\nCombined Quarterly Returns (70% Equities, 30% Cash for Options Strategy):")
print(combined_qtr_returns)

combined_cum_returns = (1 + combined_qtr_returns).cumprod()

# Align equity cumulative returns for comparison.
eq_cum_returns_aligned = eq_cum_returns.loc[common_dates]
print("\nEquity Cumulative Returns (Aligned):")
print(eq_cum_returns_aligned)
print("\nCombined Cumulative Returns (70% Equities, 30% Cash):")
print(combined_cum_returns)

# ======================================================
# 4. Benchmark: SPY Quarterly Returns
# ======================================================
spy_price = yf.download('SPY', start=Start_Date, end=End_Date, auto_adjust=True)['Close'].dropna()
spy_qtr_returns = spy_price.resample('QE').last().pct_change().dropna()
spy_qtr_returns.index = pd.to_datetime(spy_qtr_returns.index)
spy_cum_returns = (1 + spy_qtr_returns).cumprod()

# ======================================================
# 5. Display Overall Portfolio Performance & Risk Metrics
# ======================================================
print("\nPerformance Metrics:")
print("Equity Portfolio Expected Annual Return: {:.2f}%".format(equity_expected_return * 100))
print("Bond Portfolio Expected Annual Return (reference): {:.2f}%".format(bond_expected_return * 100))
combined_expected_return = equity_weight * equity_expected_return + options_weight * 0  # cash assumed 0% return
print("Combined Portfolio Expected Annual Return (70% Equities, 30% Cash): {:.2f}%".format(combined_expected_return * 100))
combined_volatility = combined_qtr_returns.std() * np.sqrt(4)
print("Combined Portfolio Annualized Volatility: {:.2f}%".format(combined_volatility * 100))

# ======================================================
# 6. Plot Performance (Cumulative Returns)
# ======================================================
bond_cum_returns_aligned = bond_cum_returns.loc[common_dates]

plt.figure(figsize=(12, 6))
plt.plot(eq_cum_returns_aligned.index, eq_cum_returns_aligned.values,
         label="Equity Portfolio", marker="o")
plt.plot(bond_cum_returns_aligned.index, bond_cum_returns_aligned.values,
         label="Bond Portfolio (Reference)", marker="o")
plt.plot(combined_cum_returns.index, combined_cum_returns.values,
         label="Combined Portfolio (70% Equity, 30% Cash)", marker="o")
plt.plot(spy_cum_returns.index, spy_cum_returns.values,
         label="Benchmark (SPY)", marker="o")

plt.title("Cumulative Returns Comparison")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.savefig("cumulative_returns_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
