import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Import our portfolio functions from equities.py and bonds.py.
from equities import get_equity_portfolio
from bonds import get_bond_portfolio

# ======================================================
# 1. Set analysis parameters
# ======================================================
start_date = '2020-01-01'
end_date   = '2024-12-31'
risk_free_rate = 0.02   # Annual risk-free rate

# ======================================================
# 2. Build the individual sub-portfolios (Quarterly)
# ======================================================

# Equity Portfolio: quarterly rebalanced returns 
eq_quarterly_returns, eq_cum_returns, equity_expected_return = get_equity_portfolio(
    start_date, end_date, risk_free_rate, plot_frontier=True
)

# Ensure the indices are datetime objects.
eq_quarterly_returns.index = pd.to_datetime(eq_quarterly_returns.index)
eq_cum_returns.index = pd.to_datetime(eq_cum_returns.index)

# *** FIX: Adjust equity portfolio dates to quarter-end.
# The equities module returns the period end as the next quarter's start.
# Subtract one day so that, for example, 2020-04-01 becomes 2020-03-31.
eq_quarterly_returns.index = eq_quarterly_returns.index - pd.Timedelta(days=1)
eq_cum_returns.index = eq_cum_returns.index - pd.Timedelta(days=1)

# Bond Portfolio:
bond_expected_df, bond_expected_return = get_bond_portfolio(start_date, end_date, T=0.25, plot_results=False)
bond_price = yf.download('TLT', start=start_date, end=end_date, auto_adjust=True)['Close'].dropna()
# Resample TLT to quarter-end dates using 'QE' then compute quarterly returns.
bond_quarterly_returns = bond_price.resample('QE').last().pct_change().dropna()
bond_quarterly_returns.index = pd.to_datetime(bond_quarterly_returns.index)

# If the bond returns are returned as a DataFrame with one column, convert to a Series.
if isinstance(bond_quarterly_returns, pd.DataFrame):
    if bond_quarterly_returns.shape[1] == 1:
        bond_quarterly_returns = bond_quarterly_returns.iloc[:, 0]

bond_cum_returns = (1 + bond_quarterly_returns).cumprod()

# ======================================================
# 3. Combine sub-portfolios into a 60/40 Overall Portfolio (Quarterly)
# ======================================================

# Find common quarter-end dates.
common_dates = eq_quarterly_returns.index.intersection(bond_quarterly_returns.index)
print("Common Dates (Quarter-End):")
print(common_dates)

# Reindex the equity and bond returns to these common dates.
eq_qtr = eq_quarterly_returns.reindex(common_dates, fill_value=0)
bond_qtr = bond_quarterly_returns.reindex(common_dates, fill_value=0)

# --- Debug: Inspect the aligned sub-portfolio returns and their types.
print("\nType of Equity Quarterly Returns:", type(eq_qtr))
print("Type of Bond Quarterly Returns:", type(bond_qtr))
print("\nEquity Quarterly Returns (Aligned):")
print(eq_qtr)
print("\nBond Quarterly Returns (Aligned):")
print(bond_qtr)

# Define allocation weights: 60% equities, 40% bonds.
equity_weight = 0.60
bond_weight = 0.40

# Compute the combined quarterly return as the weighted sum.
combined_qtr_returns = equity_weight * eq_qtr + bond_weight * bond_qtr

# --- Debug: Inspect combined quarterly returns.
print("\nCombined Quarterly Returns:")
print(combined_qtr_returns)

# Compute cumulative returns using quarterly compounding.
combined_cum_returns = (1 + combined_qtr_returns).cumprod()

# Compute overall expected annual return as a weighted average.
overall_expected_return = equity_weight * equity_expected_return + bond_weight * bond_expected_return

# ======================================================
# 4. Benchmark: SPY Quarterly Returns
# ======================================================
spy_price = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)['Close'].dropna()
# Resample SPY to quarter-end dates using 'QE' then compute quarterly returns.
spy_qtr_returns = spy_price.resample('QE').last().pct_change().dropna()
spy_qtr_returns.index = pd.to_datetime(spy_qtr_returns.index)
spy_cum_returns = (1 + spy_qtr_returns).cumprod()

# ======================================================
# 5. Display Overall Portfolio Performance & Risk Metrics
# ======================================================
print("\nPerformance Metrics:")
print("Equity Portfolio Expected Annual Return: {:.2f}%".format(equity_expected_return * 100))
print("Bond Portfolio Expected Annual Return: {:.2f}%".format(bond_expected_return * 100))
print("Combined 60/40 Portfolio Expected Annual Return: {:.2f}%".format(overall_expected_return * 100))

# Calculate annualized volatility from the quarterly returns (volatility * sqrt(4)).
combined_volatility = combined_qtr_returns.std() * np.sqrt(4)
print("Combined 60/40 Portfolio Annualized Volatility: {:.2f}%".format(combined_volatility * 100))

# ======================================================
# 6. Plot Performance (Cumulative Returns)
# ======================================================
plt.figure(figsize=(12, 8))
plt.plot(eq_cum_returns.index, eq_cum_returns, label='Equity Portfolio (60%)', linewidth=2)
plt.plot(bond_cum_returns.index, bond_cum_returns, label='Bond Portfolio (40%)', linewidth=2)
plt.plot(combined_cum_returns.index, combined_cum_returns, label='Combined 60/40 Portfolio', linewidth=2, linestyle='--')
plt.plot(spy_cum_returns.index, spy_cum_returns, label='Benchmark (SPY)', linewidth=2, linestyle=':')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('60/40 Portfolio Backtest (Quarterly Rebalancing)')
plt.legend()
plt.grid(True)
plt.show()
