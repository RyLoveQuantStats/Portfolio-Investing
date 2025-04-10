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

# For testing, set allocation weights so that the combined returns should match one asset exactly.
equity_weight = 0.7  
bond_weight = 0.3    

# ======================================================
# 2. Build the individual sub-portfolios (Quarterly)
# ======================================================

# Equity Portfolio: quarterly rebalanced returns
eq_quarterly_returns, eq_cum_returns, equity_expected_return = get_equity_portfolio(
    Start_Date, End_Date, risk_free_rate, plot_frontier=True
)

# Ensure the indices are datetime objects.
eq_quarterly_returns.index = pd.to_datetime(eq_quarterly_returns.index)
eq_cum_returns.index = pd.to_datetime(eq_cum_returns.index)

# Adjust the equity index to quarter-end (subtract one day from the next quarter's start).
eq_quarterly_returns.index = eq_quarterly_returns.index - pd.Timedelta(days=1)
eq_cum_returns.index = eq_cum_returns.index - pd.Timedelta(days=1)

# Bond Portfolio:
bond_expected_df, bond_expected_return = get_bond_portfolio(Start_Date, End_Date, T=0.25, plot_results=False)
bond_price = yf.download('TLT', start=Start_Date, end=End_Date, auto_adjust=True)['Close'].dropna()
# Resample TLT to quarter-end dates using 'QE' then compute quarterly returns.
bond_quarterly_returns = bond_price.resample('QE').last().pct_change().dropna()
bond_quarterly_returns.index = pd.to_datetime(bond_quarterly_returns.index)

# If bond_quarterly_returns is a DataFrame with one column, convert it to a Series.
if isinstance(bond_quarterly_returns, pd.DataFrame) and bond_quarterly_returns.shape[1] == 1:
    bond_quarterly_returns = bond_quarterly_returns.iloc[:, 0]

bond_cum_returns = (1 + bond_quarterly_returns).cumprod()

# ======================================================
# 3. Combine sub-portfolios into a Combined Portfolio (Quarterly)
# ======================================================

# Define common quarter-end dates from both series.
common_dates = eq_quarterly_returns.index.intersection(bond_quarterly_returns.index)
print("Common Dates (Quarter-End):")
print(common_dates)

# Use .loc[…] to extract only those common dates (without fill_value, to avoid introducing zeros).
eq_qtr = eq_quarterly_returns.loc[common_dates]
bond_qtr = bond_quarterly_returns.loc[common_dates]

# --- Debug: Inspect the aligned quarterly returns.
print("\nEquity Quarterly Returns (Aligned):")
print(eq_qtr)
print("\nBond Quarterly Returns (Aligned):")
print(bond_qtr)

# Compute the combined quarterly returns.
combined_qtr_returns = equity_weight * eq_qtr + bond_weight * bond_qtr
print("\nCombined Quarterly Returns (should equal Equity Quarterly Returns when weights are 1.0 and 0.0):")
print(combined_qtr_returns)

# Compute cumulative returns for the combined portfolio.
combined_cum_returns = (1 + combined_qtr_returns).cumprod()

# For a direct comparison, align the equity cumulative returns to the same common dates:
eq_cum_returns_aligned = eq_cum_returns.loc[common_dates]
print("\nEquity Cumulative Returns (Aligned):")
print(eq_cum_returns_aligned)
print("\nCombined Cumulative Returns (Should match Equity Cumulative Returns):")
print(combined_cum_returns)

# ======================================================
# 4. Benchmark: SPY Quarterly Returns
# ======================================================
spy_price = yf.download('SPY', start=Start_Date, end=End_Date, auto_adjust=True)['Close'].dropna()
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
combined_expected_return = equity_weight * equity_expected_return + bond_weight * bond_expected_return
print("Combined Portfolio Expected Annual Return: {:.2f}%".format(combined_expected_return * 100))

# Annualized volatility (using quarterly returns, multiply by sqrt(4)).
combined_volatility = combined_qtr_returns.std() * np.sqrt(4)
print("Combined Portfolio Annualized Volatility: {:.2f}%".format(combined_volatility * 100))

# ======================================================
# 6. Plot Performance (Cumulative Returns)
# ======================================================
plt.figure(figsize=(12, 8))
plt.plot(eq_cum_returns.index, eq_cum_returns, label='Equity Portfolio', linewidth=2)
plt.plot(bond_cum_returns.index, bond_cum_returns, label='Bond Portfolio', linewidth=2)
plt.plot(combined_cum_returns.index, combined_cum_returns, label='Combined Portfolio', linewidth=2, linestyle='--')
plt.plot(spy_cum_returns.index, spy_cum_returns, label='Benchmark (SPY)', linewidth=2, linestyle=':')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Combined Portfolio Backtest (Quarterly Rebalancing)')
plt.legend()
plt.grid(True)
plt.show()

# Optionally, save the plot.
plt.savefig('combined_portfolio_backtest.png', dpi=300, bbox_inches='tight')
