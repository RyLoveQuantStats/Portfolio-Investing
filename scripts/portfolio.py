import pandas as pd
import matplotlib.pyplot as plt
from config import Start_Date, End_Date

from equities import get_equity_portfolio
from options import simulate_covered_calls

def blend_portfolios(equity_cum, cc_cum, equity_weight=0.70, cc_weight=0.30):
    """
    Combine cumulative returns from equities and covered-call strategies based on specified weights.
    
    Parameters:
      equity_cum (pd.Series): Cumulative returns from the equity portfolio.
      cc_cum (pd.Series): Cumulative returns from the covered call strategy.
      equity_weight (float): Allocation weight for equities.
      cc_weight (float): Allocation weight for covered calls.
      
    Returns:
      blended (pd.Series): Combined portfolio cumulative returns.
    """
    # Resample the covered call cumulative returns to quarterly frequency 
    # (to match the equity portfolio data).
    cc_quarterly = cc_cum.resample('Q').last()
    cc_quarterly = cc_quarterly.reindex(equity_cum.index).ffill()
    
    # Assume an initial portfolio of 1 where each component compounds separately.
    blended = equity_weight * equity_cum + cc_weight * cc_quarterly
    return blended

def main():
    # Set risk–free rate (you can adjust as needed).
    risk_free_rate = 0.02

    # 1. Run the equity portfolio simulation (quarterly rebalancing).
    eq_quarterly_returns, eq_cum_returns, eq_exp_return = get_equity_portfolio(Start_Date, End_Date, risk_free_rate, plot_frontier=False)
    
    # 2. Run the SPY covered call simulation (monthly rebalancing).
    cc_monthly_returns, cc_cum_returns = simulate_covered_calls(strike_buffer=0.05, sigma=0.15, r=risk_free_rate)
    
    # 3. Blend the two strategies using target weights (e.g. 70% equities and 30% covered calls).
    blended_cum_returns = blend_portfolios(eq_cum_returns, cc_cum_returns, equity_weight=0.70, cc_weight=0.30)
    
    # 4. Visualize the performance of each component and the blended portfolio.
    plt.figure(figsize=(12, 8))
    plt.plot(eq_cum_returns.index, eq_cum_returns, label="Equity Portfolio (Quarterly)")
    # For plotting, also resample covered-call results to quarterly.
    cc_quarterly = cc_cum_returns.resample('Q').last()
    plt.plot(cc_quarterly.index, cc_quarterly, label="Covered Calls on SPY (Monthly, resampled Quarterly)")
    plt.plot(blended_cum_returns.index, blended_cum_returns, label="Blended Portfolio (70% Equities / 30% Covered Calls)", linewidth=2, linestyle='--')
    
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Portfolio Returns: Equities vs. Covered Calls vs. Blended Strategy")
    plt.legend()
    plt.show()
    
    # Display final portfolio performance.
    print("Final Equity Portfolio Cumulative Return: {:.2f}%".format(eq_cum_returns.iloc[-1] * 100))
    print("Final Covered Call Strategy Cumulative Return: {:.2f}%".format(cc_quarterly.iloc[-1] * 100))
    print("Final Blended Portfolio Cumulative Return: {:.2f}%".format(blended_cum_returns.iloc[-1] * 100))

if __name__ == "__main__":
    main()
