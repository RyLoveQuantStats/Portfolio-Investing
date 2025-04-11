# options.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

# Parameters for simulation
Start_Date = "2019-01-01"   # Use a longer period to capture volatility.
End_Date = "2024-12-31"
risk_free_rate = 0.01       # Annual risk-free rate (1%).
shares_per_contract = 100   # Number of shares per option contract.
strike_buffer = 0.05        # Call strike is 5% above the current price.
option_term_days = 30       # Option expires 30 days from sale.
etf_list = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV',
            'XLI', 'XLB', 'XLRE', 'XLK', 'XLU', 'XLC']

# ---------------------------
# Step 1: Define Black-Scholes Function
# ---------------------------
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European call option.
    
    Parameters:
      S (float): Spot price of the underlying asset.
      K (float): Strike price.
      T (float): Time to expiration in years.
      r (float): Annual risk-free rate.
      sigma (float): Annualized volatility.
    
    Returns:
      float: Theoretical price of the call option.
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# ---------------------------
# Step 2: Simulate Covered Call for a Single ETF
# ---------------------------
def simulate_covered_call_for_etf(ticker, start_date, end_date):
    """
    Run a daily simulation of a covered call strategy for a single ETF.
    
    Parameters:
      ticker (str): ETF ticker symbol.
      start_date (str): Start date for data download.
      end_date (str): End date for data download.
    
    Returns:
      pd.DataFrame: Daily portfolio value for the ETF.
    """
    # Download historical data for the given ticker.
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}")
    # Use the 'Close' prices.
    price_data = data["Close"].dropna()
    # Create a DataFrame from the Series and force the column name to 'Price'.
    df = pd.DataFrame(price_data)
    df.columns = ['Price']
    # Calculate log returns.
    df['Returns'] = np.log(df['Price'] / df['Price'].shift(1))
    # Calculate rolling volatility (30-day window, annualized).
    window_size = 30
    df['Volatility'] = df['Returns'].rolling(window=window_size).std() * np.sqrt(252)
    df.dropna(inplace=True)
    # Identify option selling dates: first trading day of each month.
    monthly_dates = df.resample('MS').first().index

    # === Modification: Initialize as if you already hold the underlying shares ===
    # In a covered call, you already own the stock. So start with one contract's worth:
    shares = shares_per_contract  # Already holding shares.
    cash = 0.0  # No cash initially.
    
    portfolio_value = []
    dates = []
    positions = []  # Option positions.

    for current_date in df.index:
        S = df.loc[current_date, 'Price']
        sigma = df.loc[current_date, 'Volatility']
        # Update portfolio value (should be nonzero after the first day).
        portfolio_val = cash + shares * S
        dates.append(current_date)
        portfolio_value.append(portfolio_val)
        
        # Check if any sold option has expired.
        positions_to_remove = []
        for pos in positions:
            if current_date >= pos['expiration_date']:
                K = pos['K']
                # If S > K, the option is exercised (shares are called away).
                if S > K:
                    cash += shares * K
                    shares = 0
                positions_to_remove.append(pos)
        for pos in positions_to_remove:
            positions.remove(pos)
        
        # On month start, sell a new call.
        if current_date in monthly_dates:
            # If not holding shares (because they were called away), buy a contract's worth.
            if shares == 0:
                shares = shares_per_contract
                cash -= shares * S
            # Define option parameters.
            T = option_term_days / 252.0
            K = S * (1 + strike_buffer)
            call_premium = black_scholes_call(S, K, T, risk_free_rate, sigma) * shares_per_contract
            positions.append({
                'expiration_date': current_date + pd.Timedelta(days=option_term_days),
                'K': K
            })
            cash += call_premium

    result_df = pd.DataFrame({'Date': dates, 'Portfolio Value': portfolio_value})
    result_df.set_index('Date', inplace=True)
    return result_df

# ---------------------------
# Step 3: Simulate Portfolio Covered Call Strategy
# ---------------------------
def simulate_covered_calls_portfolio(etf_list, start_date, end_date):
    """
    Run the covered call simulation for each ETF in the list and combine them with equal weighting.
    
    Parameters:
      etf_list (list): List of ETF ticker symbols.
      start_date (str): Start date for simulation.
      end_date (str): End date for simulation.
      
    Returns:
      pd.Series: Combined portfolio value time series.
    """
    portfolio_dfs = {}
    for ticker in etf_list:
        try:
            df = simulate_covered_call_for_etf(ticker, start_date, end_date)
            portfolio_dfs[ticker] = df['Portfolio Value']
            print(f"Simulated {ticker}: {df['Portfolio Value'].iloc[0]:.2f} -> {df['Portfolio Value'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"Error simulating {ticker}: {e}")
    if not portfolio_dfs:
        raise ValueError("No ETF data available for simulation.")
    # Combine individual ETF portfolio values by aligning on dates.
    combined_df = pd.concat(portfolio_dfs, axis=1)
    combined_portfolio = combined_df.mean(axis=1)
    return combined_portfolio

# ---------------------------
# Step 4: Run Simulation and Visualize Results
# ---------------------------
if __name__ == "__main__":
    combined_portfolio = simulate_covered_calls_portfolio(etf_list, Start_Date, End_Date)
    
    # Calculate buy-and-hold portfolio (equal weight across ETFs).
    price_data_all = yf.download(etf_list, start=Start_Date, end=End_Date, auto_adjust=True)['Close']
    buy_hold = price_data_all.mean(axis=1)
    initial_value = buy_hold.iloc[0]
    buy_hold_portfolio = buy_hold / initial_value  # Normalized time series.
    
    # Normalize the combined portfolio time series (starting at 1).
    combined_norm = combined_portfolio / combined_portfolio.iloc[0]
    
    plt.figure(figsize=(14, 7))
    plt.plot(combined_norm.index, combined_norm.values, label='Covered Call Strategy')
    plt.plot(buy_hold_portfolio.index, buy_hold_portfolio.values, label='Buy & Hold Strategy')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.title('Covered Call Strategy vs. Buy & Hold Strategy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    covered_call_return = (combined_norm.iloc[-1] - 1) * 100
    buy_hold_return = (buy_hold_portfolio.iloc[-1] - 1) * 100
    print(f"Total Return of Covered Call Strategy: {covered_call_return:.2f}%")
    print(f"Total Return of Buy & Hold Strategy: {buy_hold_return:.2f}%")
