import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import timedelta
# PyPortfolioOpt imports:
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- Configuration ---
Start_Date = "2019-01-01"
End_Date   = "2024-12-31"
risk_free_equity = 0.02      # For CAPM & Sharpe computations (annual)
risk_free_option = 0.01      # For Black-Scholes pricing (annual)
shares_per_contract = 100
option_term_days = 30
transaction_cost = 0.001     # 0.1% per trade cost

# Default parameters; these can be varied
calibration_years = 2        # Use the most recent 2 years of data for calibration in equity optimization.
base_strike_buffer = 0.03    # Base strike buffer (3%)
strike_buffer_multiplier = 0.05  # Additional buffer per unit volatility
max_strike_buffer = 0.10     # Maximum allowed strike buffer

# Dynamic weight allocation parameters
high_equity_weight = 0.70
low_equity_weight = 0.60
vol_threshold = 0.20         # If SPY volatility (annualized) exceeds 20%

# Same ETF list and market benchmark
etf_list = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 
            'XLI', 'XLB', 'XLRE', 'XLK', 'XLU', 'XLC']
benchmark_ticker = 'SPY'

# --- Helper Functions ---

def dynamic_strike_buffer(current_vol, base=base_strike_buffer, multiplier=strike_buffer_multiplier, cap=max_strike_buffer):
    """
    Compute a dynamic strike buffer based on the current volatility.
    For example, a higher volatility increases the buffer but is capped.
    """
    buffer = base + multiplier * current_vol  # current_vol is annualized volatility
    return min(buffer, cap)

def dynamic_weight_allocation(current_vol, low=low_equity_weight, high=high_equity_weight, threshold=vol_threshold):
    """
    Adjust the overall equity allocation dynamically based on market volatility.
    If current SPY volatility is above the threshold, reduce equity allocation.
    """
    if current_vol >= threshold:
        return low
    else:
        return high

# --- Black-Scholes Call Option Pricing ---
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European call option.
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# --- Covered Call Simulation for a Single ETF ---
def simulate_covered_call_for_etf(ticker, price_series):
    """
    Simulate daily portfolio values for a covered call strategy on one ETF.
    Incorporates dynamic strike buffer using the day's volatility.
    """
    df = pd.DataFrame({'Price': price_series.dropna()})
    df['LogReturn'] = np.log(df['Price'] / df['Price'].shift(1))
    df['Volatility'] = df['LogReturn'].rolling(window=30).std() * np.sqrt(252)  # Annualized vol.
    df.dropna(inplace=True)
    
    # Identify monthly option sale dates – first trading day of each month.
    monthly_dates = df.resample('MS').first().index
    monthly_dates = [d for d in monthly_dates if d in df.index]
    
    # Initialize: Start with 100 shares and zero cash.
    shares = shares_per_contract
    cash = 0.0
    portfolio_values = []
    dates = []
    active_calls = []  # Each call: {'strike': K, 'expiration': date}
    
    for current_date in df.index:
        S = df.at[current_date, 'Price']
        sigma = df.at[current_date, 'Volatility']
        portfolio_val = cash + shares * S
        dates.append(current_date)
        portfolio_values.append(portfolio_val)
        
        # Check for option expirations.
        for pos in list(active_calls):
            if current_date >= pos['expiration']:
                K = pos['strike']
                if S > K and shares > 0:
                    cash += shares * K
                    shares = 0
                active_calls.remove(pos)
        
        # On a monthly sale date, sell a new call.
        if current_date in monthly_dates:
            # If shares were called away, buy new shares (incur cost + transaction cost).
            if shares == 0:
                shares = shares_per_contract
                cash -= shares * S
                cash -= (shares * S) * transaction_cost
            # Calculate dynamic strike buffer using current volatility.
            dyn_buffer = dynamic_strike_buffer(sigma)
            K = S * (1 + dyn_buffer)
            expiration_date = current_date + pd.Timedelta(days=option_term_days)
            T = option_term_days / 252.0
            premium = black_scholes_call(S, K, T, risk_free_option, sigma) * shares_per_contract
            cash += premium
            cash -= (shares * S) * transaction_cost  # transaction cost on sale
            active_calls.append({'strike': K, 'expiration': expiration_date})
    
    return pd.Series(portfolio_values, index=dates)

# --- Covered Call Portfolio ---
def simulate_covered_calls_portfolio(etfs, price_data):
    """
    Run the covered call simulation for each ETF and combine equally.
    Returns a normalized daily series.
    """
    portfolios = {}
    for ticker in etfs:
        try:
            prices = price_data[ticker]
            portfolios[ticker] = simulate_covered_call_for_etf(ticker, prices)
        except Exception as e:
            print(f"Warning: Simulation for {ticker} failed: {e}")
    if not portfolios:
        raise ValueError("No ETF data available for covered call simulation.")
    combined_df = pd.DataFrame(portfolios).sort_index().ffill()
    combined_portfolio = combined_df.mean(axis=1)
    normalized = combined_portfolio / combined_portfolio.iloc[0]
    return normalized

# --- Equity Portfolio Simulation (Using PyPortfolioOpt) ---
def simulate_equity_portfolio(etfs, benchmark, price_data, calib_years=calibration_years):
    """
    Simulate a rebalanced equity portfolio using a rolling calibration window and PyPortfolioOpt.
    Returns a normalized daily series.
    """
    daily_returns = price_data.pct_change().dropna()
    rebalance_dates = daily_returns.resample('QS').first().index
    rebalance_dates = [d for d in rebalance_dates if d >= daily_returns.index[0]]
    
    dates = []
    values = []
    current_value = 1.0
    
    for i, rebalance_date in enumerate(rebalance_dates):
        if rebalance_date > daily_returns.index[-1]:
            break
        # Define the calibration window (most recent calib_years)
        window_start = rebalance_date - pd.DateOffset(years=calib_years)
        calib_data = daily_returns.loc[window_start:rebalance_date]
        # Use PyPortfolioOpt to compute expected returns and covariance.
        mu = expected_returns.mean_historical_return(price_data.loc[window_start:rebalance_date], frequency=252)
        Sigma = risk_models.sample_cov(price_data.loc[window_start:rebalance_date], frequency=252)
        # Use EfficientFrontier to maximize Sharpe Ratio.
        ef = EfficientFrontier(mu, Sigma, weight_bounds=(0, 1))
        ef.max_sharpe(risk_free_equity)
        weights = ef.clean_weights()
        optimal_weights = pd.Series(weights)
        # Apply rebalancing cost after the first rebalance.
        if i > 0:
            current_value *= (1 - transaction_cost)
        
        # Determine period end.
        if i < len(rebalance_dates) - 1:
            next_start = rebalance_dates[i + 1]
            period_idx = daily_returns.index[daily_returns.index < next_start]
            period_end = period_idx[-1] if len(period_idx) > 0 else daily_returns.index[-1]
        else:
            period_end = daily_returns.index[-1]
        
        period_returns = daily_returns.loc[rebalance_date:period_end, etfs]
        if rebalance_date not in period_returns.index:
            dates.append(rebalance_date)
            values.append(current_value)
        for day in period_returns.index:
            if day == rebalance_date:
                dates.append(day)
                values.append(current_value)
            # Compute daily portfolio return via dot product with optimal weights.
            daily_ret = np.dot(period_returns.loc[day, etfs], optimal_weights[etfs])
            current_value *= (1 + daily_ret)
            dates.append(day)
            values.append(current_value)
    
    equity_portfolio = pd.Series(values, index=dates)
    equity_portfolio = equity_portfolio[~equity_portfolio.index.duplicated(keep='first')]
    equity_index = equity_portfolio / equity_portfolio.iloc[0]
    return equity_index

# --- Bootstrap Sensitivity Analysis ---
def bootstrap_sensitivity(index_series, n_iter=1000, sample_frac=0.5):
    """
    Perform bootstrap resampling on the portfolio's daily returns (derived from index_series)
    to provide sensitivity analysis for key risk metrics.
    Returns a DataFrame with the bootstrap distribution of metrics.
    """
    returns = index_series.pct_change().dropna()
    metrics_list = []
    for _ in range(n_iter):
        sample = returns.sample(frac=sample_frac, replace=True)
        sample_series = (1 + sample).cumprod()
        cum_ret, ann_ret, ann_vol, sharpe, max_dd = compute_risk_metrics(sample_series)
        metrics_list.append((cum_ret, ann_ret, ann_vol, sharpe, max_dd))
    bootstrap_df = pd.DataFrame(metrics_list, columns=["Cumulative Return", "Annual Return", "Volatility", "Sharpe", "Max Drawdown"])
    return bootstrap_df

# --- Compute Risk Metrics ---
def compute_risk_metrics(index_series, risk_free=0.02):
    """
    Compute key risk metrics for a normalized index series.
    Returns cumulative return, annualized return, volatility, Sharpe ratio, and max drawdown.
    """
    returns = index_series.pct_change().dropna()
    cumulative_return = index_series.iloc[-1] / index_series.iloc[0] - 1
    annual_factor = 252
    annual_return = (index_series.iloc[-1] / index_series.iloc[0])**(annual_factor / len(index_series)) - 1
    ann_vol = returns.std() * np.sqrt(annual_factor)
    sharpe = (annual_return - risk_free) / (ann_vol if ann_vol != 0 else np.nan)
    running_max = index_series.cummax()
    drawdown = (index_series - running_max) / running_max
    max_drawdown = drawdown.min()
    return cumulative_return, annual_return, ann_vol, sharpe, max_drawdown

# --- Main Execution ---
if __name__ == "__main__":
    # Download historical price data
    all_tickers = etf_list + [benchmark_ticker]
    price_data = yf.download(all_tickers, start=Start_Date, end=End_Date, auto_adjust=True)['Close']
    price_data.dropna(inplace=True)
    
    # Run simulations
    equity_index = simulate_equity_portfolio(etf_list, benchmark_ticker, price_data)
    covered_call_index = simulate_covered_calls_portfolio(etf_list, price_data)
    
    # Use common full date range starting at the later of the two series' first dates.
    common_start = max(equity_index.index.min(), covered_call_index.index.min())
    full_dates = pd.date_range(start=common_start, end=price_data.index.max(), freq='B')
    
    # Reindex simulated series using both forward and backward fill.
    equity_index_full = equity_index.reindex(full_dates).ffill().bfill()
    covered_call_index_full = covered_call_index.reindex(full_dates).ffill().bfill()
    
    # Compute daily returns for each.
    equity_daily_ret = equity_index_full.pct_change().fillna(0)
    option_daily_ret = covered_call_index_full.pct_change().fillna(0)
    
    # Get current SPY volatility (using a 30-day window on SPY)
    spy_prices = price_data[benchmark_ticker]
    spy_returns = spy_prices.pct_change().dropna()
    spy_vol = spy_returns.rolling(window=30).std().iloc[-1] * np.sqrt(252)
    dyn_equity_weight = dynamic_weight_allocation(spy_vol)
    
    print(f"Dynamic equity weight based on SPY volatility ({spy_vol:.2%}): {dyn_equity_weight:.2f}")
    
    # Combined portfolio via daily returns weighted dynamically (equity weight and remainder for options).
    combined_daily_ret = dyn_equity_weight * equity_daily_ret + (1 - dyn_equity_weight) * option_daily_ret
    combined_index = (1 + combined_daily_ret).cumprod()
    
    # SPY Buy & Hold (normalized) reindexed on common dates.
    spy_index = (price_data[benchmark_ticker] / price_data[benchmark_ticker].iloc[0]).reindex(full_dates).ffill().bfill()
    
    # Plot cumulative returns.
    plt.figure(figsize=(10,6))
    plt.plot(equity_index_full.index, equity_index_full.values, label='Equity Portfolio')
    plt.plot(covered_call_index_full.index, covered_call_index_full.values, label='Covered Call Strategy')
    plt.plot(combined_index.index, combined_index.values, label='Combined Portfolio')
    plt.plot(spy_index.index, spy_index.values, label='SPY Buy & Hold', linestyle='--')
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value (Start = 1.0)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('cumulative_returns_comparison.png')
    plt.show()
    
    # Compute risk metrics.
    strategies = {
        'Equity Portfolio': equity_index_full,
        'Covered Call Strategy': covered_call_index_full,
        'Combined Portfolio': combined_index,
        'SPY Buy & Hold': spy_index
    }
    
    metrics = []
    for name, series in strategies.items():
        cum_ret, ann_ret, ann_vol, sharpe, max_dd = compute_risk_metrics(series, risk_free=risk_free_equity)
        metrics.append({
            "Strategy": name,
            "Cumulative Return (%)": cum_ret * 100,
            "Annualized Return (%)": ann_ret * 100,
            "Annualized Volatility (%)": ann_vol * 100,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd * 100
        })
    
    metrics_df = pd.DataFrame(metrics)
    print("\nRisk Metrics Comparison:")
    print(metrics_df)
    
    # Write risk metrics to Excel.
    excel_filename = "portfolio_risk_metrics.xlsx"
    metrics_df.to_excel(excel_filename, index=False)
    print(f"\nRisk metrics have been written to {excel_filename}")
    
    # Perform bootstrap sensitivity analysis on the combined portfolio.
    bootstrap_results = bootstrap_sensitivity(combined_index, n_iter=500, sample_frac=0.5)
    print("\nBootstrap Sensitivity Analysis (first 5 rows):")
    print(bootstrap_results.head())
