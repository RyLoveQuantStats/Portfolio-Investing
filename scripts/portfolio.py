import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Configuration ---
Start_Date = "2019-01-01"
End_Date   = "2024-12-31"
risk_free_equity = 0.02   # For CAPM and Sharpe computations
risk_free_option = 0.01   # For Black-Scholes pricing
shares_per_contract = 100
strike_buffer = 0.05      # Call strike 5% above current price
option_term_days = 30
transaction_cost = 0.001  # 0.1% transaction cost per trade

# Same ETF list for both strategies and market benchmark
etf_list = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 
            'XLI', 'XLB', 'XLRE', 'XLK', 'XLU', 'XLC']
benchmark_ticker = 'SPY'

# --- Black-Scholes Call Option Pricing ---
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European call option.
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call_price

# --- Covered Call Simulation for a Single ETF ---
def simulate_covered_call_for_etf(ticker, price_series):
    """
    Simulate daily portfolio values for a covered call strategy on a single ETF.
    """
    df = pd.DataFrame({'Price': price_series.dropna()})
    df['LogReturn'] = np.log(df['Price'] / df['Price'].shift(1))
    df['Volatility'] = df['LogReturn'].rolling(window=30).std() * np.sqrt(252)
    df.dropna(inplace=True)
    
    # Identify monthly option sale dates (first trading day of each month)
    monthly_dates = df.resample('MS').first().index
    monthly_dates = [d for d in monthly_dates if d in df.index]
    
    # Initialize portfolio: 100 shares and zero cash.
    shares = shares_per_contract
    cash = 0.0
    portfolio_values = []
    dates = []
    active_calls = []  # Track active call positions: {'strike': K, 'expiration': date}
    
    for current_date in df.index:
        S = df.at[current_date, 'Price']
        sigma = df.at[current_date, 'Volatility']
        portfolio_val = cash + shares * S
        dates.append(current_date)
        portfolio_values.append(portfolio_val)
        
        # Process option expirations.
        for pos in list(active_calls):
            if current_date >= pos['expiration']:
                K = pos['strike']
                if S > K and shares > 0:
                    cash += shares * K
                    shares = 0
                active_calls.remove(pos)
        
        # On monthly sale dates, sell a new call.
        if current_date in monthly_dates:
            # If shares were called away, buy new shares (apply transaction cost on purchase)
            if shares == 0:
                shares = shares_per_contract
                cash -= shares * S
                cash -= (shares * S) * transaction_cost
            K = S * (1 + strike_buffer)
            expiration_date = current_date + pd.Timedelta(days=option_term_days)
            T = option_term_days / 252.0
            premium = black_scholes_call(S, K, T, risk_free_option, sigma) * shares_per_contract
            cash += premium
            cash -= (shares * S) * transaction_cost  # transaction cost on option sale
            active_calls.append({'strike': K, 'expiration': expiration_date})
    
    return pd.Series(portfolio_values, index=dates)

# --- Covered Call Portfolio (Combined Equally Across ETFs) ---
def simulate_covered_calls_portfolio(etfs, price_data):
    """
    Run the covered call simulation for each ETF and combine them equally.
    Returns a normalized (start = 1) daily series.
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

# --- Equity Portfolio Simulation (Quarterly-Rebalanced Sharpe-Optimal) ---
def simulate_equity_portfolio(etfs, benchmark, price_data):
    """
    Simulate a quarterly-rebalanced equity portfolio using CAPM estimates and 
    Monte Carlo simulation for a Sharpe-optimal allocation.
    Returns a normalized (start = 1) daily series.
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
        calib_data = daily_returns.loc[:rebalance_date]
        etf_calib = calib_data[etfs]
        bench_calib = calib_data[benchmark]
        avg_daily_bench = bench_calib.mean()
        annual_market_return = (1 + avg_daily_bench)**252 - 1
        exp_returns = {}
        for etf in etf_calib.columns:
            cov = np.cov(etf_calib[etf], bench_calib)[0, 1]
            var = bench_calib.var()
            beta = cov / var if var != 0 else 0.0
            exp_returns[etf] = risk_free_equity + beta * (annual_market_return - risk_free_equity)
        exp_ret_series = pd.Series(exp_returns)
        cov_matrix = etf_calib.cov() * 252
        
        num_portfolios = 10000
        best_sharpe = -np.inf
        best_weights = None
        np.random.seed(42)
        for _ in range(num_portfolios):
            w = np.random.rand(len(etfs))
            w /= np.sum(w)
            port_return = np.dot(exp_ret_series.values, w)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            sharpe = (port_return - risk_free_equity) / (port_vol if port_vol != 0 else 1)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w
        optimal_weights = pd.Series(best_weights, index=etfs)
        
        # Apply a rebalancing transaction cost after the first quarter.
        if i > 0:
            current_value *= (1 - transaction_cost)
        
        if i < len(rebalance_dates) - 1:
            next_start = rebalance_dates[i+1]
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
            daily_ret = np.dot(period_returns.loc[day, etfs], optimal_weights)
            current_value *= (1 + daily_ret)
            dates.append(day)
            values.append(current_value)
    
    equity_portfolio = pd.Series(values, index=dates)
    equity_portfolio = equity_portfolio[~equity_portfolio.index.duplicated(keep='first')]
    equity_index = equity_portfolio / equity_portfolio.iloc[0]
    return equity_index

# --- Compute Risk Metrics ---
def compute_risk_metrics(index_series, risk_free=0.02):
    """
    Given a normalized index time series, compute:
      - Cumulative Return (%)
      - Annualized Return (%)
      - Annualized Volatility (%)
      - Sharpe Ratio
      - Maximum Drawdown (%)
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
    
    # Define a common full date range starting at the later of the two series' first dates:
    common_start = max(equity_index.index.min(), covered_call_index.index.min())
    full_dates = pd.date_range(start=common_start, end=price_data.index.max(), freq='B')
    
    # Reindex the simulated series using both forward and backward fill
    equity_index_full = equity_index.reindex(full_dates).ffill().bfill()
    covered_call_index_full = covered_call_index.reindex(full_dates).ffill().bfill()
    
    # Compute daily returns (fill missing with 0) for the separate series
    equity_daily_ret = equity_index_full.pct_change().fillna(0)
    option_daily_ret = covered_call_index_full.pct_change().fillna(0)
    
    # Combined portfolio via daily returns weighted 70/30 then compounded
    combined_daily_ret = 0.7 * equity_daily_ret + 0.3 * option_daily_ret
    combined_index = (1 + combined_daily_ret).cumprod()
    
    # SPY Buy & Hold: normalized series reindexed on common dates
    spy_index = price_data[benchmark_ticker] / price_data[benchmark_ticker].iloc[0]
    spy_index = spy_index.reindex(full_dates).ffill().bfill()
    
    # Plot cumulative returns for each strategy
    plt.figure(figsize=(10,6))
    plt.plot(equity_index_full.index, equity_index_full.values, label='Equity Portfolio (70%)')
    plt.plot(covered_call_index_full.index, covered_call_index_full.values, label='Covered Call Strategy (30%)')
    plt.plot(combined_index.index, combined_index.values, label='Combined Portfolio (70/30)')
    plt.plot(spy_index.index, spy_index.values, label='SPY Buy & Hold', linestyle='--')
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value (Start = 1.0)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('cumulative_returns_comparison.png')
    
    # Compute risk metrics for each strategy
    strategies = {
        'Equity Portfolio (70%)': equity_index_full,
        'Covered Call Strategy (30%)': covered_call_index_full,
        'Combined Portfolio (70/30)': combined_index,
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
    
    # Write risk metrics to Excel
    excel_filename = "portfolio_risk_metrics.xlsx"
    metrics_df.to_excel(excel_filename, index=False)
    print(f"\nRisk metrics have been written to {excel_filename}")
