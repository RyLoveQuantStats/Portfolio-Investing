# equities.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def get_equity_portfolio(start_date, end_date, risk_free_rate=0.02, plot_frontier=False):
    """
    Builds an equity portfolio based on U.S. sector ETFs using CAPM expected returns,
    with quarterly rebalancing. For each quarter, the function uses all available 
    data up to that rebalance date to compute CAPM betas, expected returns, and an 
    annualized covariance matrix. It then runs a Monte Carlo simulation (10,000 iterations) 
    to identify the optimal allocation (maximizing the Sharpe ratio) which is applied 
    for that quarter.
    
    In this version, instead of generating daily returns, the portfolio return is 
    computed for each quarter (by compounding the daily returns over the quarter).
    
    Additionally, if plot_frontier=True, the efficient frontier is plotted only for the first quarter.
    
    Finally, the function computes the quarterly returns and cumulative returns and also
    overlays the benchmark (SPY) quarterly cumulative returns on the final plot.
    
    Parameters:
        start_date (str): The start date (format 'YYYY-MM-DD').
        end_date (str): The end date (format 'YYYY-MM-DD').
        risk_free_rate (float): Annual risk-free rate assumption (default: 0.02).
        plot_frontier (bool): Whether to show the efficient frontier plot for the first quarter.
        
    Returns:
        equity_quarterly_returns (pd.Series): Quarterly returns of the equity portfolio.
        equity_cum_returns (pd.Series): Cumulative returns (quarterly compounding) of the equity portfolio.
        equity_expected_return (float): Time-weighted average expected annual return from CAPM estimates.
    """
    # Define sector ETFs and benchmark (SPY)
    sector_etfs = ['XLY',  # Consumer Discretionary
                   'XLP',  # Consumer Staples
                   'XLE',  # Energy
                   'XLF',  # Financials
                   'XLV',  # Health Care
                   'XLI',  # Industrials
                   'XLB',  # Materials
                   'XLRE', # Real Estate
                   'XLK',  # Technology
                   'XLU',  # Utilities
                   'XLC']  # Communication Services
    benchmark_ticker = 'SPY'
    
    # Download adjusted close prices.
    tickers = sector_etfs + [benchmark_ticker]
    price_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    price_data = price_data.dropna()
    
    # Calculate daily returns.
    daily_returns = price_data.pct_change().dropna()
    
    # Determine quarterly rebalancing dates using quarter start frequency.
    # These dates are the first trading day of each quarter.
    rebalance_dates = daily_returns.resample('QS').first().index
    rebalance_dates = rebalance_dates[rebalance_dates >= daily_returns.index[0]]
    
    quarterly_returns_list = []  # to store one return per quarter
    exp_ret_list = []            # list of quarter CAPM expected return estimates (averaged over ETFs)
    weight_records = []          # record optimal weights by quarter
    
    for i, rebalance_date in enumerate(rebalance_dates):
        # Calibration period: all data up to and including the current rebalance date.
        calib_data = daily_returns.loc[:rebalance_date]
        etf_calib = calib_data[sector_etfs]
        bench_calib = calib_data[benchmark_ticker]
        
        # Compute annualized benchmark return using calibration data.
        avg_daily_bench = bench_calib.mean()
        annual_market_return = (1 + avg_daily_bench)**252 - 1
        
        # Compute CAPM beta and expected annual returns for each ETF.
        betas = {}
        expected_returns = {}
        for etf in sector_etfs:
            cov = np.cov(etf_calib[etf], bench_calib)[0, 1]
            var = bench_calib.var()
            beta = cov / var if var != 0 else 0
            betas[etf] = beta
            expected_returns[etf] = risk_free_rate + beta * (annual_market_return - risk_free_rate)
        exp_ret_series = pd.Series(expected_returns)
        exp_ret_list.append(exp_ret_series.mean())
        
        # Compute the annualized covariance matrix.
        cov_daily = etf_calib.cov()
        cov_matrix = cov_daily * 252
        
        # Monte Carlo simulation for optimal portfolio allocation.
        num_portfolios = 10000
        results = []
        num_assets = len(sector_etfs)
        np.random.seed(42)
        for _ in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            port_return = np.sum(exp_ret_series.values * weights)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (port_return - risk_free_rate) / port_volatility if port_volatility != 0 else 0
            results.append({
                'Return': port_return,
                'Volatility': port_volatility,
                'Sharpe': sharpe_ratio,
                'Weights': weights
            })
        results_df = pd.DataFrame(results)
        max_sharpe_idx = results_df['Sharpe'].idxmax()
        max_sharpe_portfolio = results_df.loc[max_sharpe_idx]
        optimal_weights = pd.Series(max_sharpe_portfolio['Weights'], index=sector_etfs)
        weight_records.append((rebalance_date, optimal_weights))
        
        # Plot the efficient frontier for the first quarter if requested (optional).
        '''
        if plot_frontier and i == 0:
            plt.figure(figsize=(10, 6))
            sc = plt.scatter(results_df['Volatility'], results_df['Return'], 
                             c=results_df['Sharpe'], cmap='viridis', marker='o', s=10, alpha=0.5)
            plt.colorbar(sc, label='Sharpe Ratio')
            plt.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return'], 
                        color='red', marker='*', s=500, label='Max Sharpe')
            plt.xlabel('Annualized Volatility')
            plt.ylabel('Annualized Return')
            plt.title(f'Efficient Frontier (Rebalance: {rebalance_date.date()})')
            plt.legend()
            plt.show()
        '''
        # Determine the period over which to calculate the portfolio return.
        # Here, we want to do the calculation at the end of the period (quarter-end) before the new one starts.
        if i < len(rebalance_dates) - 1:
            # Get all trading days from the current rebalance date until the next quarter's start.
            period_data = daily_returns.loc[rebalance_date:rebalance_dates[i+1]]
            # The period_end is the last trading day in this interval.
            period_end = period_data.index[-1]
        else:
            period_end = daily_returns.index[-1]
        
        # Extract daily returns for the current quarter period.
        quarter_data = daily_returns.loc[rebalance_date:period_end, sector_etfs]
        # Compute the portfolio daily returns for the quarter.
        portfolio_daily = (quarter_data * optimal_weights).sum(axis=1)
        # Compute the quarterly return by compounding daily returns.
        quarter_return = (1 + portfolio_daily).prod() - 1
        # The quarter return is assigned the timestamp of the period_end.
        quarterly_returns_list.append(pd.Series(quarter_return, index=[period_end]))
    
    # Concatenate quarterly returns into one Series and compute cumulative returns.
    equity_quarterly_returns = pd.concat(quarterly_returns_list).sort_index()
    equity_cum_returns = (1 + equity_quarterly_returns).cumprod()
    
    # Compute time-weighted average expected annual return (weighted by number of days in each quarter).
    quarter_lengths = [
        len(daily_returns.loc[rebalance_dates[i]:rebalance_dates[i+1]])
        if i < len(rebalance_dates)-1 else len(daily_returns.loc[rebalance_dates[i]:])
        for i in range(len(rebalance_dates))
    ]
    weighted_exp_return = np.average(exp_ret_list, weights=quarter_lengths)
    
    # Compute benchmark (SPY) quarterly returns.
    spy_data = price_data[benchmark_ticker]
    spy_quarterly = spy_data.resample('Q').last().pct_change().dropna()
    spy_cum_returns = (1 + spy_quarterly).cumprod()
    
    return equity_quarterly_returns, equity_cum_returns, weighted_exp_return

# Allow the module to run standalone for testing.
if __name__ == "__main__":
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    risk_free_rate = 0.02
    
    eq_qtr, eq_cum, eq_exp_return = get_equity_portfolio(start_date, end_date, risk_free_rate, plot_frontier=False)
    
    print("\nEquity Portfolio Quarterly Testing:")
    print("Final Cumulative Return: {:.2f}".format(eq_cum.iloc[-1]))
    print("Time-Weighted Average Expected Annual Return: {:.2f}%".format(eq_exp_return * 100))
    
    plt.figure(figsize=(10,6))
    plt.plot(eq_cum.index, eq_cum, label="Equity Portfolio Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Equity Portfolio (Quarterly Rebalancing) Backtest")
    plt.legend()
    plt.show()
