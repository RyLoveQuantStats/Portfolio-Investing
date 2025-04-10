# bonds.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from datetime import datetime

def get_bond_portfolio(start_date, end_date, T=0.25, plot_results=True):
    """
    Builds a bond "portfolio" based on roll-down valuation. It downloads Treasury 
    yields for maturities 2Y, 5Y, 10Y, 20Y, and 30Y from FRED, resamples the data 
    to quarterly snapshots, and then uses a roll-down model to compute the expected 
    annualized return for each bond over a holding period T (default: 0.25 years).
    
    For each quarter, it calculates the expected return on each bond using the formula:
         Expected Return = [ current_yield * maturity - y_new * (maturity - T) ] / T
    where y_new is obtained by linear interpolation on the current yield curve.
    
    Parameters:
        start_date (str): The start date (format 'YYYY-MM-DD').
        end_date (str): The end date (format 'YYYY-MM-DD').
        T (float): The holding period in years over which to roll down the yield curve (default 0.25 for a quarter).
        plot_results (bool): If True, plot the quarterly expected returns for each maturity.
        
    Returns:
        expected_returns_quarterly (pd.DataFrame): A DataFrame indexed by quarter with columns '2Y', '5Y', '10Y', '20Y', '30Y'
                                                   representing the computed expected annualized returns.
        overall_expected_return (float): A simple average expected annualized return across bonds and quarters.
    """
    # -------------------------------------------------
    # 1. DATA ACQUISITION: Download Treasury yields from FRED
    # -------------------------------------------------
    # FRED series for U.S. Treasury yields:
    # 'DGS2'   = 2-Year, 'DGS5'  = 5-Year, 'DGS10' = 10-Year,
    # 'DGS20'  = 20-Year, 'DGS30' = 30-Year.
    series = ['DGS2', 'DGS5', 'DGS10', 'DGS20', 'DGS30']
    yields_df = web.DataReader(series, 'fred', start_date, end_date)
    
    # Drop any days with missing data and rename columns for clarity.
    yields_df = yields_df.dropna()
    yields_df.columns = ['2Y', '5Y', '10Y', '20Y', '30Y']
    
    # -------------------------------------------------
    # 2. RESAMPLE THE DATA TO A QUARTERLY SNAPSHOT
    # -------------------------------------------------
    # We'll use the last available observation in each quarter.
    quarterly_yields = yields_df.resample('Q').last()
    
    # -------------------------------------------------
    # 3. DEFINE THE ROLL-DOWN EXPECTED RETURN MODEL
    # -------------------------------------------------
    # The holding period T is 0.25 years by default.
    # For a zero-coupon bond with current yield y and maturity M, if held for T years,
    # its new yield y_new is determined by linearly interpolating the yield curve at (M - T).
    def compute_expected_return(maturity, current_yield, maturities, yield_curve, T):
        """
        Calculate the expected annualized return for a bond with a given maturity.
        
        Parameters:
            maturity (float): The bond's current maturity (in years).
            current_yield (float): The current yield (as decimal) at that maturity.
            maturities (np.array): Array of benchmark maturities (e.g., [2, 5, 10, 20, 30]).
            yield_curve (np.array): Array of yields (as decimals) corresponding to the maturities.
            T (float): The holding period in years (e.g., 0.25 for quarterly).
        
        Returns:
            float: The approximated expected annualized return.
        """
        new_maturity = maturity - T
        # For values below the minimum available maturity, use the 2Y yield.
        y_new = np.interp(new_maturity, maturities, yield_curve, left=yield_curve[0], right=yield_curve[-1])
        return (current_yield * maturity - y_new * (maturity - T)) / T
    
    # Define maturities corresponding to the yield series.
    maturities = np.array([2, 5, 10, 20, 30], dtype=float)
    
    # -------------------------------------------------
    # 4. COMPUTE EXPECTED RETURNS FOR EACH QUARTER AND BOND
    # -------------------------------------------------
    # Prepare a DataFrame to store the computed expected returns.
    expected_returns_quarterly = pd.DataFrame(index=quarterly_yields.index,
                                              columns=['2Y','5Y','10Y','20Y','30Y'])
    
    # Loop over each quarter.
    for date, row in quarterly_yields.iterrows():
        # Retrieve the yield curve for this quarter and convert from percentage to decimal.
        yield_curve = np.array([row['2Y'], row['5Y'], row['10Y'], row['20Y'], row['30Y']], dtype=float) / 100.0
        for i, mat in enumerate(maturities):
            exp_ret = compute_expected_return(mat, yield_curve[i], maturities, yield_curve, T)
            expected_returns_quarterly.loc[date, f'{int(mat)}Y'] = exp_ret
    
    # Convert expected returns to numeric format.
    expected_returns_quarterly = expected_returns_quarterly.astype(float)
    
    # Print the quarterly expected returns.
    print("Quarterly Expected Annualized Returns (based on roll-down valuation):")
    print(expected_returns_quarterly)
    
    # -------------------------------------------------
    # 5. VISUALIZATION
    # -------------------------------------------------
    if plot_results:
        plt.figure(figsize=(10, 6))
        for col in expected_returns_quarterly.columns:
            plt.plot(expected_returns_quarterly.index, expected_returns_quarterly[col],
                     marker='o', linestyle='-', label=f'{col} Bond')
        plt.xlabel('Quarter')
        plt.ylabel('Expected Annualized Return')
        plt.title('Quarterly Expected Annualized Returns by Bond Maturity')
        plt.legend()
        plt.show()
    
    # Compute an overall expected annual return as the simple average across all bonds and quarters.
    overall_expected_return = expected_returns_quarterly.mean().mean()
    
    return expected_returns_quarterly, overall_expected_return


# Allow this module to run standalone for testing.
if __name__ == "__main__":
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    expected_returns_quarterly, overall_expected_return = get_bond_portfolio(start_date, end_date, T=0.25, plot_results=True)
    
    print("\nOverall Expected Annual Return (average across bonds and quarters): {:.2f}%".format(overall_expected_return * 100))
