# bonds.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from datetime import datetime
from config import Start_Date, End_Date

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
    
    # -------------------------
    # 2. RESAMPLE THE DATA (unchanged)
    quarterly_yields = yields_df.resample('Q').last()

    # -------------------------
    # 3. DEFINE ROLL-DOWN MODEL (unchanged)
    def compute_expected_return(maturity, current_yield, maturities, yield_curve, T):
        new_maturity = maturity - T
        y_new = np.interp(new_maturity, maturities, yield_curve, left=yield_curve[0], right=yield_curve[-1])
        return (current_yield * maturity - y_new * (maturity - T)) / T

    maturities = np.array([2, 5, 10, 20, 30], dtype=float)

    # -------------------------
    # 4. COMPUTE EXPECTED RETURNS FOR EACH QUARTER
    expected_returns_quarterly = pd.DataFrame(index=quarterly_yields.index,
                                            columns=['2Y','5Y','10Y','20Y','30Y'])

    for date, row in quarterly_yields.iterrows():
        yield_curve = np.array([row['2Y'], row['5Y'], row['10Y'], row['20Y'], row['30Y']],
                            dtype=float) / 100.0
        for i, mat in enumerate(maturities):
            exp_ret = compute_expected_return(mat, yield_curve[i], maturities, yield_curve, T)
            expected_returns_quarterly.loc[date, f'{int(mat)}Y'] = exp_ret

    expected_returns_quarterly = expected_returns_quarterly.astype(float)

    print("Quarterly Expected Annualized Returns (based on roll-down valuation):")
    print(expected_returns_quarterly)

    # -------------------------
    # 5. VISUALIZATION (unchanged)
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

    # Compute overall expected annual return (simple average).
    overall_expected_return = expected_returns_quarterly.mean().mean()

    # --------------------------------------------------
    # NEW LOGIC: If the overall expected return is not profitable,
    # we assume bonds are not attractive and therefore the strategy remains in cash.
    if overall_expected_return <= 0:
        print("Bonds expected returns not profitable. Remaining in cash.")
        # Override all the expected return data to zero, reflecting a cash allocation.
        expected_returns_quarterly[:] = 0.0
        overall_expected_return = 0.0

    return expected_returns_quarterly, overall_expected_return

    # --------------------------------------------------
    # NEW LOGIC: If the overall expected return is not profitable,
    # we assume bonds are not attractive and therefore the strategy remains in cash.
    if overall_expected_return <= 0:
        print("Bonds expected returns not profitable. Remaining in cash.")
        # Override all the expected return data to zero, reflecting a cash allocation.
        expected_returns_quarterly[:] = 0.0
        overall_expected_return = 0.0

    return expected_returns_quarterly, overall_expected_return

# Allow this module to run standalone for testing.
if __name__ == "__main__":
    start_date = Start_Date
    end_date = End_Date
    expected_returns_quarterly, overall_expected_return = get_bond_portfolio(start_date, end_date, T=0.25, plot_results=True)
    
    print("\nOverall Expected Annual Return (average across bonds and quarters): {:.2f}%".format(overall_expected_return * 100))
