import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

def gph_estimator(series, bandwidth_power=0.5):
    """
    Geweke-Porter-Hudak (GPH) estimator for the fractional integration parameter (d).
    """
    y = np.asarray(series)
    n = len(y)
    fft_y = np.fft.fft(y)
    periodogram = (np.abs(fft_y) ** 2) / (2 * np.pi * n)
    j = np.arange(1, int(n ** bandwidth_power) + 1)
    frequencies = 2 * np.pi * j / n
    X = -2 * np.log(2 * np.sin(frequencies / 2))
    X = sm.add_constant(X)
    Y = np.log(periodogram[j])
    model = sm.OLS(Y, X).fit()
    return model.params[1]

def quandt_andrews_unknown_break(series, trim=0.15):
    """
    Quandt-Andrews Sup-Wald test for an unknown structural break.
    """
    y = series.dropna().values
    dates = series.dropna().index
    n = len(y)
    Y = y[1:]
    X = sm.add_constant(y[:-1])
    
    start_idx = int(n * trim)
    end_idx = int(n * (1 - trim))
    
    max_f_stat = 0
    best_break_idx = None
    
    base_model = sm.OLS(Y, X).fit()
    ssr_pooled = base_model.ssr
    k = 2 
    
    for t in range(start_idx, end_idx):
        ssr1 = sm.OLS(Y[:t], X[:t]).fit().ssr
        ssr2 = sm.OLS(Y[t:], X[t:]).fit().ssr
        numerator = (ssr_pooled - (ssr1 + ssr2)) / k
        denominator = (ssr1 + ssr2) / (len(Y) - 2 * k)
        f_stat = numerator / denominator
        
        if f_stat > max_f_stat:
            max_f_stat = f_stat
            best_break_idx = t
            
    p_val = 1 - stats.f.cdf(max_f_stat, k, len(Y) - 2 * k)
    break_date = dates[best_break_idx + 1].strftime('%Y-%m-%d') if best_break_idx else None
    
    return max_f_stat, p_val, break_date

def run_diagnostics(csv_path):
    print("\n" + "="*105)
    print(" DESIRED OUTCOMES FOR VOLATILITY FACTORS (WHY WE USE HAR)")
    print("="*105)
    print(" 1. ADF Test (Stationarity) : Want p < 0.05. Rejects pure random walk; proves series mean-reverts.")
    print(" 2. Hurst (Memory)          : Want H > 0.50. Proves 'Long Memory' (persistence), justifying HAR lags.")
    print(" 3. GPH Test (Fractional)   : Want 0 < d < 1.0. Proves Fractional Integration. DO NOT DIFFERENCE!")
    print(" The Perron Effect: I       : If stationary but structural break, long-memory parameter (d) will be inflated and bias unit root tests toward failing.")
    print("="*105 + "\n")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # We will test all columns in the CSV
    test_columns = df.columns
    
    header = f"{'Factor':<13} | {'ADF p':<7} | {'ADF Eval':<13} | {'Hurst':<7} | {'Hurst Eval':<13} | {'GPH (d)':<7} | {'GPH Eval':<14} | {'QA Break'}"
    print(header)
    print("-" * len(header))
    
    for col in test_columns:
        # Restrict to training slice so we don't cheat by looking at 2025
        series = df[col].loc[:"2024-12-31"].dropna()
        if len(series) < 100:
            continue
            
        # 1. ADF Test
        adf_pvalue = adfuller(series)[1]
        if adf_pvalue < 0.05:
            adf_eval = "Stationary"
        else:
            adf_eval = "Unit Root?"
            
        # 2. Hurst Exponent
        H, _, _ = compute_Hc(series, kind='change', simplified=True)
        if H > 0.55:
            h_eval = "Persistent"
        elif H < 0.45:
            h_eval = "Anti-Persist"
        else:
            h_eval = "Random Walk"
            
        # 3. GPH Test
        d = gph_estimator(series)
        if d < 0.5:
            gph_eval = "Stat. Frac."      # Stationary Fractional
        elif d < 1.0:
            gph_eval = "NonStat Frac."    # Non-Stationary Fractional (Mean reverting)
        else:
            gph_eval = "Unit Root"
            
        # 4. Unknown Breakpoint
        _, qa_pval, break_date = quandt_andrews_unknown_break(series)
        break_display = break_date if qa_pval < 0.05 else "No Break"
        
        print(f"{col:<13} | {adf_pvalue:>7.3f} | {adf_eval:<13} | {H:>7.3f} | {h_eval:<13} | {d:>7.3f} | {gph_eval:<14} | {break_display}")

if __name__ == "__main__":
    csv_file = "results/factors/factors.csv"
    try:
        run_diagnostics(csv_file)
    except FileNotFoundError:
        print(f"Could not find {csv_file}. Run the extraction script first!")
