import os
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from src.fitting.ssvi import SSVI
from src.fitting.data_preparation import SSVIDataProcessor

def plot_smile_fit(ssvi_model, save_path=None):
    """Replicates Melvin's _run_diagnostics for a fitted SSVI object."""
    TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 24, 16, 14
    LINE_WIDTH, MARKER_SIZE = 2.5, 50

    Ts = ssvi_model.res["maturities"]
    selected_Ts = Ts[np.linspace(0, len(Ts) - 1, min(len(Ts), 3), dtype=int)]
    
    fig, axes = plt.subplots(len(selected_Ts), 1, figsize=(10, 8), constrained_layout=True)
    if len(selected_Ts) == 1: axes = [axes]

    for idx, T in enumerate(selected_Ts):
        ax = axes[idx]
        mask = ssvi_model.df['tau'] == T
        k_mkt = ssvi_model.df.loc[mask, 'log_moneyness'].values
        iv_mkt = ssvi_model.df.loc[mask, 'implied_volatility'].values
        
        if len(k_mkt) > 0:
            k_grid = np.linspace(k_mkt.min() - 0.1, k_mkt.max() + 0.1, 100)
            
            # Use the exposed total_variance method
            w_model = np.array([ssvi_model.total_variance(T, ki) for ki in k_grid])
            iv_model = np.sqrt(np.maximum(w_model, 0.0) / T)

            ax.scatter(k_mkt, iv_mkt, s=MARKER_SIZE, alpha=0.7, color='black', marker='x', label='Market')
            ax.plot(k_grid, iv_model, color='red', lw=LINE_WIDTH, label='SSVI Fit')

        ax.set_title(f"T = {T:.3f}", fontsize=TITLE_SIZE)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.set_ylabel('Implied Vol ($\\sigma_{BS}$)', fontsize=LABEL_SIZE)
        if idx == len(selected_Ts) - 1:
            ax.set_xlabel('Log-Moneyness ($k$)', fontsize=LABEL_SIZE)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    else:
        plt.show()
    plt.close(fig)

def performance_fit(symbols, raw_pq_path="data/processed/options_surfaces_data_cleaned.parquet"):
    """Finds Best, Median, and Worst fits per asset and plots them."""
    all_rmse_data = []
    processor = SSVIDataProcessor(raw_pq_path)

    for symbol in symbols:
        path_eval = f"data/ssvi_surfaces_output/{symbol}_eval.parquet"
        if not os.path.exists(path_eval): continue

        df_eval = pd.read_parquet(path_eval)
        if 'avg_rmse' not in df_eval.columns: continue

        all_rmse_data.append(df_eval[['symbol', 'avg_rmse']])

        best_idx = df_eval['avg_rmse'].idxmin()
        worst_idx = df_eval['avg_rmse'].idxmax()
        median_rmse = df_eval['avg_rmse'].median()
        median_idx = (df_eval['avg_rmse'] - median_rmse).abs().idxmin()

        selected_days = {
            "bf": df_eval.loc[best_idx, 'quote_datetime'],
            "avg": df_eval.loc[median_idx, 'quote_datetime'],
            "wf": df_eval.loc[worst_idx, 'quote_datetime'],
        }

        print(f"\nSymbol {symbol}")
        df_real_data = processor.clean_symbol_data(symbol)

        for label, qd in selected_days.items():
            print(f"  {label.upper()} RMSE day: {qd}")
            folder = f"res/ssvi_fit/{label}"
            os.makedirs(folder, exist_ok=True)
            
            try:
                ssvi = SSVI(df_real_data, symbol=symbol, quote_datetime=qd)
                ssvi.fit()
                plot_smile_fit(ssvi, save_path=f"{folder}/{symbol.lower()}_{label}.png")
            except Exception as e:
                print(f"  Plotting failed for {symbol} on {qd}: {e}")

    # Generate the aggregate Boxplot
    if all_rmse_data:
        combined_rmse = pd.concat(all_rmse_data, ignore_index=True)
        plt.figure(figsize=(15, 6))
        sns.boxplot(x='symbol', y='avg_rmse', data=combined_rmse)
        plt.xlabel('Symbol')
        plt.ylabel('Average RMSE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs("res/ssvi_fit", exist_ok=True)
        plt.savefig("res/ssvi_fit/aggregate_rmse_boxplot.png", dpi=300)
        plt.close()

def rmse_summary_table(symbols):
    """Generates the LaTeX-ready summary statistics for the SSVI RMSE."""
    summary_rows = []

    for symbol in symbols:
        path_eval = f"data/ssvi_surfaces_output/{symbol}_eval.parquet"
        if not os.path.exists(path_eval): continue

        df_eval = pd.read_parquet(path_eval)
        if 'avg_rmse' not in df_eval.columns: continue

        rmse = df_eval['avg_rmse'].dropna()
        summary_rows.append({
            "symbol": symbol,
            "rmse_min": rmse.min(),
            "rmse_q05": rmse.quantile(0.05),
            "rmse_median": rmse.median(),
            "rmse_q95": rmse.quantile(0.95),
            "rmse_max": rmse.max(),
            "rmse_mean": rmse.mean(),
        })

    summary_df = pd.DataFrame(summary_rows).set_index("symbol").round(4).sort_index()
    print("\n--- SSVI RMSE SUMMARY ---")
    print(summary_df)
    
    os.makedirs("res/ssvi_fit", exist_ok=True)
    summary_df.to_csv("res/ssvi_fit/rmse_summary.csv")
    return summary_df