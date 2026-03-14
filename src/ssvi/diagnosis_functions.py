# %% 
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.ssvi.data_preparation import load_data_symbol_polaris
from src.ssvi.ssvi import SSVI
import os
# %%


def performance_fit(symbols):
    eval_dict, data_dict = dict(), dict()
    all_rmse_data = []

    for symbol in symbols:
        path_eval = f"data/ssvi_surfaces_output/{symbol}_eval.parquet"
        path_data = f"data/ssvi_surfaces_output/{symbol}_data.parquet"

        dataset_eval = ds.dataset(path_eval, format="parquet")
        dataset_data = ds.dataset(path_data, format="parquet")

        df_eval = dataset_eval.to_table().to_pandas()
        df_data = dataset_data.to_table().to_pandas()

        eval_dict[symbol] = df_eval

        if 'avg_rmse' not in df_eval.columns:
            continue

        all_rmse_data.append(df_eval[['symbol', 'avg_rmse']])


        best_idx = df_eval['avg_rmse'].idxmin()
        worst_idx = df_eval['avg_rmse'].idxmax()

        median_rmse = df_eval['avg_rmse'].median()
        median_idx = (df_eval['avg_rmse'] - median_rmse).abs().idxmin()

        selected_days = {
            "best": df_eval.loc[best_idx, 'quote_datetime'],
            "median": df_eval.loc[median_idx, 'quote_datetime'],
            "worst": df_eval.loc[worst_idx, 'quote_datetime'],
        }

        print(f"\nSymbol {symbol}")
        for k, v in selected_days.items():
            print(f"  {k.capitalize()} RMSE day: {v}")


        df_real_data = load_data_symbol_polaris(
            symbol=symbol,
        )

        for label, qd in selected_days.items():
            ssvi = SSVI(
                df_real_data,
                symbol=symbol,
                quote_datetime=qd
            )
            if label == "best":
                folder = f"res/ssvi_fit/bf"
                filename = f"{symbol.lower()}_bf.png"
            elif label == "median":
                folder = f"res/ssvi_fit/avg"
                filename = f"{symbol.lower()}_avg.png"
            elif label == "worst":
                folder = f"res/ssvi_fit/wf"
                filename = f"{symbol.lower()}_wf.png"

            # Ensure the folder exists
            os.makedirs(folder, exist_ok=True)

            # Call fit with the correct save path
            ssvi.fit(
                plot_diagnostics=False,
                smooth_params=False,
                save=f"{folder}/{filename}"
            )
            
            ssvi.evaluate_fit()


    if all_rmse_data:
        combined_rmse = pd.concat(all_rmse_data, ignore_index=True)

        plt.figure(figsize=(15, 6))
        sns.boxplot(x='symbol', y='avg_rmse', data=combined_rmse)
        plt.xlabel('Symbol')
        plt.ylabel('Average RMSE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def rmse_summary_table(symbols):
    summary_rows = []

    for symbol in symbols:
        path_eval = f"data/ssvi_surfaces_output/{symbol}_eval.parquet"

        dataset_eval = ds.dataset(path_eval, format="parquet")
        df_eval = dataset_eval.to_table().to_pandas()

        if 'avg_rmse' not in df_eval.columns:
            continue

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

    summary_df = pd.DataFrame(summary_rows)

    # -------------------------------------------------
    # Pretty print
    # -------------------------------------------------
    print(
        summary_df
        .set_index("symbol")
        .round(4)
        .sort_index()
    )

    return summary_df

    
# %%
if __name__ == "__main__":
    import json
    data = json.load(open('data/assets.json'))
    symbols = data['symbols']
    
    performance_fit(symbols)
    rmse_summary_table(symbols)
    

# %%
