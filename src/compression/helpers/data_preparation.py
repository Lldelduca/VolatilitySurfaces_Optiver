import pandas as pd
import pyarrow.dataset as ds    
import config.settings as g
import numpy as np

def get_clean_4d_tensor():
    """
    Loads SSVI data and returns a clean 4D numpy array: 
    (Time, Assets, Maturity, Moneyness) containing Log-Implied Volatility.
    """
    all_dfs = []
    for symbol in g.SYMBOLS:
        path_eval = f"results/fitting/{symbol}_data.parquet"
        dataset_eval = ds.dataset(path_eval, format="parquet")
        df_temp = dataset_eval.to_table(columns=[
            'quote_datetime', 'time_to_expiry', 'log_moneyness', 'implied_volatility', 'symbol'
        ]).to_pandas()
        all_dfs.append(df_temp)
    
    full_df = pd.concat(all_dfs)

    pivot_df = full_df.pivot_table(
        index='quote_datetime', 
        columns=['symbol', 'time_to_expiry', 'log_moneyness'], 
        values='implied_volatility'
    )
    
    target_columns = pd.MultiIndex.from_product(
        [g.SYMBOLS, g.T_GRID, g.K_GRID], 
        names=['symbol', 'time_to_expiry', 'log_moneyness']
    )
    
    pivot_df = pivot_df.reindex(columns=target_columns).ffill().bfill() 
    
    # Reshape to 4D Tensor: (Days, Assets, Maturity, Moneyness)
    tensor_data = pivot_df.values.reshape(g.N_OBS, g.N_ASSETS, g.N_MATURITY, g.N_MONEYNESS)
    
    # Return Log-IV to stabilize variance
    return np.log(tensor_data)