# %%
import pandas as pd
import numpy as np
import json
import pyarrow.parquet as pq
from scipy.stats import norm
import polars as pl
from pathlib import Path

# %%
CLEAN_FILE = Path("data/processed/options_surfaces_data_cleaned.parquet")


def delete_day(df, day):
    return df["quote_datetime" != day]

def uncomplete_days(symbols):

    full_range = pd.read_csv("data/days.csv")

    full_range = pd.to_datetime(full_range["days"])
    full_range = pd.DatetimeIndex(full_range)
    n = len(full_range)


    for symbol in symbols:

        path_data = f"data/ssvi_surfaces_output/{symbol}_data.parquet"
        
        df_existing = pd.read_parquet(path_data)
        dates_unique = pd.to_datetime(df_existing['quote_datetime'].unique())

        full_range = list(set(full_range) & set(dates_unique))

    print(f"Days not complete: {n-len(full_range)}")
    
    df_days =  pd.DataFrame({'days': full_range})
    df_days.to_csv("data/cleaned_days.csv")
    del df_days

    return full_range

def delete_not_all_present(days, symbols):

    for symbol in symbols:
        path_data = f"data/ssvi_surfaces_output/{symbol}_data.parquet"
        
        df_existing = pd.read_parquet(path_data)
        s_days = len(df_existing["quote_datetime"].unique())
        df_clean = df_existing[df_existing["quote_datetime"].isin(days)]
        f_days = len(df_clean["quote_datetime"].unique())

        df_clean.to_parquet(path_data)
        print(f"{symbol} CLEANED {s_days - f_days} days")

def clean_symbols():
    data_symbols = json.load(open('data/assets.json'))
    symbols = data_symbols['symbols']

    d = uncomplete_days(symbols)
    delete_not_all_present(d, symbols)


# %% Data cleaning functions

def clean_options_data(df, splits=None):
    df['quote_datetime'] = pd.to_datetime(df['quote_datetime'])
    df['expiration'] = pd.to_datetime(df['expiration'])
    df['days_to_expiry'] = (df['expiration'] - df['quote_datetime']).dt.days

    if splits:
        for split_date, f_factor in splits.items():
            mask_pre_split = df['quote_datetime'] < split_date
            price_cols = ['bid', 'ask', 'underlying_bid', 'underlying_ask', 'strike']
            for col in price_cols:
                if col in df.columns:
                    df.loc[mask_pre_split, col] = df.loc[mask_pre_split, col] / f_factor

    # Mid-price
    df['option_price'] = (df['bid'] + df['ask']) / 2
    df['underlying_price'] = (df['underlying_bid'] + df['underlying_ask']) / 2

    # Basic liquidity/sanity masks only
    mask_min_price = df['option_price'] >= 0.10
    mask_min_days = df['days_to_expiry'] >= 10
    mask_bid_ask = df['bid'] > 0
    mask_nul_iv = df['implied_volatility'] > 0

    return df[mask_min_price & mask_min_days & mask_bid_ask & mask_nul_iv].copy()


def filter_by_forward_logic(df):
    """Filters for OTM options and checks intrinsic value relative to the Forward."""
    intrinsic_f = np.where(
        df['option_type'] == 'C',
        np.maximum(df['forward_price'] - df['strike'], 0),
        np.maximum(df['strike'] - df['forward_price'], 0)
    )

    mask_intrinsic = df['option_price'] > intrinsic_f
    mask_otm = (
        ((df['option_type'] == 'C') & (df['strike'] >= df['forward_price'])) |
        ((df['option_type'] == 'P') & (df['strike'] <= df['forward_price']))
    )
    return df[mask_intrinsic & mask_otm].copy()


def estimate_forward_put_call_parity(df):
    """Estimate forward price F for one expiration using put-call parity at the ATM strike."""
    calls = df[df['option_type'] == 'C']
    puts  = df[df['option_type'] == 'P']
    common_strikes = np.intersect1d(calls['strike'].values, puts['strike'].values)

    if len(common_strikes) == 0:
        raise ValueError("No common strikes between calls and puts.")

    S = df['underlying_price'].iloc[0]
    K_atm = common_strikes[np.argmin(np.abs(common_strikes - S))]

    C = calls.loc[calls['strike'] == K_atm, 'option_price'].iloc[0]
    P = puts.loc[puts['strike'] == K_atm,  'option_price'].iloc[0]

    F = K_atm + C - P
    return F


def calculate_vega(S, K, tau, sigma, r):
    """Compute Black-Scholes vega for each option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    vega = S * np.sqrt(tau) * norm.pdf(d1)
    return vega


def load_symbol_data(symbol="AAPL", splits=None, pq_file=None, parquet_file="data/processed/options_surfaces_data_cleaned.parquet"):
    if pq_file is None:
        pq_file = pq.ParquetFile(parquet_file)

    df_list = []

    for i in range(pq_file.num_row_groups):
        df = pq_file.read_row_group(i).to_pandas()
        if symbol not in df['underlying_symbol'].values:
            continue

        df = df[df['underlying_symbol'] == symbol]

        # Step 1: Basic sanity
        df = clean_options_data(df, splits=splits)

        # Step 2: Estimate Forward per expiration
        forwards = {}
        for expiry, df_exp in df.groupby('expiration'):
            try:
                forwards[expiry] = estimate_forward_put_call_parity(df_exp)
            except:
                forwards[expiry] = None
        df['forward_price'] = df['expiration'].map(forwards)
        df = df.dropna(subset=['forward_price'])

        # Step 3: Filter OTM / intrinsic
        df = filter_by_forward_logic(df)

        # Step 4: Compute exact tau in years
        df['tau'] = (df['expiration'] - df['quote_datetime']).dt.total_seconds() / (365.25*24*3600)

        # Step 5: Compute vega and filter small-vega options
        S = df['underlying_price'].values
        K = df['strike'].values
        tau = df['tau'].values
        sigma = df['implied_volatility'].values
        r = np.zeros_like(S)  # discounting zero if not needed

        df['vega'] = calculate_vega(S, K, tau, sigma, r)
        df = df[df['vega'] > 0.01]  # small vega cutoff

        # Step 6: Log-moneyness and tau-dependent cut
        df['log_moneyness'] = np.log(df['strike'] / df['forward_price'])
        k_max = 3 * np.sqrt(df['tau'])
        df = df[np.abs(df['log_moneyness']) <= k_max]

        df_list.append(df)

    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()


def load_data_symbol_polaris(
    symbol: str,
    quote_datetime=None,
    parquet_file: Path = CLEAN_FILE
) -> pd.DataFrame:
    """
    Load already-cleaned option data for one symbol.
    
    Parameters
    ----------
    symbol : str
        Underlying symbol (e.g. "AAPL")
    quote_datetime : str | pd.Timestamp | None
        If provided, filter to a single quote datetime
    parquet_file : Path
        Path to cleaned parquet file
        
    Returns
    -------
    pd.DataFrame
    """

    df = (
        pl.scan_parquet(parquet_file)
        .filter(pl.col("underlying_symbol") == symbol)
    )

    if quote_datetime is not None:
        qd = pd.to_datetime(quote_datetime)
        df = df.filter(pl.col("quote_datetime") == qd)


    return df.collect().to_pandas().rename(columns={'underlying_mid_price': 'underlying_price',})



# %%
if __name__ == "__main__":
    clean_symbols()
# %%
