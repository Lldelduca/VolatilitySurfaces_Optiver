import numpy as np
import pandas as pd
from scipy.stats import norm
import pyarrow.parquet as pq

class SSVIDataProcessor:
    """
    Handles the ingestion and econometric filtering of raw options data 
    prior to SSVI surface calibration.
    """
    def __init__(self, raw_parquet_path):
        self.raw_path = raw_parquet_path

    @staticmethod
    def estimate_forward_put_call_parity(df_exp):
        """Estimates forward price F via Put-Call parity at the ATM strike."""
        calls = df_exp[df_exp['option_type'] == 'C']
        puts  = df_exp[df_exp['option_type'] == 'P']
        
        common_strikes = np.intersect1d(calls['strike'].values, puts['strike'].values)
        if len(common_strikes) == 0:
            return None

        S = df_exp['underlying_price'].iloc[0]
        K_atm = common_strikes[np.argmin(np.abs(common_strikes - S))]

        C = calls.loc[calls['strike'] == K_atm, 'option_price'].iloc[0]
        P = puts.loc[puts['strike'] == K_atm,  'option_price'].iloc[0]

        return K_atm + C - P

    @staticmethod
    def calculate_vega(S, K, tau, sigma, r):
        """Computes Black-Scholes vega for liquidity filtering."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        return S * np.sqrt(tau) * norm.pdf(d1)

    def clean_symbol_data(self, symbol, splits=None):
        """Executes the full econometric filtering pipeline for a single asset."""
        # 1. Load Data (Using PyArrow directly into Pandas for speed)
        dataset = pq.ParquetDataset(self.raw_path)
        table = dataset.to_table(filter=[('underlying_symbol', '=', symbol)])
        df = table.to_pandas()
        
        if df.empty:
            return df

        # 2. Standardize Dates and Prices
        df['quote_datetime'] = pd.to_datetime(df['quote_datetime'])
        df['expiration'] = pd.to_datetime(df['expiration'])
        df['tau'] = (df['expiration'] - df['quote_datetime']).dt.total_seconds() / (365.25 * 24 * 3600)
        
        df['option_price'] = (df['bid'] + df['ask']) / 2.0
        df['underlying_price'] = (df['underlying_bid'] + df['underlying_ask']) / 2.0

        # Adjust for splits if provided
        if splits:
            for split_date, f_factor in splits.items():
                mask = df['quote_datetime'] < pd.to_datetime(split_date)
                for col in ['bid', 'ask', 'underlying_bid', 'underlying_ask', 'strike', 'underlying_price', 'option_price']:
                    if col in df.columns:
                        df.loc[mask, col] /= f_factor

        # 3. Basic Liquidity & Sanity Masks
        mask_valid = (
            (df['option_price'] >= 0.10) & 
            (df['tau'] >= 10 / 365.25) & 
            (df['bid'] > 0) & 
            (df['implied_volatility'] > 0)
        )
        df = df[mask_valid].copy()

        # 4. Estimate Forward Price per Expiration
        forwards = {}
        for expiry, df_exp in df.groupby('expiration'):
            F = self.estimate_forward_put_call_parity(df_exp)
            if F is not None:
                forwards[expiry] = F
                
        df['forward_price'] = df['expiration'].map(forwards)
        df = df.dropna(subset=['forward_price'])

        # 5. Out-of-the-Money (OTM) & Intrinsic Filtering
        intrinsic = np.where(
            df['option_type'] == 'C',
            np.maximum(df['forward_price'] - df['strike'], 0),
            np.maximum(df['strike'] - df['forward_price'], 0)
        )
        mask_intrinsic = df['option_price'] > intrinsic
        mask_otm = (
            ((df['option_type'] == 'C') & (df['strike'] >= df['forward_price'])) |
            ((df['option_type'] == 'P') & (df['strike'] <= df['forward_price']))
        )
        df = df[mask_intrinsic & mask_otm].copy()

        # 6. Vega and Log-Moneyness Filtering
        r = np.zeros_like(df['underlying_price'].values) # Risk-free rate assumption from original code
        df['vega'] = self.calculate_vega(
            df['underlying_price'].values, df['strike'].values, 
            df['tau'].values, df['implied_volatility'].values, r
        )
        
        df['log_moneyness'] = np.log(df['strike'] / df['forward_price'])
        k_max = 3 * np.sqrt(df['tau'])
        
        # Final strict econometric boundaries
        df = df[(df['vega'] > 0.01) & (np.abs(df['log_moneyness']) <= k_max)].copy()

        return df.reset_index(drop=True)
    