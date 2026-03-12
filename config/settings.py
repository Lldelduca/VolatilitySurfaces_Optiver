import json
import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow.dataset as ds

# 1. DIRECTORY PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
RES_DIR = BASE_DIR / "results"

# 2. ASSET UNIVERSE
with open(CONFIG_DIR / 'assets.json', 'r') as f:
    data_symbols = json.load(f)

SYMBOLS = data_symbols['symbols']
N_ASSETS = len(SYMBOLS)

# 3. SSVI GRID DIMENSIONS
N_MATURITY = 20  
N_MONEYNESS = 31 

T_GRID = np.linspace(0.01, 2.0, N_MATURITY)
K_GRID = np.linspace(-0.5, 0.5, N_MONEYNESS)

# 4. TIME SERIES & TRAIN/TEST SPLIT
_sample_path = RES_DIR / f"fitting/{SYMBOLS[0]}_data.parquet"
_dataset = ds.dataset(_sample_path, format="parquet")
_df_dates = _dataset.to_table(columns=['quote_datetime']).to_pandas()
DATES = pd.DatetimeIndex(_df_dates['quote_datetime'].unique()).sort_values().normalize()
N_OBS = len(DATES)

# Find the exact index where the year 2025 starts
JAN_2025 = len(DATES) - len(DATES[DATES >= "2025-01-01"])

# 5. TENSOR FANOVA RANKS (Explained Variance Cutoffs)
N_PC_GLOBAL = 3
# Mapped to: AAPL, AMZN, GLD, GOOGL, IWM, JPM, MSFT, NFLX, NVDA, QQQ, SPY, TLT, TSLA, V
N_PC_LOCAL = [3,    3,    3,   4,     3,   3,   3,    3,    3,    3,   3,   5,   3,    3]

assert len(N_PC_LOCAL) == N_ASSETS, "Mismatch between number of assets and local PC ranks!"

# 6. VINE TRUNCATION LEVELS
K_HAR_GARCH = 10
K_NSDE = 13