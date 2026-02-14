import polars as pl
import numpy as np
import json
from scipy.stats import norm
from pathlib import Path
import datetime

SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_DIR = SCRIPT_DIR.parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "options_surfaces_data.parquet"
SPLITS_FILE = BASE_DIR / "data" / "splits" / "splits.json"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "options_surfaces_data_cleaned.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATES_TO_DROP = [
    datetime.datetime(2022, 5, 20, 12, 0, 0),
    datetime.datetime(2022, 6, 17, 12, 0, 0)
]

with open(SPLITS_FILE, 'r') as f:
    splits_data = json.load(f)

def calculate_vega(s: pl.Series) -> pl.Series:
    data = s.struct.unnest()
    S, K = data["underlying_mid_price"].to_numpy(), data["strike"].to_numpy()
    tau, sigma = data["tau"].to_numpy(), data["implied_volatility"].to_numpy()
    r = data["rate"].to_numpy()
    
    valid_mask = (tau > 0) & (sigma > 0) & (S > 0) & (K > 0)
    d1 = np.zeros_like(S, dtype=np.float64)
    vega = np.zeros_like(S, dtype=np.float64)
    
    d1[valid_mask] = (np.log(S[valid_mask] / K[valid_mask]) + 
                      (r[valid_mask] + 0.5 * sigma[valid_mask]**2) * tau[valid_mask]) / \
                      (sigma[valid_mask] * np.sqrt(tau[valid_mask]))
    
    vega[valid_mask] = S[valid_mask] * np.sqrt(tau[valid_mask]) * norm.pdf(d1[valid_mask]) * 0.01
    return pl.Series(vega)

MIN_TAU = 10 / 365.25

split_adjustment = pl.lit(1.0)
for symbol, dates in splits_data.items():
    for date_str, factor in dates.items():
        split_adjustment = pl.when(
            (pl.col("underlying_symbol") == symbol) & 
            (pl.col("quote_datetime") < pl.lit(date_str).str.to_datetime())
        ).then(split_adjustment * factor).otherwise(split_adjustment)

df = (
    pl.scan_parquet(INPUT_FILE)
    .filter(~pl.col("quote_datetime").is_in(DATES_TO_DROP))
    .with_columns([pl.col(pl.Float64).cast(pl.Float32)])
    .with_columns(split_adjustment.alias("_split_factor"))
    .with_columns([
        pl.col(c) / pl.col("_split_factor") 
        for c in ["bid", "ask", "underlying_bid", "underlying_ask", "strike"]
    ])
    .filter(
        (pl.col("implied_volatility") > 0) & 
        (pl.col("bid") > 0) &
        (pl.col("tau") >= MIN_TAU)
    )
    .drop(["trade_volume", "open_interest", "bid_size", "ask_size"])
    .with_columns([
        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid_price"),
        ((pl.col("underlying_bid") + pl.col("underlying_ask")) / 2).alias("underlying_mid_price"),
        (pl.col("ask") - pl.col("bid")).alias("spread")
    ])
    .filter(pl.col("mid_price") >= 0.10)
    
    .with_columns([(pl.col("strike") - pl.col("underlying_mid_price")).abs().alias("_atm_dist")])
    .with_columns([(pl.col("_atm_dist") == pl.col("_atm_dist").min().over(["underlying_symbol", "quote_datetime", "expiration"])).alias("_is_atm")])
    .with_columns([
        pl.when(pl.col("_is_atm") & (pl.col("option_type") == "C")).then(pl.col("mid_price")).otherwise(None).alias("_c_atm"),
        pl.when(pl.col("_is_atm") & (pl.col("option_type") == "P")).then(pl.col("mid_price")).otherwise(None).alias("_p_atm"),
        pl.when(pl.col("_is_atm")).then(pl.col("strike")).otherwise(None).alias("_k_atm")
    ])
    .with_columns([
        pl.col("_c_atm").max().over(["underlying_symbol", "quote_datetime", "expiration"]).alias("_c_atm_full"),
        pl.col("_p_atm").max().over(["underlying_symbol", "quote_datetime", "expiration"]).alias("_p_atm_full"),
        pl.col("_k_atm").max().over(["underlying_symbol", "quote_datetime", "expiration"]).alias("_k_atm_full")
    ])
    .with_columns([
        (pl.col("_k_atm_full") + (pl.col("_c_atm_full") - pl.col("_p_atm_full")) * (pl.col("rate") * pl.col("tau")).exp())
        .alias("forward_price")
    ])
    .drop(["_atm_dist", "_is_atm", "_c_atm", "_p_atm", "_k_atm", "_c_atm_full", "_p_atm_full", "_k_atm_full"])

    .with_columns([
        pl.when(pl.col("option_type") == "C")
        .then((pl.col("forward_price") - pl.col("strike")).clip(lower_bound=0))
        .otherwise((pl.col("strike") - pl.col("forward_price")).clip(lower_bound=0))
        .alias("intrinsic_f")
    ])
    .filter(pl.col("mid_price") > pl.col("intrinsic_f"))
    .drop("intrinsic_f")

    .with_columns([
        (pl.col("strike") / pl.col("forward_price")).log().alias("log_forward_moneyness"),
        (pl.col("strike") / pl.col("underlying_mid_price")).log().alias("log_moneyness"),
        pl.when(pl.col("delta").abs().is_between(0.45, 0.55)).then(pl.lit("ATM"))
        .when(
            ((pl.col("option_type") == "C") & (pl.col("delta") > 0.55)) |
            ((pl.col("option_type") == "P") & (pl.col("delta").abs() > 0.55))
        ).then(pl.lit("ITM"))
        .otherwise(pl.lit("OTM"))
        .alias("XTM")
    ])
    
    .filter(
        ((pl.col("option_type") == "C") & (pl.col("strike") >= pl.col("forward_price"))) |
        ((pl.col("option_type") == "P") & (pl.col("strike") <= pl.col("forward_price")))
    )
    .filter(pl.col("log_forward_moneyness").abs() <= 3 * pl.col("tau").sqrt())
    
    .with_columns([
        pl.struct(["underlying_mid_price", "strike", "tau", "implied_volatility", "rate"])
        .map_batches(calculate_vega, return_dtype=pl.Float64)
        .alias("vega")
    ])
    .drop("_split_factor")
    .collect(engine="streaming") 
)

counts_dict = df["XTM"].value_counts().to_dicts()
print("Distribution of moneyness in the cleaned dataset:")
for entry in counts_dict:
    print(f"{entry['XTM']}: {entry['count']}")

print("\nSample Data:")
print(df.head().to_dicts())

print(f"\nFinal shape: {df.shape}")
df.write_parquet(OUTPUT_FILE, compression="zstd")
print("Saved cleaned data to:", OUTPUT_FILE)