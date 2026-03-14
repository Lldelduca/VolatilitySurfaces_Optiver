# %%
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from src.get_data import load_data_symbol_polaris
import time
import os
import gc
import json

# %%
class SurfaceGenerator:
    def __init__(self,
                 class_SSVI,
                 t_grid,
                 k_grid,
                 pq_path,
                 save_path="data/ssvi_surfaces_output",
                 symbols=None,
                 splits_path="data/splits/splits.json",
                 symbols_path="data/assets.json"):
        
        self.symbols = symbols
        self.t_grid = t_grid
        self.k_grid = k_grid
        self.pq = pq_path
        self.class_SSVI = class_SSVI
        self.save_path = save_path

        if splits_path is not None:
            with open(splits_path, 'r') as f:
                self.splits = json.load(f)
        else:
            self.splits = None

        if symbols is None and symbols_path is not None:
            with open(symbols_path, 'r') as f:
                self.symbols = json.load(f)["symbols"]


    def _map_parquet_structure(self):
        print(f"Indexing {self.pq}...")
        pf = pq.ParquetFile(self.pq)
        symbol_map = defaultdict(list)
        for i in range(pf.num_row_groups):
            table = pf.read_row_group(i, columns=['underlying_symbol'])
            unique_syms = table['underlying_symbol'].unique()
            for sym in unique_syms:
                s = sym.as_py() if hasattr(sym, 'as_py') else sym
                symbol_map[s].append(i)
        print(f"Indexing complete. Found {len(symbol_map)} symbols.")
        print(f"Symbols: {list(symbol_map.keys())}")
        return dict(symbol_map)

    def _process_symbol_worker(self, df, t_flat, k_flat, symbol):
        """
        Calculates the full surface grid and evaluation metrics for a single symbol.
        """

        symbol_fits_list = [] # Canviem a llista per fer el DataFrame fàcilment
        symbol_slices = []
        valid_days = df['quote_datetime'].dt.strftime("%Y-%m-%d %H:%M:%S").sort_values().unique()
        n_days = len(valid_days)
        seconds = time.time()

        
        for idx, qd in enumerate(valid_days):
            try:
                

                ssvi = self.class_SSVI(df, symbol=symbol, quote_datetime=qd)
                ssvi.fit(smooth_params=False, plot_diagnostics=False)

                # --- EVAL ---

                symbol_fits_list.append(ssvi.evaluate_fit())

                # --- GRID ---
                total_var = ssvi.get_variance_grid(t_flat, k_flat)

                with np.errstate(divide='ignore', invalid='ignore'):
                    iv = np.where(t_flat > 0, np.sqrt(np.maximum(total_var, 0.0) / t_flat), 0.0)
                
                slice_df = pd.DataFrame({
                    'quote_datetime': pd.to_datetime(qd),
                    'time_to_expiry': t_flat,
                    'log_moneyness': k_flat,
                    'implied_volatility': iv,
                    'symbol': symbol
                })
                symbol_slices.append(slice_df)
                del ssvi

                if (idx + 1) % 50 == 0 or idx == 0:
                    time_elapsed = (time.time() - seconds)/60
                    time_missing = ((n_days -idx -1)/(idx+1)) * time_elapsed

                    print(f"[{symbol}] Fitting day {idx+1}/{n_days} ({qd}), "
                          f"time: {time_elapsed:.2f} min, left = {time_missing:.2f} min",
                          flush=True)
            
            except RuntimeWarning as w:
                print(
                    f"[{symbol}] RuntimeWarning on day {qd}: {w}",
                    flush=True
                )
                continue

            except Exception:
                print(f"[{symbol}] Fit failed for day {qd}.", flush=True)
                continue

        # --- SAVE AND CLEANUP ---

        os.makedirs(self.save_path, exist_ok=True)

        if symbol_slices:
            pd.concat(symbol_slices, ignore_index=True).to_parquet(
                os.path.join(self.save_path, f"{symbol}_data.parquet"), index=False
            )
        
        if symbol_fits_list:
            eval_df = pd.DataFrame(symbol_fits_list)
            eval_df['symbol'] = symbol
            eval_df.to_parquet(os.path.join(self.save_path, f"{symbol}_eval.parquet"), index=False)
            print(f"[{symbol}] Saved metrics to {self.save_path}/{symbol}_eval.parquet", flush=True)

        del df, symbol_slices, symbol_fits_list
        gc.collect()

        return symbol, "Success"
    
    def _generate_grid(self):
        # 1. Pre-calculate the FLATTENED Grid (Meshgrid)
        # This creates pairs for every T and every K
        t_mesh, k_mesh = np.meshgrid(self.t_grid, self.k_grid)
        t_flat = t_mesh.ravel()
        k_flat = k_mesh.ravel()

        # 3. Prepare Tasks
        tasks = []
        


        for symbol in self.symbols:
            tasks.append((
                    load_data_symbol_polaris(symbol=symbol),
                    t_flat,
                    k_flat,
                    symbol
                ))

        print(f"Prepared {len(tasks)} tasks for multiprocessing.")

        # 4. Use maxtasksperchild to prevent memory leaks
        with Pool(processes=cpu_count(), maxtasksperchild=1) as pool:
            results = pool.starmap(self._process_symbol_worker, tasks)
        
        return dict(results), self.save_path
# %%
