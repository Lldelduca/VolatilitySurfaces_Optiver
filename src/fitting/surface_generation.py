import os
import gc
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from src.fitting.data_preparation import SSVIDataProcessor

class SurfaceGenerator:
    """
    Mass-calibrates SSVI surfaces across the entire option panel using multiprocessing,
    and extracts uniform variance grids for downstream PCA compression.
    """
    def __init__(self, ssvi_class, t_grid, k_grid, raw_pq_path, save_path, symbols):
        self.SSVI = ssvi_class
        self.t_grid = t_grid
        self.k_grid = k_grid
        self.save_path = save_path
        self.symbols = symbols
        self.processor = SSVIDataProcessor(raw_pq_path)

    def _process_symbol_worker(self, symbol):
        """Worker function executed independently on each CPU core."""
        start_time = time.time()
        os.makedirs(self.save_path, exist_ok=True)

        # 1. Clean Data for Symbol
        try:
            df = self.processor.clean_symbol_data(symbol)
            if df.empty: return symbol, "Empty Data"
        except Exception as e:
            return symbol, f"Data Prep Failed: {e}"

        valid_days = np.sort(df['quote_datetime'].dt.strftime("%Y-%m-%d %H:%M:%S").unique())
        n_days = len(valid_days)
        
        # Prepare flattening grids for mfPCA extraction
        t_mesh, k_mesh = np.meshgrid(self.t_grid, self.k_grid)
        t_flat, k_flat = t_mesh.ravel(), k_mesh.ravel()

        symbol_fits_list = []
        symbol_slices = []

        # 2. Iterate and Fit Daily Surfaces
        for idx, qd in enumerate(valid_days):
            try:
                # Instantiate and Fit (uses the class we built in the previous turn)
                ssvi = self.SSVI(df, symbol=symbol, quote_datetime=qd)
                ssvi.fit()

                # Evaluate RMSE and Stability
                fit_metrics = ssvi.evaluate_fit()
                symbol_fits_list.append(fit_metrics)

                # Extract Uniform Grid for PCA
                total_var = ssvi.get_variance_grid(t_flat, k_flat)
                with np.errstate(divide='ignore', invalid='ignore'):
                    iv = np.where(t_flat > 0, np.sqrt(np.maximum(total_var, 0.0) / t_flat), 0.0)
                
                symbol_slices.append(pd.DataFrame({
                    'quote_datetime': pd.to_datetime(qd),
                    'time_to_expiry': t_flat,
                    'log_moneyness': k_flat,
                    'implied_volatility': iv,
                    'symbol': symbol
                }))

                # Memory management for large loops
                del ssvi
                
                # Logging
                if (idx + 1) % 50 == 0:
                    elapsed = (time.time() - start_time) / 60
                    remaining = ((n_days - idx - 1) / (idx + 1)) * elapsed
                    print(f"[{symbol}] Day {idx+1}/{n_days} | Elapsed: {elapsed:.1f}m | ETA: {remaining:.1f}m", flush=True)

            except Exception as e:
                continue # Skip un-fittable days gracefully

        # 3. Export Data
        if symbol_slices:
            pd.concat(symbol_slices, ignore_index=True).to_parquet(
                os.path.join(self.save_path, f"{symbol}_data.parquet"), index=False
            )
        if symbol_fits_list:
            eval_df = pd.DataFrame(symbol_fits_list)
            eval_df['symbol'] = symbol
            eval_df.to_parquet(os.path.join(self.save_path, f"{symbol}_eval.parquet"), index=False)

        del df, symbol_slices, symbol_fits_list
        gc.collect()
        
        return symbol, "Success"

    def generate_grid(self):
        """Deploys the workload across all available CPU cores."""
        print(f"[*] Deploying SSVI Calibration across {cpu_count()} cores for {len(self.symbols)} symbols...")
        
        # Spin up multiprocessing pool. 
        # maxtasksperchild=1 strictly prevents memory leaks in SciPy optimization loops
        with Pool(processes=cpu_count(), maxtasksperchild=1) as pool:
            results = pool.map(self._process_symbol_worker, self.symbols)
            
        print("\n[*] SSVI Generation Complete. Summary:")
        for res in results:
            print(f"    - {res[0]}: {res[1]}")
            
        return dict(results)