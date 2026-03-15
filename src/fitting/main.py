import numpy as np
import json
import os
from src.fitting.ssvi import SSVI
from src.fitting.surface_generation import SurfaceGenerator
from src.fitting.diagnosis_functions import performance_fit, rmse_summary_table

def main():
    print("="*60)
    print(" STARTING SSVI CALIBRATION & GRID GENERATION PIPELINE ")
    print("="*60)
    
    # 1. Define standard grid for beta-adjusted mfPCA downstream
    t_grid = np.linspace(0.01, 2.0, 20)
    k_grid = np.linspace(-0.5, 0.5, 31)

    # 2. Load universe from central assets file
    with open('data/assets.json', 'r') as f:
        symbols = json.load(f)["symbols"]

    raw_data_path = "data/processed/options_surfaces_data_cleaned.parquet"
    save_directory = "data/ssvi_surfaces_output"

    # 3. Initialize the multiprocessing generator
    gen = SurfaceGenerator(
        ssvi_class=SSVI,
        t_grid=t_grid,
        k_grid=k_grid,
        raw_pq_path=raw_data_path,
        save_path=save_directory,
        symbols=symbols
    )
    
    # 4. Generate all daily surfaces and extract uniform grids
    print("\n[STEP 1] Calibrating Surfaces and Extracting Grids...")
    results_dict = gen.generate_grid()
    print(f"\n[+] Data saved in Parquet format at: {save_directory}")

    # 5. Run Diagnostics (Plots and Tables)
    print("\n[STEP 2] Running Diagnostics & Generating Plots...")
    performance_fit(symbols, raw_pq_path=raw_data_path)

    print("\n[STEP 3] Generating RMSE Summary Table...")
    rmse_summary_table(symbols)

    print("\n" + "="*60)
    print(" PIPELINE COMPLETE ")
    print("="*60)

if __name__ == "__main__":
    main()