# %%
from src.ssvi.surface_generator import SurfaceGenerator
import numpy as np 
from src.ssvi.ssvi import SSVI

from src.ssvi.data_preparation import clean_symbols
from src.ssvi.diagnosis_functions import performance_fit, rmse_summary_table
import json

# %%
if __name__ == "__main__":
    t_grid = np.linspace(0.01, 2.0, 20)
    k_grid = np.linspace(-0.5, 0.5, 31)

    print("Starting SSVI fit algorithm")
    
    gen = SurfaceGenerator(
        symbols=["JPM"],
        t_grid=t_grid,
        k_grid=k_grid,
        pq_path="options_surfaces_data.parquet",
        save_path="data/ssvi_surfaces_output",
        class_SSVI=SSVI
    )
    
    evals, path = gen._generate_grid()
    print(f"Done! Data saved in Hive-partitioned format at: {path}")

    data = json.load(open('data/assets.json'))
    symbols = data['symbols']
    
    performance_fit(symbols)

    rmse_summary_table(symbols)

    clean_symbols()

    #interpolate_all_ssvi()
