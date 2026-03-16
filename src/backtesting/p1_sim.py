import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import sys
import pyvinecopulib as pv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

_real_torch_load = torch.load
def _cpu_safe_load(*args, **kwargs):
    kwargs.setdefault('map_location', 'cpu')
    return _real_torch_load(*args, **kwargs)
torch.load = _cpu_safe_load

from src.dynamics.HAR_GARCH import HAR_GARCH_EVT
from src.dynamics.NGARCH_T import NGARCH_T
from src.backtesting.portfolio1 import Portfolio1_RiskReversal
from src.backtesting.utils.risk_metrics import compute_risk_metrics
from src.backtesting.utils.generators import UniversalScenarioGenerator, DynamicNeuralVine
from src.backtesting.p1_backtest import RiskModelUnpickler, precompute_garch_states

def main():
    torch.set_num_threads(18) 
    print("=" * 80)
    print(" PORTFOLIO 1 -- MULTI-DAY VANNA SIMULATION (M2 vs Baselines)")
    print("=" * 80)

    DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "processed")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    ALPHA       = 0.05
    TRAIN_END   = pd.Timestamp("2024-12-31")
    OOS_START   = pd.Timestamp("2025-01-02")
    OOS_END     = pd.Timestamp("2025-12-29")
    HORIZON     = 20
    N_SCENARIOS = 10000

    # 1. Load Data & Marginals (Using HAR-GARCH for M2 and Baselines)
    df_factors = pd.read_csv(os.path.join(RESULTS_DIR, "factors", "factors.csv"), index_col=0, parse_dates=True)
    df_returns = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"), index_col=0, parse_dates=True)
    
    with open(os.path.join(RESULTS_DIR, "dynamics", "NGARCH", "fitted_marginals.pkl"), "rb") as f:
        marginals = RiskModelUnpickler(f).load()
    with open(os.path.join(RESULTS_DIR, "dynamics", "HAR_GARCH", "fitted_marginals.pkl"), "rb") as f:
        marginals.update(RiskModelUnpickler(f).load())
    
    with open(os.path.join(RESULTS_DIR, "factors", "surfaces_dict.pkl"), "rb") as f:
        surfaces_dict = pickle.load(f)
    
    _obj = next(iter(surfaces_dict.values()))
    mat_arr = np.array(_obj.maturity_labels if hasattr(_obj, 'maturity_labels') else getattr(_obj, '_Reconstruction__mat_labels'))
    mon_arr = np.array(_obj.moneyness_labels if hasattr(_obj, 'moneyness_labels') else getattr(_obj, '_Reconstruction__mon_labels'))

    valid_names = [c for c in df_returns.columns if c in marginals] + [c for c in df_factors.columns if c in marginals]
    name_to_col = {n: i for i, n in enumerate(valid_names)}
    factor_idx_map = {'G_PC': [name_to_col[n] for n in valid_names if n.startswith("G_PC_")]}
    for sym in set([s.split('_')[2] for s in valid_names if s.startswith("L_PC_")]):
        factor_idx_map[sym] = [name_to_col[n] for n in valid_names if n.startswith(f"L_PC_{sym}_")]

    oos_dates = df_returns[(df_returns.index >= TRAIN_END) & (df_returns.index <= OOS_END)].index
    all_states = precompute_garch_states(valid_names, marginals, df_returns, df_factors, oos_dates, TRAIN_END)

    # Load Uniforms for Neural Warmup
    df_unif_spot = pd.read_csv(os.path.join(RESULTS_DIR, "dynamics", "NGARCH", "uniforms_ngarch_train.csv"), index_col=0, parse_dates=True)
    df_har_uniforms = pd.read_csv(os.path.join(RESULTS_DIR, "dynamics", "HAR_GARCH", "uniforms_har_garch_evt_train.csv"), index_col=0, parse_dates=True)
    df_unif_merged = pd.concat([df_unif_spot, df_har_uniforms], axis=1).reindex(columns=valid_names).ffill().bfill()
    
    cols = ['underlying_symbol', 'quote_datetime', 'underlying_mid_price', 'tau', 'strike', 'option_type', 'delta', 'rate', 'log_moneyness', 'implied_volatility']
    df_parquet = pd.read_parquet(os.path.join(DATA_DIR, "options_surfaces_data_cleaned.parquet"), columns=cols)
    parquet_by_date = {d: grp.reset_index(drop=True) for d, grp in df_parquet.groupby('quote_datetime')}

    # 2. Setup Copulas
    m0_path = os.path.join(RESULTS_DIR, "copulas", "static", "joint_vine_spot_har_garch_evt_model.json")
    gauss_path = os.path.join(RESULTS_DIR, "copulas", "static", "gaussian_vine_spot_har_garch_evt_model.json")
    m2_path = os.path.join(RESULTS_DIR, "copulas", "neural", "neural_vine_spot_har_garch_evt_model.pth")
    
    m0_copula = pv.Vinecop.from_file(m0_path)
    gauss_copula = pv.Vinecop.from_file(gauss_path)
    
    gen_base = UniversalScenarioGenerator(valid_names, m0_copula, "M0")
    gen_base.classify_marginals(marginals)

    rolling_dates = pd.Series(oos_dates).groupby([oos_dates.year, oos_dates.month]).first().values
    rolling_dates = [pd.Timestamp(d) for d in rolling_dates if d >= OOS_START]

    results = []
    print(f"\n[!] Initiating Monthly Simulation: {len(rolling_dates)} steps | {HORIZON}-Day Horizon")

    for current_date in rolling_dates:
        print(f"\n[OOS SIMULATION] {current_date.strftime('%Y-%m-%d')}")

        available_dates = sorted(parquet_by_date.keys())
        t0 = next((d for d in reversed(available_dates) if d <= current_date), available_dates[-1])
        
        portfolio = Portfolio1_RiskReversal(parquet_by_date[t0], t0, n_contracts=1000, target_dte=30)
        if not portfolio.rrs: continue

        day_idx = oos_dates.get_loc(current_date)
        init_states = {n: all_states[n][day_idx] for n in valid_names}

        # Initialize and Warmup Neural Vine up to current date
        hist_window = df_unif_merged.loc[:current_date].iloc[-60:].values 
        neural_vine = DynamicNeuralVine(m2_path, m0_path, hist_window)
        gen_m2 = UniversalScenarioGenerator(valid_names, neural_vine, "M2")
        gen_m2.classify_marginals(marginals)

        print(f"  [-] Projecting paths...")
        paths_m2 = gen_m2.simulate_multiday(N_SCENARIOS, HORIZON, init_states, marginals, use_copula=True)
        paths_m0 = gen_base.simulate_multiday(N_SCENARIOS, HORIZON, init_states, marginals, use_copula=True)
        
        gen_base.copula = gauss_copula
        paths_gauss = gen_base.simulate_multiday(N_SCENARIOS, HORIZON, init_states, marginals, use_copula=True)
        paths_indep = gen_base.simulate_multiday(N_SCENARIOS, HORIZON, init_states, marginals, use_copula=False)

        print(f"  [-] Evaluating Daily Delta-Hedged PnL...")
        res_m2 = portfolio.evaluate_multiday_hedged_pnl(paths_m2, valid_names, surfaces_dict, name_to_col, mat_arr, mon_arr, HORIZON, factor_idx_map)
        res_m0 = portfolio.evaluate_multiday_hedged_pnl(paths_m0, valid_names, surfaces_dict, name_to_col, mat_arr, mon_arr, HORIZON, factor_idx_map)
        res_gauss = portfolio.evaluate_multiday_hedged_pnl(paths_gauss, valid_names, surfaces_dict, name_to_col, mat_arr, mon_arr, HORIZON, factor_idx_map)
        res_indep = portfolio.evaluate_multiday_hedged_pnl(paths_indep, valid_names, surfaces_dict, name_to_col, mat_arr, mon_arr, HORIZON, factor_idx_map)

        _, es_m2 = compute_risk_metrics(res_m2['he_total'], ALPHA)
        _, es_m0 = compute_risk_metrics(res_m0['he_total'], ALPHA)
        _, es_gauss = compute_risk_metrics(res_gauss['he_total'], ALPHA)
        _, es_indep = compute_risk_metrics(res_indep['he_total'], ALPHA)

        print(f"  [RESULT] M2 ES: ${es_m2:,.0f} | M0 ES: ${es_m0:,.0f} | Gauss ES: ${es_gauss:,.0f} | Indep ES: ${es_indep:,.0f}")
        print(f"  [PREMIUM] Delta-Vanna (vs Indep): ${es_m2 - es_indep:,.0f}")

        results.append({
            'Date': current_date,
            'M2_ES': es_m2, 'M0_ES': es_m0, 'Gauss_ES': es_gauss, 'Indep_ES': es_indep,
            'DeltaES_Vanna_Indep': es_m2 - es_indep,
            'DeltaES_Vanna_Gauss': es_m2 - es_gauss,
            'DeltaES_Vanna_M0': es_m2 - es_m0
        })

    df_res = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "backtests", "sim_p1_vanna_premium.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\n[+] Simulation complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()