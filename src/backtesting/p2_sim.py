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
from src.backtesting.portfolio2 import Portfolio2_Dispersion
from src.backtesting.utils.risk_metrics import compute_risk_metrics
from src.backtesting.utils.generators import UniversalScenarioGenerator, DynamicGASVine
from src.backtesting.p1_backtest import RiskModelUnpickler, precompute_garch_states

def main():
    torch.set_num_threads(18) 
    print("=" * 80)
    print(" PORTFOLIO 2 -- MULTI-DAY CORRELATION SIMULATION (M1 vs Baselines)")
    print("=" * 80)

    DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "processed")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    ALPHA       = 0.05
    TRAIN_END   = pd.Timestamp("2024-12-31")
    OOS_START   = pd.Timestamp("2025-01-02")
    OOS_END     = pd.Timestamp("2025-12-29")
    HORIZON     = 20
    N_SCENARIOS = 10000

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

    # Fetch Realized Uniforms for GAS Roll-Forward
    real_paths_matrix = np.zeros((len(oos_dates), len(valid_names)))
    for i, t in enumerate(oos_dates):
        for j, n in enumerate(valid_names):
            if n in df_returns.columns and t in df_returns.index: real_paths_matrix[i, j] = df_returns.at[t, n]
            elif n in df_factors.columns and t in df_factors.index: real_paths_matrix[i, j] = df_factors.at[t, n]

    cols = ['underlying_symbol', 'quote_datetime', 'underlying_mid_price', 'tau', 'strike', 'option_type', 'delta', 'rate', 'log_moneyness', 'implied_volatility']
    df_parquet = pd.read_parquet(os.path.join(DATA_DIR, "options_surfaces_data_cleaned.parquet"), columns=cols)
    parquet_by_date = {d: grp.reset_index(drop=True) for d, grp in df_parquet.groupby('quote_datetime')}

    m0_path = os.path.join(RESULTS_DIR, "copulas", "static", "joint_vine_spot_har_garch_evt_model.json")
    gauss_path = os.path.join(RESULTS_DIR, "copulas", "static", "gaussian_vine_spot_har_garch_evt_model.json")
    m1_path = os.path.join(RESULTS_DIR, "copulas", "gas", "gas_vine_spot_har_garch_evt_model.pth")
    
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
        
        portfolio = Portfolio2_Dispersion(parquet_by_date[t0], t0, n_contracts=1000, target_dte=30)
        if not portfolio.book['valid']: continue

        day_idx = oos_dates.get_loc(current_date)
        init_states = {n: all_states[n][day_idx] for n in valid_names}

        # Initialize fresh GAS Copula and roll forward to current day
        print(f"  [-] Fast-forwarding GAS states from OOS_START...")
        gas_vine = DynamicGASVine(m1_path, m0_path)
        gen_m1 = UniversalScenarioGenerator(valid_names, gas_vine, "M1")
        gen_m1.classify_marginals(marginals)
        
        for k in range(day_idx):
            u_realized = gen_m1.calculate_realized_uniforms(real_paths_matrix[k+1], {n: all_states[n][k] for n in valid_names}, marginals)
            gas_vine.update_states(u_realized)

        print(f"  [-] Projecting paths...")
        paths_m1 = gen_m1.simulate_multiday(N_SCENARIOS, HORIZON, init_states, marginals, use_copula=True)
        paths_m0 = gen_base.simulate_multiday(N_SCENARIOS, HORIZON, init_states, marginals, use_copula=True)
        
        gen_base.copula = gauss_copula
        paths_gauss = gen_base.simulate_multiday(N_SCENARIOS, HORIZON, init_states, marginals, use_copula=True)
        paths_indep = gen_base.simulate_multiday(N_SCENARIOS, HORIZON, init_states, marginals, use_copula=False)

        print(f"  [-] Evaluating Terminal Unhedged Dispersion PnL...")
        res_m1 = portfolio.evaluate_multiday_terminal_pnl(paths_m1, valid_names, surfaces_dict, name_to_col, mat_arr, mon_arr, HORIZON, factor_idx_map)
        res_m0 = portfolio.evaluate_multiday_terminal_pnl(paths_m0, valid_names, surfaces_dict, name_to_col, mat_arr, mon_arr, HORIZON, factor_idx_map)
        res_gauss = portfolio.evaluate_multiday_terminal_pnl(paths_gauss, valid_names, surfaces_dict, name_to_col, mat_arr, mon_arr, HORIZON, factor_idx_map)
        res_indep = portfolio.evaluate_multiday_terminal_pnl(paths_indep, valid_names, surfaces_dict, name_to_col, mat_arr, mon_arr, HORIZON, factor_idx_map)

        _, es_m1 = compute_risk_metrics(res_m1['pnl_total'], ALPHA)
        _, es_m0 = compute_risk_metrics(res_m0['pnl_total'], ALPHA)
        _, es_gauss = compute_risk_metrics(res_gauss['pnl_total'], ALPHA)
        _, es_indep = compute_risk_metrics(res_indep['pnl_total'], ALPHA)

        print(f"  [RESULT] M1 ES: ${es_m1:,.0f} | M0 ES: ${es_m0:,.0f} | Gauss ES: ${es_gauss:,.0f} | Indep ES: ${es_indep:,.0f}")
        print(f"  [PREMIUM] Delta-Corr (vs Indep): ${es_m1 - es_indep:,.0f}")

        results.append({
            'Date': current_date,
            'M1_ES': es_m1, 'M0_ES': es_m0, 'Gauss_ES': es_gauss, 'Indep_ES': es_indep,
            'DeltaES_Corr_Indep': es_m1 - es_indep,
            'DeltaES_Corr_Gauss': es_m1 - es_gauss,
            'DeltaES_Corr_M0': es_m1 - es_m0
        })

    df_res = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "backtests", "sim_p2_correlation_premium.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\n[+] Simulation complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()