import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import sys
import pyvinecopulib as pv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from src.dynamics.HAR_GARCH import HAR_GARCH_EVT
from src.dynamics.NeuralSDE import NeuralSDE
from src.dynamics.NGARCH_T import NGARCH_T
from src.backtesting.portfolio1 import Portfolio1_RiskReversal
from src.backtesting.utils.risk_metrics import tick_loss, kupiec_pof_test, christoffersen_test, mcneil_frey_test, diebold_mariano_test
from src.backtesting.utils.generators import UniversalScenarioGenerator, DynamicVineWrapper

class RiskModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'HAR_GARCH_EVT': return HAR_GARCH_EVT
        if name == 'NGARCH_T': return NGARCH_T
        if name == 'NeuralSDE': return NeuralSDE
        return super().find_class(module, name)

def precompute_garch_states(valid_names, marginals, df_returns, df_factors, oos_dates, train_end):
    states = {n: [None] * len(oos_dates) for n in valid_names}
    for n in valid_names:
        m = marginals[n]
        if hasattr(m, 'pi_drift'): 
            arr, dates = (df_returns[n].values, df_returns.index) if n in df_returns.columns else (df_factors[n].values, df_factors.index)
            for i, t_today in enumerate(oos_dates):
                loc_today = int(dates.searchsorted(t_today, side='right'))
                if loc_today >= m.n_lags: states[n][i] = {'history': arr[loc_today - m.n_lags : loc_today]}
        elif hasattr(m, 'params') and isinstance(m.params, list): 
            if n not in df_returns.columns: continue
            mu, omega, alpha, beta, theta, nu = m.params
            curr_sig2, curr_eps = m.vol[-1]**2, float(df_returns[n].loc[:train_end].iloc[-1]) - mu
            for i, rv in enumerate(df_returns[n].reindex(oos_dates).values):
                states[n][i] = {'sigma2': curr_sig2, 'resid': curr_eps}
                prev_sig = np.sqrt(curr_sig2)
                curr_sig2 = omega + alpha * (((curr_eps / max(prev_sig, 1e-6) if prev_sig > 1e-6 else 0.0) - theta)**2) * curr_sig2 + beta * curr_sig2
                curr_eps = rv - mu if not np.isnan(rv) else curr_eps
        else: 
            if n not in df_factors.columns: continue
            p, arr, fac_dates = m.params, df_factors[n].values, df_factors.index
            curr_sig2, curr_resid = m.vol[-1]**2, m.resids[-1] * m.vol[-1]
            for i, t_today in enumerate(oos_dates):
                loc_today = int(fac_dates.searchsorted(t_today, side='right'))
                if loc_today < 22: continue
                states[n][i] = {'sigma2': curr_sig2, 'resid': curr_resid, 'history': arr[loc_today - 22: loc_today]}
                if loc_today < len(arr):
                    hw = arr[loc_today - 22: loc_today]
                    mean = p['har_intercept'] + p['har_daily']*hw[-1] + p['har_weekly']*hw[-5:].mean() + p['har_monthly']*hw.mean()
                    curr_sig2 = p['garch_omega'] + p['garch_alpha']*(curr_resid**2) + p['garch_beta']*curr_sig2
                    curr_resid = arr[loc_today] - mean
    return states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['M0', 'M1', 'M2', 'M3', 'M4'])
    parser.add_argument('--paths', type=int, default=10000)
    args = parser.parse_args()

    torch.set_num_threads(18) 
    print(f"[*] Initializing Portfolio 1 Backtest for {args.model}...")

    DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "processed")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
    ALPHA       = 0.05
    TRAIN_END   = pd.Timestamp("2024-12-31")
    OOS_START   = pd.Timestamp("2025-01-02")
    OOS_END     = pd.Timestamp("2025-12-29")

    # 1. DYNAMIC PATH RESOLUTION 
    if args.model in ["M0", "M1", "M2"]:
        marg_model_name = "HAR_GARCH"
    else:
        marg_model_name = "NSDE"
        
    MARG_PATH = os.path.join("dynamics", marg_model_name, "fitted_marginals.pkl")

    COP_DICT = {
        "M0": "joint_vine_spot_har_garch_evt_model.json",
        "M1": "gas_vine_spot_har_garch_evt_model.pth",
        "M2": "neural_vine_spot_har_garch_evt_model.pth",
        "M3": "gas_vine_spot_nsde_model.pth",
        "M4": "neural_vine_spot_nsde_model.pth"
    }
    COP_PATH = os.path.join("copulas", "static" if args.model == "M0" else "gas" if args.model in ["M1", "M3"] else "neural", COP_DICT[args.model])

    # 2. DATA LOADING
    df_factors = pd.read_csv(os.path.join(RESULTS_DIR, "factors", "hierarchical_all_factors.csv"), index_col=0, parse_dates=True)
    df_returns = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"), index_col=0, parse_dates=True)
    
    with open(os.path.join(RESULTS_DIR, MARG_PATH), "rb") as f:
        marginals = RiskModelUnpickler(f).load()

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

    oos_dates = df_returns[(df_returns.index >= OOS_START) & (df_returns.index <= OOS_END)].index
    all_states = precompute_garch_states(valid_names, marginals, df_returns, df_factors, oos_dates, TRAIN_END)
    
    real_paths_matrix = np.zeros((len(oos_dates), len(valid_names)))
    for i, t in enumerate(oos_dates):
        for j, n in enumerate(valid_names):
            if n in df_returns.columns and t in df_returns.index: real_paths_matrix[i, j] = df_returns.at[t, n]
            elif n in df_factors.columns and t in df_factors.index: real_paths_matrix[i, j] = df_factors.at[t, n]

    # 3. COPULA & GENERATOR SETUP
    full_cop_path = os.path.join(RESULTS_DIR, COP_PATH)
    
    if args.model == "M0":
        copula = pv.Vinecop.from_file(full_cop_path)
    else:
        # Load Warmup Uniforms for the Rolling Window
        unif_dir = os.path.join(RESULTS_DIR, "dynamics", "NSDE" if args.model in ["M3", "M4"] else "HAR_GARCH")
        df_unif = pd.read_csv(os.path.join(unif_dir, "uniforms_train.csv"), index_col=0, parse_dates=True)
        df_spot = pd.read_csv(os.path.join(RESULTS_DIR, "dynamics", "NGARCH", "uniforms_train.csv"), index_col=0, parse_dates=True)
        df_unif = pd.concat([df_spot, df_unif], axis=1).reindex(columns=valid_names).ffill().bfill()
        
        history_window = df_unif.iloc[-60:].values 
        static_base = os.path.join(RESULTS_DIR, "copulas", "static", "joint_vine_spot_har_garch_evt_model.json")
        
        # Use our new Wrapper that handles both GAS and Neural
        copula = DynamicVineWrapper(full_cop_path, static_base, history_window, "GAS" if args.model in ["M1", "M3"] else "Neural")

    generator = UniversalScenarioGenerator(valid_names, copula, args.model)
    generator.classify_marginals(marginals)

    cols = ['underlying_symbol', 'quote_datetime', 'underlying_mid_price', 'tau', 'strike', 'option_type', 'delta', 'rate']
    df_parquet = pd.read_parquet(os.path.join(DATA_DIR, "options_surfaces_data_cleaned.parquet"), columns=cols)
    df_parquet['quote_datetime'] = pd.to_datetime(df_parquet['quote_datetime'])
    parquet_by_date = {d: grp.reset_index(drop=True) for d, grp in df_parquet.groupby('quote_datetime')}

    # 4. ROLLING BACKTEST
    results = []
    print(f"\n[!] Executing Rolling Simulation ({len(oos_dates)-1} days x {args.paths} paths)...")
    
    for i in tqdm(range(len(oos_dates) - 1)):
        t_today, t_tmrw = oos_dates[i], oos_dates[i + 1]
        dt_days = (t_tmrw - t_today).days 

        available_dates = sorted(parquet_by_date.keys())
        t0 = next((d for d in reversed(available_dates) if d <= t_today), available_dates[-1])
        df_today = parquet_by_date[t0]
        
        portfolio = Portfolio1_RiskReversal(df_today, t0, n_contracts=1000, target_dte=30)
        if not portfolio.rrs: continue

        init_states = {n: all_states[n][i] for n in valid_names if all_states[n][i] is not None}
        if len(init_states) != len(valid_names): continue

        paths_j, paths_i = generator.simulate_1day_dual(args.paths, init_states, marginals)
        real_row = real_paths_matrix[i + 1].reshape(1, -1)

        pnl_j = portfolio.evaluate_1day_pnl(paths_j, valid_names, surfaces_dict, name_to_col, mat_arr, mon_arr, dt_days, factor_idx_map)
        pnl_i = portfolio.evaluate_1day_pnl(paths_i, valid_names, surfaces_dict, name_to_col, mat_arr, mon_arr, dt_days, factor_idx_map)
        real_pnl = portfolio.evaluate_1day_pnl(real_row, valid_names, surfaces_dict, name_to_col, mat_arr, mon_arr, dt_days, factor_idx_map)[0]

        var_j, var_i = float(np.percentile(pnl_j, ALPHA * 100)), float(np.percentile(pnl_i, ALPHA * 100))
        es_j, es_i = float(pnl_j[pnl_j <= var_j].mean()), float(pnl_i[pnl_i <= var_i].mean())

        results.append({
            'Date': t_tmrw, 'Realized_PnL': real_pnl,
            'Vine_VaR': var_j, 'Vine_ES': es_j, 'Vine_Std': float(pnl_j.std()), 'Vine_Hit': int(real_pnl < var_j),
            'Indep_VaR': var_i, 'Indep_ES': es_i, 'Indep_Std': float(pnl_i.std()), 'Indep_Hit': int(real_pnl < var_i),
        })

        if args.model != "M0":
            u_realized = generator.calculate_realized_uniforms(real_row[0], init_states, marginals)
            copula.update_states(u_realized)

    # 5. ECONOMETRICS
    df_res = pd.DataFrame(results)
    rpnl, vh, ih = df_res['Realized_PnL'].values, df_res['Vine_Hit'].values, df_res['Indep_Hit'].values

    # Run all tests
    vine_kup_p = kupiec_pof_test(vh, ALPHA)
    vine_cc_p, vine_ind_p, vine_uc_p = christoffersen_test(vh, ALPHA)
    vine_mf_stat, vine_mf_p, _ = mcneil_frey_test(rpnl, df_res['Vine_ES'].values, df_res['Vine_Std'].values, vh)
    dm_stat, dm_pval, dm_diff = diebold_mariano_test(tick_loss(rpnl, df_res['Vine_VaR'].values, ALPHA), tick_loss(rpnl, df_res['Indep_VaR'].values, ALPHA), h=1)

    print(f"\n{'='*60}")
    print(f" BACKTEST COMPLETE: {args.model} | P1: RISK REVERSAL")
    print(f"{'='*60}")
    print(f" Vine Exceedances: {vh.sum()} | Target: {len(df_res)*ALPHA:.1f}")
    print(f" Kupiec POF p-value: {vine_kup_p:.4f} (PASS if > {ALPHA})")
    print(f" Christoffersen CC p-value: {vine_cc_p:.4f} (PASS if > {ALPHA})")
    print(f" McNeil-Frey p-value: {vine_mf_p:.4f} (PASS if > {ALPHA})")
    print(f" Diebold-Mariano p-value: {dm_pval:.4f}")
    
    os.makedirs(os.path.join(RESULTS_DIR, "backtests"), exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "backtests", f"p1_results_{args.model}.csv")
    df_res.to_csv(out_path, index=False)
    print(f" Saved to {out_path}")

if __name__ == "__main__":
    main()