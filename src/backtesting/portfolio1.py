import numpy as np
import pandas as pd
from src.backtesting.utils.black_scholes import bs_price_vec, bs_delta_vec, bilinear_interp_vec

from src.fitting.ssvi import SSVI 

class Portfolio1_RiskReversal:
    def __init__(self, df_today, current_date, n_contracts=1000, target_dte=30):
        self.n_contracts = n_contracts
        # Pass the specific day's DataFrame and date to build the book
        self.rrs = self._build_book(df_today, current_date, target_dte)

    def _build_book(self, df_today, current_date, target_dte):
        rrs = []
        target_tau = target_dte / 365.25
        
        for sym in df_today['underlying_symbol'].unique():
            sub = df_today[df_today['underlying_symbol'] == sym].copy()
            if sub.empty: continue
            
            S0 = sub['underlying_mid_price'].mean()
            # Default to 0.0 if rate is missing, else convert from %
            r0 = float(sub['rate'].mean() / 100.0) if 'rate' in sub.columns else 0.0

            # --- 1. FIT SSVI ON THE FLY ---
            try:
                surf = SSVI(sub, symbol=sym, quote_datetime=current_date)
                surf.fit(max_iter=5000)
            except Exception:
                continue

            # --- 2. FIND CLOSEST OPTIONS ---
            win = sub[(sub['tau'] >= 25/365.25) & (sub['tau'] <= 45/365.25)]
            if win.empty: continue

            best_tau = win.loc[(win['tau'] - target_tau).abs().idxmin()]['tau']
            win_c = win[(win['tau'] == best_tau) & (win['option_type'] == 'C')]
            win_p = win[(win['tau'] == best_tau) & (win['option_type'] == 'P')]
            if win_c.empty or win_p.empty: continue

            K_C = float(win_c.loc[(win_c['delta'] - 0.25).abs().idxmin()]['strike'])
            K_P = float(win_p.loc[(win_p['delta'] + 0.25).abs().idxmin()]['strike'])
            
            try:
                sig_C = surf.get_iv(K_C, best_tau, S_current=S0, r=r0)
                sig_P = surf.get_iv(K_P, best_tau, S_current=S0, r=r0)
                if any(s is None or s <= 0 or np.isnan(s) for s in [sig_C, sig_P]): continue
            except Exception: continue

            price_C = float(bs_price_vec(S0, K_C, best_tau, sig_C, r0, 'C'))
            price_P = float(bs_price_vec(S0, K_P, best_tau, sig_P, r0, 'P'))
            delta0  = float(bs_delta_vec(S0, K_C, best_tau, sig_C, r0, 'C') - 
                            bs_delta_vec(S0, K_P, best_tau, sig_P, r0, 'P'))
            
            rrs.append({
                'sym': sym, 'S0': S0, 'tau': best_tau, 'r0': r0,
                'K_C': K_C, 'sig_C': sig_C, 'K_P': K_P, 'sig_P': sig_P, 
                'n': self.n_contracts, 'delta0': delta0, 'val_0': price_C - price_P
            })
        return rrs

    def evaluate_1day_pnl(self, paths, valid_names, surfaces_dict, name_to_idx, mat_arr, mon_arr, dt_days, factor_idx_map):
        N = len(paths)
        pnl = np.zeros(N)
        
        g_idx = factor_idx_map.get("G_PC", [])
        g_mat = paths[:, g_idx] if g_idx else np.zeros((N, 3))

        for s in self.rrs:
            sym = s['sym']
            if sym not in name_to_idx or sym not in surfaces_dict: continue

            S1 = s['S0'] * np.exp(paths[:, name_to_idx[sym]])
            tau1 = max(s['tau'] - (dt_days / 365.25), 1e-6)
            
            l_idx = factor_idx_map.get(sym, [])
            l_mat = paths[:, l_idx] if l_idx else np.zeros((N, 3))

            try:
                grids = surfaces_dict[sym].reconstruct(global_param=g_mat, local_param=l_mat)
                sig1_C = np.exp(np.clip(bilinear_interp_vec(grids, mat_arr, mon_arr, tau1, np.log(s['K_C']/S1)), -10, 3))
                sig1_P = np.exp(np.clip(bilinear_interp_vec(grids, mat_arr, mon_arr, tau1, np.log(s['K_P']/S1)), -10, 3))
            except Exception:
                sig1_C, sig1_P = np.full(N, s['sig_C']), np.full(N, s['sig_P'])

            val1_C = bs_price_vec(S1, s['K_C'], tau1, sig1_C, s['r0'], 'C')
            val1_P = bs_price_vec(S1, s['K_P'], tau1, sig1_P, s['r0'], 'P')
            
            pnl += ((val1_C - val1_P - s['val_0']) * s['n'] + (-s['delta0'] * s['n']) * (S1 - s['S0']))
            
        return pnl