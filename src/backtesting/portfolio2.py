import numpy as np
import pandas as pd
from scipy.stats import norm
from src.fitting.ssvi import SSVI 
from src.backtesting.utils.black_scholes import bs_price_vec, bilinear_interp_vec

# --- Module-Level Static Configuration ---
INDEX_ASSET = "QQQ"
CONSTITUENTS = ["NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "NFLX"]

_RAW_W = {"NVDA": 8.63, "AAPL": 7.61, "MSFT": 5.78, "AMZN": 4.34, "TSLA": 3.92, "GOOGL": 3.47, "NFLX": 2.21}
_W_SUM = sum(_RAW_W.values())
CONSTITUENT_WEIGHTS = {k: v / _W_SUM for k, v in _RAW_W.items()}
# -----------------------------------------

def bs_straddle_price_vec(S, K, T, sigma, r):
    """ATM straddle = call + put. Vectorised over S and sigma."""
    return bs_price_vec(S, K, T, sigma, r, 'C') + bs_price_vec(S, K, T, sigma, r, 'P')

def bs_straddle_vega_vec(S, K, T, sigma, r):
    """Straddle vega = 2 * single-option vega. Vectorised over S and sigma."""
    S, sigma = np.asarray(S, dtype=float), np.asarray(sigma, dtype=float)
    vega = np.zeros_like(S)
    mask = (T >= 1e-6) & (sigma >= 1e-8) & (S >= 1e-8)
    if not np.any(mask): return vega
    d1 = (np.log(S[mask] / K) + (r + 0.5 * sigma[mask]**2) * T) / (sigma[mask] * np.sqrt(T))
    vega[mask] = 2.0 * S[mask] * norm.pdf(d1) * np.sqrt(T) 
    return vega

class Portfolio2_Dispersion:
    INDEX_ASSET = INDEX_ASSET
    CONSTITUENTS = CONSTITUENTS
    CONSTITUENT_WEIGHTS = CONSTITUENT_WEIGHTS

    def __init__(self, df_today, current_date, n_contracts=1000, target_dte=30):
        self.n_contracts = n_contracts
        self.r0 = 0.0 # Strict Forward-Parity
        self.book = self._build_book(df_today, current_date, target_dte)

    def _get_atm_straddle_data(self, df_today, current_date, sym, target_tau):
        sub = df_today[df_today['underlying_symbol'] == sym].copy()
        if sub.empty: return None
        
        S0 = sub['underlying_mid_price'].mean()
        
        taus_avail = sub['tau'].dropna().unique()
        if len(taus_avail) == 0: return None
        best_tau = float(taus_avail[np.argmin(np.abs(taus_avail - target_tau))])
        
        if best_tau < 10 / 365.25: return None

        try:
            surf = SSVI(sub, symbol=sym, quote_datetime=current_date)
            surf.fit(max_iter=5000)
            sigma_atm = surf.get_iv(S0, best_tau, S_current=S0, r=self.r0)
            if sigma_atm is None or np.isnan(sigma_atm) or sigma_atm <= 0: return None
        except Exception:
            return None

        S_arr, sig_arr = np.array([S0]), np.array([sigma_atm])
        price = float(bs_straddle_price_vec(S_arr, S0, best_tau, sig_arr, self.r0)[0])
        vega = float(bs_straddle_vega_vec(S_arr, S0, best_tau, sig_arr, self.r0)[0])

        if price <= 0 or vega <= 0: return None

        return {
            'sym': sym, 'S0': S0, 'K': S0, 'tau': best_tau,
            'sigma0': sigma_atm, 'price0': price, 'vega0': vega
        }

    def _build_book(self, df_today, current_date, target_dte):
        target_tau = target_dte / 365.25
        
        idx_data = self._get_atm_straddle_data(df_today, current_date, self.INDEX_ASSET, target_tau)
        if idx_data is None: return {'valid': False}

        const_data = {}
        for sym in self.CONSTITUENTS:
            d = self._get_atm_straddle_data(df_today, current_date, sym, target_tau)
            if d is not None: const_data[sym] = d

        if len(const_data) == 0: return {'valid': False}

        # Re-normalise weights over available constituents only
        avail_w_sum = sum(self.CONSTITUENT_WEIGHTS[s] for s in const_data)
        if avail_w_sum < 1e-8: return {'valid': False}
        w_adj = {s: self.CONSTITUENT_WEIGHTS[s] / avail_w_sum for s in const_data}

        # Vega-neutral sizing (Eq. 4.5.5)
        n_j = {s: w_adj[s] * self.n_contracts for s in const_data}
        total_const_vega = sum(n_j[s] * const_data[s]['vega0'] for s in const_data)
        n_I = total_const_vega / idx_data['vega0']

        val_0_idx = idx_data['price0']
        val_0_const = {s: const_data[s]['price0'] for s in const_data}

        return {
            'valid': True, 'idx_data': idx_data, 'const_data': const_data,
            'n_I': n_I, 'n_j': n_j, 'w_adj': w_adj,
            'val_0_idx': val_0_idx, 'val_0_const': val_0_const
        }

    def evaluate_1day_pnl(self, paths, valid_names, surfaces_dict, name_to_idx, mat_arr, mon_arr, dt_days, factor_idx_map):
        if not self.book['valid']: return np.zeros(len(paths))
        
        N = paths.shape[0]
        pnl = np.zeros(N)
        tau_step = dt_days / 365.25
        
        g_idx = factor_idx_map.get("G_PC", [])
        g_mat = paths[:, g_idx] if g_idx else np.zeros((N, 3))

        def reprice_straddle(leg):
            sym, S0, K, tau1 = leg['sym'], leg['S0'], leg['K'], max(leg['tau'] - tau_step, 1e-6)

            if sym not in name_to_idx or sym not in surfaces_dict:
                S1 = S0 * np.ones(N)
                sig1 = np.full(N, leg['sigma0'])
            else:
                S1 = S0 * np.exp(paths[:, name_to_idx[sym]])
                l_idx = factor_idx_map.get(sym, [])
                l_mat = paths[:, l_idx] if l_idx else np.zeros((N, 3))

                try:
                    grids = surfaces_dict[sym].reconstruct(global_param=g_mat, local_param=l_mat)
                    log_m = np.log(K / np.clip(S1, 1e-8, np.inf))
                    sig1 = np.exp(np.clip(bilinear_interp_vec(grids, mat_arr, mon_arr, tau1, log_m), -10, 3))
                except Exception:
                    sig1 = np.full(N, leg['sigma0'])

            return bs_straddle_price_vec(S1, K, tau1, sig1, self.r0)

        # Index leg (Short)
        price1_idx = reprice_straddle(self.book['idx_data'])
        pnl += (price1_idx - self.book['val_0_idx']) * (-self.book['n_I'])

        # Constituent legs (Long)
        for sym, leg in self.book['const_data'].items():
            price1_j = reprice_straddle(leg)
            pnl += (price1_j - self.book['val_0_const'][sym]) * self.book['n_j'][sym]

        return pnl