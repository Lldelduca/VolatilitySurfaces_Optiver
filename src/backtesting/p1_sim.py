import numpy as np
import pandas as pd
import pyvinecopulib as pv
import pickle
import os
import sys
import math
import warnings
import torch
import torch.nn as nn
import scipy.special
from scipy.stats import norm, t as student_t, ks_2samp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pickle")
torch.set_default_dtype(torch.float64)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
FACTOR_DIR   = os.path.join(SCRIPT_DIR, "dynamics")
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "data", "results")

if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

try:
    sys.path.append(SCRIPT_DIR)
    import dynamics.EVT
    sys.modules['EVT'] = dynamics.EVT
except ImportError:
    pass

from src.fitting.ssvi import SSVI

class InverseStudentT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, nu):
        u_cpu, nu_cpu = u.detach().cpu().numpy(), nu.detach().cpu().numpy()
        u_cpu = np.clip(u_cpu, 1e-12, 1 - 1e-12)
        x_tensor = torch.from_numpy(np.asarray(scipy.special.stdtrit(nu_cpu, u_cpu))).to(u.device, dtype=u.dtype)
        ctx.save_for_backward(x_tensor, u, nu)
        return x_tensor

    @staticmethod
    def backward(ctx, grad_output):
        x, u, nu = ctx.saved_tensors
        pi = torch.tensor(3.141592653589793, device=x.device, dtype=x.dtype)
        log_const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * pi)
        log_kernel = -((nu + 1) / 2) * torch.log(1 + (x**2) / nu)
        pdf = torch.exp(log_const + log_kernel)
        grad_u = grad_output / torch.clamp(pdf, min=1e-100)
        return grad_u, None

def inverse_t_cdf(u, nu): return InverseStudentT.apply(u, nu)

class NeuralPairCopula(nn.Module):
    def __init__(self, family, rotation=0, hidden_dim=8, num_layers=1, dropout=0.0):
        super().__init__()
        self.family = str(family).split('.')[-1].lower()
        self.rotation = int(rotation)
        self.rnn = nn.GRU(input_size=2, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)
        self.f_init = nn.Parameter(torch.tensor(0.0))
        
        if 'student' in self.family: self.nu_param = nn.Parameter(torch.tensor(2.0))
        else: self.register_parameter('nu_param', None)
        
        self.hidden_state = None

    def get_nu(self):
        if self.nu_param is None: return None
        return torch.nn.functional.softplus(self.nu_param) + 2.01

    def rotate_data(self, u, v):
        if self.rotation == 90: return 1-u, v
        if self.rotation == 180: return 1-u, 1-v
        if self.rotation == 270: return u, 1-v
        return u, v
    
    def transform_parameter(self, f_t):
        if 'gaussian' in self.family or 'student' in self.family: return torch.tanh(f_t) * 0.999 
        elif 'clayton' in self.family: return torch.nn.functional.softplus(f_t) + 1e-5 
        elif 'gumbel' in self.family: return torch.nn.functional.softplus(f_t) + 1.0001 
        elif 'frank' in self.family: return torch.where(torch.abs(f_t) < 1e-4, torch.sign(f_t + 1e-12) * 1e-4, f_t)
        return f_t

    def compute_h_func(self, u, v, theta, nu=None):
        u_rot, v_rot = self.rotate_data(u, v)
        eps = 1e-9
        u_rot, v_rot = torch.clamp(u_rot, eps, 1-eps), torch.clamp(v_rot, eps, 1-eps)
        h_val = torch.zeros_like(u_rot)
        
        if 'gaussian' in self.family:
            n = torch.distributions.Normal(0, 1)
            x, y = n.icdf(u_rot), n.icdf(v_rot)
            h_val = n.cdf((x - theta*y) / torch.sqrt(1 - theta**2 + 1e-8))
        elif 'student' in self.family:
            x, y = inverse_t_cdf(u_rot, nu), inverse_t_cdf(v_rot, nu)
            factor = torch.sqrt((nu + 1) / (nu + y**2) / (1 - theta**2 + 1e-8))
            h_val = torch.tensor(scipy.special.stdtr((nu+1).detach().cpu().numpy(), ((x - theta * y) * factor).detach().cpu().numpy()))
        elif 'clayton' in self.family:
            h_val = torch.pow(v_rot, -theta-1) * torch.pow(torch.pow(u_rot, -theta) + torch.pow(v_rot, -theta) - 1, -1/theta - 1)
        elif 'gumbel' in self.family:
            x, y = -torch.log(u_rot), -torch.log(v_rot)
            A = torch.pow(x**theta + y**theta, 1/theta)
            h_val = torch.exp(-A) * torch.pow(y, theta-1) / v_rot * torch.pow(x**theta + y**theta, 1/theta - 1)
        elif 'frank' in self.family:
            et, eu, ev = torch.exp(-theta), torch.exp(-theta*u_rot), torch.exp(-theta*v_rot)
            h_val = ((eu - 1) * ev) / ((et - 1) + (eu - 1) * (ev - 1) + 1e-20)

        if self.rotation in [90, 270]: h_val = 1 - h_val
        return torch.clamp(h_val, eps, 1 - eps)

    @torch.no_grad()
    def step_forward(self, u_val, v_val):
        if 'indep' in self.family: return torch.tensor(0.0), torch.tensor(0.0), u_val, v_val
        u_t, v_t, nu = u_val.view(-1), v_val.view(-1), self.get_nu()
        
        eps = 1e-6
        u_clamped, v_clamped = torch.clamp(u_t, eps, 1-eps), torch.clamp(v_t, eps, 1-eps)
        x_in = torch.erfinv(2 * u_clamped - 1) * math.sqrt(2)
        y_in = torch.erfinv(2 * v_clamped - 1) * math.sqrt(2)
        inputs = torch.stack([x_in, y_in], dim=1).unsqueeze(0) 

        rnn_out, self.hidden_state = self.rnn(inputs, self.hidden_state)
        f_t_seq = self.head(rnn_out).squeeze(0).squeeze(1)
        theta_next = self.transform_parameter(f_t_seq)
        
        h_dir = self.compute_h_func(u_t, v_t, theta_next, nu)
        h_indir = self.compute_h_func(v_t, u_t, theta_next, nu)
        return theta_next, nu, h_dir, h_indir

class DynamicNeuralVine:
    def __init__(self, pth_path, static_json_path):
        self.base_copula = pv.Vinecop.from_file(static_json_path)
        self.matrix = np.array(self.base_copula.matrix, dtype=np.int64)
        self.N = self.matrix.shape[0]
        self.models = {}
        
        is_dir = os.path.isdir(pth_path)
        neural_dict = {}
        if not is_dir:
            neural_dict = torch.load(pth_path, map_location='cpu', weights_only=False)

        print(f"[+] Initializing Neural Vine: Loading from {'Folder' if is_dir else 'Single File'}")
        
        for tree in range(self.N - 1):
            for edge in range(self.N - 1 - tree):
                edge_key = f"T{tree}_E{edge}"
                
                if is_dir:
                    edge_file = os.path.join(pth_path, f"{edge_key}.pth")
                    info = torch.load(edge_file, map_location='cpu') if os.path.exists(edge_file) else None
                else:
                    info = neural_dict.get(edge_key, None)

                if not info or info['family'] == 'indep':
                    self.models[edge_key] = None
                    continue
                
                model = NeuralPairCopula(
                    family=info['family'], 
                    rotation=info.get('rotation', 0),
                    hidden_dim=info.get('hidden_dim', 8),
                    num_layers=info.get('num_layers', 1),
                    dropout=info.get('dropout', 0.0)
                )
                
                if 'state_dict' in info:
                    model.load_state_dict(info['state_dict'], strict=True)
                else:
                    state_dict = {k: v for k, v in info.items() if isinstance(v, torch.Tensor)}
                    if state_dict: model.load_state_dict(state_dict, strict=False)
                
                model.eval()
                self.models[edge_key] = model

    def warm_up_and_push(self, history_window):
        for model in self.models.values():
            if model: model.hidden_state = None
            
        for t in range(history_window.shape[0]):
            self._update_states(history_window[t:t+1])
            
        for tree in range(self.N - 1):
            for edge in range(self.N - 1 - tree):
                model = self.models[f"T{tree}_E{edge}"]
                if model:
                    with torch.no_grad():
                        dummy_in = torch.zeros(1, 1, 2)
                        out, _ = model.rnn(dummy_in, model.hidden_state)
                        f_t = model.head(out).squeeze()
                        theta = float(model.transform_parameter(f_t).item())
                        nu = model.get_nu()
                        self.base_copula.get_pair_copula(tree, edge).parameters = np.array([[theta], [float(nu.item())]]) if nu is not None else np.array([[theta]])

    def _update_states(self, u_realized_np):
        u_tensor = torch.tensor(u_realized_np, dtype=torch.float64).view(1, -1)
        M = self.matrix
        if M.max() == self.N: M -= 1 
        if np.sum(M[0] >= 0) > np.sum(M[-1] >= 0): M = np.flipud(M)
            
        h_storage = {(i, -1): u_tensor[:, i] for i in range(self.N)}
        for tree in range(self.N - 1):
            for edge in range(self.N - 1 - tree):
                row, col = self.N - 1 - tree, edge
                u_vec = h_storage[(M[row, col], -1)] if tree == 0 else h_storage[(col, tree-1)]
                var_2, partner_col = M[col, col], -1
                
                if tree == 0: v_vec = h_storage[(var_2, -1)]
                else:
                    for k in range(self.N):
                        if M[row+1, k] == var_2: partner_col = k; break
                    v_vec = h_storage[(partner_col, tree-1)]
                    
                model = self.models[f"T{tree}_E{edge}"]
                if model is None:
                    h_storage[(edge, tree)] = u_vec
                    if tree < self.N - 2: h_storage[(partner_col, tree)] = v_vec
                else:
                    _, _, h_dir, h_indir = model.step_forward(u_vec, v_vec)
                    h_storage[(edge, tree)] = h_dir
                    if tree < self.N - 2: h_storage[(partner_col, tree)] = h_indir
class HAR_GARCH_EVT:
    def __init__(self): self.params = {}; self.evt_model = None; self.vol = []; self.resids = []

class NGARCH_T:
    def __init__(self): self.params = []; self.vol = []; self.resids = []

def _patched_generator_ctor(*args, **kwargs):
    import numpy as np
    return np.random.default_rng()

class RiskModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'HAR_GARCH_EVT': return HAR_GARCH_EVT
        if name == 'NGARCH_T':      return NGARCH_T
        if name == '_patched_generator_ctor': return _patched_generator_ctor
        return super().find_class(module, name)

def load_csv_robust(filename, search_dirs):
    for d in search_dirs:
        path = os.path.join(d, filename) if d else filename
        if os.path.exists(path):
            print(f"  [-] Found {filename} at: {path}")
            return pd.read_csv(path, index_col=0, parse_dates=True)
    raise FileNotFoundError(f"Could not find {filename} in {search_dirs}")

def get_initial_market_state_from_df(df, anchor_date):
    available = sorted(df['quote_datetime'].unique())
    t0 = next((d for d in reversed(available) if d <= anchor_date), available[-1])
    
    df_t0    = df[df['quote_datetime'] == t0].copy()
    spots    = {}
    rates    = {}
    surfaces = {}

    for sym in df_t0['underlying_symbol'].unique():
        sub = df_t0[df_t0['underlying_symbol'] == sym]
        spots[sym] = sub['underlying_mid_price'].mean()
        rates[sym] = sub['rate'].mean()  
        try:
            ssvi = SSVI(sub, symbol=sym, quote_datetime=t0)
            ssvi.fit()
            surfaces[sym] = ssvi
        except Exception:
            surfaces[sym] = None

    return {'spots': spots, 'rates': rates, 'surfaces': surfaces, 'date': t0, 'df_raw': df_t0}

def precompute_garch_states_and_uniforms(valid_names, marginals, df_returns, df_factors, oos_dates, TRAIN_END):
    states = {n: [None] * len(oos_dates) for n in valid_names}
    uniforms_matrix = np.zeros((len(oos_dates), len(valid_names)))
    
    for j, n in enumerate(valid_names):
        m = marginals[n]
        is_ngarch = hasattr(m, 'params') and isinstance(m.params, list)
        
        if is_ngarch:
            if n not in df_returns.columns: continue
            mu, omega, alpha, beta, theta, nu = m.params
            curr_sig2, curr_eps = m.vol[-1]**2, float(df_returns[n].loc[:TRAIN_END].iloc[-1]) - mu
            
            for i, rv in enumerate(df_returns[n].reindex(oos_dates).values):
                states[n][i] = {'sigma2': curr_sig2, 'resid': curr_eps}
                prev_sig = np.sqrt(curr_sig2)
                curr_sig2 = omega + alpha * (((curr_eps / max(prev_sig, 1e-6) if prev_sig > 1e-6 else 0.0) - theta)**2) * curr_sig2 + beta * curr_sig2
                z_real = (rv - mu) / np.sqrt(curr_sig2) if not np.isnan(rv) else 0
                uniforms_matrix[i, j] = student_t.cdf(z_real, df=nu)
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
                    z_real = (arr[loc_today] - mean) / np.sqrt(curr_sig2)
                    u = m.evt_model.transform(np.array([z_real]))[0] if hasattr(m, 'evt_model') else norm.cdf(z_real)
                    uniforms_matrix[i, j] = u
                    curr_resid = arr[loc_today] - mean
                    
    return states, np.clip(uniforms_matrix, 1e-6, 1 - 1e-6)

class ScenarioGenerator:
    def __init__(self, factor_order):
        self.factor_order = factor_order

    def simulate(self, n_scenarios, horizon, init_states, marginals, copula=None):
        dim    = len(self.factor_order)
        paths  = np.zeros((n_scenarios, horizon, dim))

        n_total = n_scenarios * horizon
        if copula is None:
            U_all = np.random.uniform(1e-6, 1 - 1e-6, size=(n_total, dim))
        else:
            U_all = np.clip(copula.simulate(n_total), 1e-6, 1 - 1e-6)
        
        U_all = U_all.reshape((horizon, n_scenarios, dim))

        ngarch_idx, har_idx = [], []
        ngarch_params, har_params = [], []
        har_models = []
        
        for i, name in enumerate(self.factor_order):
            m = marginals[name]
            is_ngarch = hasattr(m, 'params') and isinstance(m.params, list)
            if is_ngarch:
                ngarch_idx.append(i)
                ngarch_params.append(m.params)
            else:
                har_idx.append(i)
                p = m.params
                har_params.append([
                    p['har_intercept'], p['har_daily'], p['har_weekly'], p['har_monthly'],
                    p['garch_omega'], p['garch_alpha'], p['garch_beta']
                ])
                har_models.append(m.evt_model if hasattr(m, 'evt_model') and m.evt_model is not None else None)

        if ngarch_idx:
            ngarch_names = [self.factor_order[i] for i in ngarch_idx]
            sig2_n = np.tile(np.array([init_states[n]['sigma2'] for n in ngarch_names]), (n_scenarios, 1))
            eps_n  = np.tile(np.array([init_states[n]['resid'] for n in ngarch_names]), (n_scenarios, 1))
            p_n = np.array(ngarch_params)
            mu_n, om_n, al_n, be_n, th_n, nu_n = [p_n[:, k].reshape(1, -1) for k in range(6)]

        if har_idx:
            har_names = [self.factor_order[i] for i in har_idx]
            sig2_h = np.tile(np.array([init_states[n]['sigma2'] for n in har_names]), (n_scenarios, 1))
            resid_h = np.tile(np.array([init_states[n]['resid'] for n in har_names]), (n_scenarios, 1))
            hist_h = np.zeros((n_scenarios, len(har_idx), 22))
            for j, n in enumerate(har_names):
                hist_h[:, j, :] = np.tile(init_states[n]['history'][-22:], (n_scenarios, 1))
            p_h = np.array(har_params)
            h_int, h_d, h_w, h_m, g_om, g_al, g_be = [p_h[:, k].reshape(1, -1) for k in range(7)]

        for t in range(horizon):
            U = U_all[t]
            if ngarch_idx:
                U_n = U[:, ngarch_idx]
                z_n = student_t.ppf(U_n, df=nu_n)
                prev_sig = np.sqrt(sig2_n)
                prev_z_val = np.where(prev_sig > 1e-6, eps_n / np.maximum(prev_sig, 1e-6), 0.0)
                next_sig2_n = om_n + al_n * ((prev_z_val - th_n) ** 2) * sig2_n + be_n * sig2_n
                shock_n = np.sqrt(next_sig2_n) * z_n
                paths[:, t, ngarch_idx] = mu_n + shock_n
                sig2_n, eps_n = next_sig2_n, shock_n

            if har_idx:
                U_h = U[:, har_idx]
                z_h = np.zeros_like(U_h)
                for j, model in enumerate(har_models):
                    if model is not None: z_h[:, j] = model.inverse_transform(U_h[:, j])
                    else: z_h[:, j] = norm.ppf(U_h[:, j])
                next_sig2_h = g_om + g_al * (resid_h ** 2) + g_be * sig2_h
                mean_h = (h_int + h_d * hist_h[:, :, -1] + h_w * hist_h[:, :, -5:].mean(axis=2) + h_m * hist_h.mean(axis=2))
                shock_h = np.sqrt(next_sig2_h) * z_h
                val_h   = mean_h + shock_h
                paths[:, t, har_idx] = val_h
                sig2_h, resid_h = next_sig2_h, shock_h
                hist_h = np.concatenate([hist_h[:, :, 1:], val_h[:, :, np.newaxis]], axis=2)

        return paths

def bs_price_vec(S, K, T, sigma, r, option_type):
    S, sigma = np.asarray(S, dtype=float), np.asarray(sigma, dtype=float)
    price = np.zeros_like(S)
    mask = (T >= 1e-6) & (sigma >= 1e-8) & (S >= 1e-8)
    inv_mask = ~mask
    if np.any(inv_mask):
        if option_type == 'C': price[inv_mask] = np.maximum(0.0, S[inv_mask] - K)
        else: price[inv_mask] = np.maximum(0.0, K - S[inv_mask])
    if not np.any(mask): return price
    S_m, sig_m = S[mask], sigma[mask]
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S_m / K) + (r + 0.5 * sig_m**2) * T) / (sig_m * sqrt_T)
    d2 = d1 - sig_m * sqrt_T
    if option_type == 'C': val = S_m * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else: val = K * np.exp(-r*T) * norm.cdf(-d2) - S_m * norm.cdf(-d1)
    price[mask] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return price

def bs_delta_vec(S, K, T, sigma, r, option_type):
    S, sigma = np.asarray(S, dtype=float), np.asarray(sigma, dtype=float)
    delta = np.zeros_like(S)
    mask = (T >= 1e-6) & (sigma >= 1e-8) & (S >= 1e-8)
    if not np.any(mask): return delta
    S_m, sig_m = S[mask], sigma[mask]
    d1 = (np.log(S_m / K) + (r + 0.5 * sig_m**2) * T) / (sig_m * np.sqrt(T))
    val = norm.cdf(d1) if option_type == 'C' else norm.cdf(d1) - 1.0
    delta[mask] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return delta

def vega_bs_vec(S, K, T, sigma, r):
    S, sigma = np.asarray(S, dtype=float), np.asarray(sigma, dtype=float)
    vega = np.zeros_like(S)
    mask = (T >= 1e-6) & (sigma >= 1e-8) & (S >= 1e-8)
    if not np.any(mask): return vega
    S_m, sig_m = S[mask], sigma[mask]
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S_m / K) + (r + 0.5 * sig_m**2) * T) / (sig_m * sqrt_T)
    val = S_m * norm.pdf(d1) * sqrt_T
    vega[mask] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return vega

def vanna_bs_vec(S, K, T, sigma, r):
    S, sigma = np.asarray(S, dtype=float), np.asarray(sigma, dtype=float)
    vanna = np.zeros_like(S)
    mask = (T >= 1e-6) & (sigma >= 1e-8) & (S >= 1e-8)
    if not np.any(mask): return vanna
    S_m, sig_m = S[mask], sigma[mask]
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S_m / K) + (r + 0.5 * sig_m**2) * T) / (sig_m * sqrt_T)
    d2 = d1 - sig_m * sqrt_T
    val = -norm.pdf(d1) * d2 / sig_m
    vanna[mask] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return vanna

def gamma_bs_vec(S, K, T, sigma, r):
    S, sigma = np.asarray(S, dtype=float), np.asarray(sigma, dtype=float)
    gamma = np.zeros_like(S)
    mask = (T >= 1e-6) & (sigma >= 1e-8) & (S >= 1e-8)
    if not np.any(mask): return gamma
    S_m, sig_m = S[mask], sigma[mask]
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S_m / K) + (r + 0.5 * sig_m**2) * T) / (sig_m * sqrt_T)
    val = norm.pdf(d1) / (S_m * sig_m * sqrt_T)
    gamma[mask] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return gamma

def build_risk_reversal_book(market_t0, df_raw, target_tau_days=30, n_contracts=1000):
    tau_lo = (target_tau_days - 5)  / 365.0
    tau_hi = (target_tau_days + 15) / 365.0
    rrs = []

    for sym in sorted(market_t0['spots'].keys()):
        S0   = market_t0['spots'][sym]
        r0   = market_t0['rates'].get(sym, 0.0) 
        surf = market_t0['surfaces'].get(sym)
        if surf is None: continue

        sub = df_raw[df_raw['underlying_symbol'] == sym].copy()
        win = sub[(sub['tau'] >= tau_lo) & (sub['tau'] <= tau_hi)].copy()
        if win.empty: win = sub[sub['tau'] >= tau_lo].copy()
        if win.empty: continue

        win['_t_diff'] = (win['tau'] - target_tau_days/365.0).abs()
        best_tau = win.loc[win['_t_diff'].idxmin()]['tau']
        win = win[win['tau'] == best_tau].copy()

        win_c = win[win['option_type'] == 'C'].copy()
        win_p = win[win['option_type'] == 'P'].copy()
        if win_c.empty or win_p.empty: continue

        win_c['_dd'] = (win_c['delta'] - 0.25).abs()
        K_C = float(win_c.loc[win_c['_dd'].idxmin()]['strike'])

        win_p['_dd'] = (win_p['delta'] - (-0.25)).abs()
        K_P = float(win_p.loc[win_p['_dd'].idxmin()]['strike'])
        
        tau = float(best_tau)

        try:
            sigma0_C = surf.get_iv(K_C, tau, S_current=S0)
            sigma0_P = surf.get_iv(K_P, tau, S_current=S0)
            if any(s is None or s <= 0 or np.isnan(s) for s in [sigma0_C, sigma0_P]): raise ValueError("bad vol")
        except Exception: continue

        price_C = float(bs_price_vec(S0, K_C, tau, sigma0_C, r0, 'C'))
        price_P = float(bs_price_vec(S0, K_P, tau, sigma0_P, r0, 'P'))
        delta0 = float(bs_delta_vec(S0, K_C, tau, sigma0_C, r0, 'C') - bs_delta_vec(S0, K_P, tau, sigma0_P, r0, 'P'))
        vega0  = float(vega_bs_vec(S0, K_C, tau, sigma0_C, r0) - vega_bs_vec(S0, K_P, tau, sigma0_P, r0))
        vanna0 = float(vanna_bs_vec(S0, K_C, tau, sigma0_C, r0) - vanna_bs_vec(S0, K_P, tau, sigma0_P, r0))
        gamma0 = float(gamma_bs_vec(S0, K_C, tau, sigma0_C, r0) - gamma_bs_vec(S0, K_P, tau, sigma0_P, r0))

        rrs.append({
            'sym': sym, 'S0': S0, 'r_0': r0, 'tau_0': tau,
            'K_C': K_C, 'sigma_0_C': sigma0_C, 'K_P': K_P, 'sigma_0_P': sigma0_P,
            'n': n_contracts, 'val_0_C': price_C, 'val_0_P': price_P,
            'rr_val_0': price_C - price_P, 'delta_0': delta0, 'vega_0': vega0, 'vanna_0': vanna0, 'gamma_0': gamma0
        })
    return rrs

def _bilinear_interp_vec(grid_3d, mat_arr, mon_arr, tau, kappa_vec):
    N = len(kappa_vec)
    tau = float(np.clip(tau, mat_arr[0], mat_arr[-1]))
    kappa_vec = np.clip(kappa_vec, mon_arr[0], mon_arr[-1])
    i1 = int(np.clip(np.searchsorted(mat_arr, tau, side='right') - 1, 0, len(mat_arr)-2))
    i2 = i1 + 1
    j1 = np.searchsorted(mon_arr, kappa_vec, side='right') - 1
    j1 = np.clip(j1, 0, len(mon_arr)-2).astype(int)
    j2 = j1 + 1
    t = (tau - mat_arr[i1]) / (mat_arr[i2] - mat_arr[i1] + 1e-12)
    u = (kappa_vec - mon_arr[j1]) / (mon_arr[j2] - mon_arr[j1] + 1e-12)
    batch_idx = np.arange(N)
    return (1-t)*(1-u)*grid_3d[batch_idx, i1, j1] + t*(1-u)*grid_3d[batch_idx, i2, j1] + (1-t)*u*grid_3d[batch_idx, i1, j2] + t*u*grid_3d[batch_idx, i2, j2]

def evaluate_vanna_book(paths, rrs, market_t0, factor_idx_map, surfaces_dict, N_SCENARIOS, HORIZON):
    dt = 1.0 / 252.0
    N  = N_SCENARIOS

    _obj    = next(iter(surfaces_dict.values()))
    mat_arr = np.array(_obj._BA_MFPCA_Reconstruction__mat_labels, dtype=float)
    mon_arr = np.array(_obj._BA_MFPCA_Reconstruction__mon_labels, dtype=float)

    S = {s['sym']: np.full(N, market_t0['spots'][s['sym']], dtype=float) for s in rrs}
    hedge_qty  = {}
    hedge_cash = {}
    
    sigma_prev_C = {s['sym']: np.full(N, s['sigma_0_C'], dtype=float) for s in rrs}
    sigma_prev_P = {s['sym']: np.full(N, s['sigma_0_P'], dtype=float) for s in rrs}
    he_per_asset = {s['sym']: np.zeros(N, dtype=float) for s in rrs}

    for s in rrs:
        sym    = s['sym']
        init_q = -s['delta_0'] * s['n']
        hedge_qty[sym]  = np.full(N, init_q, dtype=float)
        initial_outlay = (init_q * market_t0['spots'][sym]) + (s['rr_val_0'] * s['n'])
        hedge_cash[sym] = np.full(N, -initial_outlay, dtype=float)

    for t in range(HORIZON):
        S_prev = {sym: arr.copy() for sym, arr in S.items()}

        for sym in S:
            col_idx = factor_idx_map.get(f"Return_{sym}")
            if col_idx is not None:
                S[sym] = np.clip(S[sym] * np.exp(paths[:, t, col_idx]), 1e-4, 1e10)

        g_idx = factor_idx_map.get("G_PC", [])
        g_mat = paths[:, t, g_idx] if g_idx else np.zeros((N, 3))

        for s in rrs:
            sym   = s['sym']
            K_C, K_P, n_c, r_0 = s['K_C'], s['K_P'], s['n'], s['r_0']
            tau_t = s['tau_0'] - (t + 1) * dt
            
            if tau_t <= 1e-5: continue

            hedge_cash[sym] *= np.exp(r_0 * dt)
            hfpca = surfaces_dict.get(sym)
            if hfpca is None: continue

            l_idx = factor_idx_map.get(sym, [])
            l_mat = paths[:, t, l_idx] if l_idx else np.zeros((N, 3))
            S_vec  = S[sym]
            dS_vec = S_vec - S_prev[sym]

            try:
                log_iv_grids = hfpca.reconstruct(global_param=g_mat, local_param=l_mat)
                
                kappa_vec_C    = np.log(K_C / S_vec)
                log_iv_vec_C   = np.clip(_bilinear_interp_vec(log_iv_grids, mat_arr, mon_arr, tau_t, kappa_vec_C), -10.0, 3.0)
                sigma_t_C      = np.exp(log_iv_vec_C)
                bad_C = np.isnan(sigma_t_C) | np.isinf(sigma_t_C) | (sigma_t_C <= 0)
                sigma_t_C[bad_C] = sigma_prev_C[sym][bad_C]
                
                kappa_vec_P    = np.log(K_P / S_vec)
                log_iv_vec_P   = np.clip(_bilinear_interp_vec(log_iv_grids, mat_arr, mon_arr, tau_t, kappa_vec_P), -10.0, 3.0)
                sigma_t_P      = np.exp(log_iv_vec_P)
                bad_P = np.isnan(sigma_t_P) | np.isinf(sigma_t_P) | (sigma_t_P <= 0)
                sigma_t_P[bad_P] = sigma_prev_P[sym][bad_P]
                
            except Exception: 
                sigma_t_C = sigma_prev_C[sym].copy()
                sigma_t_P = sigma_prev_P[sym].copy()

            delta_C = bs_delta_vec(S_vec, K_C, tau_t, sigma_t_C, r_0, 'C')
            delta_P = bs_delta_vec(S_vec, K_P, tau_t, sigma_t_P, r_0, 'P')
            
            target_qty = -(delta_C - delta_P) * n_c
            hedge_cash[sym] -= (target_qty - hedge_qty[sym]) * S_vec
            hedge_qty[sym]   = target_qty

            vanna_C = vanna_bs_vec(S_vec, K_C, tau_t, sigma_t_C, r_0)
            vanna_P = vanna_bs_vec(S_vec, K_P, tau_t, sigma_t_P, r_0)
            
            dsigma_C = sigma_t_C - sigma_prev_C[sym]
            dsigma_P = sigma_t_P - sigma_prev_P[sym]
            
            asset_daily_he = (vanna_C * dsigma_C - vanna_P * dsigma_P) * n_c * dS_vec
            he_per_asset[sym] += np.nan_to_num(asset_daily_he, nan=0.0, posinf=0.0, neginf=0.0)
            
            sigma_prev_C[sym] = sigma_t_C
            sigma_prev_P[sym] = sigma_t_P

    pnl_hedge  = np.zeros(N, dtype=float)
    for s in rrs:
        sym, K_C, K_P, n_c, r_0 = s['sym'], s['K_C'], s['K_P'], s['n'], s['r_0']
        S_vec = S[sym]
        pnl_hedge  += np.nan_to_num(hedge_cash[sym] + hedge_qty[sym] * S_vec, nan=0.0, posinf=0.0, neginf=0.0)

    he_tot  = sum(he_per_asset.values())

    tau_end = rrs[0]['tau_0'] - HORIZON * dt
    pnl_total_unhedged = np.zeros(N, dtype=float)
    
    for s in rrs:
        sym, K_C, K_P, n_c, r_0 = s['sym'], s['K_C'], s['K_P'], s['n'], s['r_0']
        S_vec = S[sym]
        
        term_price_C = bs_price_vec(S_vec, K_C, tau_end, sigma_prev_C[sym], r_0, 'C')
        term_price_P = bs_price_vec(S_vec, K_P, tau_end, sigma_prev_P[sym], r_0, 'P')
        
        term_rr_val = term_price_C - term_price_P
        pnl_total_unhedged += (term_rr_val - s['rr_val_0']) * n_c

    return {'he_total': he_tot, 'hedge_cash_final': pnl_hedge, 'he_per_asset': he_per_asset, 'pnl_total_unhedged': pnl_total_unhedged}

def compute_risk_metrics(pnl, alpha=0.05):
    var = float(np.percentile(pnl, alpha * 100))
    es  = float(pnl[pnl <= var].mean())
    return var, es

def bootstrap_es(pnl, alpha=0.05, n_boot=1000):
    n = len(pnl)
    idx = np.random.randint(0, n, size=(n_boot, n))
    samples = pnl[idx]
    vars_boot = np.percentile(samples, alpha * 100, axis=1)
    es_boot = np.array([samples[i, samples[i] <= vars_boot[i]].mean() for i in range(n_boot)])
    return np.percentile(es_boot, 2.5), np.percentile(es_boot, 97.5)

def plot_rolling_backtest(df):
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "savefig.dpi": 300})
    fig, ax1 = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True, facecolor="white")
    ax1.set_facecolor("white")
 
    ax1.plot(df['Date'], df['M2_HE_ES'].abs(), color="#005b96", marker='^', linestyle='-',
             linewidth=2, label='M2 Dynamic Neural Vine')
    ax1.plot(df['Date'], df['M0_HE_ES'].abs(), color="#8B0000", marker='D', linestyle='-',
             label='M0 Static Vine Copula')
    ax1.plot(df['Date'], df['Gaussian_HE_ES'].abs(), color="#2ca02c", marker='o', linestyle='-.',
             label='Gaussian Copula')
    ax1.plot(df['Date'], df['Indep_HE_ES'].abs(), color="#888888", marker='s', linestyle='--',
             label='Independence Copula')
    ax1.fill_between(df['Date'], df['M0_HE_ES'].abs(), df['M2_HE_ES'].abs(),
                     color="#005b96", alpha=0.08)
 
    ax1.set_xlabel("OOS Rebalancing Date (2025)")
    ax1.set_ylabel("95% Expected Shortfall - Hedging Error ($)")
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
 
    ax2 = ax1.twinx()
    ax2.plot(df['Date'], df['Port_Vanna'], color="#444444", linestyle=':',
             label='Portfolio Initial Vanna')
    ax2.set_ylabel("Portfolio Initial Vanna Exposure", color="#444444")
 
    ax1.set_title("Hedging Risk & Capital Allocation: M2 vs M0 vs Gaussian vs Indep",
                  loc="left", pad=10)
 
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(
        lines_1 + lines_2, labels_1 + labels_2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),   # below the figure
        ncol=3,
        frameon=True, framealpha=1,
        edgecolor="#BBBBBB", facecolor="white",
        fontsize=9,
    )
    plt.show()
 
 
def plot_component_es(asset_es_dict):
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "savefig.dpi": 300})
    sorted_assets = sorted(asset_es_dict.items(), key=lambda item: item[1])
    labels = [x[0] for x in sorted_assets]
    values = [x[1] for x in sorted_assets]
    fig, ax = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True, facecolor="white")
    ax.set_facecolor("white")
    bars = ax.barh(labels, values, color="#8B0000", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Mean Component Expected Shortfall ($)")
    ax.set_title("Vanna Risk Drivers: Asset Contribution to Tail Loss (M2 Model)",
                  loc="left", pad=10)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    x_min = min(values)
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
 
    for bar, val in zip(bars, values):
        bar_width = bar.get_width()         
        bar_left  = bar.get_x()             
        y_center  = bar.get_y() + bar.get_height() / 2
 
        label_str = f'${val:,.0f}'
        PADDING = abs(x_min) * 0.03          
        inside_x = bar_width + PADDING      
 
        bar_fraction = abs(bar_width) / (abs(x_min) + 1e-9)
        if bar_fraction < 0.12:             
            ax.text(bar_width - PADDING * 0.5, y_center,
                    label_str, va='center', ha='right', fontsize=9, color='black')
        else:                               
            ax.text(inside_x, y_center,
                    label_str, va='center', ha='left', fontsize=9, color='black')
 
    ax.set_xlim(left=x_min * 1.04)
 
    plt.show()

def main():
    PARQUET_FILE = os.path.join(DATA_DIR, "options_surfaces_data_cleaned.parquet")
    TRAIN_END    = pd.Timestamp("2024-12-31 23:59:59")
    OOS_START    = pd.Timestamp("2025-01-02")
    OOS_END      = pd.Timestamp("2025-12-29")
    N_SCENARIOS  = 10000
    HORIZON      = 20
    N_CONTRACTS  = 1000
    ALPHA        = 0.05
    SEARCH_DIRS  = [FACTOR_DIR, DATA_DIR, "", "data"]

    with open(os.path.join(RESULTS_DIR, "fitted_marginals0.pkl"), "rb") as f:
        marginals = RiskModelUnpickler(f).load()

    with open(os.path.join(RESULTS_DIR, "surfaces_dict.pkl"), "rb") as f:
        surfaces_dict = pickle.load(f)

    df_factors = load_csv_robust("hierarchical_all_factors.csv", SEARCH_DIRS)
    df_returns = load_csv_robust("returns.csv",                   SEARCH_DIRS)

    df_factors.index = pd.to_datetime(df_factors.index).normalize()
    df_returns.index = pd.to_datetime(df_returns.index).normalize()

    print(f"[-] Pre-loading parquet database into memory...")
    cols = ['underlying_symbol', 'quote_datetime', 'underlying_mid_price', 'tau', 'strike', 'option_type', 'delta', 'mid_price', 'XTM', 'log_moneyness', 'implied_volatility', 'vanna', 'vega', 'rate']
    df_parquet = pd.read_parquet(PARQUET_FILE, columns=cols)
    df_parquet['quote_datetime'] = pd.to_datetime(df_parquet['quote_datetime'])

    valid_names = [c for c in df_returns.columns if c in marginals] + [c for c in df_factors.columns if c in marginals]
    
    name_to_col = {n: i for i, n in enumerate(valid_names)}
    factor_idx_map = {'G_PC': [name_to_col[n] for n in valid_names if n.startswith("G_PC_")]}
    unique_syms = set([s.split('_')[2] for s in valid_names if s.startswith("L_PC_")])
    for sym in unique_syms:
        factor_idx_map[sym] = [name_to_col[n] for n in valid_names if n.startswith(f"L_PC_{sym}_")]
        factor_idx_map[f"Return_{sym}"] = name_to_col[sym]

    print(f"[-] Pre-computing OOS GARCH states and realized uniforms for GRU Warmup...")
    oos_dates = df_returns[(df_returns.index >= OOS_START) & (df_returns.index <= OOS_END)].index
    all_states, all_uniforms = precompute_garch_states_and_uniforms(valid_names, marginals, df_returns, df_factors, oos_dates, TRAIN_END)
    df_oos_uniforms = pd.DataFrame(all_uniforms, index=oos_dates, columns=valid_names)

    df_har_train = pd.read_csv(os.path.join(RESULTS_DIR, "uniforms_har_garch_evt_train.csv"), index_col=0, parse_dates=True)
    df_spot_train = pd.read_csv(os.path.join(RESULTS_DIR, "uniforms_ngarch_t_train.csv"), index_col=0, parse_dates=True)
    df_train_uniforms = pd.concat([df_spot_train, df_har_train], axis=1).reindex(columns=valid_names).ffill().bfill()
    df_full_uniforms = pd.concat([df_train_uniforms, df_oos_uniforms])

    print(f"[-] Initializing Models (M0 Static, M2 Neural, Gaussian)...")
    pth_path = os.path.join(RESULTS_DIR, "neural_vine_spot_har_garch_evt_model.pth.zip")
    json_path = os.path.join(RESULTS_DIR, "joint_vine_spot_har_garch_evt_model.json")
    gaussian_json_path = os.path.join(RESULTS_DIR, "gaussian_vine_spot_har_garch_evt_model.json")
    
    m0_static_copula = pv.Vinecop.from_file(json_path)
    gaussian_copula  = pv.Vinecop.from_file(gaussian_json_path)
    
    dynamic_vine     = DynamicNeuralVine(pth_path, json_path)
    
    generator = ScenarioGenerator(factor_order=valid_names)

    rolling_dates = pd.Series(oos_dates).groupby([oos_dates.year, oos_dates.month]).first().values
    rolling_dates = [pd.Timestamp(d) for d in rolling_dates]

    print(f"\n[!] Initiating Monthly Rolling Simulation over {len(rolling_dates)} steps in 2025...")
    results = []
    
    asset_es_contributions = {} 

    for current_date in rolling_dates:
        print(f"\n{'='*70}\n[OOS STEP] Advancing to: {current_date.strftime('%Y-%m-%d')}\n{'='*70}")

        market_t0 = get_initial_market_state_from_df(df_parquet, current_date)
        rrs = build_risk_reversal_book(market_t0, market_t0['df_raw'], target_tau_days=30, n_contracts=N_CONTRACTS)
        if not rrs: continue

        day_idx = oos_dates.get_loc(current_date)
        init_states = {n: all_states[n][day_idx] for n in valid_names}

        port_vanna = sum(s['vanna_0'] * s['n'] for s in rrs)
        port_vega  = sum(s['vega_0']  * s['n'] for s in rrs)
        port_gamma = sum(s['gamma_0'] * s['n'] for s in rrs)
        
        vega_vanna_ratio  = port_vega / port_vanna if port_vanna != 0 else 0
        gamma_vanna_ratio = port_gamma / port_vanna if port_vanna != 0 else 0

        hist_window = df_full_uniforms.loc[:current_date].iloc[-60:].values
        dynamic_vine.warm_up_and_push(hist_window)

        print(f"  [-] Simulating {N_SCENARIOS:,} paths forward for {HORIZON} days...")
        paths_m2       = generator.simulate(N_SCENARIOS, HORIZON, init_states, marginals, copula=dynamic_vine.base_copula)
        paths_m0       = generator.simulate(N_SCENARIOS, HORIZON, init_states, marginals, copula=m0_static_copula)
        paths_gaussian = generator.simulate(N_SCENARIOS, HORIZON, init_states, marginals, copula=gaussian_copula)
        paths_indep    = generator.simulate(N_SCENARIOS, HORIZON, init_states, marginals, copula=None)

        res_m2       = evaluate_vanna_book(paths_m2, rrs, market_t0, factor_idx_map, surfaces_dict, N_SCENARIOS, HORIZON)
        res_m0       = evaluate_vanna_book(paths_m0, rrs, market_t0, factor_idx_map, surfaces_dict, N_SCENARIOS, HORIZON)
        res_gaussian = evaluate_vanna_book(paths_gaussian, rrs, market_t0, factor_idx_map, surfaces_dict, N_SCENARIOS, HORIZON)
        res_indep    = evaluate_vanna_book(paths_indep, rrs, market_t0, factor_idx_map, surfaces_dict, N_SCENARIOS, HORIZON)

        _, es_m2_he       = compute_risk_metrics(res_m2['he_total'], ALPHA)
        _, es_m0_he       = compute_risk_metrics(res_m0['he_total'], ALPHA)
        _, es_gaussian_he = compute_risk_metrics(res_gaussian['he_total'], ALPHA)
        _, es_indep_he    = compute_risk_metrics(res_indep['he_total'], ALPHA)

        _, es_m2_unhedged = compute_risk_metrics(res_m2['pnl_total_unhedged'], ALPHA)
        _, es_m0_unhedged = compute_risk_metrics(res_m0['pnl_total_unhedged'], ALPHA)

        var_m2 = np.percentile(res_m2['he_total'], ALPHA * 100)
        tail_mask = res_m2['he_total'] <= var_m2  
        
        for sym, pnl_arr in res_m2['he_per_asset'].items():
            if sym not in asset_es_contributions:
                asset_es_contributions[sym] = []
            asset_es_contributions[sym].append(float(pnl_arr[tail_mask].mean()))

        ks_stat, ks_pval = ks_2samp(res_m2['he_total'], res_m0['he_total'])
        m2_es_low, m2_es_high = bootstrap_es(res_m2['he_total'], ALPHA)
        m0_es_low, m0_es_high = bootstrap_es(res_m0['he_total'], ALPHA)

        print(f"  [METRICS] Port Initial Vanna:   ${port_vanna:,.0f}")
        print(f"            Vega/Vanna Ratio:      {vega_vanna_ratio:,.4f}")
        print(f"            Gamma/Vanna Ratio:     {gamma_vanna_ratio:,.4f}")
        print(f"\n  [RESULT]  M2 Neural HE ES:    ${es_m2_he:,.0f}  | 95% CI: [${m2_es_low:,.0f}, ${m2_es_high:,.0f}]")
        print(f"            M0 Static HE ES:    ${es_m0_he:,.0f}  | 95% CI: [${m0_es_low:,.0f}, ${m0_es_high:,.0f}]")
        print(f"            Gaussian HE ES:     ${es_gaussian_he:,.0f}")
        print(f"            Indep HE ES:        ${es_indep_he:,.0f}")
        print(f"\n  [UNHEDGED] M2 Unhedged ES:     ${es_m2_unhedged:,.0f}")
        print(f"             M0 Unhedged ES:     ${es_m0_unhedged:,.0f}")
        print(f"\n  [STATS]   K-S Test (M2 vs M0) -> Stat: {ks_stat:.4f}, p-value: {ks_pval:.4e}")

        results.append({
            'Date': current_date,
            'M2_Unhedged_ES': es_m2_unhedged,
            'M2_HE_ES': es_m2_he,
            'M0_Unhedged_ES': es_m0_unhedged,
            'M0_HE_ES': es_m0_he,
            'Gaussian_HE_ES': es_gaussian_he,
            'Indep_HE_ES': es_indep_he, 
            'Port_Vanna': port_vanna,
            'Vega_Ratio': vega_vanna_ratio,
            'Gamma_Ratio': gamma_vanna_ratio,
            'KS_p_value': ks_pval
        })

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 155)
    print(f"{'ROLLING 2025 OUT-OF-SAMPLE SIMULATION RESULTS (M2 vs M0 vs Gaussian vs Independence)':^155}")
    print("=" * 155)
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:,.4f}" if isinstance(x, float) and abs(x) < 1 else (f"{x:,.2f}" if isinstance(x, float) else x)))
    
    plot_rolling_backtest(df_res)
    
    mean_asset_es = {sym: sum(vals)/len(vals) for sym, vals in asset_es_contributions.items()}
    plot_component_es(mean_asset_es)

if __name__ == "__main__":
    main()