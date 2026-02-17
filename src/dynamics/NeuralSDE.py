import os
import subprocess
import sys

# --- ENVIRONMENT SETUP ---
# Quietly install required packages for background execution
os.system('pip install -q optuna torch-ema')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import optuna
import json
import pickle
import warnings
import shutil
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import genpareto, gaussian_kde, kstest

# Robust tqdm for background logs
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

# Force non-interactive backend for Kaggle "Save & Run All" stability
plt.switch_backend('agg')
plt.rcParams['mathtext.fontset'] = 'cm'
np.seterr(all='ignore')
warnings.filterwarnings("ignore")

# --- SILENCE OPTUNA SPAM ---
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- HARDWARE SETUP ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- HARDWARE CHECK ---")
if device == 'cuda':
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}\n")
else:
    print("⚠️ Using CPU (GPU recommended for faster training)\n")

GLOBAL_N_LAGS = 40

# %% --- EXTREME VALUE THEORY CLASS ---

class EVT:
    """Three-region semi-parametric EVT (GPD for tails, KDE for body)"""
    def __init__(self):
        self.u_lower = None
        self.u_upper = None
        self.params_lower = None
        self.params_upper = None
        self.body_kde = None
        self.eta_l = None
        self.eta_u = None
        self.body_min_cdf = 0.0
        self.body_max_cdf = 1.0
        self._is_fitted = False

    def fit(self, z, lower_quantile=0.10, upper_quantile=0.10):
        self.eta_l = lower_quantile
        self.eta_u = upper_quantile
        sorted_z = np.sort(z)
        n = len(z)
        idx_lower = int(self.eta_l * n)
        idx_upper = int((1 - self.eta_u) * n)
        if idx_lower >= idx_upper:
            raise ValueError("Tail thresholds overlap or dataset too small.")
        self.u_lower = sorted_z[idx_lower]
        self.u_upper = sorted_z[idx_upper]

        # Lower Tail (GPD)
        lower_data = sorted_z[sorted_z < self.u_lower]
        if len(lower_data) >= 10:
            excess_lower = self.u_lower - lower_data
            self.params_lower = genpareto.fit(excess_lower, floc=0)
        else:
            self.params_lower = None

        # Upper Tail (GPD)
        upper_data = sorted_z[sorted_z > self.u_upper]
        if len(upper_data) >= 10:
            excess_upper = upper_data - self.u_upper
            self.params_upper = genpareto.fit(excess_upper, floc=0)
        else:
            self.params_upper = None

        # Body (Gaussian KDE)
        mask_body = (sorted_z >= self.u_lower) & (sorted_z <= self.u_upper)
        body_data = sorted_z[mask_body]
        if len(body_data) > 2:
            self.body_kde = gaussian_kde(body_data)
            self.body_min_cdf = self.body_kde.integrate_box_1d(-np.inf, self.u_lower)
            self.body_max_cdf = self.body_kde.integrate_box_1d(-np.inf, self.u_upper)
        else:
            self.body_kde = None

        self._is_fitted = True
        return self

    def transform(self, z):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before transform.")
        u = np.zeros_like(z, dtype=float)
        mask_l = z < self.u_lower
        if np.any(mask_l):
            if self.params_lower:
                xi, _, sigma = self.params_lower
                excess = self.u_lower - z[mask_l]
                cdf_gpd = genpareto.cdf(excess, xi, 0, sigma)
                u[mask_l] = self.eta_l * (1 - cdf_gpd)
            else:
                u[mask_l] = self.eta_l * 0.5
        mask_u = z > self.u_upper
        if np.any(mask_u):
            if self.params_upper:
                xi, _, sigma = self.params_upper
                excess = z[mask_u] - self.u_upper
                cdf_gpd = genpareto.cdf(excess, xi, 0, sigma)
                u[mask_u] = (1 - self.eta_u) + self.eta_u * cdf_gpd
            else:
                u[mask_u] = 1.0 - (self.eta_u * 0.5)
        mask_b = (~mask_l) & (~mask_u)
        if np.any(mask_b) and self.body_kde:
            raw_cdf = np.array([self.body_kde.integrate_box_1d(-np.inf, x) for x in z[mask_b]])
            target_range = (1 - self.eta_u) - self.eta_l
            raw_range = self.body_max_cdf - self.body_min_cdf
            if raw_range > 1e-9:
                u[mask_b] = self.eta_l + (raw_cdf - self.body_min_cdf) * (target_range / raw_range)
            else:
                u[mask_b] = 0.5
        elif np.any(mask_b):
            u[mask_b] = 0.5
        return np.clip(u, 1e-6, 1-1e-6)

# %% --- CORE MODEL CLASSES ---

class GRU_Encoder(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers, dropout_rate=0.0):
        super().__init__()
        gru_drop = dropout_rate if n_layers > 1 else 0.0
        self.gru = nn.GRU(n_in, n_hidden, n_layers, batch_first=True, dropout=gru_drop)

    def forward(self, x):
        self.gru.flatten_parameters()
        _, h = self.gru(x)
        return h.transpose(0, 1).flatten(start_dim=1)

class DriftNet(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers, n_out, dropout_rate=0.0):
        super().__init__()
        self.encoder = GRU_Encoder(n_in, n_hidden, n_layers, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(n_hidden * n_layers, n_out)
    def forward(self, x):
        features = self.encoder(x)
        out = self.head(self.dropout(features))
        return torch.clamp(out, min=-20.0, max=20.0)

class DiffusionNet(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers, n_out, dropout_rate=0.0):
        super().__init__()
        self.encoder = GRU_Encoder(n_in, n_hidden, n_layers, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(n_hidden * n_layers, n_out)
        self.softplus = nn.Softplus()
    def forward(self, x):
        features = self.encoder(x)
        out = self.head(self.dropout(features))
        out = torch.clamp(out, min=-20.0, max=20.0)
        return self.softplus(out) + 1e-4

class NeuralSDE(nn.Module):
    def __init__(self, params, device=device):
        super().__init__()
        self.params, self.device = params, device
        self.n_lags = params.get('n_lags', 10)
        self.alpha_pit = params.get('alpha_pit', 100.0)
        n_features, dropout_rate = params.get('n_features', 1), params.get('dropout', 0.0)
        self.pi_drift = DriftNet(n_features, params['hidden_size'], params['n_layers'], n_features, dropout_rate).to(device)
        self.pi_diff = DiffusionNet(n_features, params['hidden_size'], params['n_layers'], n_features, dropout_rate).to(device)
        self.optimizer = optim.AdamW(list(self.pi_drift.parameters()) + list(self.pi_diff.parameters()), lr=params.get('lr', 1e-3), weight_decay=params.get('weight_decay', 1e-4))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.register_buffer('sqrt_2pi', torch.tensor(2.50662827).to(device))
        self.loss_history = {'total': [], 'nll': [], 'penalty': []}

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def get_pit(self, nu, sigma, dX, dT):
        t = torch.clamp((dX - nu * dT) / (sigma * torch.sqrt(dT)), min=-20.0, max=20.0)
        return 0.5 + t * (t**2 + 6) / (2 * (t**2 + 4)**1.5)

    def density_penalty(self, pit_values):
        batch_size, n_features = pit_values.shape
        u_grid = torch.linspace(0, 1, 100, device=self.device).view(1, -1)
        du, error = 1.0 / 100, 0.0
        for i in range(n_features):
            p_i = pit_values[:, i].view(-1, 1)
            h = 1.06 * torch.std(p_i.detach()) * (batch_size ** -0.2) + 1e-5
            phi_z = torch.exp(-0.5 * ((u_grid - p_i) / h)**2) / self.sqrt_2pi
            f_hat = (1.0 / (batch_size * h)) * phi_z.sum(dim=0)
            error += torch.sum((f_hat - 1.0)**2 * du)
        return error / n_features

    def train_step(self, data_batch, dX, dT, loss_type='combined'):
        self.pi_drift.train(); self.pi_diff.train()
        self.optimizer.zero_grad()
        nu, sigma = self.pi_drift(data_batch), self.pi_diff(data_batch)
        if torch.isnan(nu).any() or torch.isnan(sigma).any(): return float('inf'), float('inf'), float('inf')
        dist = torch.distributions.StudentT(torch.tensor(4.0, device=self.device), loc=nu * dT, scale=sigma * torch.sqrt(dT))
        nll = -dist.log_prob(dX).mean()
        if torch.isnan(nll) or torch.isinf(nll): return float('inf'), float('inf'), float('inf')
        loss, penalty_val = 0.0, 0.0
        if loss_type == 'mse': loss = torch.mean((nu * dT - dX)**2)
        elif loss_type == 'combined':
            pit_vals = self.get_pit(nu, sigma, dX, dT)
            penalty_val = self.density_penalty(pit_vals)
            loss = nll + self.alpha_pit * penalty_val
        if torch.isnan(loss) or torch.isinf(loss): return float('inf'), float('inf'), float('inf')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pi_drift.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.pi_diff.parameters(), 1.0)
        self.optimizer.step(); self.scheduler.step()
        return loss.item(), nll.item(), (penalty_val if isinstance(penalty_val, float) else penalty_val.item())

# %% --- HELPER FUNCTIONS ---

def prepare_data(data, n_lags, device):
    data = torch.tensor(data, dtype=torch.float32).to(device)
    T_len = data.shape[0]

    if T_len <= n_lags:
        raise ValueError(f"Data length {T_len} is too short for n_lags={n_lags}")

    # FIX: Slice to [:-1] to maintain the first valid window and prevent day-skipping
    data_seq = data.unfold(0, n_lags, 1)[:-1].transpose(1, 2)
    
    # FIX: Shift targets and next_vals to perfectly align with Day 40
    targets = data[n_lags-1:-1, :]
    next_vals = data[n_lags:, :]
    
    dX = next_vals - targets
    dT = torch.ones((dX.shape[0], 1)).to(device) * (1.0/252.0)

    return data_seq, dX, dT, data

def train_model(model, data_numpy, n_epochs=1000, batch_size=256, verbose=True, trial=None, val_data=None):
    try:
        X, dX, dT, _ = prepare_data(data_numpy, model.n_lags, model.device)
        if val_data is not None: X_val, dX_val, dT_val, _ = prepare_data(val_data, model.n_lags, model.device)
    except: return None, None, None, None
    iterator = tqdm(range(n_epochs), desc="  ↳ Training", leave=False) if verbose else range(n_epochs)
    for epoch in iterator:
        idx = torch.randperm(X.shape[0])[:batch_size]
        loss, nll, pen = model.train_step(X[idx], dX[idx], dT[idx], 'mse' if epoch < (n_epochs*0.1) else 'combined')
        if loss == float('inf'):
            if trial: raise optuna.exceptions.TrialPruned()
            break
        model.loss_history['total'].append(loss); model.loss_history['nll'].append(nll); model.loss_history['penalty'].append(pen)
        if verbose and epoch % 10 == 0: iterator.set_postfix({'Loss': f"{loss:.4f}"})
    return X, dX, dT, data_numpy

def optimize_hyperparameters(data_numpy, column_name, n_trials=75):
    print(f"  ↳ Running Bayesian Optimization ({n_trials} trials)...", end="", flush=True)
    tscv = TimeSeriesSplit(n_splits=3)
    def objective(trial):
        cfg = {'n_features': 1, 'n_lags': GLOBAL_N_LAGS, 'hidden_size': trial.suggest_categorical("hidden_size", [8, 16, 32]), 
               'n_layers': trial.suggest_int("n_layers", 1, 2), 'alpha_pit': trial.suggest_float("alpha_pit", 50, 150), 
               'lr': trial.suggest_float("lr", 1e-4, 1e-3, log=True), 'dropout': trial.suggest_float("dropout", 0.1, 0.4), 
               'weight_decay': 1e-4}
        scores = []
        for train_idx, val_idx in tscv.split(data_numpy):
            m = NeuralSDE(cfg, device=device)
            v_start = max(0, val_idx[0] - GLOBAL_N_LAGS)
            train_model(m, data_numpy[train_idx], 150, 128, False, trial, data_numpy[v_start:val_idx[-1]+1])
            try:
                X_v, dX_v, dT_v, _ = prepare_data(data_numpy[v_start:val_idx[-1]+1], GLOBAL_N_LAGS, device)
                with torch.no_grad():
                    nu, sig = m.pi_drift(X_v), m.pi_diff(X_v)
                    scores.append((-torch.distributions.StudentT(torch.tensor(4.0, device=device), nu*dT_v, sig*torch.sqrt(dT_v)).log_prob(dX_v).mean() + cfg['alpha_pit'] * m.density_penalty(m.get_pit(nu, sig, dX_v, dT_v))).item())
            except: scores.append(1e6)
        return np.mean(scores)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=20))
    study.optimize(objective, n_trials=n_trials)
    print(" Done!"); return study.best_params

def fit_evt_on_residuals(Z_numpy):
    U, models = np.zeros_like(Z_numpy), []
    for i in range(Z_numpy.shape[1]):
        evt = EVT().fit(Z_numpy[:, i])
        U[:, i] = evt.transform(Z_numpy[:, i])
        models.append(evt)
    return U, models

def run_diagnostics(model, X, dX, dT, raw_data, column_name, save_path=None, U_numpy=None, phase="Train"):
    model.eval()
    
    # --- 1. PLOT CONVERGENCE ---
    losses = model.loss_history
    if losses['total'] and phase == "Train":
        df_loss = pd.DataFrame(losses)
        window = min(50, len(df_loss))
        df_loss['total_smooth'] = df_loss['total'].rolling(window=window, min_periods=1).mean()
        df_loss['penalty_smooth'] = df_loss['penalty'].rolling(window=window, min_periods=1).mean()

        fig, ax1 = plt.subplots(figsize=(10, 4))
        color = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Total Loss (SymLog)', color=color)
        ax1.plot(df_loss['total'], color=color, alpha=0.15)
        ax1.plot(df_loss['total_smooth'], color=color, linewidth=2)
        ax1.set_yscale('symlog')
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('PIT Penalty', color=color)
        ax2.plot(df_loss['penalty'], color=color, alpha=0.15, linestyle='--')
        ax2.plot(df_loss['penalty_smooth'], color=color, linewidth=2, linestyle='--')
        
        plt.title(f"Training Convergence - {column_name}")
        if save_path: 
            plt.savefig(os.path.join(save_path, "convergence.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # --- 2. GET PIT VALUES ---
    with torch.no_grad():
        nu, sig = model.pi_drift(X), model.pi_diff(X)
        pit = model.get_pit(nu, sig, dX, dT).cpu().numpy()
        
    pit_flat = pit[:, 0]
    ks_pit = kstest(pit_flat, 'uniform')[1]
    
    # --- 3. PLOT PRE-EVT PIT DIST ---
    plt.figure(figsize=(6, 4))
    plt.hist(pit_flat, bins=30, density=True, color='purple', alpha=0.6, edgecolor='black')
    plt.axhline(1.0, color='red', lw=2, linestyle='--')
    plt.title(f"PIT Distribution (Pre-EVT) - {column_name} ({phase})")
    plt.text(0.98, 0.97, f'KS p-value: {ks_pit:.4f}', transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    if save_path: 
        plt.savefig(os.path.join(save_path, f"pit_dist_{phase.lower()}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 4. PLOT POST-EVT UNIFORMS ---
    if U_numpy is not None:
        u_flat = U_numpy[:, 0]
        ks_evt = kstest(u_flat, 'uniform')[1]
        
        plt.figure(figsize=(6, 4))
        plt.hist(u_flat, bins=30, density=True, color='orange', alpha=0.6, edgecolor='black')
        plt.axhline(1.0, color='red', lw=2, linestyle='--')
        plt.title(f"EVT Uniforms (Post-EVT) - {column_name} ({phase})")
        plt.text(0.98, 0.97, f'KS p-value: {ks_evt:.4f}', transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', horizontalalignment='right', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        if save_path: 
            plt.savefig(os.path.join(save_path, f"evt_uniforms_dist_{phase.lower()}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ↳ [{phase}] KS p: Pre-EVT={ks_pit:.4f}, Post-EVT={ks_evt:.4f}")
        return ks_evt
        
    return ks_pit

def save_artifacts(model, X, dX, dT, params, col, base):
    folder = os.path.join(base, col); os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/best_nsde_params.json", "w") as f: json.dump(params, f, indent=4)
    torch.save(model.state_dict(), f"{folder}/best_nsde_model.pth")
    with torch.no_grad():
        nu, sig = model.pi_drift(X), model.pi_diff(X)
        Z = ((dX - nu * dT) / (sig * torch.sqrt(dT))).cpu().numpy()
    np.save(f"{folder}/nsde_standardized_residuals_train.npy", Z)
    return folder, Z

def get_predictions_and_residuals(model, data_numpy, dates, n_lags=40):
    X, dX, dT, _ = prepare_data(data_numpy, n_lags, model.device)
    model.eval()
    with torch.no_grad():
        nu, sig = model.pi_drift(X), model.pi_diff(X)
        Z = ((dX - nu * dT) / (sig * torch.sqrt(dT))).cpu().numpy()
    # FIX: Slice shifted to match new prepare_data logic
    return Z, dates[n_lags:]

# %% --- MAIN EXECUTION ---

if __name__ == "__main__":
    df_full = pd.read_csv('/kaggle/input/datasets/lucaleimbeckdelduca/factors/factors.csv', index_col=0, parse_dates=True).dropna().select_dtypes(include=[np.number])
    split_date = pd.Timestamp("2025-01-02")
    train_df, test_df = df_full[df_full.index < split_date], df_full[df_full.index >= split_date]
    results_base_dir = "results_nsde_train_test"
    os.makedirs(results_base_dir, exist_ok=True)
    
    all_best_params, failed_uniformity = {}, []
    all_uniforms_train, all_uniforms_test = {}, {}
    total_cols, alpha = len(df_full.columns), 0.05

    for idx, col in enumerate(df_full.columns, 1):
        print(f"▶ [{idx}/{total_cols}] Processing: {col}")
        col_train = train_df[[col]].values
        
        best_p = optimize_hyperparameters(col_train, col, n_trials=75)
        all_best_params[col] = best_p
        
        final_model = NeuralSDE({**best_p, 'n_features': 1, 'n_lags': GLOBAL_N_LAGS}, device=device)
        X_tr, dX_tr, dT_tr, _ = train_model(final_model, col_train, 800, best_p.get('batch_size', 128))
        
        if X_tr is None: continue
        
        # --- TRAIN SET EVT ---
        Z_tr, tr_dates = get_predictions_and_residuals(final_model, col_train, train_df.index, GLOBAL_N_LAGS)
        U_tr, evt_models = fit_evt_on_residuals(Z_tr)
        
        save_p, _ = save_artifacts(final_model, X_tr, dX_tr, dT_tr, best_p, col, results_base_dir)
        run_diagnostics(final_model, X_tr, dX_tr, dT_tr, col_train, col, save_path=save_p, U_numpy=U_tr, phase="Train")

        # --- TEST SET EVALUATION ---
        # Generate the test inputs by appending the holdout set to the last n_lags of the train set
        test_inputs = np.vstack([col_train[-GLOBAL_N_LAGS:], test_df[[col]].values])
        test_index = train_df.index[-GLOBAL_N_LAGS:].append(test_df.index)
        
        Z_te, te_dates = get_predictions_and_residuals(final_model, test_inputs, test_index, GLOBAL_N_LAGS)

        # --- STATIC EVT FOR TEST SET (MATCHES HAR/NGARCH METHODOLOGY) ---
        U_te = np.zeros_like(Z_te)
        for i in range(Z_te.shape[1]):
            U_te[:, i] = evt_models[i].transform(Z_te[:, i])

        try:
            # Diagnostics on the test set
            X_te_d, dX_te_d, dT_te_d, _ = prepare_data(test_inputs, GLOBAL_N_LAGS, final_model.device)
            p_val = run_diagnostics(final_model, X_te_d, dX_te_d, dT_te_d, None, col, save_path=save_p, U_numpy=U_te, phase="Test")
            
            if p_val < alpha: 
                failed_uniformity.append((col, p_val))
                
        except Exception as e: 
            print(f"  ↳ ⚠️ Test Diag Fail: {e}")

        all_uniforms_train[col], all_uniforms_test[col] = U_tr[:, 0], U_te[:, 0]
        
        # Memory cleanup between iterations
        del final_model
        torch.cuda.empty_cache()
        print("-" * 40)

    # ==========================================================
    # --- AGGREGATE AND SAVE (MATCHING HAR/NGARCH INDEXING) ---
    # ==========================================================
    with open(f"{results_base_dir}/all_columns_params.json", "w") as f: 
        json.dump(all_best_params, f, indent=4)
        
    if all_uniforms_train: 
        train_df_out = pd.DataFrame(all_uniforms_train, index=tr_dates)
        train_df_out.index = pd.to_datetime(train_df_out.index).date
        train_df_out.index.name = "Date"
        train_df_out.to_csv(f"{results_base_dir}/uniforms_nsde_train.csv")
        
    if all_uniforms_test: 
        test_df_out = pd.DataFrame(all_uniforms_test, index=te_dates)
        test_df_out.index = pd.to_datetime(test_df_out.index).date
        test_df_out.index.name = "Date"
        test_df_out.to_csv(f"{results_base_dir}/uniforms_nsde_test.csv")
    
    shutil.make_archive('nsde_results', 'zip', results_base_dir)
    
    print("\n" + "="*50)
    print("📊 FINAL UNIFORMITY REPORT (Test Set)")
    print("="*50)
    if not failed_uniformity:
        print(f"✅ All {total_cols} columns passed uniformity (p > {alpha}).")
    else:
        print(f"❌ {len(failed_uniformity)}/{total_cols} columns failed (p < {alpha}):")
        for c, p in failed_uniformity: print(f"   - {c:.<20} p: {p:.5f}")
    print("="*50)
    print(f"✅ Processing Complete. Archive: nsde_results.zip")
