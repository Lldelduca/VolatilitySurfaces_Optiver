import json
import os
import warnings
import random

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genpareto, gaussian_kde, kstest
from scipy.stats import t as scipy_t
from scipy.special import gammaln
from sklearn.model_selection import TimeSeriesSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch_ema import ExponentialMovingAverage

import optuna
from tqdm.auto import tqdm

from src.dynamics.EVT import EVT


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

seed_everything(42)

plt.switch_backend('agg')
plt.rcParams['mathtext.fontset'] = 'cm'
np.seterr(all='ignore')
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Initialized Neural SDE with device: {device}")

GLOBAL_N_LAGS = 40


# =============================================================================
# --- Architecture ---
# =============================================================================

class GRU_Encoder(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers, dropout_rate=0.0):
        super().__init__()
        gru_drop = dropout_rate if n_layers > 1 else 0.0
        self.gru = nn.GRU(n_in, n_hidden, n_layers, batch_first=True, dropout=gru_drop)
        self.out_dim = n_hidden * n_layers + n_hidden

    def forward(self, x):
        self.gru.flatten_parameters()
        seq_out, h = self.gru(x)
        h_flat = h.transpose(0, 1).flatten(start_dim=1)
        pooled = seq_out.mean(dim=1)
        return torch.cat([h_flat, pooled], dim=1)


class DriftNet(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers, n_out, dropout_rate=0.0):
        super().__init__()
        self.encoder = GRU_Encoder(n_in, n_hidden, n_layers, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(self.encoder.out_dim, n_out)

    def forward(self, x):
        return torch.clamp(self.head(self.dropout(self.encoder(x))), -20.0, 20.0)


class DiffusionNet(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers, n_out, dropout_rate=0.0):
        super().__init__()
        self.encoder = GRU_Encoder(n_in, n_hidden, n_layers, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(self.encoder.out_dim, n_out)
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(torch.clamp(self.head(self.dropout(self.encoder(x))), -20.0, 20.0)) + 1e-4


class NeuralSDE(nn.Module):
    def __init__(self, params, device=device):
        super().__init__()
        self.params = params
        self.device = device
        self.n_lags = params.get('n_lags', GLOBAL_N_LAGS)
        self.alpha_pit = params.get('alpha_pit', 100.0)
        n_features = params.get('n_features', 1)
        dropout_rate = params.get('dropout', 0.0)

        self.pi_drift = DriftNet(n_features, params['hidden_size'], params['n_layers'], n_features, dropout_rate).to(device)
        self.pi_diff = DiffusionNet(n_features, params['hidden_size'], params['n_layers'], n_features, dropout_rate).to(device)
        self.log_nu = nn.Parameter(torch.tensor(np.log(4.0), dtype=torch.float32, device=device))

        all_params = list(self.pi_drift.parameters()) + list(self.pi_diff.parameters()) + [self.log_nu]

        self.optimizer = optim.AdamW(all_params, lr=params.get('lr', 1e-3), weight_decay=params.get('weight_decay', 1e-4))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.ema = ExponentialMovingAverage(all_params, decay=0.999)
        self.scaler = GradScaler()
        self.sqrt_2pi = torch.tensor(2.50662827, device=device)
        self.loss_history = {'total': [], 'nll': [], 'penalty': []}

    @property
    def nu(self):
        return torch.clamp(torch.exp(self.log_nu), min=2.1, max=30.0)

    def _studentt_cdf(self, x, loc, scale):
        class _CDF(torch.autograd.Function):
            @staticmethod
            def forward(ctx, z, nu_val):
                z_np = z.detach().cpu().float().numpy()
                cdf_np = scipy_t.cdf(z_np, df=nu_val).astype(np.float32)
                pdf_np = scipy_t.pdf(z_np, df=nu_val).astype(np.float32)
                ctx.save_for_backward(torch.from_numpy(pdf_np).to(z.device))
                return torch.from_numpy(cdf_np).to(z.device)

            @staticmethod
            def backward(ctx, grad_out):
                (pdf,) = ctx.saved_tensors
                return grad_out * pdf, None

        z = (x - loc) / scale
        cdf = _CDF.apply(z, float(self.nu.item()))
        return torch.clamp(cdf, 1e-6, 1 - 1e-6)

    def get_pit(self, nu, sigma, dX, dT):
        return self._studentt_cdf(dX, loc=nu * dT, scale=sigma * torch.sqrt(dT))

    def density_penalty(self, pit_values):
        batch_size, n_features = pit_values.shape
        n_grid = 100
        u_grid = torch.linspace(0, 1, n_grid, device=self.device)
        du = 1.0 / n_grid
        error = 0.0
        for i in range(n_features):
            p_i = pit_values[:, i]
            h = 1.06 * p_i.detach().std() * (batch_size ** -0.2) + 1e-5
            phi = torch.exp(-0.5 * ((u_grid.unsqueeze(0) - p_i.unsqueeze(1)) / h) ** 2) / self.sqrt_2pi
            f_hat = phi.sum(dim=0) / (batch_size * h)
            error += ((f_hat - 1.0) ** 2 * du).sum()
        return error / n_features

    def train_step(self, data_batch, dX, dT, loss_type='combined'):
        self.pi_drift.train()
        self.pi_diff.train()
        self.optimizer.zero_grad()

        with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
            nu = self.pi_drift(data_batch)
            sigma = self.pi_diff(data_batch)
            if torch.isnan(nu).any() or torch.isnan(sigma).any():
                return float('inf'), float('inf'), float('inf')

            dist = torch.distributions.StudentT(self.nu, loc=nu * dT, scale=sigma * torch.sqrt(dT))
            nll = -dist.log_prob(dX).mean()
            if torch.isnan(nll) or torch.isinf(nll):
                return float('inf'), float('inf'), float('inf')

            penalty_val = 0.0
            if loss_type == 'mse':
                loss = torch.mean((nu * dT - dX) ** 2)
            else:
                pit_vals = self.get_pit(nu, sigma, dX, dT)
                penalty_val = self.density_penalty(pit_vals)
                loss = nll + self.alpha_pit * penalty_val

            if torch.isnan(loss) or torch.isinf(loss):
                return float('inf'), float('inf'), float('inf')

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.pi_drift.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.pi_diff.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.ema.update()

        return (loss.item(), nll.item(), penalty_val if isinstance(penalty_val, float) else penalty_val.item())


# =============================================================================
# --- Data Preparation ---
# =============================================================================

def prepare_data(data, n_lags, dev):
    data = torch.tensor(data, dtype=torch.float32).to(dev)
    T_len = data.shape[0]
    if T_len <= n_lags:
        raise ValueError(f"Data length {T_len} is too short for n_lags={n_lags}")
    data_seq = data.unfold(0, n_lags, 1)[:-1].transpose(1, 2)
    targets = data[n_lags - 1:-1, :]
    next_vals = data[n_lags:, :]
    dX = next_vals - targets
    dT = torch.ones((dX.shape[0], 1), device=dev) * (1.0 / 252.0)
    return data_seq, dX, dT


def train_model(model, data_numpy, n_epochs=1000, batch_size=256, verbose=True, trial=None):
    X, dX, dT = prepare_data(data_numpy, model.n_lags, model.device)
    iterator = tqdm(range(n_epochs), desc="Training", leave=False) if verbose else range(n_epochs)
    for epoch in iterator:
        idx = torch.randperm(X.shape[0])[:batch_size]
        loss_type = 'mse' if epoch < (n_epochs * 0.1) else 'combined'
        loss, nll, pen = model.train_step(X[idx], dX[idx], dT[idx], loss_type)
        if loss == float('inf'):
            if trial:
                raise optuna.exceptions.TrialPruned()
            break
        model.loss_history['total'].append(loss)
        model.loss_history['nll'].append(nll)
        model.loss_history['penalty'].append(pen)
        if verbose and epoch % 10 == 0:
            iterator.set_postfix({'Loss': f"{loss:.4f}", 'nu': f"{model.nu.item():.2f}"})
    return X, dX, dT


def get_residuals(model, data_numpy, n_lags):
    """Extract standardized residuals z = (dX - mu*dT) / (sigma*sqrt(dT))."""
    X, dX, dT = prepare_data(data_numpy, n_lags, model.device)
    with model.ema.average_parameters():
        model.eval()
        with torch.no_grad():
            nu = model.pi_drift(X)
            sig = model.pi_diff(X)
            Z = ((dX - nu * dT) / (sig * torch.sqrt(dT))).cpu().numpy()
    return Z, X, dX, dT


def compute_oos_metrics(model, X, dX, dT):
    """Compute OOS log-likelihood and MSE using the learned Student-t distribution."""
    with model.ema.average_parameters():
        model.eval()
        with torch.no_grad():
            nu_pred = model.pi_drift(X)
            sig_pred = model.pi_diff(X)
            mse = torch.mean((nu_pred * dT - dX) ** 2).item()
            loglik = torch.distributions.StudentT(
                model.nu, loc=nu_pred * dT, scale=sig_pred * torch.sqrt(dT)
            ).log_prob(dX).mean().item()
    return mse, loglik


# =============================================================================
# --- Hyperparameter Optimization ---
# =============================================================================

def optimize_hyperparameters(data_numpy, column_name, n_trials=75):
    print(f"  HPO for {column_name} ({n_trials} trials, 3-fold TSCV)...")
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        cfg = {
            'n_features': 1, 'n_lags': GLOBAL_N_LAGS,
            'hidden_size': trial.suggest_categorical("hidden_size", [8, 16, 32]),
            'n_layers': trial.suggest_int("n_layers", 1, 2),
            'alpha_pit': trial.suggest_float("alpha_pit", 50, 150),
            'lr': trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            'dropout': trial.suggest_float("dropout", 0.1, 0.4),
            'weight_decay': 1e-4,
        }
        scores = []
        for train_idx, val_idx in tscv.split(data_numpy):
            m = NeuralSDE(cfg, device=device)
            train_model(m, data_numpy[train_idx], 150, 128, False, trial)
            try:
                v_start = max(0, val_idx[0] - GLOBAL_N_LAGS)
                X_v, dX_v, dT_v = prepare_data(data_numpy[v_start:val_idx[-1] + 1], GLOBAL_N_LAGS, device)
                with torch.no_grad():
                    nu, sig = m.pi_drift(X_v), m.pi_diff(X_v)
                    nll_v = -torch.distributions.StudentT(m.nu, nu * dT_v, sig * torch.sqrt(dT_v)).log_prob(dX_v).mean()
                    pen_v = m.density_penalty(m.get_pit(nu, sig, dX_v, dT_v))
                    scores.append((nll_v + cfg['alpha_pit'] * pen_v).item())
            except Exception:
                scores.append(1e6)
        return np.mean(scores)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)
    
    print(f"    Best: hidden={study.best_params.get('hidden_size')}, "
          f"layers={study.best_params.get('n_layers')}, "
          f"lr={study.best_params.get('lr', 0):.5f}")
    return study.best_params


# =============================================================================
# --- Diagnostics ---
# =============================================================================

def plot_fitted_curve(model, data_numpy, dates, column_name, n_lags, save_path, phase="Train"):
    X, dX, dT = prepare_data(data_numpy, n_lags, model.device)
    with model.ema.average_parameters():
        model.eval()
        with torch.no_grad():
            nu_all = model.pi_drift(X).cpu().numpy()
            sig_all = model.pi_diff(X).cpu().numpy()

    dt_val = 1.0 / 252.0
    nu_val = float(model.nu.item())
    actual = data_numpy[n_lags:, 0]
    center = data_numpy[n_lags - 1:-1, 0] + nu_all[:, 0] * dt_val
    scale = sig_all[:, 0] * np.sqrt(dt_val)
    plot_dates = dates[n_lags:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})
    ax = axes[0]
    for level, color in [(0.50, '#cce5ff'), (0.90, '#66b0ff'), (0.95, '#1a6fc4')]:
        tail = (1 - level) / 2
        lo = scipy_t.ppf(tail, df=nu_val, loc=center, scale=scale)
        hi = scipy_t.ppf(1 - tail, df=nu_val, loc=center, scale=scale)
        ax.fill_between(plot_dates, lo, hi, alpha=0.35, color=color, label=f'{int(level * 100)}% CI')
    ax.plot(plot_dates, actual, color='#e63946', lw=0.8, alpha=0.9, label='Actual')
    ax.plot(plot_dates, center, color='#1d3557', lw=1.3, alpha=0.95, label='Fitted mean')
    ax.set_title(f"NeuralSDE [{phase}]: {column_name}", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    residuals = actual - center
    colors = np.where(residuals >= 0, '#2a9d8f', '#e76f51')
    ax2.bar(plot_dates, residuals, width=1, color=colors, alpha=0.65)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_title("Residuals", fontsize=10)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{column_name}_{phase.lower()}_fitted.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_diagnostics(u_values, column_name, phase, save_path):
    """PIT histogram with KS test — matches HAR/NGARCH diagnostic style."""
    ks_stat, ks_p = kstest(u_values, 'uniform')
    
    plt.figure(figsize=(6, 4))
    plt.hist(u_values, bins=50, density=True, alpha=0.6, color='purple', ec='black')
    plt.axhline(1.0, color='r', ls='--', lw=2, label='Uniform')
    plt.text(0.05, 0.95, f'KS: {ks_stat:.3f}\np={ks_p:.3f}',
             transform=plt.gca().transAxes, va='top',
             bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
    plt.title(f'NSDE Uniform Transform ({phase}): {column_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_path, f"{column_name}_{phase.lower()}_pit.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return ks_p


# =============================================================================
# --- Main Pipeline ---
# =============================================================================

if __name__ == "__main__":
    seed_everything(42)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    file_path = os.path.join(project_root, "results", "factors", "factors.csv")
    res_dir = os.path.join(project_root, "results", "dynamics", "NSDE")
    diag_dir = os.path.join(res_dir, "plots")
    os.makedirs(diag_dir, exist_ok=True)

    SPLIT_DATE = pd.Timestamp("2025-01-02")
    KS_ALPHA = 0.05

    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        exit(1)

    df_full = (pd.read_csv(file_path, index_col=0, parse_dates=True)
               .dropna()
               .select_dtypes(include=[np.number]))

    train_df = df_full[df_full.index < SPLIT_DATE]
    test_df = df_full[df_full.index >= SPLIT_DATE]
    holdout_days = len(test_df)

    print(f"Processing {df_full.shape[1]} factors.")
    print(f"Train: {train_df.shape[0]} days | Test: {test_df.shape[0]} days\n")

    # Storage
    all_uniforms_train = {}
    all_uniforms_test = {}
    all_params = []
    all_best_hpo = {}
    ks_train_pass, ks_test_pass = 0, 0
    total_cols = len(df_full.columns)

    for idx, col in enumerate(df_full.columns, 1):
        print(f"[{idx}/{total_cols}] {col}")

        col_train = train_df[[col]].values
        col_test_with_context = np.vstack([col_train[-GLOBAL_N_LAGS:], test_df[[col]].values])

        # --- Step 1: HPO ---
        best_p = optimize_hyperparameters(col_train, col, n_trials=75)
        all_best_hpo[col] = best_p

        # --- Step 2: Final training ---
        seed_everything(42)
        final_cfg = {**best_p, 'n_features': 1, 'n_lags': GLOBAL_N_LAGS}
        model = NeuralSDE(final_cfg, device=device)
        X_tr, dX_tr, dT_tr = train_model(model, col_train, n_epochs=800, batch_size=best_p.get('batch_size', 128))

        # --- Step 3: Extract residuals ---
        Z_tr, _, _, _ = get_residuals(model, col_train, GLOBAL_N_LAGS)
        Z_te, _, _, _ = get_residuals(model, col_test_with_context, GLOBAL_N_LAGS)
        
        tr_dates = train_df.index[GLOBAL_N_LAGS:]
        te_dates = test_df.index

        # --- Step 4: EVT transform ---
        evt = EVT()
        evt.fit(Z_tr[:, 0], lower_quantile=0.10, upper_quantile=0.10)

        U_tr = np.clip(evt.transform(Z_tr[:, 0]), 1e-6, 1 - 1e-6)
        U_te = np.clip(evt.transform(Z_te[:, 0]), 1e-6, 1 - 1e-6)

        all_uniforms_train[col] = pd.Series(U_tr, index=tr_dates)
        all_uniforms_test[col] = pd.Series(U_te, index=te_dates)

        # --- Step 5: OOS metrics ---
        mse_tr, loglik_tr = compute_oos_metrics(model, X_tr, dX_tr, dT_tr)
        
        mse_te, loglik_te = float('nan'), float('nan')
        try:
            X_te, dX_te, dT_te = prepare_data(col_test_with_context, GLOBAL_N_LAGS, model.device)
            mse_te, loglik_te = compute_oos_metrics(model, X_te, dX_te, dT_te)
        except Exception as e:
            print(f"  Warning: OOS metrics failed for {col}: {e}")

        # --- Step 6: Diagnostics ---
        ks_p_train = plot_diagnostics(U_tr, col, "Train", diag_dir)
        ks_p_test = plot_diagnostics(U_te, col, "Test", diag_dir)
        
        plot_fitted_curve(model, col_train, train_df.index, col, GLOBAL_N_LAGS, diag_dir, "Train")
        try:
            plot_fitted_curve(model, col_test_with_context, 
                              train_df.index[-GLOBAL_N_LAGS:].append(test_df.index),
                              col, GLOBAL_N_LAGS, diag_dir, "Test")
        except Exception:
            pass

        if ks_p_train >= KS_ALPHA: ks_train_pass += 1
        if ks_p_test >= KS_ALPHA: ks_test_pass += 1

        train_passed = "PASS" if ks_p_train >= KS_ALPHA else "FAIL"
        test_passed = "PASS" if ks_p_test >= KS_ALPHA else "FAIL"
        print(f"  KS train p={ks_p_train:.3f} {train_passed} | KS test p={ks_p_test:.3f} {test_passed}")

        all_params.append({
            'factor': col,
            'nu': round(model.nu.item(), 3),
            'hidden_size': best_p.get('hidden_size'),
            'n_layers': best_p.get('n_layers'),
            'alpha_pit': round(best_p.get('alpha_pit', 0), 1),
            'lr': best_p.get('lr'),
            'dropout': best_p.get('dropout'),
            'loglikelihood': round(loglik_tr, 4),
            'loglikelihood_oos': round(loglik_te, 4),
            'mse_train': round(mse_tr, 6),
            'mse_test': round(mse_te, 6),
            'ks_p_train': round(ks_p_train, 4),
            'ks_p_test': round(ks_p_test, 4),
        })

        # Save per-factor model weights
        factor_dir = os.path.join(res_dir, "models", col)
        os.makedirs(factor_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(factor_dir, "model.pth"))
        model.ema.store()
        model.ema.copy_to()
        torch.save(model.state_dict(), os.path.join(factor_dir, "model_ema.pth"))
        model.ema.restore()

        # Cleanup GPU memory
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # --- Save CSVs (matching HAR_GARCH and NGARCH format) ---
    train_out = pd.DataFrame(all_uniforms_train)
    test_out = pd.DataFrame(all_uniforms_test)
    train_out.index = pd.to_datetime(train_out.index).date
    test_out.index = pd.to_datetime(test_out.index).date
    train_out.index.name = "Date"
    test_out.index.name = "Date"

    train_out.to_csv(os.path.join(res_dir, "uniforms_nsde_train.csv"))
    test_out.to_csv(os.path.join(res_dir, "uniforms_nsde_test.csv"))

    p_df = pd.DataFrame(all_params)
    p_df.to_csv(os.path.join(res_dir, "params_nsde.csv"), index=False)

    with open(os.path.join(res_dir, "best_hpo_params.json"), "w") as f:
        json.dump(all_best_hpo, f, indent=4)

    # --- Uniformity Report (matching HAR/NGARCH format) ---
    print(f"\n{'='*50}")
    print(f"FINAL UNIFORMITY REPORT (Neural SDE)")
    print(f"{'='*50}")
    print(f"  Train: {ks_train_pass}/{total_cols} passed (p >= {KS_ALPHA})")
    print(f"  Test:  {ks_test_pass}/{total_cols} passed (p >= {KS_ALPHA})")
    print(f"{'='*50}")
    print(f"\nResults saved to: {res_dir}")