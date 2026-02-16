import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm # Use tqdm from standard terminal if not in Jupyter: from tqdm import tqdm
import copy
import optuna
from sklearn.model_selection import TimeSeriesSplit
import json
import os
import pickle
from scipy.stats import genpareto, gaussian_kde, kstest
import warnings

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

        # Lower Tail
        mask_l = z < self.u_lower
        if np.any(mask_l):
            if self.params_lower:
                xi, _, sigma = self.params_lower
                excess = self.u_lower - z[mask_l]
                cdf_gpd = genpareto.cdf(excess, xi, 0, sigma)
                u[mask_l] = self.eta_l * (1 - cdf_gpd)
            else:
                u[mask_l] = self.eta_l * 0.5

        # Upper Tail
        mask_u = z > self.u_upper
        if np.any(mask_u):
            if self.params_upper:
                xi, _, sigma = self.params_upper
                excess = z[mask_u] - self.u_upper
                cdf_gpd = genpareto.cdf(excess, xi, 0, sigma)
                u[mask_u] = (1 - self.eta_u) + self.eta_u * cdf_gpd
            else:
                u[mask_u] = 1.0 - (self.eta_u * 0.5)

        # Body
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

# %% --- CORE MODEL CLASSES (WITH CLAMPING & DROPOUT) ---

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
        self.params = params
        self.device = device
        self.n_lags = params.get('n_lags', 10)
        self.alpha_pit = params.get('alpha_pit', 100.0)

        n_features = params.get('n_features', 1)
        dropout_rate = params.get('dropout', 0.0)

        self.pi_drift = DriftNet(
            n_in=n_features,
            n_hidden=params['hidden_size'],
            n_layers=params['n_layers'],
            n_out=n_features,
            dropout_rate=dropout_rate
        ).to(device)

        self.pi_diff = DiffusionNet(
            n_in=n_features,
            n_hidden=params['hidden_size'],
            n_layers=params['n_layers'],
            n_out=n_features,
            dropout_rate=dropout_rate
        ).to(device)

        self.optimizer = optim.AdamW(
            list(self.pi_drift.parameters()) + list(self.pi_diff.parameters()),
            lr=params.get('lr', 1e-3),
            weight_decay=params.get('weight_decay', 1e-4)
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

        self.register_buffer('sqrt_2pi', torch.tensor(2.50662827).to(device))
        self.loss_history = {'total': [], 'nll': [], 'penalty': []}

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def get_pit(self, nu, sigma, dX, dT):
        t = (dX - nu * dT) / (sigma * torch.sqrt(dT))
        t = torch.clamp(t, min=-20.0, max=20.0)
        return 0.5 + t * (t**2 + 6) / (2 * (t**2 + 4)**1.5)

    def density_penalty(self, pit_values):
        batch_size, n_features = pit_values.shape
        u_grid = torch.linspace(0, 1, 100, device=self.device).view(1, -1)
        du = 1.0 / 100

        error = 0.0
        for i in range(n_features):
            p_i = pit_values[:, i].view(-1, 1)
            std_dev = torch.std(p_i.detach())
            h = 1.06 * std_dev * (batch_size ** -0.2) + 1e-5
            z = (u_grid - p_i) / h
            phi_z = torch.exp(-0.5 * z**2) / self.sqrt_2pi
            f_hat = (1.0 / (batch_size * h)) * phi_z.sum(dim=0)
            error += torch.sum((f_hat - 1.0)**2 * du)

        return error / n_features

    def train_step(self, data_batch, dX, dT, loss_type='combined'):
        self.pi_drift.train()
        self.pi_diff.train()
        self.optimizer.zero_grad()

        nu = self.pi_drift(data_batch)
        sigma = self.pi_diff(data_batch)

        if torch.isnan(nu).any() or torch.isnan(sigma).any():
            return float('inf'), float('inf'), float('inf')

        df_tensor = torch.tensor(4.0, device=self.device)
        dist = torch.distributions.StudentT(df_tensor, loc=nu * dT, scale=sigma * torch.sqrt(dT))
        nll = -dist.log_prob(dX).mean()

        if torch.isnan(nll) or torch.isinf(nll):
            return float('inf'), float('inf'), float('inf')

        loss = 0.0
        penalty_val = 0.0

        if loss_type == 'mse':
            loss = torch.mean((nu * dT - dX)**2)
        elif loss_type == 'combined':
            pit_vals = self.get_pit(nu, sigma, dX, dT)
            penalty_val = self.density_penalty(pit_vals)
            loss = nll + self.alpha_pit * penalty_val

        if torch.isnan(loss) or torch.isinf(loss):
            return float('inf'), float('inf'), float('inf')

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.pi_drift.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.pi_diff.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()

        p_val = penalty_val if isinstance(penalty_val, float) else penalty_val.item()
        return loss.item(), nll.item(), p_val

# %% --- TRAINING & OPTIMIZATION FUNCTIONS ---

def prepare_data(data, n_lags, device):
    data = torch.tensor(data, dtype=torch.float32).to(device)
    T_len = data.shape[0]

    if T_len <= n_lags + 1:
        raise ValueError(f"Data length {T_len} is too short for n_lags={n_lags}")

    data_seq = data.unfold(0, n_lags, 1)[:-2].transpose(1, 2)

    targets = data[n_lags:-1, :]
    next_vals = data[n_lags+1:, :]
    dX = next_vals - targets
    dT = torch.ones((dX.shape[0], 1)).to(device) * (1.0/252.0)

    return data_seq, dX, dT, data

def train_model(model, data_numpy, n_epochs=1000, batch_size=256, verbose=True, trial=None, val_data=None):
    try:
        X, dX, dT, raw_data = prepare_data(data_numpy, model.n_lags, model.device)
        if val_data is not None:
            X_val, dX_val, dT_val, _ = prepare_data(val_data, model.n_lags, model.device)
    except ValueError as e:
        if verbose: print(f"  ↳ ⚠️ {e}")
        return None, None, None, None

    dataset_size = X.shape[0]
    # Use tqdm only if verbose, nicely formatted
    iterator = tqdm(range(n_epochs), desc="  ↳ Training", leave=False) if verbose else range(n_epochs)

    for epoch in iterator:
        model.pi_drift.train()
        model.pi_diff.train()

        indices = torch.randperm(dataset_size)[:batch_size]
        batch_X = X[indices]
        batch_dX = dX[indices]
        batch_dT = dT[indices]

        mode = 'mse' if epoch < (n_epochs * 0.1) else 'combined'
        loss, nll, pen = model.train_step(batch_X, batch_dX, batch_dT, loss_type=mode)

        if loss == float('inf'):
            if trial is not None:
                raise optuna.exceptions.TrialPruned()
            else:
                if verbose: print("\n  ↳ ⚠️ Warning: Final model generated NaN. Stopping early.")
                break

        model.loss_history['total'].append(loss)
        model.loss_history['nll'].append(nll)
        model.loss_history['penalty'].append(pen)

        if verbose and epoch % 10 == 0:
            iterator.set_postfix({'Loss': f"{loss:.4f}", 'PIT': f"{pen:.4f}"})

        # --- OPTUNA PRUNING LOGIC ---
        if trial is not None and val_data is not None and epoch % 10 == 0:
            model.pi_drift.eval()
            model.pi_diff.eval()
            with torch.no_grad():
                nu = model.pi_drift(X_val)
                sigma = model.pi_diff(X_val)

                if torch.isnan(nu).any() or torch.isnan(sigma).any():
                    raise optuna.exceptions.TrialPruned()

                df_tensor = torch.tensor(4.0, device=model.device)
                dist = torch.distributions.StudentT(df_tensor, loc=nu * dT_val, scale=sigma * torch.sqrt(dT_val))
                val_nll = -dist.log_prob(dX_val).mean()

                pit_vals = model.get_pit(nu, sigma, dX_val, dT_val)
                val_penalty = model.density_penalty(pit_vals)

                total_val_loss = val_nll + model.alpha_pit * val_penalty

            if torch.isnan(total_val_loss) or torch.isinf(total_val_loss):
                raise optuna.exceptions.TrialPruned()

            trial.report(total_val_loss.item(), epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return X, dX, dT, raw_data

def optimize_hyperparameters(data_numpy, column_name, n_trials=75):
    print(f"  ↳ Running Bayesian Optimization ({n_trials} trials)...", end="", flush=True)
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [8, 16, 32, 64])
        n_layers = trial.suggest_int("n_layers", 1, 3)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        alpha_pit = trial.suggest_float("alpha_pit", 1.0, 200.0)
        dropout = trial.suggest_float("dropout", 0.0, 0.6)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        config = {
            'n_features': 1, 'n_lags': GLOBAL_N_LAGS, 'hidden_size': hidden_size,
            'n_layers': n_layers, 'alpha_pit': alpha_pit, 'lr': lr,
            'batch_size': batch_size, 'dropout': dropout, 'weight_decay': weight_decay
        }

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(data_numpy)):
            train_data = data_numpy[train_idx]
            val_start_adjusted = max(0, val_idx[0] - GLOBAL_N_LAGS)
            val_data_extended = data_numpy[val_start_adjusted : val_idx[-1] + 1]

            model = NeuralSDE(config, device=device)

            try:
                train_model(
                    model, train_data, n_epochs=200, batch_size=batch_size,
                    verbose=False, trial=trial, val_data=val_data_extended
                )
            except optuna.exceptions.TrialPruned:
                raise optuna.exceptions.TrialPruned()
            except Exception:
                return float('inf')

            try:
                X_val, dX_val, dT_val, _ = prepare_data(val_data_extended, GLOBAL_N_LAGS, model.device)
            except ValueError:
                return float('inf')

            model.pi_drift.eval()
            model.pi_diff.eval()

            with torch.no_grad():
                nu = model.pi_drift(X_val)
                sigma = model.pi_diff(X_val)

                if torch.isnan(nu).any() or torch.isnan(sigma).any():
                    raise optuna.exceptions.TrialPruned()

                df_tensor = torch.tensor(4.0, device=model.device)
                dist = torch.distributions.StudentT(df_tensor, loc=nu * dT_val, scale=sigma * torch.sqrt(dT_val))
                val_nll = -dist.log_prob(dX_val).mean()

                pit_vals = model.get_pit(nu, sigma, dX_val, dT_val)
                val_penalty = model.density_penalty(pit_vals)

                total_val_loss = val_nll + alpha_pit * val_penalty

                if torch.isnan(total_val_loss) or torch.isinf(total_val_loss):
                    raise optuna.exceptions.TrialPruned()

                fold_scores.append(total_val_loss.item())

        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=30)
    )
    study.optimize(objective, n_trials=n_trials)

    print(" Done!")
    return study.best_params


def fit_evt_on_residuals(Z_numpy):
    n_samples, n_features = Z_numpy.shape
    U_numpy = np.zeros_like(Z_numpy)
    evt_models = []

    for i in range(n_features):
        z_col = Z_numpy[:, i]
        evt = EVT()
        evt.fit(z_col, lower_quantile=0.10, upper_quantile=0.10)
        u_col = evt.transform(z_col)
        evt_models.append(evt)
        U_numpy[:, i] = u_col

    return U_numpy, evt_models

# %% --- DIAGNOSTICS & SAVING ---

def run_diagnostics(model, X, dX, dT, raw_data, column_name, save_path=None, U_numpy=None, phase="Train"):
    model.pi_drift.eval()
    model.pi_diff.eval()

    losses = model.loss_history
    df_loss = pd.DataFrame(losses)
    window = 50
    df_loss['total_smooth'] = df_loss['total'].rolling(window=window).mean()
    df_loss['penalty_smooth'] = df_loss['penalty'].rolling(window=window).mean()

    # 1. Convergence Plot
    fig, ax1 = plt.subplots(figsize=(10, 4))
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Total Loss (SymLog)', color=color)
    if 'total' in df_loss:
        ax1.plot(df_loss['total'], color=color, alpha=0.15)
        ax1.plot(df_loss['total_smooth'], color=color, linewidth=2)
    ax1.set_yscale('symlog')
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('PIT Penalty', color=color)
    if 'penalty' in df_loss:
        ax2.plot(df_loss['penalty'], color=color, alpha=0.15, linestyle='--')
        ax2.plot(df_loss['penalty_smooth'], color=color, linewidth=2, linestyle='--')
    plt.title(f"Training Convergence - {column_name}")
    if save_path: plt.savefig(os.path.join(save_path, "convergence.png"), dpi=150)
    # plt.show() # Uncomment if you want plots to pop up; usually suppressed during long loops
    plt.close()

    # 2. PIT Histogram (Pre-EVT)
    with torch.no_grad():
        nu = model.pi_drift(X)
        sigma = model.pi_diff(X)
        pit = model.get_pit(nu, sigma, dX, dT).cpu().numpy()

    pit_flat = pit[:, 0]
    ks_stat_pit, ks_pval_pit = kstest(pit_flat, 'uniform')
    print(f"  ↳ [KS Test {phase}] Pre-EVT Stat: {ks_stat_pit:.4f} (p={ks_pval_pit:.4f})", end="")
    ks_results = {'pit_ks_stat': float(ks_stat_pit), 'pit_ks_pval': float(ks_pval_pit)}

    # 2b. EVT Uniforms Histogram (Post-EVT)
    if U_numpy is not None:
        u_flat = U_numpy[:, 0]
        ks_stat_evt, ks_pval_evt = kstest(u_flat, 'uniform')
        print(f" | Post-EVT Stat: {ks_stat_evt:.4f} (p={ks_pval_evt:.4f})")
        ks_results['evt_ks_stat'] = float(ks_stat_evt)
        ks_results['evt_ks_pval'] = float(ks_pval_evt)
    else:
        print() # Line break if no Post-EVT

    if save_path:
        with open(os.path.join(save_path, f"ks_test_results_{phase.lower()}.json"), "w") as f:
            json.dump(ks_results, f, indent=4)


def save_artifacts(model, X, dX, dT, params, column_name, base_folder):
    folder = os.path.join(base_folder, column_name)
    os.makedirs(folder, exist_ok=True)

    with open(f"{folder}/best_nsde_params.json", "w") as f:
        json.dump(params, f, indent=4)

    torch.save(model.state_dict(), f"{folder}/best_nsde_model.pth")

    model.pi_drift.eval()
    model.pi_diff.eval()

    with torch.no_grad():
        nu = model.pi_drift(X)
        sigma = model.pi_diff(X)
        Z = (dX - nu * dT) / (sigma * torch.sqrt(dT))
        Z_numpy = Z.cpu().numpy()

    np.save(f"{folder}/nsde_standardized_residuals_train.npy", Z_numpy)
    return folder, Z_numpy


def get_predictions_and_residuals(model, data_numpy, dates, n_lags=40):
    data_tensor = torch.tensor(data_numpy, dtype=torch.float32).to(model.device)
    T_len = data_tensor.shape[0]

    data_seq = data_tensor.unfold(0, n_lags, 1)[:-2].transpose(1, 2)

    targets = data_tensor[n_lags:-1, :]
    next_vals = data_tensor[n_lags+1:, :]
    dX = next_vals - targets
    dT = torch.ones((dX.shape[0], 1)).to(model.device) * (1.0/252.0)

    model.pi_drift.eval()
    model.pi_diff.eval()

    with torch.no_grad():
        nu = model.pi_drift(data_seq)
        sigma = model.pi_diff(data_seq)
        Z = (dX - nu * dT) / (sigma * torch.sqrt(dT))

    Z_numpy = Z.cpu().numpy()
    residual_dates = dates[n_lags+1:]

    return Z_numpy, residual_dates


# %% --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":

    df_full = pd.read_csv('all_factors.csv', index_col=0, parse_dates=True)
    df_full = df_full.dropna()
    df_full = df_full.select_dtypes(include=[np.number])

    print(f"Total Data Shape: {df_full.shape}")
    print(f"Date Range: {df_full.index[0].date()} to {df_full.index[-1].date()}\n")

    split_date = pd.Timestamp("2025-01-02")
    train_df = df_full[df_full.index < split_date]
    test_df = df_full[df_full.index >= split_date]

    print(f"Train Set: {train_df.shape} | Test Set: {test_df.shape}")
    print(f"⭐ Using GLOBAL_N_LAGS = {GLOBAL_N_LAGS}\n")

    results_base_dir = "results_nsde_train_test"
    os.makedirs(results_base_dir, exist_ok=True)

    all_best_params = {}
    all_uniforms_train = {}
    all_uniforms_test = {}

    total_cols = len(df_full.columns)

    for idx, col in enumerate(df_full.columns, 1):
        print(f"▶ [{idx}/{total_cols}] Processing: {col}")

        col_train_data = train_df[[col]].values
        col_full_data = df_full[[col]].values

        best_params = optimize_hyperparameters(col_train_data, col, n_trials=75)
        all_best_params[col] = best_params

        best_config = best_params.copy()
        best_config['n_features'] = 1
        best_config['n_lags'] = GLOBAL_N_LAGS

        final_model = NeuralSDE(best_config, device=device)
        best_batch_size = best_config.get('batch_size', 256)

        X_train, dX_train, dT_train, raw_data_train = train_model(
            final_model,
            col_train_data,
            n_epochs=800,
            batch_size=best_batch_size,
            verbose=True
        )

        if X_train is None:
            print(f"  ↳ ⚠️ Training failed for {col}")
            print("-" * 40)
            continue

        Z_train, train_dates = get_predictions_and_residuals(
            final_model,
            col_train_data,
            train_df.index,
            n_lags=GLOBAL_N_LAGS
        )

        try:
            U_train, evt_models = fit_evt_on_residuals(Z_train)
        except Exception as e:
            print(f"  ↳ ⚠️ EVT fitting failed: {str(e)}")
            U_train, evt_models = None, None
            continue

        save_path, _ = save_artifacts(
            final_model, X_train, dX_train, dT_train, best_params, col,
            base_folder=results_base_dir
        )

        run_diagnostics(
            final_model, X_train, dX_train, dT_train, raw_data_train, col,
            save_path=save_path, U_numpy=U_train, phase="Train"
        )

        # Test Set Residuals
        Z_test, test_dates = get_predictions_and_residuals(
            final_model,
            col_full_data,
            df_full.index,
            n_lags=GLOBAL_N_LAGS
        )

        test_indices = [i for i, d in enumerate(test_dates) if d >= split_date]
        Z_test_subset = Z_test[test_indices]
        test_dates_subset = test_dates[test_indices]

        # Rolling EVT
        U_test = np.zeros_like(Z_test_subset)
        window_size = 252
        Z_combined = np.vstack([Z_train, Z_test_subset])
        test_start_idx = len(Z_train)

        for i in range(Z_test_subset.shape[1]):
            for t in range(len(Z_test_subset)):
                current_idx = test_start_idx + t
                window_residuals = Z_combined[current_idx - window_size : current_idx, i]

                local_evt = EVT()
                try:
                    local_evt.fit(window_residuals, lower_quantile=0.10, upper_quantile=0.10)
                    U_test[t, i] = local_evt.transform(np.array([Z_test_subset[t, i]]))[0]
                except:
                    U_test[t, i] = evt_models[i].transform(np.array([Z_test_subset[t, i]]))[0]

        try:
            if len(U_test) >= 50:
                n_diag = min(len(U_test), len(Z_test))
                X_test_approx, dX_test_approx, dT_test_approx, _ = prepare_data(
                    col_full_data[-n_diag-100:, :], GLOBAL_N_LAGS, final_model.device
                )
                n_use = min(n_diag, X_test_approx.shape[0])
                X_test_approx = X_test_approx[-n_use:, :, :]
                dX_test_approx = dX_test_approx[-n_use:, :]
                dT_test_approx = dT_test_approx[-n_use:, :]
                U_test_diag = U_test[-n_use:] if n_use <= len(U_test) else U_test

                run_diagnostics(
                    final_model, X_test_approx, dX_test_approx, dT_test_approx, None, col,
                    save_path=save_path, U_numpy=U_test_diag.reshape(-1, 1), phase="Test"
                )
            else:
                print(f"  ↳ ⚠️ Insufficient test samples for diagnostics ({len(U_test)} < 50)")
        except Exception as e:
            print(f"  ↳ ⚠️ Test diagnostics skipped: {str(e)}")

        all_uniforms_train[col] = U_train[:, 0]
        all_uniforms_test[col] = U_test[:, 0]
        print("-" * 40)

    print("\n✅ All columns processed. Saving aggregated parameters & uniforms...")

    json_path = os.path.join(results_base_dir, "all_columns_params.json")
    with open(json_path, "w") as f:
        json.dump(all_best_params, f, indent=4)

    csv_path = os.path.join(results_base_dir, "all_columns_params.csv")
    param_df = pd.DataFrame.from_dict(all_best_params, orient='index')
    param_df.index.name = 'Column'
    param_df.to_csv(csv_path)

    if all_uniforms_train:
        uniforms_train_df = pd.DataFrame(all_uniforms_train, index=train_dates[GLOBAL_N_LAGS+1:])
        uniforms_train_df.index.name = 'Date'
        uniforms_train_csv = os.path.join(results_base_dir, "uniforms_nsde_train.csv")
        uniforms_train_df.to_csv(uniforms_train_csv)

    if all_uniforms_test:
        uniforms_test_df = pd.DataFrame(all_uniforms_test, index=test_dates_subset)
        uniforms_test_df.index.name = 'Date'
        uniforms_test_csv = os.path.join(results_base_dir, "uniforms_nsde_test.csv")
        uniforms_test_df.to_csv(uniforms_test_csv)

    print(f"✅ Processing Complete. Results saved in: ./{results_base_dir}")