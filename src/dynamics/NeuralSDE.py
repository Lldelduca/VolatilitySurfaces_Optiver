import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import copy
import optuna
from sklearn.model_selection import TimeSeriesSplit
import json
import os
import pickle
from EVT import EVT

# %%
class GRU_Encoder(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers):
        super().__init__()
        self.gru = nn.GRU(n_in, n_hidden, n_layers, batch_first=True)
        
    def forward(self, x):
        self.gru.flatten_parameters()
        _, h = self.gru(x)
        return h.transpose(0, 1).flatten(start_dim=1)

class DriftNet(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers, n_out):
        super().__init__()
        self.encoder = GRU_Encoder(n_in, n_hidden, n_layers)
        self.head = nn.Linear(n_hidden * n_layers, n_out)
        
    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)

class DiffusionNet(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers, n_out):
        super().__init__()
        self.encoder = GRU_Encoder(n_in, n_hidden, n_layers)
        self.head = nn.Linear(n_hidden * n_layers, n_out)
        self.softplus = nn.Softplus() 
        
    def forward(self, x):
        features = self.encoder(x)
        return self.softplus(self.head(features)) + 1e-4

class NeuralSDE(nn.Module):
    def __init__(self, params, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.params = params
        self.device = device
        self.n_lags = params.get('n_lags', 10)
        self.alpha_pit = params.get('alpha_pit', 100.0)
        
        n_features = params.get('n_features', 1)

        self.pi_drift = DriftNet(
            n_in=n_features, 
            n_hidden=params['hidden_size'], 
            n_layers=params['n_layers'], 
            n_out=n_features
        ).to(device)
        
        self.pi_diff = DiffusionNet(
            n_in=n_features, 
            n_hidden=params['hidden_size'], 
            n_layers=params['n_layers'], 
            n_out=n_features
        ).to(device)
        
        self.optimizer = optim.AdamW(
            list(self.pi_drift.parameters()) + list(self.pi_diff.parameters()),
            lr=params.get('lr', 1e-3)
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        self.register_buffer('sqrt_2', torch.tensor(1.41421356).to(device))
        self.register_buffer('sqrt_2pi', torch.tensor(2.50662827).to(device))
        self.loss_history = {'total': [], 'nll': [], 'penalty': []}

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def get_pit(self, nu, sigma, dX, dT):
        std_res = (dX - nu * dT) / (sigma * torch.sqrt(dT))
        return 0.5 * (1 + torch.erf(std_res / self.sqrt_2))

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
        
        dist = torch.distributions.Normal(nu * dT, sigma * torch.sqrt(dT))
        nll = -dist.log_prob(dX).mean()
        
        loss = 0.0
        penalty_val = 0.0
        
        if loss_type == 'mse':
            loss = torch.mean((nu * dT - dX)**2)
        elif loss_type == 'combined':
            pit_vals = self.get_pit(nu, sigma, dX, dT)
            penalty_val = self.density_penalty(pit_vals)
            loss = nll + self.alpha_pit * penalty_val

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.pi_drift.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.pi_diff.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        p_val = penalty_val if isinstance(penalty_val, float) else penalty_val.item()
        return loss.item(), nll.item(), p_val

# %%

def prepare_data(data, n_lags, device):
    data = torch.tensor(data, dtype=torch.float32).to(device)
    T_len = data.shape[0]
    
    if T_len <= n_lags + 1:
        raise ValueError(f"Data length {T_len} is too short for n_lags={n_lags}")

    data_seq = torch.zeros((T_len - n_lags - 1, n_lags, data.shape[1])).to(device)
    for i in range(T_len - n_lags - 1):
        data_seq[i, :, :] = data[i : i + n_lags, :]
        
    targets = data[n_lags:-1, :]
    next_vals = data[n_lags+1:, :]
    dX = next_vals - targets
    dT = torch.ones((dX.shape[0], 1)).to(device) * (1.0/252.0)
    
    return data_seq, dX, dT, data

def train_model(model, data_numpy, n_epochs=1000, batch_size=256, verbose=True):
    if verbose:
        print(f"Starting Training on Device: {model.device}")
        
    try:
        X, dX, dT, raw_data = prepare_data(data_numpy, model.n_lags, model.device)
    except ValueError as e:
        if verbose: print(e)
        return None, None, None, None

    dataset_size = X.shape[0]
    iterator = tqdm(range(n_epochs), desc="Training") if verbose else range(n_epochs)
    
    for epoch in iterator:
        indices = torch.randperm(dataset_size)[:batch_size]
        batch_X = X[indices]
        batch_dX = dX[indices]
        batch_dT = dT[indices]
        
        mode = 'mse' if epoch < (n_epochs * 0.1) else 'combined'
        loss, nll, pen = model.train_step(batch_X, batch_dX, batch_dT, loss_type=mode)
        
        model.loss_history['total'].append(loss)
        model.loss_history['nll'].append(nll)
        model.loss_history['penalty'].append(pen)
        
        if verbose and epoch % 10 == 0:
            iterator.set_postfix({'Loss': f"{loss:.4f}", 'PIT': f"{pen:.4f}", 'Mode': mode})

    return X, dX, dT, raw_data


def optimize_hyperparameters(data_numpy, column_name, n_trials=20):
    print(f"\n--- Starting Optimization for '{column_name}' ({n_trials} trials) ---")

    tscv = TimeSeriesSplit(n_splits=3)
    
    def objective(trial):
        n_lags = trial.suggest_int("n_lags", 5, min(100, len(data_numpy)//3))
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        alpha_pit = trial.suggest_float("alpha_pit", 10.0, 200.0)
        
        config = {
            'n_features': data_numpy.shape[1], # Should be 1
            'n_lags': n_lags,
            'hidden_size': hidden_size,
            'n_layers': n_layers,
            'alpha_pit': alpha_pit,
            'lr': lr
        }
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(data_numpy)):
            
            train_data = data_numpy[train_idx]
            
            # Adjust validation start to include lookback window
            val_start_adjusted = max(0, val_idx[0] - n_lags)
            val_data_extended = data_numpy[val_start_adjusted : val_idx[-1] + 1]
            
            model = NeuralSDE(config)
            
            try:
                # Reduced epochs for optimization speed
                train_model(model, train_data, n_epochs=100, batch_size=256, verbose=False)
            except Exception:
                return float('inf')
            
            try:
                X_val, dX_val, dT_val, _ = prepare_data(val_data_extended, n_lags, model.device)
            except ValueError:
                return float('inf')
                
            model.pi_drift.eval()
            model.pi_diff.eval()
            
            with torch.no_grad():
                nu = model.pi_drift(X_val)
                sigma = model.pi_diff(X_val)
                
                dist = torch.distributions.Normal(nu * dT_val, sigma * torch.sqrt(dT_val))
                val_nll = -dist.log_prob(dX_val).mean()
                
                pit_vals = model.get_pit(nu, sigma, dX_val, dT_val)
                val_penalty = model.density_penalty(pit_vals)
                
                total_val_loss = val_nll + alpha_pit * val_penalty
                fold_scores.append(total_val_loss.item())
        
        return np.mean(fold_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n--- Optimization Finished for '{column_name}' ---")
    print(f"Best Params: {study.best_params}")
    return study.best_params

def run_diagnostics(model, X, dX, dT, raw_data, column_name, save_path=None):
    model.pi_drift.eval()
    model.pi_diff.eval()
    
    losses = model.loss_history
    df_loss = pd.DataFrame(losses)
    window = 50  
    df_loss['total_smooth'] = df_loss['total'].rolling(window=window).mean()
    df_loss['penalty_smooth'] = df_loss['penalty'].rolling(window=window).mean()
    
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
        plt.savefig(os.path.join(save_path, "convergence.png"))
    plt.close()

    with torch.no_grad():
        nu = model.pi_drift(X)
        sigma = model.pi_diff(X)
        pit = model.get_pit(nu, sigma, dX, dT).cpu().numpy()
        
    # Plot PIT Histogram for single feature
    plt.figure(figsize=(6, 4))
    plt.hist(pit[:, 0], bins=30, density=True, color='purple', alpha=0.6, edgecolor='black')
    plt.axhline(1.0, color='red', lw=2, linestyle='--')
    plt.title(f"PIT Distribution - {column_name}")
    if save_path:
        plt.savefig(os.path.join(save_path, "pit_dist.png"))
    plt.close()

    # Dynamics Check
    lookback = 200
    with torch.no_grad():
        subset_X = X[-lookback:]
        subset_dT = dT[-lookback:]
        all_drift = model.pi_drift(subset_X).cpu().numpy()
        all_vol = model.pi_diff(subset_X).cpu().numpy()
        
    dt_vals = subset_dT.cpu().numpy().flatten()
    
    actual_levels = raw_data[-lookback:, 0].cpu().numpy()
    prev_levels = raw_data[-lookback-1:-1, 0].cpu().numpy()
    
    expected_path = prev_levels + all_drift[:, 0] * dt_vals
    margin = 1.96 * all_vol[:, 0] * np.sqrt(dt_vals)
    
    plt.figure(figsize=(10, 5))
    plt.plot(actual_levels, 'k.-', label='Actual', lw=1, alpha=0.7)
    plt.plot(expected_path, 'r--', label='Drift', lw=1)
    plt.fill_between(range(lookback), 
                     expected_path - margin, 
                     expected_path + margin, 
                     color='green', alpha=0.2, label='95% Band')
    plt.title(f"Dynamics Check - {column_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, "dynamics.png"))
    plt.close()

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

def save_artifacts(model, X, dX, dT, params, column_name, base_folder="results"):
    # Create column-specific folder
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
        
    np.save(f"{folder}/nsde_standardized_residuals.npy", Z_numpy)
    
    U_numpy, evt_models = fit_evt_on_residuals(Z_numpy)
        
    np.save(f"{folder}/nsde_uniforms.npy", U_numpy)
        
    with open(f"{folder}/evt_models.pkl", "wb") as f:
        pickle.dump(evt_models, f)
        
    print(f"   [Saved] {column_name} -> {folder}")
    return folder

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "all_factors.csv")
    
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        print("Make sure the CSV is in the same folder as this script.")
        exit()

    df_full = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df_full = df_full.dropna()
    df_full = df_full.select_dtypes(include=[np.number])
    
    print(f"Total Data Shape: {df_full.shape}")
    print(f"Columns to process: {list(df_full.columns)}")
    
    results_base_dir = os.path.join(script_dir, "results_per_column")
    os.makedirs(results_base_dir, exist_ok=True)

    # --- MAIN LOOP PER COLUMN ---
    for col in df_full.columns:
        print(f"\n{'='*50}")
        print(f"PROCESSING COLUMN: {col}")
        print(f"{'='*50}")
        
        # Extract single column as (T, 1) array
        col_data = df_full[[col]].values
        
        # 1. Optimize Hyperparameters for this specific column
        # Increase n_trials for better results in production
        best_params = optimize_hyperparameters(col_data, col, n_trials=20)
        
        # 2. Configure Model
        best_config = best_params.copy()
        best_config['n_features'] = 1  # Strictly univariate
        
        # 3. Train Final Model
        print(f"Training Final Model for {col}...")
        final_model = NeuralSDE(best_config)
        X_train, dX_train, dT_train, raw_data_train = train_model(
            final_model, 
            col_data, 
            n_epochs=1000, # Full training epochs
            verbose=True
        )
        
        # 4. Save Artifacts & Diagnostics
        if X_train is not None:
            save_path = save_artifacts(final_model, X_train, dX_train, dT_train, best_params, col, base_folder=results_base_dir)
            
            run_diagnostics(final_model, X_train, dX_train, dT_train, raw_data_train, col, save_path=save_path)
