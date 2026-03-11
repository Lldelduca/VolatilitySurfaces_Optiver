import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.special
from scipy.stats import kendalltau, norm
from tqdm import tqdm
import pandas as pd
import pyvinecopulib as pv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import random
import optuna
from joblib import Parallel, delayed
import sys

torch.set_default_dtype(torch.float64)

# ====================================================================================
# --- CUSTOM AUTOGRAD FOR STUDENT-T ---
# ====================================================================================
class InverseStudentT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, nu):
        u_cpu = u.detach().cpu().numpy()
        nu_cpu = nu.detach().cpu().numpy()
        u_cpu = np.clip(u_cpu, 1e-12, 1 - 1e-12)
        x = np.asarray(scipy.special.stdtrit(nu_cpu, u_cpu))
        x_tensor = torch.from_numpy(x).to(u.device, dtype=u.dtype)
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
        grad_nu = None # Simplified for speed
        return grad_u, grad_nu

def inverse_t_cdf(u, nu):
    return InverseStudentT.apply(u, nu)

# ====================================================================================
# --- NEURAL PAIR COPULA MODEL ---
# ====================================================================================
class NeuralPairCopula(nn.Module):
    def __init__(self, family, rotation=0, hidden_dim=8, num_layers=1, dropout=0.0):
        super().__init__()
        self.family = str(family).split('.')[-1].lower()
        self.rotation = int(rotation)
        
        # SOTA GRU Architecture
        self.rnn = nn.GRU(
            input_size=2, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0.0, 
            batch_first=True
        )
        self.head = nn.Linear(hidden_dim, 1)
        self.f_init = nn.Parameter(torch.tensor(0.0))
        
        if 'student' in self.family:
            self.nu_param = nn.Parameter(torch.tensor(2.0))
        else:
            self.register_parameter('nu_param', None)

    def warm_start(self, static_theta, static_nu=None):
        if static_theta is not None:
            f_init = 0.0
            if 'gaussian' in self.family or 'student' in self.family:
                val = static_theta / 0.999
                f_init = float(np.arctanh(np.clip(val, -0.999, 0.999)))
            elif 'clayton' in self.family:
                f_init = float(np.log(np.exp(max(static_theta - 1e-5, 1e-6)) - 1))
            elif 'gumbel' in self.family:
                f_init = float(np.log(np.exp(max(static_theta - 1.0001, 1e-6)) - 1))
            elif 'frank' in self.family:
                f_init = float(static_theta)

            with torch.no_grad():
                self.f_init.copy_(torch.tensor(f_init))
                if static_nu is not None and self.nu_param is not None:
                    nu_init = float(np.log(np.exp(max(static_nu - 2.01, 1e-6)) - 1))
                    self.nu_param.copy_(torch.tensor(nu_init))

    def get_nu(self):
        if self.nu_param is None: return None
        return torch.nn.functional.softplus(self.nu_param) + 2.01

    def rotate_data(self, u, v):
        if self.rotation == 90: return 1-u, v
        if self.rotation == 180: return 1-u, 1-v
        if self.rotation == 270: return u, 1-v
        return u, v
    
    def transform_parameter(self, f_t):
        if 'gaussian' in self.family or 'student' in self.family:
            return torch.tanh(f_t) * 0.999 
        elif 'clayton' in self.family:
            return torch.nn.functional.softplus(f_t) + 1e-5 
        elif 'gumbel' in self.family:
            return torch.nn.functional.softplus(f_t) + 1.0001 
        elif 'frank' in self.family:
            val = f_t 
            mask = torch.abs(val) < 1e-4
            val = torch.where(mask, torch.sign(val + 1e-12) * 1e-4, val)
            return val
        return f_t

    def log_likelihood_pair(self, u, v, theta, nu=None):
        u_rot, v_rot = self.rotate_data(u, v)
        eps = 1e-9
        u_rot = torch.clamp(u_rot, eps, 1 - eps)
        v_rot = torch.clamp(v_rot, eps, 1 - eps)

        if 'gaussian' in self.family:
            rho = theta
            n = torch.distributions.Normal(0, 1)
            x, y = n.icdf(u_rot), n.icdf(v_rot)
            z = x**2 + y**2 - 2*rho*x*y
            log_det = 0.5 * torch.log(1 - rho**2 + 1e-8)
            log_exp = -0.5 * (z / (1 - rho**2 + 1e-8) - (x**2 + y**2))
            return -log_det + log_exp

        elif 'student' in self.family:
            rho = theta
            x = inverse_t_cdf(u_rot, nu)
            y = inverse_t_cdf(v_rot, nu)
            zeta = (x**2 + y**2 - 2*rho*x*y) / (1 - rho**2)
            term1 = -((nu + 2)/2) * torch.log(1 + zeta/nu)
            term2 = ((nu + 1)/2) * (torch.log(1 + x**2/nu) + torch.log(1 + y**2/nu))
            log_det = 0.5 * torch.log(1 - rho**2)
            lgamma = torch.lgamma
            const = lgamma((nu + 2)/2) + lgamma(nu/2) - 2*lgamma((nu+1)/2)
            return const - log_det + term1 + term2

        elif 'clayton' in self.family:
            t = theta
            a = torch.log(1 + t) - (1 + t) * (torch.log(u_rot) + torch.log(v_rot))
            b = torch.pow(u_rot, -t) + torch.pow(v_rot, -t) - 1
            return a - (2 + 1/t) * torch.log(torch.clamp(b, min=eps))

        elif 'gumbel' in self.family:
            t = theta
            x = -torch.log(u_rot)
            y = -torch.log(v_rot)
            A = torch.pow(x**t + y**t, 1/t)
            term1 = torch.log(A + t - 1); term2 = -A
            term3 = (t - 1) * (torch.log(x) + torch.log(y))
            term4 = (1/t - 2) * torch.log(x**t + y**t)
            jacobian = -torch.log(u_rot) - torch.log(v_rot)
            return term1 + term2 + term3 + term4 + jacobian
        
        elif 'frank' in self.family:
            t = theta
            exp_t = torch.exp(-t)
            exp_tu = torch.exp(-t * u_rot)
            exp_tv = torch.exp(-t * v_rot)
            log_num = torch.log(torch.abs(t) + eps) + torch.log(torch.abs(1 - exp_t) + eps) - t*(u_rot + v_rot)
            denom_inner = (1 - exp_t) - (1 - exp_tu) * (1 - exp_tv)
            log_denom = 2.0 * torch.log(torch.abs(denom_inner) + eps)
            return log_num - log_denom
        
        return torch.zeros_like(u)

    def compute_h_func(self, u, v, theta, nu=None):
        u_rot, v_rot = self.rotate_data(u, v)
        eps = 1e-9
        u_rot = torch.clamp(u_rot, eps, 1-eps)
        v_rot = torch.clamp(v_rot, eps, 1-eps)
        h_val = torch.zeros_like(u_rot)
        
        if 'gaussian' in self.family:
            n = torch.distributions.Normal(0, 1)
            x, y = n.icdf(u_rot), n.icdf(v_rot)
            h_val = n.cdf((x - theta*y) / torch.sqrt(1 - theta**2 + 1e-8))
        elif 'student' in self.family:
            x = inverse_t_cdf(u_rot, nu)
            y = inverse_t_cdf(v_rot, nu)
            factor = torch.sqrt((nu + 1) / (nu + y**2) / (1 - theta**2 + 1e-8))
            arg = (x - theta * y) * factor
            h_val = torch.tensor(scipy.special.stdtr((nu+1).detach().cpu().numpy(), arg.detach().cpu().numpy()))
        elif 'clayton' in self.family:
            t = theta
            term = torch.pow(v_rot, -t-1) * torch.pow(torch.pow(u_rot, -t) + torch.pow(v_rot, -t) - 1, -1/t - 1)
            h_val = term
        elif 'gumbel' in self.family:
            t = theta
            x = -torch.log(u_rot); y = -torch.log(v_rot)
            A = torch.pow(x**t + y**t, 1/t)
            h_val = torch.exp(-A) * torch.pow(y, t-1) / v_rot * torch.pow(x**t + y**t, 1/t - 1)
        elif 'frank' in self.family:
            t = theta
            et = torch.exp(-t); eu = torch.exp(-t*u_rot); ev = torch.exp(-t*v_rot)
            num = (eu - 1) * ev
            den = (et - 1) + (eu - 1) * (ev - 1)
            h_val = num / (den + 1e-20)

        if self.rotation in [90, 270]: h_val = 1 - h_val
        return torch.clamp(h_val, eps, 1 - eps)

    def forward(self, u_vec, v_vec):
        eps = 1e-6
        u_clamped = torch.clamp(u_vec, eps, 1-eps)
        v_clamped = torch.clamp(v_vec, eps, 1-eps)
        
        x_in = torch.erfinv(2 * u_clamped - 1) * 1.414
        y_in = torch.erfinv(2 * v_clamped - 1) * 1.414
        
        inputs = torch.stack([x_in, y_in], dim=1).unsqueeze(0) 
        
        rnn_out, _ = self.rnn(inputs)
        f_t_seq = self.head(rnn_out).squeeze(0).squeeze(1)
        
        thetas_pred = self.transform_parameter(f_t_seq)
        
        # --- THE PREDICTIVE SHIFT ---
        theta_0 = self.transform_parameter(self.f_init).unsqueeze(0)
        theta_aligned = torch.cat([theta_0, thetas_pred[:-1]])
        
        nu = self.get_nu()
        loss_vec = self.log_likelihood_pair(u_vec, v_vec, theta_aligned, nu)
        
        self.oos_forecast = thetas_pred[-1].detach()
        
        # Return negative vector so we can slice and sum properly outside!
        return -loss_vec, theta_aligned

# ====================================================================================
# --- OPTUNA HYPERPARAMETER OPTIMIZATION ---
# ====================================================================================
def objective(trial, u_full, v_full, split_idx, fam_str, rotation, static_theta, static_nu):
    hidden_dim = trial.suggest_categorical("hidden_dim", [1, 2, 4, 8, 16, 32])
    num_layers = trial.suggest_int("num_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.0, 0.3) if num_layers == 2 else 0.0
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    device = torch.device("cpu")
    model = NeuralPairCopula(fam_str, rotation=rotation, hidden_dim=hidden_dim, 
                             num_layers=num_layers, dropout=dropout).to(device)
    model.warm_start(static_theta, static_nu)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 1. Train only on the first 80%
    u_train = u_full[:split_idx]
    v_train = v_full[:split_idx]
    
    model.train()
    best_train_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(150):
        optimizer.zero_grad()
        loss_vec, _ = model(u_train, v_train)
        loss = torch.sum(loss_vec) # Sum to get total NLL
        
        if torch.isnan(loss): 
            raise optuna.exceptions.TrialPruned()
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if loss.item() < best_train_loss - 1e-4:
            best_train_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > 10: break

    # 2. Evaluate on the tail 20%, maintaining the continuous memory state!
    model.eval()
    with torch.no_grad():
        full_loss_vec, _ = model(u_full, v_full)
        # Sum ONLY the validation tail for Optuna to minimize
        val_loss = torch.sum(full_loss_vec[split_idx:])
        
        if torch.isnan(val_loss):
            raise optuna.exceptions.TrialPruned()
            
    return val_loss.item()

def run_hyperparameter_search(u_matrix, static_model):
    T = u_matrix.shape[0]
    split_idx = int(np.floor(T * 0.8))

    M = np.array(static_model.matrix, dtype=np.int64)
    if M.max() == u_matrix.shape[1]: M -= 1 
    if np.sum(M[0] >= 0) > np.sum(M[-1] >= 0): M = np.flipud(M)
    
    var_1 = M[u_matrix.shape[1]-1, 0]
    var_2 = M[0, 0]
    
    u_full = torch.tensor(np.nan_to_num(u_matrix[:, var_1], 0.5), dtype=torch.float64)
    v_full = torch.tensor(np.nan_to_num(u_matrix[:, var_2], 0.5), dtype=torch.float64)
    
    pc = static_model.pair_copulas[0][0]
    fam_str = str(pc.family)
    rotation = int(pc.rotation)
    params = pc.parameters.flatten()
    static_theta = float(params[0]) if len(params) > 0 else None
    static_nu = float(params[1]) if len(params) > 1 else None

    print(f"\n--- Starting Optuna HPO on Root Node ({fam_str}) ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    
    study.optimize(lambda trial: objective(trial, u_full, v_full, split_idx, 
                                           fam_str, rotation, static_theta, static_nu), 
                   n_trials=30)
    
    print("Best Hyperparameters Found:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
        
    return study.best_trial.params

# ====================================================================================
# --- PARALLEL VINE FITTING ---
# ====================================================================================
def fit_single_neural_edge(tree, edge, partner_col, fam_str, rotation, static_theta, static_nu, h_storage_subset, T, hpo_params):
    u_vec = torch.tensor(h_storage_subset['u'], dtype=torch.float64)
    v_vec = torch.tensor(h_storage_subset['v'], dtype=torch.float64)

    if 'indep' in fam_str.lower():
        result_dict = {'loss_history': [], 'path': np.zeros(T), 'family': 'indep', 'rotation': 0,
                       'oos_forecast': 0.0, 'nll': 0.0, 'aic': 0.0, 'bic': 0.0}
        return edge, partner_col, result_dict, u_vec, v_vec

    torch.set_num_threads(1)
    device = torch.device("cpu")
    
    model = NeuralPairCopula(fam_str, rotation=rotation, 
                             hidden_dim=hpo_params['hidden_dim'], 
                             num_layers=hpo_params['num_layers'], 
                             dropout=hpo_params['dropout'] if 'dropout' in hpo_params else 0.0).to(device)
    model.warm_start(static_theta, static_nu)
    
    optimizer = optim.AdamW(model.parameters(), lr=hpo_params['lr'], weight_decay=hpo_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    
    loss_hist = []
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 15
    max_epochs = 200

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        loss_vec, _ = model(u_vec, v_vec) 
        loss = torch.sum(loss_vec) # Sum to get total NLL
        
        if torch.isnan(loss): break
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_loss = loss.item()
        loss_hist.append(current_loss)
        scheduler.step(current_loss)
        
        if current_loss < best_loss - 1e-4:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit: break
            
    _, theta_path = model(u_vec, v_vec)
    theta_path = theta_path.detach()
    
    nu_val = model.get_nu()
    if nu_val is not None: nu_val = nu_val.detach()
        
    h_direct = model.compute_h_func(u_vec, v_vec, theta_path, nu_val)
    h_indirect = model.compute_h_func(v_vec, u_vec, theta_path, nu_val)
    
    # AIC / BIC Calculations
    if 'student' in fam_str: param_base = 4 
    else: param_base = 3 
    
    final_nll = loss_hist[-1] if loss_hist else float('nan')
    k = sum(p.numel() for p in model.parameters() if p.requires_grad) # Total NN weights
    
    aic = 2 * k + 2 * final_nll
    bic = k * np.log(T) + 2 * final_nll

    result_dict = {
        'loss_history': loss_hist,
        'path': theta_path.numpy(),
        'family': fam_str,
        'rotation': rotation,
        'oos_forecast': model.oos_forecast.numpy(),
        'nll': final_nll,
        'aic': aic,
        'bic': bic
    }
    
    return edge, partner_col, result_dict, h_direct, h_indirect

def fit_neural_vine(u_matrix, structure, hpo_params):
    T, N = u_matrix.shape
    M = np.array(structure.matrix, dtype=np.int64)
    if M.max() == N: M -= 1 
    if np.sum(M[0] >= 0) > np.sum(M[-1] >= 0): M = np.flipud(M)
    
    fams = structure.pair_copulas
    h_storage = {(i, -1): u_matrix[:, i] for i in range(N)}
    fitted_models = {}
    
    active_cores = max(1, os.cpu_count() - 2)
    print(f"Parallelizing Neural Vine Fitting across {active_cores} CPU cores...")

    for tree in range(N - 1):
        edges = N - 1 - tree
        tasks = []
        
        for edge in range(edges):
            row = N - 1 - tree; col = edge
            var_1 = M[row, col]
            u_vec = h_storage[(var_1, -1)] if tree == 0 else h_storage[(col, tree-1)]
            
            var_2 = M[col, col]
            partner_col = -1
            if tree == 0:
                v_vec = h_storage[(var_2, -1)]
            else:
                for k in range(N):
                    if M[row+1, k] == var_2: partner_col = k; break
                v_vec = h_storage[(partner_col, tree-1)]

            if tree < len(fams):
                pc = fams[tree][edge]
                fam_str = str(pc.family)
                rotation = int(pc.rotation)
                params = pc.parameters.flatten()
                static_theta = float(params[0]) if len(params) > 0 else None
                static_nu = float(params[1]) if len(params) > 1 else None
            else:
                fam_str, rotation, static_theta, static_nu = "indep", 0, None, None

            u_np = np.nan_to_num(u_vec, 0.5)
            v_np = np.nan_to_num(v_vec, 0.5)
            
            tasks.append((tree, edge, partner_col, fam_str, rotation, static_theta, static_nu, {'u': u_np, 'v': v_np}, T, hpo_params))
            
        results = Parallel(n_jobs=active_cores, return_as="generator")(delayed(fit_single_neural_edge)(*t) for t in tasks)

        for res in tqdm(results, total=edges, desc=f"Tree {tree+1}/{N-1}"):
            edge_idx, partner_col, model_dict, h_dir, h_indir = res
            fitted_models[f"T{tree}_E{edge_idx}"] = model_dict
            h_storage[(edge_idx, tree)] = h_dir
            if tree < N - 2: h_storage[(partner_col, tree)] = h_indir
                
    return fitted_models

# ====================================================================================
# --- DIAGNOSTICS SUITE ---
# ====================================================================================
def theta_to_tau(family_str, theta_array):
    fam = family_str.lower()
    theta = np.array(theta_array, dtype=float)
    if 'gaussian' in fam or 'student' in fam: return (2 / np.pi) * np.arcsin(theta)
    elif 'clayton' in fam: return theta / (theta + 2)
    elif 'gumbel' in fam: return 1 - (1 / theta)
    elif 'frank' in fam: return theta / 10.0 
    elif 'indep' in fam: return np.zeros_like(theta)
    return theta

def plot_dynamic_tau_paths(fitted_models, dates, name, save_path):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    top_edges = [k for k in fitted_models.keys() if k.startswith("T0_E")][:3]
    if not top_edges: return
    
    fig, axes = plt.subplots(len(top_edges), 1, figsize=(10, 2.5 * len(top_edges)), sharex=True)
    if len(top_edges) == 1: axes = [axes]
    
    for i, edge_key in enumerate(top_edges):
        model_info = fitted_models[edge_key]
        path = model_info['path'].flatten()
        family = model_info['family'].capitalize()
        tau_path = theta_to_tau(family, path)
        if model_info.get('rotation', 0) in [90, 270]: tau_path = -tau_path
            
        ax = axes[i]
        ax.plot(dates, tau_path, color='#d62728', lw=1.5, label=f'{family} Copula (Neural)')
        ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.7)
        ax.set_ylabel(f"Kendall's Tau", fontsize=12)
        ax.set_title(f"Tree 1, Edge {i+1} Dynamics", fontsize=12, fontweight='bold')
        ax.set_ylim(-1.05, 1.05)
        ax.legend(loc='upper right')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
    plt.xlabel("Date", fontsize=12)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_neural_convergence(fitted_models, name, save_path):
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    top_edges = [k for k in fitted_models.keys() if k.startswith("T0_E")][:4]
    if not top_edges: return
    
    plt.figure(figsize=(8, 5))
    line_styles = ['-', '--', '-.', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, edge_key in enumerate(top_edges):
        loss_hist = fitted_models[edge_key].get('loss_history', [])
        fam = fitted_models[edge_key]["family"].capitalize()
        if loss_hist:
            plt.plot(loss_hist, lw=2, linestyle=line_styles[idx % 4], color=colors[idx % 4],
                     label=f'{edge_key} ({fam})')
            
    plt.title(f"Neural GRU Optimization Convergence - {name}", fontsize=14, fontweight='bold')
    plt.xlabel("AdamW Epochs", fontsize=12)
    plt.ylabel("Negative Log-Likelihood", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, loc='best')
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ====================================================================================
# --- MAIN EXECUTION ---
# ====================================================================================
if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    # Aligning paths with your GAS structure
    res_dir = os.path.join(project_root, "results", "dynamics")
    static_out_dir = os.path.join(project_root, "results", "copulas", "static")
    out_dir = os.path.join(project_root, "results", "copulas", "neural")
    graph_dir = os.path.join(project_root, "results", "copulas", "neural", "plots")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    
    # 1. Load Train Data
    u_spot_file = os.path.join(res_dir, "NGARCH", "uniforms_ngarch_train.csv")
    u_har_file = os.path.join(res_dir, "HAR_GARCH", "uniforms_har_garch_evt_train.csv")
    u_nsde_file = os.path.join(res_dir, "NSDE", "uniforms_nsde_train.csv")

    u_spot = pd.read_csv(u_spot_file, index_col='Date', parse_dates=True)
    u_har = pd.read_csv(u_har_file, index_col='Date', parse_dates=True)
    u_nsde = pd.read_csv(u_nsde_file, index_col='Date', parse_dates=True)

    # 2. Synchronize Dates
    global_valid_dates = u_spot.index.intersection(u_har.index).intersection(u_nsde.index)
    u_spot = u_spot.loc[global_valid_dates]
    u_har = u_har.loc[global_valid_dates]
    u_nsde = u_nsde.loc[global_valid_dates]
    
    print(f"Neural Vine Evaluation Period: {global_valid_dates[0].date()} to {global_valid_dates[-1].date()}")

    # 3. Capture Terminal Argument (HAR, NSDE, or ALL)
    model_choice = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    
    if model_choice == "HAR":
        factor_sets = {"HAR-GARCH-EVT": u_har}
    elif model_choice == "NSDE":
        factor_sets = {"NSDE": u_nsde}
    else:
        factor_sets = {"HAR-GARCH-EVT": u_har, "NSDE": u_nsde}

    # 4. Sequential/Parallel execution of the outer loop
    for factor_name, u_factors in factor_sets.items():
        print(f"\n{'='*60}")
        print(f"--- FITTING NEURAL VINE: Spot + {factor_name} ---")
        print(f"{'='*60}")
        
        combined_u = pd.concat([u_spot, u_factors], axis=1)
        np_data = combined_u.to_numpy()

        # Route to the correct static JSON baseline
        static_json_name = f"joint_vine_spot_{factor_name.lower().replace('-', '_')}_model.json"
        static_json_path = os.path.join(static_out_dir, static_json_name)
        
        if not os.path.exists(static_json_path):
            print(f"ERROR: Static baseline not found at {static_json_path}")
            continue
            
        print("Step 1: Loading static structure...")
        static_model = pv.Vinecop.from_file(static_json_path)

        print("Step 2: Optuna HPO on Root Node...")
        best_hpo_params = run_hyperparameter_search(np_data, static_model)

        print("Step 3: Multi-Core Neural Vine Fitting...")
        neural_fitted_models = fit_neural_vine(np_data, static_model, best_hpo_params)
        
        # Save & Plot
        save_prefix = f"neural_vine_spot_{factor_name.lower().replace('-', '_')}"
        
        plot_dynamic_tau_paths(neural_fitted_models, global_valid_dates, f"Neural {factor_name}", 
                               os.path.join(graph_dir, f"{save_prefix}_dynamic_tau.png"))
        
        plot_neural_convergence(neural_fitted_models, f"Neural {factor_name}", 
                                os.path.join(graph_dir, f"{save_prefix}_convergence.png"))

        torch.save(neural_fitted_models, os.path.join(out_dir, f"{save_prefix}_model.pth"))
        print(f"\nDONE: Saved {factor_name} Neural Model to {out_dir}")
