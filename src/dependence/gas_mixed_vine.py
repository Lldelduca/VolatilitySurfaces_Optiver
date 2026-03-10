import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.special
from scipy.stats import kendalltau, norm
from tqdm import tqdm
import matplotlib.dates as mdates
import pyvinecopulib as pv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from joblib import Parallel, delayed
import multiprocessing
import json
import sys
import os
import random

torch.set_default_dtype(torch.float64)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True, warn_only=True)

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
        pdf = torch.clamp(pdf, min=1e-100)
        grad_u = grad_output / pdf
        
        grad_nu = None
        if ctx.needs_input_grad[1]:
            eps = 1e-4
            u_cpu = u.detach().cpu().numpy()
            nu_p = (nu + eps).detach().cpu().numpy()
            nu_m = (nu - eps).detach().cpu().numpy()
            x_p = scipy.special.stdtrit(nu_p, u_cpu)
            x_m = scipy.special.stdtrit(nu_m, u_cpu)
            dx_dnu = torch.from_numpy(np.asarray((x_p - x_m) / (2 * eps))).to(x.device, dtype=x.dtype)
            grad_nu = (grad_output * dx_dnu).sum()
        
        return grad_u, grad_nu

def inverse_t_cdf(u, nu):
    return InverseStudentT.apply(u, nu)

class GASPairCopula(nn.Module):
    def __init__(self, family, rotation=0):
        super().__init__()
        self.family = str(family).split('.')[-1].lower()
        self.rotation = int(rotation)
        
        self.omega = nn.Parameter(torch.tensor(0.0))
        self.A = nn.Parameter(torch.tensor(0.05))
        self.B_logit = nn.Parameter(torch.tensor(3.0))
        
        if 'student' in self.family:
            self.nu_param = nn.Parameter(torch.tensor(2.0))
        else:
            self.register_parameter('nu_param', None)

    def get_nu(self):
        if self.nu_param is None: return None
        return torch.nn.functional.softplus(self.nu_param) + 2.01
    
    def get_B(self):
        return torch.sigmoid(self.B_logit)

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

    def warm_start(self, u_vec, v_vec, static_theta=None, static_nu=None):
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
                self.omega.copy_(torch.tensor(f_init))
                if static_nu is not None and self.nu_param is not None:
                    nu_init = float(np.log(np.exp(max(static_nu - 2.01, 1e-6)) - 1))
                    self.nu_param.copy_(torch.tensor(nu_init))
        else:
            if isinstance(u_vec, torch.Tensor):
                u_vec = u_vec.detach().cpu().numpy()
                v_vec = v_vec.detach().cpu().numpy()
            
            tau, _ = kendalltau(u_vec, v_vec)
            if self.rotation in [90, 270]: tau = -tau
            
            f_init = 0.0
            if 'gaussian' in self.family or 'student' in self.family:
                theta = np.sin(tau * np.pi / 2)
                f_init = np.arctanh(np.clip(theta, -0.99, 0.99))
            elif 'clayton' in self.family:
                theta = 2 * tau / (1 - tau) if tau < 1 else 0.1
                f_init = np.log(np.exp(max(theta, 1e-4)) - 1)
            elif 'gumbel' in self.family:
                theta = 1 / (1 - tau) if tau < 1 else 1.1
                f_init = np.log(np.exp(max(theta - 1.0, 1e-4)) - 1)
            elif 'frank' in self.family:
                f_init = 5 * tau

            with torch.no_grad():
                self.omega.copy_(torch.tensor(f_init))

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
            term1 = torch.log(A + t - 1)
            term2 = -A
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
            denom = torch.sqrt(1 - theta**2 + 1e-8)
            arg = (x - theta*y) / denom
            arg = torch.nan_to_num(arg, nan=0.0)
            h_val = n.cdf(arg)

        elif 'student' in self.family:
            x = inverse_t_cdf(u_rot, nu)
            y = inverse_t_cdf(v_rot, nu)
            factor = torch.sqrt((nu + 1) / (nu + y**2) / (1 - theta**2))
            arg = (x - theta * y) * factor
            h_val = torch.tensor(scipy.special.stdtr((nu+1).detach().cpu().numpy(), arg.detach().cpu().numpy()))

        elif 'clayton' in self.family:
            t = theta
            term = torch.pow(v_rot, -t-1) * torch.pow(torch.pow(u_rot, -t) + torch.pow(v_rot, -t) - 1, -1/t - 1)
            h_val = term

        elif 'gumbel' in self.family:
            t = theta
            x = -torch.log(u_rot)
            y = -torch.log(v_rot)
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

    def forward(self, u_data, v_data):
        T = u_data.shape[0]
        f_t = self.omega.clone()
        B = self.get_B()
        nu = self.get_nu()
        score_variance = torch.tensor(1.0, device=u_data.device)
        alpha = 0.99 
        
        log_likes = []
        thetas = []
        
        for t in range(T):
            theta_t = self.transform_parameter(f_t)
            thetas.append(theta_t.view(-1)) 
            
            u_t = u_data[t:t+1].view(-1)
            v_t = v_data[t:t+1].view(-1)

            u_rot, v_rot = self.rotate_data(u_t, v_t)
            u_rot = torch.clamp(u_rot, 1e-9, 1 - 1e-9)
            v_rot = torch.clamp(v_rot, 1e-9, 1 - 1e-9)
            
            if 'gaussian' in self.family:
                rho = theta_t
                n = torch.distributions.Normal(0, 1)
                x = n.icdf(u_rot)
                y = n.icdf(v_rot)
                score = rho + (x * y * (1 + rho**2) - rho * (x**2 + y**2)) / (1 - rho**2 + 1e-8)
                score = score.view(-1)
                
            elif 'student' in self.family:
                rho = theta_t
                x = inverse_t_cdf(u_rot, nu)
                y = inverse_t_cdf(v_rot, nu)
                zeta = (x**2 - 2*rho*x*y + y**2) / (1 - rho**2 + 1e-8)
                w_t = (nu + 2) / (nu + zeta)
                score = rho + w_t * (x * y * (1 + rho**2) - rho * (x**2 + y**2)) / (1 - rho**2 + 1e-8)
                score = score.view(-1)

            else:
                f_t_leaf = f_t.detach().requires_grad_(True)
                theta_leaf = self.transform_parameter(f_t_leaf)
                ll = self.log_likelihood_pair(u_t, v_t, theta_leaf, nu).sum()
                score = torch.autograd.grad(ll, f_t_leaf, create_graph=False)[0]
                score = score.view(-1)
            
            with torch.no_grad():
                score_val = score.item()
                score_variance = alpha * score_variance + (1 - alpha) * (score_val**2)
                scale = 1.0 / (torch.sqrt(score_variance) + 1e-8)
            
            scaled_score = score.detach() * scale
            scaled_score = torch.clamp(scaled_score, min=-10.0, max=10.0) 
            
            f_next = self.omega + self.A * scaled_score.squeeze() + B * (f_t - self.omega)
            
            ll_connected = self.log_likelihood_pair(u_t, v_t, theta_t, nu).view(-1)
            log_likes.append(ll_connected)
            
            f_t = f_next
            
        return -torch.sum(torch.stack(log_likes)), torch.stack(thetas)


# ====================================================================================

def fit_single_edge(tree, edge, partner_col, fam_str, rotation, static_theta, static_nu, h_storage_subset, T, model_type):
    u_vec = torch.tensor(h_storage_subset['u'], dtype=torch.float64)
    v_vec = torch.tensor(h_storage_subset['v'], dtype=torch.float64)

    if 'indep' in fam_str.lower():
        result_dict = {'loss_history': [], 'path': np.zeros(T), 'family': 'indep', 'rotation': 0,
                       'omega': 0.0, 'A': 0.0, 'B': 0.0, 'nu': float('nan'), 'nll': 0.0, 'aic': 0.0, 'bic': 0.0}
        return edge, partner_col, result_dict, u_vec, v_vec

    torch.set_num_threads(1)
    
    device = torch.device("cpu")
    model = GASPairCopula(fam_str, rotation=rotation).to(device)
    model.warm_start(u_vec, v_vec, static_theta, static_nu)
    
    # --- DYNAMIC OPTIMIZER SELECTION ---
    if model_type == "NSDE":
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
        )
        patience = 35 # Allow a bit of time to settle after aggressive decay
    else:
        # Default for HAR
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
        )
        patience = 35

    loss_hist = []
    max_epochs = 300
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        loss, _ = model(u_vec.unsqueeze(1), v_vec.unsqueeze(1))
        
        if torch.isnan(loss): 
            break
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        current_loss = loss.item()
        loss_hist.append(current_loss)
        scheduler.step(current_loss)
        
        normalized_loss = current_loss / T
        if normalized_loss < best_loss - 1e-6:
            best_loss = normalized_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    _, theta_path = model(u_vec.unsqueeze(1), v_vec.unsqueeze(1))
    theta_path = theta_path.detach().squeeze()
    theta_path = torch.nan_to_num(theta_path, nan=0.0)
    
    nu_val = model.get_nu()
    if nu_val is not None: 
        nu_val = torch.nan_to_num(nu_val.detach(), nan=5.0)
        
    h_direct = model.compute_h_func(u_vec, v_vec, theta_path, nu_val)
    h_indirect = model.compute_h_func(v_vec, u_vec, theta_path, nu_val)

    if 'student' in fam_str: k = 4 
    else: k = 3 
    
    final_nll = loss_hist[-1] if loss_hist else float('nan')
    
    aic = 2 * k + 2 * final_nll
    bic = k * np.log(T) + 2 * final_nll
    
    result_dict = {'loss_history': loss_hist, 'path': theta_path.numpy(),'family': fam_str, 'rotation': rotation,
                   'omega': model.omega.item(), 'A': model.A.item(), 'B': model.get_B().item(), 
                   'nu': nu_val.item() if nu_val is not None else float('nan'),
                   'nll': final_nll, 'aic': aic, 'bic': bic}
    
    return edge, partner_col, result_dict, h_direct, h_indirect

def fit_mixed_gas_vine(u_matrix, structure, model_type):
    T, N = u_matrix.shape
    device = torch.device("cpu") 
    u_tensor = torch.tensor(u_matrix, dtype=torch.float64).to(device)
    
    M = np.array(structure.matrix, dtype=np.int64)
    if M.max() == N:
        M -= 1 
    
    top_density = np.sum(M[0] >= 0)
    bot_density = np.sum(M[-1] >= 0)
    if top_density > bot_density:
        M = np.flipud(M)
    
    fams = structure.pair_copulas
    h_storage = {} 
    
    for i in range(N):
        h_storage[(i, -1)] = u_tensor[:, i]

    fitted_models = {}
    active_cores = 48
    print(f"Parallelizing Vine Fitting across {active_cores} CPU cores...")

    for tree in range(N - 1):
        edges = N - 1 - tree
        print(f"Fitting Tree {tree+1}/{N-1} ({edges} edges)...")
        
        tasks = []
        for edge in range(edges):
            row = N - 1 - tree 
            col = edge
            var_1 = M[row, col]
            u_vec = h_storage[(var_1, -1)] if tree == 0 else h_storage[(col, tree-1)]
            
            var_2 = M[col, col]
            partner_col = -1
            if tree == 0:
                v_vec = h_storage[(var_2, -1)]
            else:
                for k in range(N):
                    if M[row+1, k] == var_2:
                        partner_col = k; break
                v_vec = h_storage[(partner_col, tree-1)]

            u_vec = torch.nan_to_num(u_vec, 0.5)
            v_vec = torch.nan_to_num(v_vec, 0.5)

            if tree < len(fams):
                pc = fams[tree][edge]
                fam_str = str(pc.family)
                rotation = int(pc.rotation)
                params = pc.parameters.flatten()
                static_theta = float(params[0]) if len(params) > 0 else None
                static_nu = float(params[1]) if len(params) > 1 else None
            else:
                fam_str = "indep"
                rotation = 0
                static_theta, static_nu = None, None

            u_np = torch.nan_to_num(u_vec, 0.5).numpy()
            v_np = torch.nan_to_num(v_vec, 0.5).numpy()
            
            # Pass the model_type down to the worker thread
            tasks.append((tree, edge, partner_col, fam_str, rotation, static_theta, static_nu, {'u': u_np, 'v': v_np}, T, model_type))
            
        results = Parallel(n_jobs=48, return_as="generator")(delayed(fit_single_edge)(*t) for t in tasks)

        for res in tqdm(results, total=edges, desc=f"Tree {tree+1}/{N-1}"):
            edge_idx, partner_col, model_dict, h_dir, h_indir = res
            
            edge_key = f"T{tree}_E{edge_idx}"
            fitted_models[edge_key] = model_dict
            
            h_storage[(edge_idx, tree)] = h_dir
            if tree < N - 2: 
                h_storage[(partner_col, tree)] = h_indir
                
    return fitted_models

# Visualizations
def print_gas_short_summary(fitted_gas):
    total_nll = 0.0
    total_aic = 0.0
    total_bic = 0.0
    
    for info in fitted_gas.values():
        total_nll += info.get('nll', 0.0)
        total_aic += info.get('aic', 0.0)
        total_bic += info.get('bic', 0.0)
        
    log_likelihood = -total_nll
    
    print(f"In-Sample Log-Likelihood: {log_likelihood:.2f}")
    print(f"In-Sample AIC:            {total_aic:.2f}")
    print(f"In-Sample BIC:            {total_bic:.2f}")

def theta_to_tau(family_str, theta_array):
    fam = family_str.lower()
    theta = np.array(theta_array, dtype=float)
    if 'gaussian' in fam or 'student' in fam: return (2 / np.pi) * np.arcsin(theta)
    elif 'clayton' in fam: return theta / (theta + 2)
    elif 'gumbel' in fam: return 1 - (1 / theta)
    elif 'frank' in fam: return theta / 10.0
    elif 'indep' in fam: return np.zeros_like(theta)
    return theta

def plot_dynamic_tau_paths(fitted_gas, dates, name, save_path):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    top_edges = [k for k in fitted_gas.keys() if k.startswith("T0_E")][:3]
    if not top_edges: return
    
    fig, axes = plt.subplots(len(top_edges), 1, figsize=(10, 2.5 * len(top_edges)), sharex=True)
    if len(top_edges) == 1: axes = [axes]
    
    for i, edge_key in enumerate(top_edges):
        model_info = fitted_gas[edge_key]
        path = model_info['path'].flatten()
        family = model_info['family'].capitalize()
        
        tau_path = theta_to_tau(family, path)
        if model_info['rotation'] in [90, 270]: 
            tau_path = -tau_path
            
        ax = axes[i]
        ax.plot(dates, tau_path, color='#1f77b4', lw=1.5, label=f'{family} Copula')
        ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.7)
        ax.set_ylabel(f"Kendall", fontsize=12)
        ax.set_title(f"Tree 1, Edge {i+1} Dynamics", fontsize=12, fontweight='bold')
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.legend(loc='upper right', frameon=True, shadow=False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
    plt.xlabel("Year", fontsize=12)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_gas_convergence(fitted_gas, name, save_path):
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    top_edges = [k for k in fitted_gas.keys() if k.startswith("T0_E")][:4]
    if not top_edges: return
    
    plt.figure(figsize=(8, 5))
    line_styles = ['-', '--', '-.', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, edge_key in enumerate(top_edges):
        loss_hist = fitted_gas[edge_key]['loss_history']
        fam = fitted_gas[edge_key]["family"].capitalize()
        if loss_hist:
            plt.plot(loss_hist, lw=2, linestyle=line_styles[idx % 4], color=colors[idx % 4],
                     label=f'{edge_key} ({fam})')
            
    plt.title(f"Optimization Convergence: {name}", fontsize=14, fontweight='bold')
    plt.xlabel("Optimization Epoch", fontsize=12)
    plt.ylabel("Negative Log-Likelihood", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, loc='best')
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_vine_summary_csv(fitted_gas, save_path):
    rows = []
    total_nll, total_aic, total_bic = 0.0, 0.0, 0.0
    
    for edge_key, info in fitted_gas.items():
        if info['family'] == 'indep':
            rows.append([edge_key, 'Indep', 0, 0.0, 0.0, 0.0, float('nan'), 0.0, 0.0, 0.0])
            continue
            
        rows.append([
            edge_key, 
            info['family'].capitalize(), 
            info['rotation'], 
            info['omega'], 
            info['A'], 
            info['B'], 
            info['nu'], 
            info['nll'], 
            info['aic'], 
            info['bic']
        ])
        
        total_nll += info['nll']
        total_aic += info['aic']
        total_bic += info['bic']
        
    df = pd.DataFrame(rows, columns=['Edge', 'Family', 'Rotation', 'Omega', 'A', 'B', 'Nu', 'NLL', 'AIC', 'BIC'])
    totals_row = pd.DataFrame([['TOTAL', '-', '-', '-', '-', '-', '-', total_nll, total_aic, total_bic]], columns=df.columns)
    df = pd.concat([df, totals_row], ignore_index=True)
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    res_dir = os.path.join(project_root, "results", "dynamics")
    static_out_dir = os.path.join(project_root, "results", "copulas", "static")
    out_dir = os.path.join(project_root, "results", "copulas", "gas")
    graph_dir = os.path.join(project_root, "results", "copulas", "gas", "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    u_spot_file = os.path.join(res_dir, "NGARCH", "uniforms_ngarch_train.csv")
    u_har_file = os.path.join(res_dir, "HAR_GARCH", "uniforms_har_garch_evt_train.csv")
    u_nsde_file = os.path.join(res_dir, "NSDE", "uniforms_nsde_train.csv")

    u_spot = pd.read_csv(u_spot_file, index_col='Date', parse_dates=True)
    u_har = pd.read_csv(u_har_file, index_col='Date', parse_dates=True)
    u_nsde = pd.read_csv(u_nsde_file, index_col='Date', parse_dates=True)

    global_valid_dates = u_spot.index.intersection(u_har.index).intersection(u_nsde.index)
    u_spot = u_spot.loc[global_valid_dates]
    u_har = u_har.loc[global_valid_dates]
    u_nsde = u_nsde.loc[global_valid_dates]
    print(f"Evaluation Period: {global_valid_dates[0].date()} to {global_valid_dates[-1].date()}")

    # Capture the argument passed in the terminal
    model_choice = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    
    if model_choice == "HAR":
        factor_sets = {"HAR-GARCH-EVT": u_har}
    elif model_choice == "NSDE":
        factor_sets = {"NSDE": u_nsde}
    else:
        factor_sets = {"HAR-GARCH-EVT": u_har, "NSDE": u_nsde}

    for factor_name, u_factors in factor_sets.items():
        print("")
        print(f"--- Fitting Joint Copula: Spot + {factor_name} ---")

        combined_u = pd.concat([u_spot, u_factors], axis=1)
        np_data = combined_u.to_numpy()

        static_json_path = os.path.join(static_out_dir, f"joint_vine_spot_{factor_name.lower().replace('-', '_')}_model.json")
        static_model = pv.Vinecop.from_file(static_json_path)

        # Pass the factor_name down to determine optimizer behavior
        gas_fitted_models = fit_mixed_gas_vine(np_data, static_model, factor_name)
        print_gas_short_summary(gas_fitted_models)

        save_prefix = f"gas_vine_spot_{factor_name.lower().replace('-', '_')}"
        csv_path = os.path.join(out_dir, f"{save_prefix}_summary.csv")
        save_vine_summary_csv(gas_fitted_models, csv_path)

        plot_dynamic_tau_paths(gas_fitted_models, global_valid_dates, f"GAS {factor_name}", os.path.join(graph_dir, f"{save_prefix}_dynamic_tau.png"))
        plot_gas_convergence(gas_fitted_models, f"GAS {factor_name}", os.path.join(graph_dir, f"{save_prefix}_convergence.png"))
        
        torch.save(gas_fitted_models, os.path.join(out_dir, f"{save_prefix}_model.pth"))