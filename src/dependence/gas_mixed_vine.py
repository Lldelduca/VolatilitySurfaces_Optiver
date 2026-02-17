import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import stdtrit, stdtr
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
import os

torch.set_default_dtype(torch.float64)


# CUSTOM AUTOGRAD FUNCTION: Inverse Student-t CDF
# Implements the inverse CDF of Student-t distribution with custom gradient computation.
# This is necessary because scipy's stdtrit doesn't have automatic differentiation support.
class InverseStudentT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, nu):
        u_cpu = u.detach().cpu().numpy()
        nu_cpu = nu.detach().cpu().numpy()
        u_cpu = np.clip(u_cpu, 1e-12, 1 - 1e-12)
        x = stdtrit(nu_cpu, u_cpu)  # scipy's inverse Student-t CDF
        x_tensor = torch.from_numpy(x).to(u.device, dtype=u.dtype)
        # Save tensors for backward pass
        ctx.save_for_backward(x_tensor, u, nu)
        return x_tensor

    @staticmethod
    def backward(ctx, grad_output):
        x, u, nu = ctx.saved_tensors
        pi = torch.tensor(3.141592653589793, device=x.device, dtype=x.dtype)
        
        # Compute Student-t PDF using log-space for numerical stability
        log_const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * pi)
        log_kernel = -((nu + 1) / 2) * torch.log(1 + (x**2) / nu)
        pdf = torch.exp(log_const + log_kernel)
        pdf = torch.clamp(pdf, min=1e-100)
        grad_u = grad_output / pdf
        
        grad_nu = None
        if ctx.needs_input_grad[1]:
            eps = 1e-4
            u_cpu = u.detach().cpu().numpy()
            # Central difference: [F_inv(u, nu+eps) - F_inv(u, nu-eps)] / (2*eps)
            nu_p = (nu + eps).detach().cpu().numpy()
            nu_m = (nu - eps).detach().cpu().numpy()
            x_p = stdtrit(nu_p, u_cpu)
            x_m = stdtrit(nu_m, u_cpu)
            dx_dnu = torch.from_numpy((x_p - x_m) / (2 * eps)).to(x.device, dtype=x.dtype)
            grad_nu = grad_output * dx_dnu
        
        return grad_u, grad_nu

def inverse_t_cdf(u, nu):
    return InverseStudentT.apply(u, nu)

# ==============================================================================================================================

# GAS Copula 
class GASPairCopula(nn.Module):
    def __init__(self, family, rotation=0):
        super().__init__()
        self.family = str(family).split('.')[-1].lower()
        self.rotation = int(rotation)
        
        # GAS parameters
        self.omega = nn.Parameter(torch.tensor(0.0))   # Constant level
        self.A = nn.Parameter(torch.tensor(0.05))      # Score coefficient 
        self.B_logit = nn.Parameter(torch.tensor(3.0)) # Persistence
        
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
            # Correlation parameter for Gaussian/Student: bounded in (-1, 1)
            return torch.tanh(f_t) * 0.999
        elif 'clayton' in self.family:
            # Clayton parameter must be > 0
            return torch.nn.functional.softplus(f_t) + 1e-5
        elif 'gumbel' in self.family:
            # Gumbel parameter must be >= 1
            return torch.nn.functional.softplus(f_t) + 1.0001
        elif 'frank' in self.family:
            # Frank parameter can be any real number except 0; add small threshold to avoid singularity
            val = f_t 
            mask = torch.abs(val) < 1e-4
            val = torch.where(mask, torch.sign(val) * 1e-4, val)
            return val
        return f_t
    
    def warm_start(self, u_vec, v_vec):
        if isinstance(u_vec, torch.Tensor):
            u_vec = u_vec.detach().cpu().numpy()
            v_vec = v_vec.detach().cpu().numpy()
        
        # Compute empirical Kendall's tau
        tau, _ = kendalltau(u_vec, v_vec)
        if self.rotation in [90, 270]: tau = -tau
        
        # Initialize f_t
        f_init = 0.0
        if 'gaussian' in self.family or 'student' in self.family:
            # For Gaussian/Student: rho = sin(pi*tau/2)
            theta = np.sin(tau * np.pi / 2)
            f_init = np.arctanh(np.clip(theta, -0.99, 0.99))
        elif 'clayton' in self.family:
            # Clayton: tau = theta / (theta + 2)
            theta = 2 * tau / (1 - tau) if tau < 1 else 0.1
            f_init = np.log(np.exp(max(theta, 1e-4)) - 1)
        elif 'gumbel' in self.family:
            # Gumbel: tau = 1 - 1/theta
            theta = 1 / (1 - tau) if tau < 1 else 1.1
            f_init = np.log(np.exp(max(theta - 1.0, 1e-4)) - 1)
        elif 'frank' in self.family:
            # Frank: approximate relationship
            f_init = 5 * tau 

        with torch.no_grad():
            self.omega.copy_(torch.tensor(f_init * (1 - 0.95)))

    def log_likelihood_pair(self, u, v, theta, nu=None):
        u_rot, v_rot = self.rotate_data(u, v)
        eps = 1e-9
        u_rot = torch.clamp(u_rot, eps, 1 - eps)
        v_rot = torch.clamp(v_rot, eps, 1 - eps)

        if 'gaussian' in self.family:
            # Gaussian copula: uses bivariate normal CDF with correlation rho
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
            # Log-likelihood of bivariate Student-t
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

    # Compute the conditional CDF (h-function) of u given v (h(u|v,theta) = dC(u,v)/dv)
    def compute_h_func(self, u, v, theta, nu=None):
        u_rot, v_rot = self.rotate_data(u, v)
        eps = 1e-9
        u_rot = torch.clamp(u_rot, eps, 1-eps)
        v_rot = torch.clamp(v_rot, eps, 1-eps)
        h_val = torch.zeros_like(u_rot)
        
        if 'gaussian' in self.family:
            n = torch.distributions.Normal(0, 1)
            x, y = n.icdf(u_rot), n.icdf(v_rot)
            h_val = n.cdf((x - theta*y) / torch.sqrt(1 - theta**2))

        elif 'student' in self.family:
            x = inverse_t_cdf(u_rot, nu)
            y = inverse_t_cdf(v_rot, nu)
            factor = torch.sqrt((nu + 1) / (nu + y**2) / (1 - theta**2))
            arg = (x - theta * y) * factor
            h_val = torch.tensor(stdtr((nu+1).detach().cpu().numpy(), arg.detach().cpu().numpy()))

        elif 'clayton' in self.family:
            t = theta
            term = torch.pow(v_rot, -t-1) * torch.pow(torch.pow(u_rot, -t) + torch.pow(v_rot, -t) - 1, -1/t - 1)
            h_val = term

        elif 'gumbel' in self.family:
            t = theta
            # FIX: Separate lines here too
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
        score_variance = torch.tensor(1.0)
        alpha = 0.99 
        
        log_likes = []
        thetas = []
        
        for t in range(T):
            theta_t = self.transform_parameter(f_t)
            thetas.append(theta_t)
            
            f_t_leaf = f_t.detach().requires_grad_(True)
            theta_leaf = self.transform_parameter(f_t_leaf)
            u_t = u_data[t:t+1]
            v_t = v_data[t:t+1]
            
            ll = self.log_likelihood_pair(u_t, v_t, theta_leaf, nu)
            
            # Autograd.grad is essential for GAS score
            score = torch.autograd.grad(ll, f_t_leaf, create_graph=False)[0]

            with torch.no_grad():
                score_val = score.item()
                score_variance = alpha * score_variance + (1 - alpha) * (score_val**2)
                scale = 1.0 / (torch.sqrt(score_variance) + 1e-8)
            
            scaled_score = score.detach() * scale
            f_next = self.omega + self.A * scaled_score + B * f_t
            
            ll_connected = self.log_likelihood_pair(u_t, v_t, theta_t, nu)
            log_likes.append(ll_connected)
            f_t = f_next
            
        return -torch.sum(torch.stack(log_likes)), torch.stack(thetas)


# ====================================================================================

# VINE FITTER
def fit_single_edge(tree, edge, M, fams, h_storage_subset, T):
    # (We pass a subset of h_storage to avoid memory locking issues)
    row = M.shape[0] - 1 - tree
    col = edge
    
    var_1 = M[row, col]
    u_vec = h_storage_subset['u']
    v_vec = h_storage_subset['v']

    pc = fams[tree][edge]
    fam_str = str(pc.family)

    if 'indep' in fam_str.lower():
        return f"T{tree}_E{edge}", {'path': np.zeros(T), 'family': 'indep'}, u_vec, v_vec

    # Force PyTorch to use 1 thread internally so multiple processes don't fight over cores
    torch.set_num_threads(1)
    
    device = torch.device("cpu")
    model = GASPairCopula(fam_str, rotation=pc.rotation).to(device)
    model.warm_start(u_vec, v_vec)
    
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    loss_hist = []

    for _ in range(30):
        optimizer.zero_grad()
        loss, _ = model(u_vec.unsqueeze(1), v_vec.unsqueeze(1))
        if torch.isnan(loss): break
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
    
    _, theta_path = model(u_vec.unsqueeze(1), v_vec.unsqueeze(1))
    theta_path = theta_path.detach()
    
    nu_val = model.get_nu()
    if nu_val is not None: nu_val = nu_val.detach()
        
    h_direct = model.compute_h_func(u_vec, v_vec, theta_path, nu_val)
    h_indirect = model.compute_h_func(v_vec, u_vec, theta_path, nu_val)
    
    result_dict = {
        'loss_history': loss_hist,
        'path': theta_path.numpy(),
        'family': fam_str,
        'rotation': pc.rotation
    }
    
    return f"T{tree}_E{edge}", result_dict, h_direct, h_indirect

def fit_mixed_gas_vine(u_matrix, structure):
    T, N = u_matrix.shape
    device = torch.device("cpu") 
    u_tensor = torch.tensor(u_matrix, dtype=torch.float64).to(device)
    
    # Fix Dimensions and Matrix structure
    M = np.array(structure.matrix, dtype=np.int64)
    if M.max() == N:
        M -= 1 
    
    top_density = np.sum(M[0] >= 0)
    bot_density = np.sum(M[-1] >= 0)
    if top_density > bot_density:
        M = np.flipud(M)
    
    fams = structure.pair_copulas
    h_storage = {} 
    
    # Initialize Tree -1 (Raw Data)
    for i in range(N):
        h_storage[(i, -1)] = u_tensor[:, i]

    fitted_models = {}
    num_cores = multiprocessing.cpu_count() - 1 # Leave 1 core for your OS
    print(f"Parallelizing Vine Fitting across {num_cores} CPU cores...")

    for tree in range(N - 1):
        edges = N - 1 - tree
        print(f"Fitting Tree {tree+1}/{N-1} ({edges} edges)...")
        
        # Prepare the data for parallel processing
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
            
            tasks.append((tree, edge, M, fams, {'u': u_vec, 'v': v_vec}, T))
            
        # Execute edges in parallel USING THREADING to avoid pickling errors with PyTorch Autograd
        results = Parallel(n_jobs=num_cores, prefer="threads")(
            delayed(fit_single_edge)(*task) for task in tasks
        )
        
        # Unpack results and update h_storage for the next tree
        for res in results:
            edge_key, model_dict, h_dir, h_indir = res
            fitted_models[edge_key] = model_dict
            
            # Recalculate edge index for h_storage mapping
            edge_idx = int(edge_key.split('_E')[1])
            row = N - 1 - tree
            var_2 = M[col, col]
            partner_col = -1
            if tree > 0:
                for k in range(N):
                    if M[row+1, k] == var_2:
                        partner_col = k; break
                        
            h_storage[(edge_idx, tree)] = h_dir
            if tree < N - 2: 
                h_storage[(partner_col, tree)] = h_indir
                
    return fitted_models

# Visualizations
def theta_to_tau(family_str, theta_array):
    """
    Converts raw copula parameters into Kendall's Tau [-1, 1] for standardized plotting.
    """
    fam = family_str.lower()
    # Ensure it's a numpy array
    theta = np.array(theta_array, dtype=float)
    
    if 'gaussian' in fam or 'student' in fam:
        return (2 / np.pi) * np.arcsin(theta)
    elif 'clayton' in fam:
        return theta / (theta + 2)
    elif 'gumbel' in fam:
        return 1 - (1 / theta)
    elif 'frank' in fam:
        return theta / 10.0 # simplified visual scaler
    elif 'indep' in fam:
        return np.zeros_like(theta)
    return theta

def plot_dynamic_tau_paths(fitted_gas, dates, name, save_path):
    """
    Plots the standardized time-varying Kendall's Tau (\tau_t) paths.
    """
    # Grab the top 3 edges connecting to the root node
    top_edges = [k for k in fitted_gas.keys() if k.startswith("T0_E")][:3]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Dynamic Kendall's Tau Paths (GAS) - {name}", fontsize=18, fontweight='bold')
    
    for i, edge_key in enumerate(top_edges):
        model_info = fitted_gas[edge_key]
        path = model_info['path'].flatten()
        family = model_info['family'].capitalize()
        
        # Convert raw parameter to Kendall's Tau
        tau_path = theta_to_tau(family, path)
        
        # Handle rotations (if the copula was rotated 90 or 270 degrees, the correlation is negative)
        if model_info['rotation'] in [90, 270]:
            tau_path = -tau_path
            
        ax = axes[i]
        ax.plot(dates, tau_path, color='darkred', lw=1.5, label=f'{family} ($\\tau_t$)')
        
        # Add a zero-line for reference
        ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
        
        # Formatting
        ax.set_title(f"Tree 1, Edge {i+1}", fontsize=14)
        ax.set_ylim(-1, 1) # Standardized limits!
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
    plt.xlabel("Date", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_gas_convergence(fitted_gas, name, save_path):
    """
    Plots the Log-Likelihood loss history of the PyTorch Adam optimizer.
    Proves to reviewers that the Neural/Autograd GAS update rule actually converged.
    """
    top_edges = [k for k in fitted_gas.keys() if k.startswith("T0_E")][:4]
    
    plt.figure(figsize=(10, 6))
    for edge_key in top_edges:
        loss_hist = fitted_gas[edge_key]['loss_history']
        if loss_hist:
            plt.plot(loss_hist, lw=2, label=f'{edge_key} ({fitted_gas[edge_key]["family"]})')
            
    plt.title(f"GAS Autograd Optimization Convergence - {name}", fontsize=16)
    plt.xlabel("Adam Epochs")
    plt.ylabel("Negative Log-Likelihood")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    res_dir = os.path.join(project_root, "results", "dynamics")
    static_out_dir = os.path.join(project_root, "results", "copulas", "static")
    out_dir = os.path.join(project_root, "results", "copulas", "gas")
    graph_dir = os.path.join(project_root, "results", "copulas", "gas", "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    u_spot_file = os.path.join(res_dir, "NGARCH", "train_uniforms_ngarch_t.csv")
    u_har_file = os.path.join(res_dir, "HAR_GARCH", "train_uniforms_har_garch_evt.csv")
    u_nsde_file = os.path.join(res_dir, "NSDE", "train_nsde_uniforms.csv")

    u_spot = pd.read_csv(u_spot_file, index_col='Date', parse_dates=True)
    u_har = pd.read_csv(u_har_file, index_col='Date', parse_dates=True)
    u_nsde = pd.read_csv(u_nsde_file, index_col='Date', parse_dates=True)

    # Find common dates and truncate to ensure comparability
    global_valid_dates = u_spot.index.intersection(u_har.index).intersection(u_nsde.index)
    u_spot = u_spot.loc[global_valid_dates]
    u_har = u_har.loc[global_valid_dates]
    u_nsde = u_nsde.loc[global_valid_dates]
    print(f"Evaluation Period: {global_valid_dates[0].date()} to {global_valid_dates[-1].date()}")

    factor_sets = {"HAR-GARCH-EVT": u_har, "NSDE": u_nsde}

    for factor_name, u_factors in factor_sets.items():
        print("")
        print(f"--- Fitting Joint Copula: Spot + {factor_name} ---")

        combined_u = pd.concat([u_spot, u_factors], axis=1)
        np_data = combined_u.to_numpy()

        # STEP 1: Determine the Structure via Static Vine
        static_json_path = os.path.join(static_out_dir, f"joint_vine_spot_{factor_name.lower().replace('-', '_')}_model.json")
        static_model = pv.Vinecop.from_file(static_json_path)

        # STEP 2: Fit GAS
        gas_fitted_models = fit_mixed_gas_vine(np_data, static_model)

        save_prefix = f"gas_vine_spot_{factor_name.lower().replace('-', '_')}"
        plot_dynamic_tau_paths(gas_fitted_models, global_valid_dates, f"GAS {factor_name}", os.path.join(graph_dir, f"{save_prefix}_dynamic_tau.png"))
        
        plot_gas_convergence(gas_fitted_models, f"GAS {factor_name}", os.path.join(graph_dir, f"{save_prefix}_convergence.png"))
        
        torch.save(gas_fitted_models, os.path.join(out_dir, f"{save_prefix}_model.pth"))
