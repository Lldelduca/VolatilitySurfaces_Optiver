import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import stdtrit, stdtr
from scipy.stats import kendalltau, norm
from tqdm import tqdm
import pandas as pd
import pyvinecopulib as pv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

torch.set_default_dtype(torch.float64)

# CUSTOM AUTOGRAD FOR STUDENT-T
class InverseStudentT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, nu):
        u_cpu = u.detach().cpu().numpy()
        nu_cpu = nu.detach().cpu().numpy()
        u_cpu = np.clip(u_cpu, 1e-12, 1 - 1e-12)
        x = stdtrit(nu_cpu, u_cpu)
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

# NEURAL PAIR COPULA MODEL
class NeuralPairCopula(nn.Module):
    def __init__(self, family, rotation=0, hidden_dim=10):
        super().__init__()
        self.family = str(family).split('.')[-1].lower()
        self.rotation = int(rotation)
        
        # --- ARCHITECTURE: GRU ---
        self.rnn = nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)
        self.f_init = nn.Parameter(torch.tensor(0.0))
        
        if 'student' in self.family:
            self.nu_param = nn.Parameter(torch.tensor(2.0))
        else:
            self.register_parameter('nu_param', None)

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
            val = torch.where(mask, torch.sign(val) * 1e-4, val)
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
        
        # Probit transform
        x_in = torch.erfinv(2 * u_clamped - 1) * 1.414
        y_in = torch.erfinv(2 * v_clamped - 1) * 1.414
        
        inputs = torch.stack([x_in, y_in], dim=1).unsqueeze(0) 
        
        # rnn_out shape: (1, T, hidden_dim)
        rnn_out, _ = self.rnn(inputs)
        
        # f_t_seq shape: (T,)
        f_t_seq = self.head(rnn_out).squeeze(0).squeeze(1)
        
        # thetas_pred[t] is the forecast generated AFTER observing day t.
        # Therefore, it is the prediction for day t+1.
        thetas_pred = self.transform_parameter(f_t_seq)
        
        # --- THE PREDICTIVE SHIFT ---
        # To evaluate the likelihood of day t, we MUST use the prediction made on day t-1.
        # For the very first day (t=0), we use our learnable initial parameter.
        theta_0 = self.transform_parameter(self.f_init).unsqueeze(0)
        
        # theta_aligned[t] is now correctly the parameter applied to u_vec[t]
        theta_aligned = torch.cat([theta_0, thetas_pred[:-1]])
        
        # Calculate valid Log-Likelihood
        nu = self.get_nu()
        loss_vec = self.log_likelihood_pair(u_vec, v_vec, theta_aligned, nu)
        
        # We save the very last prediction (thetas_pred[-1]) to the object itself.
        # This is the "tomorrow" forecast! We will need this later for the OOS VaR Simulation.
        self.oos_forecast = thetas_pred[-1].detach()
        
        # Return theta_aligned so the h_functions in the Vine builder use the correct historical paths
        return -torch.sum(loss_vec), theta_aligned

def fit_neural_vine(u_matrix, structure, hidden_dim=10):
    T, N = u_matrix.shape
    device = torch.device("cpu")
    print(f"Fitting Neural GRU Vine on {device}...")
    u_tensor = torch.tensor(u_matrix, dtype=torch.float64).to(device)
    
    M = np.array(structure.matrix, dtype=np.int64)
    if M.max() == N: M -= 1 
    top_density = np.sum(M[0] >= 0)
    bot_density = np.sum(M[-1] >= 0)
    if top_density > bot_density: M = np.flipud(M)
    
    fams = structure.pair_copulas
    h_storage = {} 
    
    for i in range(N):
        h_storage[(i, -1)] = u_tensor[:, i]

    fitted_models = {}
    
    for tree in range(N - 1):
        edges = N - 1 - tree
        pbar = tqdm(range(edges), desc=f"Tree {tree+1}/{N-1}")
        
        for edge in pbar:
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

            u_vec = torch.nan_to_num(u_vec, 0.5)
            v_vec = torch.nan_to_num(v_vec, 0.5)

            pc = fams[tree][edge]
            fam_str = str(pc.family)

            if 'indep' in fam_str.lower():
                h_direct = u_vec; h_indirect = v_vec
                fitted_models[f"T{tree}_E{edge}"] = {'path': np.zeros(T), 'family': 'indep'}
            else:
                model = NeuralPairCopula(fam_str, rotation=pc.rotation, hidden_dim=hidden_dim).to(device)
                optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
                loss_hist = []

                for _ in range(50): 
                    optimizer.zero_grad()
                    loss, _ = model(u_vec, v_vec) 
                    if torch.isnan(loss): break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    loss_hist.append(loss.item())
                
                _, theta_path = model(u_vec, v_vec)
                theta_path = theta_path.detach()
                
                nu_val = model.get_nu()
                if nu_val is not None: nu_val = nu_val.detach()
                    
                h_direct = model.compute_h_func(u_vec, v_vec, theta_path, nu_val)
                h_indirect = model.compute_h_func(v_vec, u_vec, theta_path, nu_val)
                
                fitted_models[f"T{tree}_E{edge}"] = {
                    'loss_history': loss_hist,
                    'path': theta_path.numpy(),
                    'family': fam_str,
                    'rotation': pc.rotation
                }

            h_storage[(col, tree)] = h_direct
            if tree < N - 2: h_storage[(partner_col, tree)] = h_indirect
                
    return fitted_models

# DIAGNOSTICS SUITE
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
    top_edges = [k for k in fitted_models.keys() if k.startswith("T0_E")][:3]
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Dynamic Kendall's Tau Paths (Neural) - {name}", fontsize=18, fontweight='bold')
    
    for i, edge_key in enumerate(top_edges):
        model_info = fitted_models[edge_key]
        path = model_info['path'].flatten()
        family = model_info['family'].capitalize()
        tau_path = theta_to_tau(family, path)
        if model_info.get('rotation', 0) in [90, 270]: tau_path = -tau_path
            
        ax = axes[i]
        ax.plot(dates, tau_path, color='darkgreen', lw=1.5, label=f'{family} ($\\tau_t$)')
        ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
        ax.set_title(f"Tree 1, Edge {i+1}", fontsize=14)
        ax.set_ylim(-1, 1) 
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
    plt.xlabel("Date", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_neural_convergence(fitted_models, name, save_path):
    top_edges = [k for k in fitted_models.keys() if k.startswith("T0_E")][:4]
    plt.figure(figsize=(10, 6))
    for edge_key in top_edges:
        loss_hist = fitted_models[edge_key].get('loss_history', [])
        if loss_hist:
            plt.plot(loss_hist, lw=2, label=f'{edge_key} ({fitted_models[edge_key]["family"]})')
            
    plt.title(f"Neural GRU Optimization Convergence - {name}", fontsize=16)
    plt.xlabel("AdamW Epochs")
    plt.ylabel("Negative Log-Likelihood")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    res_dir = os.path.join(project_root, "outputs", "dynamics")
    static_out_dir = os.path.join(project_root, "outputs", "copulas")
    out_dir = os.path.join(project_root, "outputs", "copulas", "neural")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load Train Data
    u_spot_file = os.path.join(res_dir, "train_uniforms_ngarch_t.csv")
    u_har_file = os.path.join(res_dir, "train_uniforms_har_garch_evt.csv")
    u_nsde_file = os.path.join(res_dir, "train_nsde_uniforms.csv")

    try:
        u_spot = pd.read_csv(u_spot_file, index_col='Date', parse_dates=True)
        u_har = pd.read_csv(u_har_file, index_col='Date', parse_dates=True)
        u_nsde = pd.read_csv(u_nsde_file, index_col='Date', parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error loading files.\n{e}")
        exit()

    # Intersect
    global_valid_dates = u_spot.index.intersection(u_har.index).intersection(u_nsde.index)
    u_spot = u_spot.loc[global_valid_dates]
    u_har = u_har.loc[global_valid_dates]
    u_nsde = u_nsde.loc[global_valid_dates]

    factor_sets = {"HAR-GARCH-EVT": u_har, "NSDE": u_nsde}

    for factor_name, u_factors in factor_sets.items():
        print(f"\n--- Fitting Dynamic Neural Copula: Spot + {factor_name} ---")
        
        combined_u = pd.concat([u_spot, u_factors], axis=1)
        np_data = combined_u.to_numpy()

        # Load Static Baseline JSON
        static_json_path = os.path.join(static_out_dir, f"joint_vine_spot_{factor_name.lower().replace('-', '_')}_model.json")
        
        if not os.path.exists(static_json_path):
            print(f"ERROR: Cannot find static baseline JSON at {static_json_path}")
            continue
            
        print("1. Loading frozen structure from Static Baseline...")
        static_model = pv.Vinecop(static_json_path)

        # Fit Neural Model
        print("2. Fitting Dynamic Neural Updates via GRU...")
        neural_fitted_models = fit_neural_vine(np_data, static_model, hidden_dim=10)
        
        # Diagnostics
        save_prefix = f"neural_vine_spot_{factor_name.lower().replace('-', '_')}"
        
        plot_dynamic_tau_paths(neural_fitted_models, global_valid_dates, f"Neural {factor_name}", 
                               os.path.join(out_dir, f"{save_prefix}_dynamic_tau.png"))
        
        plot_neural_convergence(neural_fitted_models, f"Neural {factor_name}", 
                             os.path.join(out_dir, f"{save_prefix}_convergence.png"))

        torch.save(neural_fitted_models, os.path.join(out_dir, f"{save_prefix}_model.pth"))
        print(f"Saved Neural Model -> {save_prefix}_model.pth")
