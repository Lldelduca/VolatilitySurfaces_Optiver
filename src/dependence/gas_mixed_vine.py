import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import stdtrit, stdtr
from scipy.stats import kendalltau, norm
from tqdm import tqdm

# Set double precision for stability
torch.set_default_dtype(torch.float64)

# DIFFERENTIABLE MATH UTILITIES
class InverseStudentT(torch.autograd.Function):
    """
    Differentiable Inverse Student-t CDF.
    Forward: Uses Scipy (accurate).
    Backward: Uses PyTorch analytic PDF (differentiable).
    
    This custom function is needed because scipy's inverse CDF is not
    differentiable in PyTorch. We use scipy for accuracy in forward pass and
    analytical gradients in backward pass.
    """
    @staticmethod
    def forward(ctx, u, nu):
        u_cpu = u.detach().cpu().numpy()
        nu_cpu = nu.detach().cpu().numpy()
        # Clamp strictly to avoid inf
        u_cpu = np.clip(u_cpu, 1e-12, 1 - 1e-12)
        x = stdtrit(nu_cpu, u_cpu)
        x_tensor = torch.from_numpy(x).to(u.device, dtype=u.dtype)
        ctx.save_for_backward(x_tensor, nu)
        return x_tensor

    @staticmethod
    def backward(ctx, grad_output):
        x, nu = ctx.saved_tensors
        pi = torch.tensor(3.1415926535, device=x.device, dtype=x.dtype)
        
        # Log-space PDF for numerical stability
        # PDF formula: Gamma((nu+1)/2) / (Gamma(nu/2) * sqrt(nu*pi)) * (1 + x^2/nu)^(-(nu+1)/2)
        log_const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * pi)
        log_kernel = -((nu + 1) / 2) * torch.log(1 + (x**2) / nu)
        pdf = torch.exp(log_const + log_kernel)
        
        pdf = torch.clamp(pdf, min=1e-100) # Avoid div by zero
        grad_u = grad_output / pdf
        return grad_u, None

def inverse_t_cdf(u, nu):
    return InverseStudentT.apply(u, nu)


# GAS PAIR COPULA MODEL
class GASPairCopula(nn.Module):
    """
    Pair Copula Model with GAS (Generalized Autoregressive Score) Dynamics.
    
    This model adapts copula parameters over time based on the score (gradient of
    log-likelihood). The dynamics follow:
        f_t = omega + A * scaled_score_t + B * f_{t-1}
    where f_t maps to the copula parameter via a transformation function.
    
    Supported copula families: Gaussian, Student-t, Clayton, Gumbel, Frank
    Rotations enable modeling of tail dependence in different directions.
    """
    
    def __init__(self, family, rotation=0):
        super().__init__()
        self.family = str(family).split('.')[-1].lower()
        self.rotation = int(rotation)

        # GAS Parameters
        # omega: Mean reversion level (long-run copula parameter)
        # A: Sensitivity/alpha parameter (immediate response to shocks)
        # B_logit: Persistence/beta parameter (memory in the process)
        self.omega = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.A = nn.Parameter(torch.tensor(0.05, dtype=torch.float64)) 
        self.B_logit = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))

        # Student-t Degrees of Freedom (Learnable, > 2)
        if 'student' in self.family:
            self.nu_param = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        else:
            self.register_parameter('nu_unconstrained', None)

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
        """Maps GAS factor f_t to valid parameter space"""
        if 'gaussian' in self.family or 'student' in self.family:
            # Correlation constrained to (-1, 1)
            return torch.tanh(f_t) * 0.9999
        elif 'clayton' in self.family:
            # Clayton dependence > 0
            return torch.nn.functional.softplus(f_t) + 1e-5
        elif 'gumbel' in self.family or 'joe' in self.family:
            # Gumbel/Joe dependence >= 1
            return torch.nn.functional.softplus(f_t) + 1.0001
        elif 'frank' in self.family:
            # Frank parameter: real-valued but avoid exactly zero
            return f_t + torch.sign(f_t) * 1e-4 if torch.abs(f_t) < 1e-4 else f_t
        return f_t
    
    def warm_start(self, u, v):
        """
        Initializes omega (long-run parameter) using unconditional correlation.
        
        This provides a good starting point for optimization by setting omega
        to an estimate of the long-run copula parameter based on Kendall's tau.
        """
        u_cpu = u.detach().cpu().numpy()
        v_cpu = v.detach().cpu().numpy()
        tau, _ = kendalltau(u_cpu, v_cpu)
        
        if self.rotation in [90, 270]: tau = -tau
            
        # Convert Kendall's tau to approximate copula parameter
        theta_init = 0.0
        if 'gaussian' in self.family or 'student' in self.family:
            # Gaussian: tau = (2/pi) * arcsin(rho)
            theta_init = np.sin(tau * np.pi / 2)
        elif 'clayton' in self.family:
            # Clayton: tau = theta / (theta + 2)
            theta_init = 2 * tau / (1 - tau) if tau < 1 else 0.1
        elif 'gumbel' in self.family:
            # Gumbel: tau = 1 - 1/theta
            theta_init = 1 / (1 - tau) if tau < 1 else 1.1
        elif 'frank' in self.family:
            theta_init = 5 * tau 

        # Invert Link Function: Find f_init such that transform_parameter(f_init) = theta_init
        f_init = 0.0
        if 'gaussian' in self.family or 'student' in self.family:
            f_init = np.arctanh(np.clip(theta_init, -0.99, 0.99))
        elif 'clayton' in self.family:
            f_init = np.log(np.exp(max(theta_init, 1e-4)) - 1)
        elif 'gumbel' in self.family:
            f_init = np.log(np.exp(max(theta_init - 1.0, 1e-4)) - 1)
        elif 'frank' in self.family:
            f_init = theta_init

        # Set omega assuming steady state: f = omega / (1 - B)
        with torch.no_grad():
            self.omega.copy_(torch.tensor(f_init * (1 - 0.95)))

    def log_likelihood_pair(self, u, v, theta, nu=None):
        """
        Compute log-likelihood of the pair copula density.
        
        The copula density c(u,v|theta) is the second derivative of the CDF.
        We compute this for different families with numerical stability in mind.
        """
        u_rot, v_rot = self.rotate_data(u, v)
        eps = 1e-9
        u_rot = torch.clamp(u_rot, eps, 1 - eps)
        v_rot = torch.clamp(v_rot, eps, 1 - eps)

        if 'gaussian' in self.family:
            # Gaussian Copula: uses correlation parameter rho
            rho = theta
            n = torch.distributions.Normal(0, 1)
            # Convert uniforms to standard normals using inverse CDF
            x, y = n.icdf(u_rot), n.icdf(v_rot)
            # Bivariate normal exponent
            z = x**2 + y**2 - 2*rho*x*y
            log_det = 0.5 * torch.log(1 - rho**2 + 1e-8)
            log_exp = -0.5 * (z / (1 - rho**2 + 1e-8) - (x**2 + y**2))
            return -log_det + log_exp

        elif 'student' in self.family:
            # Student-t Copula: uses correlation rho and degrees of freedom nu
            rho = theta
            # Convert uniforms to Student-t quantiles
            x = inverse_t_cdf(u_rot, nu)
            y = inverse_t_cdf(v_rot, nu)
            # Student-t copula density computation
            zeta = (x**2 + y**2 - 2*rho*x*y) / (1 - rho**2)
            term1 = -((nu + 2)/2) * torch.log(1 + zeta/nu)
            term2 = ((nu + 1)/2) * (torch.log(1 + x**2/nu) + torch.log(1 + y**2/nu))
            log_det = 0.5 * torch.log(1 - rho**2)
            
            # Log normalization constant using log-gamma for stability
            lgamma = torch.lgamma
            const = lgamma((nu + 2)/2) + lgamma(nu/2) - 2*lgamma((nu+1)/2)
            return const - log_det + term1 + term2

        elif 'clayton' in self.family:
            # Clayton Copula: lower tail dependence
            # Parameter theta > 0: higher theta = stronger dependence
            t = theta
            a = torch.log(1 + t) - (1 + t) * (torch.log(u_rot) + torch.log(v_rot))
            # CDF term: (u^-t + v^-t - 1)
            b = torch.pow(u_rot, -t) + torch.pow(v_rot, -t) - 1
            return a - (2 + 1/t) * torch.log(torch.clamp(b, min=eps))

        elif 'gumbel' in self.family:
            # Gumbel Copula: upper tail dependence
            # Parameter theta >= 1: theta=1 is independence, theta>1 is dependence
            t = theta
            x, y = -torch.log(u_rot), -torch.log(v_rot)
            # Pickands dependence function A(w) = (w^theta + (1-w)^theta)^(1/theta)
            A = torch.pow(x**t + y**t, 1/t)
            term1 = torch.log(A + t - 1)
            term2 = -A
            term3 = (t - 1) * (torch.log(x) + torch.log(y))
            term4 = (1/t - 2) * torch.log(x**t + y**t)
            jacobian = -torch.log(u_rot) - torch.log(v_rot)
            return term1 + term2 + term3 + term4 + jacobian
        
        elif 'frank' in self.family:
            # Frank Copula: symmetric dependence (both tails or none)
            # Parameter theta: can be positive or negative
            t = theta
            exp_t = torch.exp(-t)
            exp_tu = torch.exp(-t * u_rot)
            exp_tv = torch.exp(-t * v_rot)
            
            # Frank density involves exponential terms
            log_num = torch.log(torch.abs(t) + eps) + torch.log(torch.abs(1 - exp_t) + eps) - t*(u_rot + v_rot)
            # Denominator: (1 - e^-t - (1 - e^-tu)(1 - e^-tv))
            denom_inner = (1 - exp_t) - (1 - exp_tu) * (1 - exp_tv)
            log_denom = 2.0 * torch.log(torch.abs(denom_inner) + eps)
            return log_num - log_denom
        
        return torch.zeros_like(u)

    def compute_h_func(self, u, v, theta, nu=None):
        """
        Compute h-function (conditional distribution) of pair copula.
        
        The h-function is: h(v|u,theta) = dC(u,v|theta)/du
        
        Each vine tree uses h-functions to transform variables for the next tree.
        """
        u_rot, v_rot = self.rotate_data(u, v)
        eps = 1e-9
        u_rot = torch.clamp(u_rot, eps, 1-eps)
        v_rot = torch.clamp(v_rot, eps, 1-eps)
        h_val = torch.zeros_like(u_rot)
        
        if 'gaussian' in self.family:
            # Gaussian h-function: conditional normal
            n = torch.distributions.Normal(0, 1)
            x, y = n.icdf(u_rot), n.icdf(v_rot)
            # h(v|u) = Phi((x - rho*y) / sqrt(1-rho^2))
            h_val = n.cdf((x - theta*y) / torch.sqrt(1 - theta**2))

        elif 'student' in self.family:
            # Student-t h-function: conditional Student-t similar to Gaussian
            x = inverse_t_cdf(u_rot, nu)
            y = inverse_t_cdf(v_rot, nu)
            factor = torch.sqrt((nu + 1) / (nu + y**2) / (1 - theta**2))
            arg = (x - theta * y) * factor
            # Use Scipy for Student-t CDF on CPU
            h_val = torch.tensor(stdtr((nu+1).detach().cpu().numpy(), arg.detach().cpu().numpy())).to(u.device)

        elif 'clayton' in self.family:
            # Clayton h-function: derived from CDF by partial derivative
            t = theta
            term = torch.pow(v_rot, -t-1) * torch.pow(torch.pow(u_rot, -t) + torch.pow(v_rot, -t) - 1, -1/t - 1)
            h_val = term

        elif 'gumbel' in self.family:
            # Gumbel h-function: computed from Pickands function
            t = theta
            x = -torch.log(u_rot); y = -torch.log(v_rot)
            A = torch.pow(x**t + y**t, 1/t)
            # Derivative term needed for h-function
            h_val = torch.exp(-A) * torch.pow(y, t-1) / v_rot * torch.pow(x**t + y**t, 1/t - 1)
        
        elif 'frank' in self.family:
            # Frank h-function: conditional probability for Frank copula
            t = theta
            et = torch.exp(-t); eu = torch.exp(-t*u_rot); ev = torch.exp(-t*v_rot)
            num = (eu - 1) * ev
            # Denominator from Frank copula formula
            den = (et - 1) + (eu - 1) * (ev - 1)
            h_val = num / (den + 1e-20)

        # Adjust for rotation
        if self.rotation in [90, 270]: h_val = 1 - h_val
        return torch.clamp(h_val, eps, 1 - eps)

    def forward(self, u_data, v_data):
        """
        Compute GAS dynamics and log-likelihood over time.
        
        The GAS model updates parameters dynamically:
            score_t = d(log-likelihood)/d(f_t)
            f_t = omega + A * scaled_score_t + B * f_{t-1}
        
        We use a simplified Fisher scaling: approximate Fisher matrix as variance of scores.
        """
        T = u_data.shape[0]
        f_t = self.omega.detach()  # Initialize with omega
        B = self.get_B()
        nu = self.get_nu()

        # Fisher Scaling Accumulator
        score_variance = torch.tensor(1.0, device=u_data.device, dtype=torch.float64) 
        alpha = 0.95  # Exponential smoothing parameter for variance
        
        log_likes = []
        thetas = []
        
        for t in range(T):
            # Transform raw factor to copula parameter
            theta_t = self.transform_parameter(f_t)
            thetas.append(theta_t)
            
            # Get current marginal values
            u_t = u_data[t:t+1]
            v_t = v_data[t:t+1]
            
            # Compute score: gradient of log-likelihood with respect to f_t
            f_t_leaf = f_t.detach().requires_grad_(True)
            theta_leaf = self.transform_parameter(f_t_leaf)
            
            ll = self.log_likelihood_pair(u_t, v_t, theta_leaf, nu)
            # Backpropagate to get score
            score = torch.autograd.grad(ll, f_t_leaf, create_graph=True)[0]

            # Approximate Fisher Scaling
            with torch.no_grad():
                score_val = score.item()
                score_variance = alpha * score_variance + (1 - alpha) * (score_val**2)
                scale = 1.0 / (torch.sqrt(score_variance) + 1e-6)
            
            # GAS Update
            scaled_score = score * scale
            f_next = self.omega + self.A * scaled_score + B * f_t
            
            log_likes.append(ll)
            f_t = f_next.detach()  # Detach for next iteration
            
        return -torch.sum(torch.stack(log_likes)), torch.stack(thetas)

# VINE FITTER PIPELINE
def fit_mixed_gas_vine(u_matrix, structure):
    """
    Fit a complete vine copula with GAS dynamics for each pair copula.
    
    1. Iterates through each tree in the vine
    2. Fits a GAS pair copula to each edge
    3. Computes h-functions for the next tree
    """
    T, N = u_matrix.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fitting GAS Vine on {device}...")
    
    u_tensor = torch.tensor(u_matrix, dtype=torch.float64).to(device)
    
    # Extract vine structure
    M = np.array(structure.matrix, dtype=np.int64)
    if M.max() == N:
        M[M > 0] -= 1
    
    # Storage for h-functions across trees
    fams = structure.pair_copulas
    h_storage = {} 
    for i in range(N):
        h_storage[(i, -1)] = u_tensor[:, i]

    fitted_models = {}
    for tree in range(N - 1):
        edges = N - 1 - tree
        pbar = tqdm(range(edges), desc=f"Tree {tree+1}")
        
        for edge in pbar:
            # Extract copula pairing from vine structure
            row = N - 1 - tree
            col = edge
            
            # Direct Input: From the cell itself in previous tree
            var_1 = M[row, col]
            u_vec = h_storage[(var_1, -1)] if tree == 0 else h_storage[(col, tree-1)]
            
            # Indirect Input: Search for partner
            var_2 = M[col, col] # Diagonal element
            partner_col = -1
            
            if tree == 0:
                v_vec = h_storage[(var_2, -1)]
            else:
                # Find which column contains var_2 in the current tree
                for k in range(N):
                    if M[row+1, k] == var_2:
                        partner_col = k; break
                    
                if partner_col == -1:
                    raise ValueError(f"Partner var {var_2} not found in row {row+1}")
                
                v_vec = h_storage[(partner_col, tree-1)]

            # Numerical stability: replace NaNs with 0.5 (independence)
            u_vec = torch.nan_to_num(u_vec, 0.5)
            v_vec = torch.nan_to_num(v_vec, 0.5)

            # Get copula family for this edge
            pc = fams[tree][edge]
            fam_str = str(pc.family)

            # Special case: Independence copula
            if 'indep' in fam_str.lower():
                # Independence: h(v|u) = v (no dependence)
                h_direct = u_vec; h_indirect = v_vec
                fitted_models[f"T{tree}_E{edge}"] = {'loss_history': [], 'path': np.zeros(T)}
            else:
                model = GASPairCopula(fam_str, rotation=pc.rotation).to(device)
                model.warm_start(u_vec, v_vec)
                
                optimizer = optim.Adam(model.parameters(), lr=0.03)
                loss_hist = []

                for _ in range(30):
                    optimizer.zero_grad()
                    # Forward pass: compute GAS model
                    loss, _ = model(u_vec.unsqueeze(1), v_vec.unsqueeze(1))
                    if torch.isnan(loss): break
                    loss.backward()
                    optimizer.step()
                    loss_hist.append(loss.item())
                
                # Get final theta path and h-functions
                _, theta_path = model(u_vec.unsqueeze(1), v_vec.unsqueeze(1))
                theta_path = theta_path.detach()
                nu_val = model.get_nu()
                
                # Compute conditional distributions for next tree
                # h_direct: h(v|u) - used for direct inference
                # h_indirect: h(u|v) - used for indirect inference
                with torch.no_grad():
                    h_direct = model.compute_h_func(u_vec, v_vec, theta_path, nu_val)
                    h_indirect = model.compute_h_func(v_vec, u_vec, theta_path, nu_val)
                
                # Store results
                fitted_models[f"T{tree}_E{edge}"] = {
                    'loss_history': loss_hist,
                    'path': theta_path.cpu().numpy(),
                    'family': fam_str
                }

            # Update h-function storage for next tree
            h_storage[(col, tree)] = h_direct
            if tree < N - 2: h_storage[(partner_col, tree)] = h_indirect
                
    return fitted_models


# VISUALIZATION
def plot_learning_curves(fitted_models, num_curves=4):
    """Plots loss history for the first few edges to verify convergence."""
    keys = [k for k in fitted_models.keys() if 'loss_history' in fitted_models[k] and len(fitted_models[k]['loss_history']) > 0]
    if not keys: return

    fig, axes = plt.subplots(1, min(len(keys), num_curves), figsize=(15, 3))
    if num_curves == 1: axes = [axes]
    
    for i, key in enumerate(keys[:num_curves]):
        hist = fitted_models[key]['loss_history']
        axes[i].plot(hist)
        axes[i].set_title(f"Loss: {key} ({fitted_models[key]['family']})")
        axes[i].set_xlabel("Epoch")
        
    plt.tight_layout()
    plt.show()

def plot_dynamic_results(fitted_models, true_break_point):
    """Plots the inferred GAS path against the Regime Switch."""
    keys = list(fitted_models.keys())
    if not keys: return
    
    # Pick the first edge (usually the most correlated one)
    key = keys[0] 
    path = fitted_models[key]['path']
    
    plt.figure(figsize=(10, 5))
    plt.plot(path, label='Inferred GAS Parameter (Theta)', color='blue', linewidth=1.5)
    
    # Add Regime Lines
    plt.axvline(x=true_break_point, color='red', linestyle='--', label='True Regime Switch')
    plt.text(true_break_point - 100, np.max(path)*0.9, "Low Corr", color='red', ha='right')
    plt.text(true_break_point + 50, np.max(path)*0.9, "High Corr", color='red', ha='left')
    
    plt.title(f"Result: {key} Dynamic Adjustment")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================================
# 5. Simulation and Testing
# ==========================================

def simulate_regime_switch_data(T=1000, N=5):
    """Simulates data with a structural break in correlation at T/2."""
    print("Simulating Regime Switching Data...")
    data = np.zeros((T, N))
    
    # Regime 1: Low Correlation (Low crisis time)
    # Diagonal-heavy covariance matrix: variables mostly independent
    cov1 = np.eye(N) * 0.8 + 0.2
    np.fill_diagonal(cov1, 1.0)
    
    # Regime 2: High Correlation (Crisis/Crash)
    # Off-diagonal elements large: all variables covary
    cov2 = np.eye(N) * 0.1 + 0.9
    np.fill_diagonal(cov2, 1.0)
    
    # Generate multivariate normal data
    break_point = T // 2
    data[:break_point] = np.random.multivariate_normal(np.zeros(N), cov1, size=break_point)
    data[break_point:] = np.random.multivariate_normal(np.zeros(N), cov2, size=T - break_point)
    
    # Transform to Uniforms via Probability Integral Transform
    # This ensures marginals are U(0,1) while preserving dependence structure
    return norm.cdf(data)

# ==========================================
# MAIN: END-TO-END EXAMPLE
# ==========================================

if __name__ == "__main__":
    import pyvinecopulib as pv
    
    # ==========================================
    # Step 1: Simulate Synthetic Data
    # ==========================================
    T, N = 1200, 5 # Keep N small for quick testing
    data_u = simulate_regime_switch_data(T, N)
    print(f"Generated {T}x{N} data with regime switch at t={T//2}")
    
    # ==========================================
    # Step 2: Structure Selection (Static)
    # ==========================================
    # Fit a static vine structure using a goodness-of-fit criterion
    # This determines which variables to pair in each tree
    print("Selecting Structure...")
    controls = pv.FitControlsVinecop(
        family_set=[
            pv.BicopFamily.gaussian, 
            pv.BicopFamily.student, 
            pv.BicopFamily.clayton, 
            pv.BicopFamily.gumbel, 
            pv.BicopFamily.frank
        ]
    )
    structure = pv.Vinecop(d=N)
    structure.select(data_u, controls=controls)
    print(f"Selected vine structure with {N} variables and {N-1} trees")
    
    # ==========================================
    # Step 3: Fit Dynamic GAS Model
    # ==========================================
    # For each pair in the vine, fit a GAS model that adapts parameters over time
    fitted_models = fit_mixed_gas_vine(data_u, structure)
    print(f"Fitted {len(fitted_models)} pair copulas with GAS dynamics")
    
    # ==========================================
    # Step 4: Visualizations
    # ==========================================
    print("\n--- Visualizing Learning Curves ---")
    plot_learning_curves(fitted_models)
    print("(Check that losses decrease - indicates convergence)")
    
    print("\n--- Visualizing Dynamic Results ---")
    plot_dynamic_results(fitted_models, T // 2)
    print("(Look for parameter shift at the vertical red line - regime detection)")
    
    print("\nSuccess: End-to-end example completed.")