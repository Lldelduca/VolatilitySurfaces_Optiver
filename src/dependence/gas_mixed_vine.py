import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import stdtrit, stdtr
from scipy.stats import kendalltau, norm
from tqdm import tqdm

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
def fit_mixed_gas_vine(u_matrix, structure):
    T, N = u_matrix.shape
    device = torch.device("cpu") 
    print(f"Fitting GAS Vine on {device}...")
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
    
    for tree in range(N - 1):
        edges = N - 1 - tree
        pbar = tqdm(range(edges), desc=f"Tree {tree+1}/{N-1}")
        
        for edge in pbar:
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

            # Stability
            u_vec = torch.nan_to_num(u_vec, 0.5)
            v_vec = torch.nan_to_num(v_vec, 0.5)

            pc = fams[tree][edge]
            fam_str = str(pc.family)

            if 'indep' in fam_str.lower():
                h_direct = u_vec; h_indirect = v_vec
                fitted_models[f"T{tree}_E{edge}"] = {'path': np.zeros(T), 'family': 'indep'}
            else:
                model = GASPairCopula(fam_str, rotation=pc.rotation).to(device)
                model.warm_start(u_vec, v_vec)
                
                optimizer = optim.Adam(model.parameters(), lr=0.02)
                loss_hist = []

                # Optimization
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
                
                fitted_models[f"T{tree}_E{edge}"] = {
                    'loss_history': loss_hist,
                    'path': theta_path.numpy(),
                    'family': fam_str,
                    'rotation': pc.rotation
                }

            h_storage[(col, tree)] = h_direct
            if tree < N - 2: h_storage[(partner_col, tree)] = h_indirect
                
    return fitted_models

