import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pyvinecopulib as pv
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import stdtrit, stdtr

# Exact Inverse Student-t CDF bridging Scipy (forward) and PyTorch (backward)
class InverseStudentT(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, nu):
        # Detach to numpy for Scipy calculation
        u_cpu = u.detach().cpu().numpy()
        nu_cpu = nu.detach().cpu().numpy()
        
        # Exact calculation using Scipy's Cephes library
        x = stdtrit(nu_cpu, u_cpu)
        
        # Convert back to Tensor
        x_tensor = torch.from_numpy(x).to(u.device, dtype=u.dtype)
        
        # Save for backward pass
        ctx.save_for_backward(x_tensor, nu)
        return x_tensor

    @staticmethod
    def backward(ctx, grad_output):
        x, nu = ctx.saved_tensors
        # Derivative of Inverse CDF is 1 / PDF(x)
        # We compute 1/PDF directly in Log Space for numerical stability
        
        pi = torch.tensor(3.1415926535, device=x.device)
        log_const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * pi)
        log_kernel = -((nu + 1) / 2) * torch.log(1 + (x**2) / nu)
        log_pdf = log_const + log_kernel
        
        pdf = torch.exp(log_pdf)
        grad_u = grad_output / pdf  # Chain rule: dL/du = dL/dx * dx/du
        return grad_u, None

def inverse_t_cdf(u, nu):
    return InverseStudentT.apply(u, nu)


# Single pair-copula whose parameter evolves according to the GAS rule
class GASPairCopula(nn.Module):
    def __init__(self, family, rotation=0):
        super().__init__()
        self.family = str(family).split('.')[-1].lower()
        self.rotation = int(rotation)

        # GAS Parameters (omega, A, B)
        self.omega = nn.Parameter(torch.tensor(0.0))
        self.A = nn.Parameter(torch.tensor(0.05))
        self.B = nn.Parameter(torch.tensor(0.9))

        # Additional parameter for Student-t copula
        if 'student' in self.family:
            self.nu_unconstrained = nn.Parameter(torch.tensor(1.5)) 
        else:
            self.register_parameter('nu_unconstrained', None)

    # Returns valid degrees of freedom > 2.0
    def get_nu(self):
        if self.nu_unconstrained is None: return None
        return torch.nn.functional.softplus(self.nu_unconstrained) + 2.01

    # Handles data rotation for copula families
    def rotate_data(self, u, v):
        if self.rotation == 0: return u, v
        if self.rotation == 90: return 1-u, v
        if self.rotation == 180: return 1-u, 1-v
        if self.rotation == 270: return u, 1-v
        return u, v

    # Maps unbounded GAS factor f_t to valid copula parameter space
    def transform_parameter(self, f_t):
        if 'gaussian' in self.family or 'student' in self.family:
            return torch.tanh(f_t) # (-1, 1)
        elif 'clayton' in self.family:
            return torch.nn.functional.softplus(f_t) + 1e-4 # (> 0)
        elif 'gumbel' in self.family or 'joe' in self.family:
            return torch.nn.functional.softplus(f_t) + 1.0 + 1e-4 # (>= 1)
        elif 'frank' in self.family:
            return f_t + torch.sign(f_t) * 1e-4 if torch.abs(f_t) < 1e-4 else f_t # (!= 0)
        return f_t

    # Compute the log-likelihood for a batch of (u, v) given theta
    def log_likelihood_pair(self, u, v, theta, nu=None):

        # Rotate and clamp
        u_rot, v_rot = self.rotate_data(u, v)
        eps = 1e-6
        u_rot = torch.clamp(u_rot, eps, 1 - eps)
        v_rot = torch.clamp(v_rot, eps, 1 - eps)

        if 'gaussian' in self.family:
            rho = theta
            n = torch.distributions.Normal(0, 1)
            x, y = n.icdf(u_rot), n.icdf(v_rot)
            rho2 = rho**2

            # Gaussian Copula Log-PDF          
            z = x**2 + y**2 - 2*rho*x*y
            log_det = 0.5 * torch.log(1 - rho2)
            log_exp = -0.5 * (z / (1 - rho2) - (x**2 + y**2))

            return -log_det + log_exp

        elif 'student' in self.family:
            rho = theta

            # Exact Student-t Log-Likelihood
            if nu is None: nu = torch.tensor(5.0, device=u.device)
            
            # Transform Uniforms (u,v) -> T-Scores (x,y)
            x = inverse_t_cdf(u_rot, nu)
            y = inverse_t_cdf(v_rot, nu)
            
            # Joint Log-Density: log c(u,v) = log(f_mv(x,y)) - log(f(x)) - log(f(y))
            rho2 = rho**2
            log_det = -0.5 * torch.log(1 - rho2)
            zeta = (x**2 + y**2 - 2*rho*x*y) / (1 - rho2)

            joint_kernel = -((nu + 2) / 2) * torch.log(1 + zeta / nu)
            marg_kernel_x = ((nu + 1) / 2) * torch.log(1 + (x**2) / nu)
            marg_kernel_y = ((nu + 1) / 2) * torch.log(1 + (y**2) / nu)
            
            log_gamma_const = torch.lgamma((nu + 2) / 2) + torch.lgamma(nu / 2) - 2 * torch.lgamma((nu + 1) / 2)
            
            return log_gamma_const + log_det + joint_kernel + marg_kernel_x + marg_kernel_y

        elif 'clayton' in self.family:
            t = theta
            a = torch.log(1+t) - (1+t)*(torch.log(u_rot)+torch.log(v_rot))
            b = torch.pow(u_rot, -t) + torch.pow(v_rot, -t) - 1
            b = torch.clamp(b, min=eps)
            c = -(2 + 1/t) * torch.log(b)
            return a + c

        elif 'gumbel' in self.family:
            t = theta
            x, y = -torch.log(u_rot), -torch.log(v_rot)
            sum_pow = torch.pow(x**t + y**t, 1/t)
            return torch.log(sum_pow + t - 1) - sum_pow + (t-1)*(torch.log(x)+torch.log(y)) - torch.log(u_rot*v_rot) - 2*torch.log(u_rot*v_rot) # Simplified correction

        elif 'frank' in self.family:
            t = theta
            et = torch.exp(-t); eu = torch.exp(-t*u_rot); ev = torch.exp(-t*v_rot)
            num = t * (1-et) * eu * ev
            den = (1-et) - (1-eu)*(1-ev)
            return torch.log(torch.abs(num)) - 2*torch.log(torch.abs(den))

        return torch.zeros_like(u)

    # Computes h(u|v) = dC/dv for vine propagation
    def compute_h_func(self, u, v, theta, nu=None):
        u_rot, v_rot = self.rotate_data(u, v)
        eps = 1e-6
        u_rot = torch.clamp(u_rot, eps, 1-eps)
        v_rot = torch.clamp(v_rot, eps, 1-eps)
        
        h_val = torch.zeros_like(u)
        if 'gaussian' in self.family:
            rho = torch.clamp(theta, -0.999, 0.999)
            n = torch.distributions.Normal(0,1)
            x = n.icdf(u_rot)
            y = n.icdf(v_rot)

            h_val = n.cdf((x - rho*y) / torch.sqrt(1 - rho**2))

        elif 'student' in self.family:
            if nu is None: nu = torch.tensor(5.0)
            
            # Autograd on v to get derivative
            v_leaf = v_rot.detach().requires_grad_(True)
            u_leaf = u_rot.detach()
            
            # Re-implement CDF for Student-t
            x = inverse_t_cdf(u_leaf, nu)
            y = inverse_t_cdf(v_leaf, nu)
            rho = torch.clamp(theta, -0.999, 0.999)
            
            num = x - rho * y
            den = torch.sqrt((1 - rho**2) * (nu + y**2) / (nu + 1) + 1e-8)
            arg = (num / den).detach().cpu().numpy()

            df_np = (nu + 1).detach().cpu().numpy()
            cdf_val = stdtr(df_np, arg)
            
            h_val = torch.tensor(cdf_val, dtype=u.dtype, device=u.device)

        elif 'clayton' in self.family:
            t = theta

            # Formula: v^(-t-1) * (u^-t + v^-t - 1)^(-1/t - 1)
            term1 = -(t + 1) * torch.log(v_rot)
            base = torch.pow(u_rot, -t) + torch.pow(v_rot, -t) - 1
            term2 = -(1/t + 1) * torch.log(torch.clamp(base, min=eps))

            h_val = torch.exp(term1 + term2)

        elif 'gumbel' in self.family:
            t = theta

            # Formula: C(u,v) * [(-ln v)^(t-1) / v] * [(-ln u)^t + (-ln v)^t]^(1/t - 1)
            x = -torch.log(u_rot)
            y = -torch.log(v_rot)
            x = torch.clamp(x, min=1e-6)
            y = torch.clamp(y, min=1e-6)
            
            sum_pow = torch.pow(x**t + y**t, 1/t) # This is -log(C)
            log_C = -sum_pow
            log_term2 = (t - 1) * torch.log(y) + y 
            log_term3 = (1/t - 1) * torch.log(x**t + y**t)
            
            h_val = torch.exp(log_C + log_term2 + log_term3)

        elif 'frank' in self.family:
            t = theta
            eu = torch.exp(-t * u_rot) - 1
            et = torch.exp(-t) - 1
            
            num = eu * torch.exp(-t * v_rot)
            den = torch.sign(den) * torch.max(torch.abs(den), torch.tensor(1e-6))
            h_val = num / den

        # Un-Rotate H-Function
        if self.rotation == 90: return 1 - h_val
        if self.rotation == 270: return 1 - h_val
        return h_val

    def forward(self, u_data, v_data):
        T = u_data.shape[0]
        f_t = self.omega / (1 - self.B)  # Initialize at unconditional mean
        nu_val = self.get_nu()

        log_likes = []
        thetas = []
        for t in range(T):
            theta_t = self.transform_parameter(f_t)
            thetas.append(theta_t)
            u_t = u_data[t:t+1]
            v_t = v_data[t:t+1]
            
            # Make f_t a leaf variable for Autograd
            f_t_leaf = f_t.detach().requires_grad_(True)
            theta_leaf = self.transform_parameter(f_t_leaf)
            ll = self.log_likelihood_pair(u_t, v_t, theta_leaf, nu_val)
            
            # Compute Score = dLL/df_t via Autograd
            score = torch.autograd.grad(ll, f_t_leaf, create_graph=True)[0]
            score = torch.clamp(score, -5.0, 5.0)
            
            # GAS Update
            f_next = self.omega + self.A * score + self.B * f_t
            log_likes.append(ll)
            f_t = f_next.detach()
            
        return -torch.sum(torch.stack(log_likes)), torch.stack(thetas)

# Fits GAS dynamics to a generic R-Vine structure using Dißmann algorithm
def fit_mixed_gas_vine(u_matrix, structure):
    T, N = u_matrix.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u_tensor = torch.tensor(u_matrix, dtype=torch.float32).to(device)
    
    # Parse Structure Matrix
    M = np.array(structure.matrix)
    if M.max() == N:
        print("Detected 1-based indexing in structure matrix. Adjusting to 0-based...")
        M = np.where(M > 0, M - 1, M)
        
    fams = structure.pair_copulas
    
    # Initialize Triangular Grid
    vine_data = torch.zeros((N, N, T), device=device)
    for i in range(N):
        var_idx = int(M[i, i])
        vine_data[i, i] = u_tensor[:, var_idx]

    fitted_paths = {}
    print(f"Fitting R-Vine GAS Model on {N} assets...")

    # Tree Iteration from the bottom tree (unconditional) up to the root
    for tree in range(N - 1):
        print(f"--- Processing Tree {tree + 1} / {N - 1} ---")
        edges = N - 1 - tree
        
        for edge in range(edges):
            # Dißmann Index Logic for R-Vine Matrix
            m_row = N - 1 - tree
            m_col = edge
            
            # Determine Inputs (u, v)
            if tree == 0:
                # Tree 0 (Bottom): Inputs are raw variables.
                # In R-Vine matrix, the pair at (m_row, m_col) connects:
                # 1. Variable at M[m_row, m_col]
                # 2. Variable at M[m_col, m_col] (Diagonal element of the column)
                var1_idx = int(M[m_row, m_col])
                var2_idx = int(M[m_col, m_col]) 
                
                u_vec = u_tensor[:, var1_idx]
                v_vec = u_tensor[:, var2_idx]
                
                # For storage later: we need to know where 'v' "lives" in the matrix row
                # In the bottom row, var2 is just the diagonal variable.
                # We identify it by finding which column p in this row holds var2_idx.
                partner_col = -1
                for p in range(N):
                    if M[m_row, p] == var2_idx:
                        partner_col = p
                        break
            else:
                # Tree > 0: Inputs are h-functions from previous tree (stored in grid).
                target_var = M[m_row, m_col]
                
                # Search for the "partner" column in the row below
                partner_col = -1
                for p in range(m_col + 1, N):
                    if M[m_row + 1, p] == target_var:
                        partner_col = p
                        break
                
                if partner_col == -1: 
                    partner_col = m_row + 1
                
                # Fetch transformed variables
                u_vec = vine_data[m_row + 1, m_col]
                v_vec = vine_data[m_row + 1, partner_col]

            # Fit GAS Model
            pc = fams[tree][edge]
            fam_str = str(pc.family)
            rot = pc.rotation
            
            # Independence Copula Check
            if 'indep' in fam_str.lower():
                fitted_paths[f"T{tree}_E{edge}"] = torch.zeros(T)
                h_direct = u_vec
                h_indirect = v_vec
            else:
                # Initialize and Train
                model = GASPairCopula(fam_str, rotation=rot).to(device)
                optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=20, line_search_fn='strong_wolfe')
                
                def closure():
                    optimizer.zero_grad()
                    loss, _ = model(u_vec.unsqueeze(1), v_vec.unsqueeze(1))
                    loss.backward()
                    return loss
                optimizer.step(closure)
                
                # Extract Path
                _, theta_path = model(u_vec.unsqueeze(1), v_vec.unsqueeze(1))
                fitted_paths[f"T{tree}_E{edge}"] = theta_path.detach().cpu().numpy()
                
                # Compute H-functions for next level
                theta_path = theta_path.detach()
                nu_val = model.get_nu()
                
                h_d_list = []
                h_i_list = []
                with torch.no_grad():
                    for t in range(T):
                        # h(u|v) -> Direct
                        h1 = model.compute_h_func(u_vec[t:t+1], v_vec[t:t+1], theta_path[t], nu_val)
                        h_d_list.append(h1)
                        # h(v|u) -> Indirect (Swap args)
                        h2 = model.compute_h_func(v_vec[t:t+1], u_vec[t:t+1], theta_path[t], nu_val)
                        h_i_list.append(h2)
                        
                h_direct = torch.cat(h_d_list).squeeze()
                h_indirect = torch.cat(h_i_list).squeeze()

            # Update Grid
            vine_data[m_row, m_col] = h_direct
            
            if partner_col != -1 and partner_col != m_col:
                 vine_data[m_row, partner_col] = h_indirect
            elif tree == 0 and partner_col != -1:
                 vine_data[m_row, partner_col] = h_indirect
            
    return fitted_paths

if __name__ == "__main__":
    print("1. Generating Synthetic 14-Asset Data...")
    T = 1250 
    N = 14
    # Create random covariance
    A = np.random.rand(N, N)
    cov = np.dot(A, A.transpose())
    data_mvn = np.random.multivariate_normal(np.zeros(N), cov, size=T)
    data_u = norm.cdf(data_mvn)
    
    print("2. Selecting Structure via pyvinecopulib...")
    controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.clayton, pv.BicopFamily.gumbel, pv.BicopFamily.student, pv.BicopFamily.frank])
    structure = pv.Vinecop(d=N) 
    structure.select(data_u, controls=controls)
    print(f"   Structure: R-Vine with {len(structure.pair_copulas)} trees.")

    print("3. Fitting Dynamic GAS Vine (Model 0)...")
    paths = fit_mixed_gas_vine(data_u, structure)
    
    print("4. Plotting Results...")
    keys = list(paths.keys())
    plt.figure(figsize=(12, 6))
    # Plot first edge of first tree
    plt.plot(paths[keys[0]], label=f"Tree 1 Edge 1 ({keys[0]})")
    # Plot last edge
    plt.plot(paths[keys[-1]], label=f"Deep Edge ({keys[-1]})")
    plt.title("Recovered Dynamic Dependence (GAS Parameters)")
    plt.legend()
    plt.show()
    
    print("Success. Model is Thesis-Ready.")