import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve
from numba import njit, prange

Pi = np.pi
# ==========================================
# Kernel function definitions
# ==========================================

@njit
def njit_afunc(kappa, Omega0):
    return Omega0 - (kappa**2) / (4 * Omega0)

@njit
def njit_Bfunc(kappa, Omega0):
    return - (kappa**2) / (4 * Omega0)

@njit
def njit_k0func(t, kx, kyc, kappa, Omega0):
    a = njit_afunc(kappa, Omega0)
    return (2 * a * t * kyc + kx)

@njit
def njit_Kfunc(t, kx, kyc, kappa, Omega0):
    a = njit_afunc(kappa, Omega0)
    return np.sqrt(kyc**2 + (2 * a * t * kyc + kx)**2)

@njit
def njit_ssqfunc(t, kx, kyc, Sigma_g, kappa, Omega0, cssq, G):
    if np.abs(kyc) < 1e-10:
        return cssq*(kx*kx) - 2*kx*np.pi*Sigma_g*G + kappa*kappa

    else:
        K = njit_Kfunc(t, kx, kyc, kappa, Omega0)
        A = njit_afunc(kappa, Omega0)
        
        return pow(kappa,2) + (12*pow(A,2)*pow(kyc,4))/pow(K,4) - (2*A*pow(kyc,2)*(4*A*Omega0 + pow(kappa,2)))/(Omega0*pow(K,2)) - 2*G*Pi*Sigma_g*K + cssq*pow(K,2)
        

@njit
def njit_K_kernel_func(t, s, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G):
        if(np.abs(kyc) < 1e-10 and np.abs(kx) < 1e-10):
            return 0
        else:
            C_ = np.cos(kappa*(s-t)) 
            S_ = np.sin(kappa*(s-t))
            B = njit_Bfunc(kappa, Omega0)

            k0t = njit_k0func(t, kx, kyc, kappa, Omega0)
            k0s = njit_k0func(s, kx, kyc, kappa, Omega0)

            Kt = njit_Kfunc(t, kx, kyc, kappa, Omega0)
            Ks = njit_Kfunc(s, kx, kyc, kappa, Omega0)

            Y1 = k0t - k0s

            Y2 = 4*kyc*kyc*Omega0*Omega0 + kappa*kappa*k0s*k0t

            r1_times_kyc= (kyc*Y1*C_)/(8.*B) + (Y2*S_)/(16.*B*Omega0*kappa)

            r2 = Y2 + (pow(Y1,2)*pow(kappa,2))/2. - Y2*C_ + 2*kyc*Omega0*Y1*kappa*S_

            factor = (8*G*Pi*r1_times_kyc*Sigma_s)/Ks

            exparg = (-sigma_x*sigma_x)/(pow(kappa,4)) * r2

            return factor* np.exp(exparg)

@njit
def njit_Ktot_kernel_func(t, s, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G):

        if(np.abs(kyc) < 1e-10 and np.abs(kx) < 1e-10):
            return 0
        else:
            C_ = np.cos(kappa*(s-t)) 
            S_ = np.sin(kappa*(s-t))
            B = njit_Bfunc(kappa, Omega0)

            k0t = njit_k0func(t, kx, kyc, kappa, Omega0)
            k0s = njit_k0func(s, kx, kyc, kappa, Omega0)

            Kt = njit_Kfunc(t, kx, kyc, kappa, Omega0)
            Ks = njit_Kfunc(s, kx, kyc, kappa, Omega0)

            Y1 = k0t - k0s

            Y2 = 4*kyc*kyc*Omega0*Omega0 + kappa*kappa*k0s*k0t

            r1_times_kyc= (kyc*Y1*C_)/(8.*B) + (Y2*S_)/(16.*B*Omega0*kappa)

            r2 = Y2 + (pow(Y1,2)*pow(kappa,2))/2. - Y2*C_ + 2*kyc*Omega0*Y1*kappa*S_

            factor = (8*G*Pi*r1_times_kyc*Sigma_s)/Kt

            exparg = (-sigma_x*sigma_x)/(pow(kappa,4)) * r2

            return factor* np.exp(exparg)


# ==========================================
# 2. JIT-Compiled Matrix Construction
# ==========================================

@njit
def build_h_matrix_njit(t_grid, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, cssq, G):
    n = len(t_grid) - 1
    h = t_grid[1] - t_grid[0]
    h_mat = np.zeros((3 * n, 3 * n))
    
    for i in range(n):
        t_next = t_grid[i + 1]
        for j in range(i + 1):
            tp_next = t_grid[j + 1]
            
            # Physics at this step
            kmat = njit_Ktot_kernel_func(t_next, tp_next, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G)
            ssq = njit_ssqfunc(tp_next, kx, kyc, Sigma_g, kappa, Omega0, cssq, G)
            Ktp = njit_Kfunc(tp_next, kx, kyc, kappa, Omega0)
            
            # Define Amat 3x3 blocks
            # Row index 3*i, Col index 3*j
            r, c = 3 * i, 3 * j
            
            # Populate Amat/Identity logic
            if i == j:
                # Identity - (h/2)*Amat
                h_mat[r, c] = 1.0
                h_mat[r, c+1] = -(h / 2.0) * 1.0
                
                h_mat[r+1, c] = -(h / 2.0) * (-ssq)
                h_mat[r+1, c+1] = 1.0
                h_mat[r+1, c+2] = -(h / 2.0) * (2 * np.pi * Sigma_g * Ktp)
                
                h_mat[r+2, c] = -(h / 2.0) * kmat
                h_mat[r+2, c+2] = 1.0 - (h / 2.0) * kmat
            else:
                # -h * Amat
                h_mat[r, c+1] = -h * 1.0
                
                h_mat[r+1, c] = -h * (-ssq)
                h_mat[r+1, c+2] = -h * (2 * np.pi * Sigma_g * Ktp)
                
                h_mat[r+2, c] = -h * kmat
                h_mat[r+2, c+2] = -h * kmat
                
    return h_mat


@njit
def build_h_matrix_stars_only_njit(t_grid, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G):
    n = len(t_grid) - 1
    h = t_grid[1] - t_grid[0]
    h_mat = np.zeros((n, n))
    
    for i in range(n):
        t_next = t_grid[i + 1]
        for j in range(i + 1):
            tp_next = t_grid[j + 1]
            
            # Physics at this step
            kmat = njit_Ktot_kernel_func(t_next, tp_next, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G)
            
            r, c = i, j
            
            # Populate Amat/Identity logic
            if i == j:
                # Identity - (h/2)*Amat
                h_mat[r, c] = 1.0 -(h / 2.0) * kmat
            else:
                # -h * Amat
                h_mat[r, c] = -h * kmat
                
    return h_mat

@njit
def build_h_matrix_gas_only_njit(t_grid, kx, kyc, Sigma_g, kappa, Omega0, cssq, G):
    n = len(t_grid) - 1
    h = t_grid[1] - t_grid[0]
    h_mat = np.zeros((2 * n, 2 * n))
    
    for i in range(n):
        t_next = t_grid[i + 1]
        for j in range(i + 1):
            tp_next = t_grid[j + 1]
            
            ssq = njit_ssqfunc(tp_next, kx, kyc, Sigma_g, kappa, Omega0, cssq, G)
            
            r, c = 2 * i, 2 * j

            if i == j:
                # Identity - (h/2)*Amat
                h_mat[r, c] = 1.0
                h_mat[r, c+1] = -(h / 2.0) * 1.0
                
                h_mat[r+1, c] = -(h / 2.0) * (-ssq)
                h_mat[r+1, c+1] = 1.0
            else:
                # -h * Amat
                h_mat[r, c+1] = -h * 1.0
                
                h_mat[r+1, c] = -h * (-ssq)

    return h_mat
#=============================
# Source functions : Stellar case
#=============================
@njit
def trapezoidal_rule_for_stellar_only_forcing(t_target, t_grid_subset, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G, fext_vals):
    """
    JIT-compatible trapezoidal integration to replace scipy.quad
    """
    total = 0.0
    n_pts = len(t_grid_subset)
    if n_pts < 2:
        return 0.0
    
    # Pre-calculate kernel at first point
    k_prev = njit_Ktot_kernel_func(t_target, t_grid_subset[0], kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G)
    f_prev = fext_vals[0]
    
    for i in range(1, n_pts):
        tp = t_grid_subset[i]
        k_curr = njit_Ktot_kernel_func(t_target, tp, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G)
        f_curr = fext_vals[i]
        
        # Trapezoidal rule step
        h = t_grid_subset[i] - t_grid_subset[i-1]
        total += 0.5 * h * (k_prev * f_prev + k_curr * f_curr)
        
        k_prev = k_curr
        f_prev = f_curr
        
    return total

@njit(parallel=True) # Enable parallel threads for the O(N^2) loop
def build_bb_vec_stellar_njit(t_grid, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G, fext_vals, impulse_val):
    n = len(t_grid) - 1
    bb_vec = np.zeros(n)
    t_start = t_grid[0]
    
    for i in prange(n): # Parallel loop
        t_target = t_grid[i+1]
        
        # 1. Integration part (Forcing)
        t_subset = t_grid[:i+2] # Integration from t_start to t_target
        f_subset = fext_vals[:i+2]
        forcing_res = trapezoidal_rule_for_stellar_only_forcing(t_target, t_subset, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G, f_subset)
        
        # 2. Impulse part
        k_start = njit_Ktot_kernel_func(t_target, t_start, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G)
        impulse_res = k_start * impulse_val
        
        bb_vec[i] = forcing_res + impulse_res
        
    return bb_vec

#=============================
# Source functions : Gas case
#=============================
@njit
def gas_forcing_njit(t_target, t_grid_subset, kx, kyc, Sigma_g, kappa, Omega0, fext_vals):
    total = np.zeros(2)
    n_pts = len(t_grid_subset)
    if n_pts < 2:
        return total
    
    # Pre-calculate at start
    ktp_prev = njit_Kfunc(t_grid_subset[0], kx, kyc, kappa, Omega0)
    # fext_vals is expected to be (N, 2)
    val_prev = 2 * np.pi * Sigma_g * ktp_prev * fext_vals[0, 1]
    
    for i in range(1, n_pts):
        tp = t_grid_subset[i]
        ktp_curr = njit_Kfunc(tp, kx, kyc, kappa, Omega0)
        val_curr = 2 * np.pi * Sigma_g * ktp_curr * fext_vals[i, 1]
        
        h = tp - t_grid_subset[i-1]
        total[1] += 0.5 * h * (val_prev + val_curr)
        
        val_prev = val_curr
        
    return total


@njit(parallel=True)
def build_bb_vec_gas_njit(t_grid, kx, kyc, Sigma_g, kappa, Omega0, cssq, G, fext_vals, impulse_val):
    n = len(t_grid) - 1
    bb_vec = np.zeros(2 * n)
    t_start = t_grid[0]
    kt0 = njit_Kfunc(t_start, kx, kyc, kappa, Omega0)
    
    for i in prange(n):
        t_target = t_grid[i+1]
        
        forcing = gas_forcing_njit(t_target, t_grid[:i+2], kx, kyc, Sigma_g, kappa, Omega0, fext_vals)
        
        impulse_res = np.array([0.0, 2 * np.pi * Sigma_g * kt0 * impulse_val])
        
        res = forcing + impulse_res
        bb_vec[2*i] = res[0]
        bb_vec[2*i+1] = res[1]
        
    return bb_vec

#=============================
# Source functions : Gas  + Stars case
#=============================
@njit
def fast_combined_forcing_njit(t_target, t_grid_subset, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, G, fext_vals):
    total = np.zeros(3)
    n_pts = len(t_grid_subset)
    if n_pts < 2:
        return total
    
    # helper for the integrand
    def get_val(tp, f_val_2):
        kmat = njit_Ktot_kernel_func(t_target, tp, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G)
        ktp = njit_Kfunc(tp, kx, kyc, kappa, Omega0)
        return np.array([0.0, (2 * np.pi * Sigma_g * ktp) * f_val_2, kmat * f_val_2])

    val_prev = get_val(t_grid_subset[0], fext_vals[0, 2])
    
    for i in range(1, n_pts):
        tp = t_grid_subset[i]
        val_curr = get_val(tp, fext_vals[i, 2])
        
        h = tp - t_grid_subset[i-1]
        total += 0.5 * h * (val_prev + val_curr)
        val_prev = val_curr
        
    return total

@njit(parallel=True)
def build_bb_vec_combined_njit(t_grid, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, cssq, G, fext_vals, impulse_val):
    n = len(t_grid) - 1
    bb_vec = np.zeros(3 * n)
    t_start = t_grid[0]
    kt0 = njit_Kfunc(t_start, kx, kyc, kappa, Omega0)
    
    for i in prange(n):
        t_target = t_grid[i+1]
        
        forcing = fast_combined_forcing_njit(t_target, t_grid[:i+2], kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, G, fext_vals)
        
        kmat0 = njit_Ktot_kernel_func(t_target, t_start, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G)
        impulse_res = np.array([0.0, 2 * np.pi * Sigma_g * kt0 * impulse_val, kmat0 * impulse_val])
        
        res = forcing + impulse_res
        bb_vec[3*i] = res[0]
        bb_vec[3*i+1] = res[1]
        bb_vec[3*i+2] = res[2]
        
    return bb_vec
# ==========================================
# 3. Class Wrapper
# ==========================================

class Stellar_integrator:
    def __init__(self, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, G, 
                 fext_func=None, impulse_func=None):
        self.params = (kx, kyc, Sigma_s, kappa, Omega0, sigma_x,  G)
        self.kx, self.kyc, self.Sigma_s, self.kappa, self.Omega0, self.sigma_x, self.G = self.params
        
        self._fext_func = fext_func if fext_func else self.default_gaussian_pulse
        self._impulse_func = impulse_func if impulse_func else self.default_impulse

    @staticmethod
    def default_gaussian_pulse(t, inst, delta=0.05, amplitude=1.0):
        Kt = njit_Kfunc(t, inst.kx, inst.kyc, inst.kappa, inst.Omega0)
        if np.abs(Kt)<1e-3:
            return 0.0
        else:
            
            pulse = -2 * np.pi * amplitude * (np.exp(-(Kt**2 * delta**2) / 2.0) / Kt)
            return pulse

    @staticmethod
    def default_impulse(inst, t_0, Sigma_impluse):

        Kt0 = njit_Kfunc(t_0, inst.kx, inst.kyc, inst.kappa, inst.Omega0)
        if np.abs(Kt0)<1e-3:
            return 0.0
        else:
            return -(2*inst.G*np.pi)/Kt0 * Sigma_impluse

    def get_surface_density(self, results, t_grid):
        psi_values = results
        # Vectorized call to JIT function
        # Kts = np.array([njit_Kfunc(t, self.kx, self.kyc, self.kappa, self.Omega0) for t in t_grid[1:]])

        Kts = njit_Kfunc(t_grid[1:], self.kx, self.kyc, self.kappa, self.Omega0)
        return psi_values * (Kts / (-2.0 * np.pi))

    def solve(self, t_grid, fext_params=None, impulse_params=None):
        fext_p = fext_params if fext_params else {}
        imp_p = impulse_params if impulse_params else {}
        n = len(t_grid) - 1
        
        # Pre-calculate fext for all t in t_grid to avoid repeated Python calls
        fext_vals = np.array([self._fext_func(t, self, **fext_p) for t in t_grid])
        impulse_val = self._impulse_func(self, **imp_p)

        # 1. JIT Matrix Construction
        h_mat = build_h_matrix_stars_only_njit(t_grid, *self.params)
        
        # 2. JIT B-Vector Construction 
        bb_vec = build_bb_vec_stellar_njit(t_grid, *self.params, fext_vals, impulse_val)
            
        # 3. Solve Linear System
        return solve(h_mat, bb_vec)
        

#===================================================================================
class Gas_integrator:
    def __init__(self, kx, kyc, Sigma_g, kappa, Omega0, cssq, G, 
                 fext_func=None, impulse_func=None):
        self.params = (kx, kyc, Sigma_g, kappa, Omega0, cssq,  G)
        self.kx, self.kyc, self.Sigma_g, self.kappa, self.Omega0, self.cssq, self.G = self.params
        
        self._fext_func = fext_func if fext_func else self.default_gaussian_pulse
        self._impulse_func = impulse_func if impulse_func else self.default_impulse

    @staticmethod
    def default_gaussian_pulse(t, inst, delta=0.05, amplitude=1.0):
        Kt = njit_Kfunc(t, inst.kx, inst.kyc, inst.kappa, inst.Omega0)
        if np.abs(Kt)<1e-3:
            return np.array([0.0, 0.0])
        else:
            
            pulse = -2 * np.pi * amplitude * (np.exp(-(Kt**2 * delta**2) / 2.0) / Kt)
            return np.array([0.0, pulse])

    @staticmethod
    def default_impulse(inst, t_0, Sigma_impluse):

        Kt0 = njit_Kfunc(t_0, inst.kx, inst.kyc, inst.kappa, inst.Omega0)

        if np.abs(Kt0)<1e-3:
            return 0.0
        else:
            return -(2*inst.G*np.pi)/Kt0 * Sigma_impluse

    def get_surface_density(self, results, t_grid):
        psi_values = results[:,0]
        # Vectorized call to JIT function
        Kts = np.array([njit_Kfunc(t, self.kx, self.kyc, self.kappa, self.Omega0) for t in t_grid[1:]])
        return psi_values * (Kts / (-2.0 * np.pi))

    def solve(self, t_grid, fext_params=None, impulse_params=None):
        fext_p = fext_params if fext_params else {}
        imp_p = impulse_params if impulse_params else {}
        n = len(t_grid) - 1
        
        # 1. Pre-calculate forcing values in Python once
        # This avoids calling Python objects inside the Numba prange loop
        fext_vals = np.zeros((len(t_grid), 2))
        for idx, t in enumerate(t_grid):
            fext_vals[idx] = self._fext_func(t, self, **fext_p)
            
        impulse_val = self._impulse_func(self, **imp_p)
        
        # 2. Build Matrix (Already JIT-accelerated in your original code)
        h_mat = build_h_matrix_gas_only_njit(t_grid, *self.params)
        
        # 3. Build B Vector (Newly optimized with prange and trapezoidal integration)
        # We pass only the relevant parameters to the JIT function
        bb_vec = build_bb_vec_gas_njit(
            t_grid, self.kx, self.kyc, self.Sigma_g, self.kappa, self.Omega0, 
            fext_vals, impulse_val
        )
            
        # 4. Solve Linear System
        sol_flat = solve(h_mat, bb_vec)
        return sol_flat.reshape(n, 2)
#===================================================================================

class Gas_plus_stars_integrator:
    def __init__(self, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, cssq, G, 
                 fext_func=None, impulse_func=None):
        self.params = (kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, cssq, G)
        self.kx, self.kyc, self.Sigma_g, self.Sigma_s, self.kappa, self.Omega0, self.sigma_x, self.cssq, self.G = self.params
        
        self._fext_func = fext_func if fext_func else self.default_gaussian_pulse
        self._impulse_func = impulse_func if impulse_func else self.default_impulse

    @staticmethod
    def default_gaussian_pulse(t, inst, delta=0.05, amplitude=1.0):
        Kt = njit_Kfunc(t, inst.kx, inst.kyc, inst.kappa, inst.Omega0)
        if np.abs(Kt)<1e-3:
            return np.array([0.0, 0.0, 0])
        else:
            
            pulse = -2 * np.pi * amplitude * (np.exp(-(Kt**2 * delta**2) / 2.0) / Kt)
            return np.array([0.0, 0.0, pulse])

    @staticmethod
    def default_impulse(inst, t_0, Sigma_impluse):

        Kt0 = njit_Kfunc(t_0, inst.kx, inst.kyc, inst.kappa, inst.Omega0)
        if np.abs(Kt0)<1e-3:
            return 0.0
        else:
            return -(2*inst.G*np.pi)/Kt0 * Sigma_impluse

    def get_stellar_surface_density(self, results, t_grid):
        psi_values = results[:, 2]
        # Vectorized call to JIT function
        Kts = np.array([njit_Kfunc(t, self.kx, self.kyc, self.kappa, self.Omega0) for t in t_grid[1:]])
        return psi_values * (Kts / (-2.0 * np.pi))
    
    def get_gas_surface_density(self, results, t_grid):
        psi_values = results[:, 0]
        # Vectorized call to JIT function
        Kts = np.array([njit_Kfunc(t, self.kx, self.kyc, self.kappa, self.Omega0) for t in t_grid[1:]])
        return psi_values * (Kts / (-2.0 * np.pi))

    def solve(self, t_grid, fext_params=None, impulse_params=None):
        fext_p = fext_params if fext_params else {}
        imp_p = impulse_params if impulse_params else {}
        n = len(t_grid) - 1
        
        # Pre-calculate vectorized pulse values (assuming pulse returns [0, 0, val])
        fext_vals = np.array([self._fext_func(t, self, **fext_p) for t in t_grid])
        impulse_val = self._impulse_func(self, **imp_p)
        
        h_mat = build_h_matrix_njit(t_grid, *self.params)
        bb_vec = build_bb_vec_combined_njit(t_grid, *self.params, fext_vals, impulse_val)
        
        sol_flat = solve(h_mat, bb_vec)
        return sol_flat.reshape(n, 3)
    
