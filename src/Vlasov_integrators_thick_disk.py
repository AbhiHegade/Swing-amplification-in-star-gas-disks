import numpy as np
from scipy.linalg import solve
from numba import njit, prange

# =================================================================
# Kernel function definitions: Common for most of the calculations
# =================================================================

@njit
def njit_afunc(kappa, Omega0):
    """Oort A."""
    return Omega0 - (kappa**2) / (4 * Omega0)

@njit
def njit_Bfunc(kappa, Omega0):
    """Oort B."""
    return - (kappa**2) / (4 * Omega0)

@njit
def njit_k0func(t, kx, kyc, kappa, Omega0):
    """k0 function from Eq.(24) of the paper."""
    a = njit_afunc(kappa, Omega0)
    return (2 * a * t * kyc + kx)

@njit
def njit_Kfunc(t, kx, kyc, kappa, Omega0):
    """k(t) function from Eq.(25) of the paper."""
    a = njit_afunc(kappa, Omega0)
    return np.sqrt(kyc**2 + (2 * a * t * kyc + kx)**2)

@njit
def njit_Kfunc_vec(t_arr, kx, kyc, kappa, Omega0):
    """Vectorized version of Kfunc to safely handle t_grid array slices."""
    a = njit_afunc(kappa, Omega0)
    return np.sqrt(kyc**2 + (2.0 * a * t_arr * kyc + kx)**2)

@njit
def njit_s1func(t, kx, kyc, Sigma_g, kappa, Omega0, cssq, Hgas, G):
    """S1 function from Eq. (C51) of the paper."""
    if np.abs(kyc) < 1e-10:
        return 0.0
    else:
        K = njit_Kfunc(t, kx, kyc, kappa, Omega0)
        A = njit_afunc(kappa, Omega0)
        k0t = njit_k0func(t, kx, kyc, kappa, Omega0)
        return (4*A*Hgas*kyc*k0t)/K
        
@njit
def njit_ssqfunc(t, kx, kyc, Sigma_g, kappa, Omega0, cssq, Hgas, G):
    """Ssq function from Eq. (C52) of the paper."""
    if np.abs(kyc) < 1e-10:
        return cssq*kx**2 - 2*G*kx*np.pi*Sigma_g + kappa**2 + Hgas*kx*(cssq*kx**2 + kappa**2)
    else:
        K = njit_Kfunc(t, kx, kyc, kappa, Omega0)
        A = njit_afunc(kappa, Omega0)
        return (kappa**2 + (12*A**2 * kyc**4)/K**4 
                - (2*A*kyc**2 * (4*A*Omega0 + kappa**2))/(Omega0*K**2) 
                - 2*G*np.pi*Sigma_g*K + cssq*K**2 
                + Hgas*((16*A**2 * kyc**4)/K**3 
                - (2*A*kyc**2 * (4*A*Omega0 + kappa**2))/(Omega0*K) 
                + kappa**2 * K + cssq*K**3))

@njit
def njit_Kpot_kernel_func(t, s, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G):
    """Kernel for the potential response of the stellar disk, see Eq. (C53) of the paper."""
    if np.abs(kyc) < 1e-10 and np.abs(kx) < 1e-10:
        return 0.0
    else:
        C_ = np.cos(kappa*(s-t)) 
        S_ = np.sin(kappa*(s-t))
        B = njit_Bfunc(kappa, Omega0)

        k0t = njit_k0func(t, kx, kyc, kappa, Omega0)
        k0s = njit_k0func(s, kx, kyc, kappa, Omega0)

        Kt = njit_Kfunc(t, kx, kyc, kappa, Omega0)

        Y1 = k0t - k0s
        Y2 = 4*kyc**2 * Omega0**2 + kappa**2 * k0s * k0t

        r1_times_kyc = (kyc*Y1*C_)/(8.*B) + (Y2*S_)/(16.*B*Omega0*kappa)
        r2 = Y2 + (Y1**2 * kappa**2)/2. - Y2*C_ + 2*kyc*Omega0*Y1*kappa*S_

        factor = (8*G*np.pi*r1_times_kyc*Sigma_s*kappa)/(Kt*(1 + Hs*Kt))
        exparg = (-sigma_x*sigma_x)/(kappa**4) * r2

        return factor * np.exp(exparg)
    
@njit
def njit_K_kernel_func(t, s, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G):
    """
    Passive stellar density response to a razor-thin external surface-density impulse.
    Since the source is external, only the target-time potential-to-density conversion
    applies: K = Kpot_thick * k(t) * (1 + Hs*k(t)) / k(s).
    
    """
    if np.abs(kyc) < 1e-10 and np.abs(kx) < 1e-10:
        return 0.0
    else:

        Kpotthick = njit_Kpot_kernel_func(t, s, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G)

        Kt = njit_Kfunc(t, kx, kyc, kappa, Omega0)
        Ks = njit_Kfunc(s, kx, kyc, kappa, Omega0)

        return Kpotthick*Kt*(1+Hs*Kt)/Ks

# ==========================================
# 2. JIT-Comnp.piled Matrix Construction
# ==========================================
@njit
def build_h_matrix_njit(t_grid, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, cssq, Hg, Hs, G):
    """
    Build H matrix from Eq. (B46) of the paper from the combined stellar and gas dynamics.
    We assume that t_grid = [t_0, t_1, t_2, t_3, .., t_n].
    """
    n = len(t_grid)-1 # because of t_grid convention.
    h = t_grid[1] - t_grid[0]
    h_mat = np.zeros((3 * n, 3 * n))
#-------------------------------------------------
    for i in range(n):
        t = t_grid[i+1] # i+1 because of t_grid convention.
        for j in range(i + 1):
            tp = t_grid[j+1] # j+1 because of t_grid convention.
            
            kmat = njit_Kpot_kernel_func(t, tp, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G)
            ssqthick = njit_ssqfunc(tp, kx, kyc, Sigma_g, kappa, Omega0, cssq, Hg, G)
            Ktp = njit_Kfunc(tp, kx, kyc, kappa, Omega0)
            s1thick = njit_s1func(tp, kx, kyc, Sigma_g, kappa, Omega0, cssq, Hg, G)
            
            r, c = 3 * i, 3 * j
            
            if j==i:
                h_mat[r, c] = 1.0
                h_mat[r, c+1] = -(h / 2.0)
                h_mat[r+1, c] = -(h / 2.0) * (-ssqthick)/(1 + Hg*Ktp)
                h_mat[r+1, c+1] = 1.0 - (h/2.0)*(-s1thick/((1 + Hg*Ktp)))
                h_mat[r+1, c+2] = -(h / 2.0) * (2 * np.pi * G * Sigma_g * Ktp)/(1 + Hg*Ktp)
                h_mat[r+2, c] = -(h / 2.0) * kmat
                h_mat[r+2, c+2] = 1.0 - (h / 2.0) * kmat
            else:
                h_mat[r, c+1] = -h
                h_mat[r+1, c] = -h * (-ssqthick)/(1 + Hg*Ktp)
                h_mat[r+1, c+1] = -(h)*(-s1thick/((1 + Hg*Ktp)))
                h_mat[r+1, c+2] = -h * (2 * np.pi * G * Sigma_g * Ktp)/(1 + Hg*Ktp)
                h_mat[r+2, c] = -h * kmat
                h_mat[r+2, c+2] = -h * kmat
                
    return h_mat

@njit
def build_h_matrix_stars_only_njit(t_grid, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G):
    """
    Build H matrix from Eq. (B46) of the paper for stellar perturbations only.
    We assume that t_grid = [t_0, t_1, t_2, t_3, .., t_n].
    """

    n = len(t_grid)-1 # because of t_grid convention.
    h = t_grid[1] - t_grid[0]
    h_mat = np.zeros(( n,  n))
#-------------------------------------------------
    for i in range(n):
        t = t_grid[i+1] # i+1 because of t_grid convention.
        for j in range(i + 1):
            tp = t_grid[j+1] # j+1 because of t_grid convention.

            kmat = njit_Kpot_kernel_func(t, tp, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G)
            
            if i == j:
                h_mat[i, j] = 1.0 - (h / 2.0) * kmat
            else:
                h_mat[i, j] = -h * kmat
                
    return h_mat

@njit
def build_h_matrix_gas_only_njit(t_grid, kx, kyc, Sigma_g, kappa, Omega0, cssq, Hg, G):

    """
    Build H matrix from Eq. (B46) of the paper for gas perturbations only.
    We assume that t_grid = [t_0, t_1, t_2, t_3, .., t_n].
    """

    n = len(t_grid)-1 # because of t_grid convention.
    h = t_grid[1] - t_grid[0]
    h_mat = np.zeros((2 * n, 2 * n))
#-------------------------------------------------
    for i in range(n):
        t = t_grid[i+1] # i+1 because of t_grid convention.
        for j in range(i + 1):
            tp = t_grid[j+1] # j+1 because of t_grid convention.
            
            ssqthick = njit_ssqfunc(tp, kx, kyc, Sigma_g, kappa, Omega0, cssq, Hg, G)
            Ktp = njit_Kfunc(tp, kx, kyc, kappa, Omega0)
            s1thick = njit_s1func(tp, kx, kyc, Sigma_g, kappa, Omega0, cssq, Hg, G)
            
            r, c = 2 * i, 2 * j

            if i == j:
                h_mat[r, c] = 1.0
                h_mat[r, c+1] = -(h / 2.0)
                h_mat[r+1, c] = -(h / 2.0) * (-ssqthick)/(1 + Hg*Ktp)
                h_mat[r+1, c+1] = 1.0 - (h/2.0)*(-s1thick/((1 + Hg*Ktp)))
            else:
                h_mat[r, c+1] = -h
                h_mat[r+1, c] =-h * (-ssqthick)/(1 + Hg*Ktp)
                h_mat[r+1, c+1] = -(h)*(-s1thick/((1 + Hg*Ktp)))

    return h_mat


@njit
def build_h_matrix_two_fluid_njit(t_grid, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, cssq, Hg, Hs, G):
    """
    Build H matrix from Eq. (B46) of the paper from the combined two gas dynamics.
    We assume that t_grid = [t_0, t_1, t_2, t_3, .., t_n].
    """
    n = len(t_grid)-1 # because of t_grid convention.
    h = t_grid[1] - t_grid[0]
    h_mat = np.zeros((4 * n, 4 * n))
#-------------------------------------------------
    for i in range(n):
        t = t_grid[i+1] # i+1 because of t_grid convention.
        for j in range(i + 1):
            tp = t_grid[j+1] # j+1 because of t_grid convention.
            
            ssqthick = njit_ssqfunc(tp, kx, kyc, Sigma_g, kappa, Omega0, cssq, Hg, G)
            ssqthick_stars = njit_ssqfunc(tp, kx, kyc, Sigma_s, kappa, Omega0, sigma_x*sigma_x, Hs, G)
            Ktp = njit_Kfunc(tp, kx, kyc, kappa, Omega0)
            s1thick = njit_s1func(tp, kx, kyc, Sigma_g, kappa, Omega0, cssq, Hg, G)
            s1thick_stars = njit_s1func(tp, kx, kyc, Sigma_s, kappa, Omega0, sigma_x*sigma_x, Hs, G)
            
            r, c = 4 * i, 4 * j
            
            if j==i:
                h_mat[r, c] = 1.0
                h_mat[r, c+1] = -(h / 2.0)
                h_mat[r+1, c] = -(h / 2.0) * (-ssqthick)/(1 + Hg*Ktp)
                h_mat[r+1, c+1] = 1.0 - (h/2.0)*(-s1thick/((1 + Hg*Ktp)))
                h_mat[r+1, c+2] = -(h / 2.0) * (2 * np.pi * G * Sigma_g * Ktp)/(1 + Hg*Ktp)
                h_mat[r+2, c+2] = 1.0 
                h_mat[r+2, c+3] = -(h/2.0)
                h_mat[r+3, c] = -(h / 2.0) * (2 * np.pi * G * Sigma_s * Ktp)/(1 + Hs*Ktp)
                h_mat[r+3,c+2] = -(h / 2.0) * (-ssqthick_stars)/(1 + Hs*Ktp)
                h_mat[r+3,c+3] = 1.0 - (h/2.0)*(-s1thick_stars/((1 + Hs*Ktp)))
            else:
                h_mat[r, c+1] = -h

                h_mat[r+1, c] = -h * (-ssqthick) / (1.0 + Hg * Ktp)
                h_mat[r+1, c+1] = -h * (-s1thick) / (1.0 + Hg * Ktp)
                h_mat[r+1, c+2] = -h*((2 * np.pi * G * Sigma_g * Ktp)/(1 + Hg*Ktp))
                h_mat[r+2, c+3] = -h

                h_mat[r+3, c] = - h *(2 * np.pi * G * Sigma_s * Ktp)/(1 + Hs*Ktp)
                h_mat[r+3, c+2] = -h *  (-ssqthick_stars)/(1 + Hs*Ktp)
                h_mat[r+3, c+3] = -h * (-s1thick_stars/((1 + Hs*Ktp)))
                
    return h_mat
# =============================
# Source functions
# =============================

@njit
def stellar_only_forcing(tf, t_grid_for_integration, limit_idx, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G, Sigma_ext_vals):
    """
    Returns int_{t_0}^{tf} Kpot(t,tp) * (-2*np.pi*G *Sigma_ext(tp))/k(tp).

    Sigma_ext_vals contains [Sigma_ext(t_0), Sigma_ext(t_1), Sigma_ext(t_2),...].

    t_grid_for_integration is [t_0, t_1, t_2, t_3, t_4 ,..].

    limit_idx contains the index where t_grid_for_integration[limit_idx-1] = tf
    """
    total = 0.0
    if limit_idx < 2:
        return 0.0

    h = t_grid_for_integration[1] - t_grid_for_integration[0]

    tp = t_grid_for_integration[0]
    ktp = njit_Kfunc(tp, kx, kyc, kappa, Omega0)
    k_prev = njit_Kpot_kernel_func(tf, t_grid_for_integration[0], kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G)* (-2*np.pi*G)/ktp
    f_prev = Sigma_ext_vals[0]
    
    for i in range(1, limit_idx):
        tp = t_grid_for_integration[i]
        ktp = njit_Kfunc(tp, kx, kyc, kappa, Omega0)

        k_curr = njit_Kpot_kernel_func(tf, tp, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G)* (-2*np.pi*G)/ktp

        f_curr = Sigma_ext_vals[i]
        
        total += 0.5 * h * (k_prev * f_prev + k_curr * f_curr)
        
        k_prev = k_curr
        f_prev = f_curr
        
    return total

@njit(parallel=True)
def build_bb_vec_stellar_njit(t_grid, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G, Sigma_ext_vals, impulse_val):
    """
    Build B_{n,1} vec from Eq. (B43) for stellar dynamics only.
    """
    n = len(t_grid) - 1
    bb_vec = np.zeros(n)
    t_start = t_grid[0]

    kt0 = njit_Kfunc(t_start, kx, kyc, kappa, Omega0)
    
    for i in prange(n):
        t_target = t_grid[i+1]
        forcing_res = stellar_only_forcing(t_target, t_grid, i+2, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G, Sigma_ext_vals)

        k_start = njit_Kpot_kernel_func(t_target, t_start, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G) *(-2*np.pi*G)/kt0

        bb_vec[i] = forcing_res + k_start * impulse_val
        
    return bb_vec

@njit
def gas_forcing_njit(t_grid, limit_idx, kx, kyc, Sigma_g, kappa, Omega0, Hg, G, Sigma_ext_vals):

    """
    Build gas dynamics forcing function.
    """

    total = np.zeros(2)
    if limit_idx < 2:
        return total
    
    ktp_prev = njit_Kfunc(t_grid[0], kx, kyc, kappa, Omega0)
    val_prev = -(2 * np.pi * G)**2 * Sigma_g / (1 + Hg*ktp_prev) * Sigma_ext_vals[0]

    h = t_grid[1] - t_grid[0]
    
    for i in range(1, limit_idx):
        tp = t_grid[i]
        ktp_curr = njit_Kfunc(tp, kx, kyc, kappa, Omega0)
        val_curr = -(2 * np.pi * G)**2 * Sigma_g /(1 + Hg*ktp_curr) *  Sigma_ext_vals[i]
        
        
        total[1] += 0.5 * h * (val_prev + val_curr)
        val_prev = val_curr
        
    return total

@njit(parallel=True)
def build_bb_vec_gas_njit(t_grid, kx, kyc, Sigma_g, kappa, Omega0, cssq, Hg, G, Sigma_ext_vals, impulse_val):
    """
    Build B_{n,1} vec from Eq. (B43) for gas dynamics only.
    """

    n = len(t_grid) - 1
    bb_vec = np.zeros(2 * n)
    t_start = t_grid[0]
    kt0 = njit_Kfunc(t_start, kx, kyc, kappa, Omega0)
    
    for i in prange(n):
        # t_target = t_grid[i+1]
        forcing = gas_forcing_njit(t_grid, i+2, kx, kyc, Sigma_g, kappa, Omega0, Hg, G, Sigma_ext_vals)
        impulse_res = np.array([0.0, -(2 * np.pi * G)**2 * Sigma_g * impulse_val/(1+Hg*kt0) ])
        
        res = forcing + impulse_res
        bb_vec[2*i] = res[0]
        bb_vec[2*i+1] = res[1]
        
    return bb_vec

@njit
def fast_combined_forcing_njit(t_target, t_grid, limit_idx, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, Hg, Hs, G, Sigma_ext_vals):
    """
    Build gas + stellar dynamics forcing function.
    """
    total = np.zeros(3)
    if limit_idx < 2:
        return total
    
    # def get_val(tp, sigma_ext_val):
    #     kmat = njit_Kpot_kernel_func(t_target, tp, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G)
    #     ktp = njit_Kfunc(tp, kx, kyc, kappa, Omega0)
    #     return np.array([0.0,  -(2 * np.pi * G)**2 * Sigma_g * 1.0/(1+Hg*ktp) * sigma_ext_val, - (2* np.pi* G)* kmat * 1.0/ktp * sigma_ext_val])

    # Calculate index 0 directly 
    tp_prev = t_grid[0]
    kmat_prev = njit_Kpot_kernel_func(t_target, tp_prev, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G)
    ktp_prev = njit_Kfunc(tp_prev, kx, kyc, kappa, Omega0)
    
    val_prev = np.array([
        0.0,  
        -(2 * np.pi * G)**2 * Sigma_g / (1 + Hg * ktp_prev) * Sigma_ext_vals[0],
        -(2 * np.pi * G) * kmat_prev / ktp_prev * Sigma_ext_vals[0]
    ])

    h = t_grid[1] - t_grid[0]
    for i in range(1, limit_idx):
        tp_curr = t_grid[i]
        kmat_curr = njit_Kpot_kernel_func(t_target, tp_curr, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G)
        ktp_curr = njit_Kfunc(tp_curr, kx, kyc, kappa, Omega0)
        
        val_curr = np.array([
            0.0,  
            -(2 * np.pi * G)**2 * Sigma_g / (1 + Hg * ktp_curr) * Sigma_ext_vals[i],
            -(2 * np.pi * G) * kmat_curr / ktp_curr * Sigma_ext_vals[i]
        ])

        total += 0.5 * h * (val_prev + val_curr)
        val_prev = val_curr
        
    return total

@njit(parallel=True)
def build_bb_vec_combined_njit(t_grid, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, cssq, Hg, Hs, G, fext_vals, impulse_val):
    """
    Build B_{n,1} vec from Eq. (B43) for stellar + gas dynamics.
    """
     
    n = len(t_grid) - 1
    bb_vec = np.zeros(3 * n)
    t_start = t_grid[0]
    kt0 = njit_Kfunc(t_start, kx, kyc, kappa, Omega0)
    
    for i in prange(n):
        t_target = t_grid[i+1]
        forcing = fast_combined_forcing_njit(t_target, t_grid, i+2, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, Hg, Hs, G, fext_vals)
        kmat0 = njit_Kpot_kernel_func(t_target, t_start, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G)
        
        impulse_res = np.array([0.0, -(2 * np.pi * G)**2 * Sigma_g /(1 + Hg*kt0) * impulse_val, -(2*np.pi*G)* (1.0/kt0) * kmat0 * impulse_val])
        res = forcing + impulse_res
        
        bb_vec[3*i] = res[0]
        bb_vec[3*i+1] = res[1]
        bb_vec[3*i+2] = res[2]
        
    return bb_vec

@njit
def fast_two_fluid_combined_forcing_njit(t_target, t_grid, limit_idx, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, Hg, Hs, G, Sigma_ext_vals):
    """
    Build two fluid dynamics forcing function.
    """
    total = np.zeros(4)
    if limit_idx < 2:
        return total
    
    # def get_val(tp, sigma_ext_val):
    #     kmat = njit_Kpot_kernel_func(t_target, tp, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G)
    #     ktp = njit_Kfunc(tp, kx, kyc, kappa, Omega0)
    #     return np.array([0.0,  -(2 * np.pi * G)**2 * Sigma_g * 1.0/(1+Hg*ktp) * sigma_ext_val, - (2* np.pi* G)**2 * Sigma_s * 1.0/(1+Hs*ktp) * sigma_ext_val])

    # Calculate index 0 directly 
    tp_prev = t_grid[0]
    ktp_prev = njit_Kfunc(tp_prev, kx, kyc, kappa, Omega0)
    
    val_prev = np.array([
        0.0,  
        -(2 * np.pi * G)**2 * Sigma_g / (1 + Hg * ktp_prev) * Sigma_ext_vals[0],
       0.0,
        -(2 * np.pi * G)**2 * Sigma_s / (1 + Hs * ktp_prev) * Sigma_ext_vals[0],
    ])

    h = t_grid[1] - t_grid[0]
    for i in range(1, limit_idx):
        tp_curr = t_grid[i]
        ktp_curr = njit_Kfunc(tp_curr, kx, kyc, kappa, Omega0)
        
        val_curr = np.array([
            0.0,  
            -(2 * np.pi * G)**2 * Sigma_g / (1 + Hg * ktp_curr) * Sigma_ext_vals[i],
           0.0,
           -(2 * np.pi * G)**2 * Sigma_s / (1 + Hs * ktp_curr) * Sigma_ext_vals[i],
        ])

        total += 0.5 * h * (val_prev + val_curr)
        val_prev = val_curr
        
    return total

@njit(parallel=True)
def build_bb_vec_two_fluid_combined_njit(t_grid, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, cssq, Hg, Hs, G, fext_vals, impulse_val):
    """
    Build B_{n,1} vec from Eq. (B43) for two fluid dynamics.
    """
     
    n = len(t_grid) - 1
    bb_vec = np.zeros(4 * n)
    t_start = t_grid[0]
    kt0 = njit_Kfunc(t_start, kx, kyc, kappa, Omega0)
    
    for i in prange(n):
        t_target = t_grid[i+1]
        forcing = fast_two_fluid_combined_forcing_njit(t_target, t_grid, i+2, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, Hg, Hs, G, fext_vals)
        
        impulse_res = np.array([0.0, -(2 * np.pi * G)**2 * Sigma_g /(1 + Hg*kt0) * impulse_val, 0.0, -(2 * np.pi * G)**2 * Sigma_s /(1 + Hs*kt0) * impulse_val])

        res = forcing + impulse_res
        
        bb_vec[4*i] = res[0]
        bb_vec[4*i+1] = res[1]
        bb_vec[4*i+2] = res[2]
        bb_vec[4*i+3] = res[3]
        
    return bb_vec

# ==========================================
# Class Wrappers
# ==========================================

class Stellar_integrator:
    def __init__(self, kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G, fext_func=None, impulse_func=None):
        self.kx, self.kyc, self.Sigma_s, self.kappa, self.Omega0, self.sigma_x, self.Hs, self.G = kx, kyc, Sigma_s, kappa, Omega0, sigma_x, Hs, G
        self._fext_func = fext_func if fext_func else self.default_gaussian_pulse
        self._impulse_func = impulse_func if impulse_func else self.default_impulse

    @staticmethod
    def default_gaussian_pulse(t, inst, delta=0.05, amplitude=1.0):
        Kt = njit_Kfunc(t, inst.kx, inst.kyc, inst.kappa, inst.Omega0)
        return 0.0 if np.abs(Kt)<1e-3 else  amplitude * (np.exp(-(Kt**2 * delta**2) / 2.0) )/(4.0*(np.pi)**2)

    @staticmethod
    def default_impulse(inst, t_0, Sigma_impluse):
        Kt0 = njit_Kfunc(t_0, inst.kx, inst.kyc, inst.kappa, inst.Omega0)
        return 0.0 if np.abs(Kt0)<1e-3 else Sigma_impluse

    def get_surface_density(self, results, t_grid):
        Kts = njit_Kfunc_vec(t_grid[1:], self.kx, self.kyc, self.kappa, self.Omega0)
        return results * (Kts*(1 + self.Hs*Kts) / (-2.0 * np.pi * self.G))

    def solve(self, t_grid, fext_params=None, impulse_params=None):
        fext_p = fext_params if fext_params else {}
        imp_p = impulse_params if impulse_params else {}
        n = len(t_grid) - 1
        
        fext_vals = np.array([self._fext_func(t, self, **fext_p) for t in t_grid])
        impulse_val = self._impulse_func(self, **imp_p)

        h_mat = build_h_matrix_stars_only_njit(t_grid, self.kx, self.kyc, self.Sigma_s, self.kappa, self.Omega0, self.sigma_x, self.Hs, self.G)
        bb_vec = build_bb_vec_stellar_njit(t_grid, self.kx, self.kyc, self.Sigma_s, self.kappa, self.Omega0, self.sigma_x, self.Hs, self.G, fext_vals, impulse_val)
            
        return solve(h_mat, bb_vec)


class Gas_integrator:
    def __init__(self, kx, kyc, Sigma_g, kappa, Omega0, cssq, Hg, G, fext_func=None, impulse_func=None):
        self.kx, self.kyc, self.Sigma_g, self.kappa, self.Omega0, self.cssq, self.Hg, self.G = kx, kyc, Sigma_g, kappa, Omega0, cssq, Hg, G
        self._fext_func = fext_func if fext_func else self.default_gaussian_pulse
        self._impulse_func = impulse_func if impulse_func else self.default_impulse

    @staticmethod
    def default_gaussian_pulse(t, inst, delta=0.05, amplitude=1.0):
        Kt = njit_Kfunc(t, inst.kx, inst.kyc, inst.kappa, inst.Omega0)
        if np.abs(Kt)<1e-3: return 0.0
        pulse = amplitude * (np.exp(-(Kt**2 * delta**2) / 2.0) )/(4.0*(np.pi)**2)
        return pulse

    @staticmethod
    def default_impulse(inst, t_0, Sigma_impluse):
        Kt0 = njit_Kfunc(t_0, inst.kx, inst.kyc, inst.kappa, inst.Omega0)
        return 0.0 if np.abs(Kt0)<1e-3 else Sigma_impluse

    def get_surface_density(self, results, t_grid):
        psi_values = results[:,0]
        Kts = njit_Kfunc_vec(t_grid[1:], self.kx, self.kyc, self.kappa, self.Omega0)
        return psi_values * (Kts*(1+self.Hg*Kts) / (-2.0 * np.pi *self.G))

    def solve(self, t_grid, fext_params=None, impulse_params=None):
        fext_p = fext_params if fext_params else {}
        imp_p = impulse_params if impulse_params else {}
        n = len(t_grid) - 1
        
        fext_vals = np.zeros(len(t_grid))
        for idx, t in enumerate(t_grid):
            fext_vals[idx] = self._fext_func(t, self, **fext_p)
            
        impulse_val = self._impulse_func(self, **imp_p)
        h_mat = build_h_matrix_gas_only_njit(t_grid, self.kx, self.kyc, self.Sigma_g, self.kappa, self.Omega0, self.cssq, self.Hg, self.G)
        
        bb_vec = build_bb_vec_gas_njit(
            t_grid, self.kx, self.kyc, self.Sigma_g, self.kappa, self.Omega0, self.cssq, self.Hg, self.G,
            fext_vals, impulse_val
        )
            
        sol_flat = solve(h_mat, bb_vec)
        return sol_flat.reshape(n, 2)


class Gas_plus_stars_integrator:
    def __init__(self, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, cssq, Hg, Hs, G, fext_func=None, impulse_func=None):
        self.kx, self.kyc, self.Sigma_g, self.Sigma_s, self.kappa, self.Omega0, self.sigma_x, self.cssq, self.Hg, self.Hs, self.G = kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, cssq, Hg, Hs, G
        self._fext_func = fext_func if fext_func else self.default_gaussian_pulse
        self._impulse_func = impulse_func if impulse_func else self.default_impulse

    @staticmethod
    def default_gaussian_pulse(t, inst, delta=0.05, amplitude=1.0):
        Kt = njit_Kfunc(t, inst.kx, inst.kyc, inst.kappa, inst.Omega0)
        if np.abs(Kt)<1e-3: return 0.0 
        pulse = amplitude * (np.exp(-(Kt**2 * delta**2) / 2.0) )/(4.0*(np.pi)**2)
        return pulse

    @staticmethod
    def default_impulse(inst, t_0, Sigma_impluse):
        Kt0 = njit_Kfunc(t_0, inst.kx, inst.kyc, inst.kappa, inst.Omega0)
        return 0.0 if np.abs(Kt0)<1e-3 else Sigma_impluse

    def get_stellar_surface_density(self, results, t_grid):
        psi_values = results[:, 2]
        Kts = njit_Kfunc_vec(t_grid[1:], self.kx, self.kyc, self.kappa, self.Omega0)
        return psi_values * (Kts *(1 + self.Hs*Kts) / (-2.0 * np.pi *self.G))
    
    def get_gas_surface_density(self, results, t_grid):
        psi_values = results[:, 0]
        Kts = njit_Kfunc_vec(t_grid[1:], self.kx, self.kyc, self.kappa, self.Omega0)
        return psi_values * (Kts*(1 + self.Hg*Kts) / (-2.0 * np.pi *self.G))

    def solve(self, t_grid, fext_params=None, impulse_params=None):
        fext_p = fext_params if fext_params else {}
        imp_p = impulse_params if impulse_params else {}
        n = len(t_grid) - 1
        
        fext_vals = np.array([self._fext_func(t, self, **fext_p) for t in t_grid])
        impulse_val = self._impulse_func(self, **imp_p)
        
        h_mat = build_h_matrix_njit(t_grid, self.kx, self.kyc, self.Sigma_g, self.Sigma_s, self.kappa, self.Omega0, self.sigma_x, self.cssq, self.Hg, self.Hs, self.G)
        bb_vec = build_bb_vec_combined_njit(t_grid, self.kx, self.kyc, self.Sigma_g, self.Sigma_s, self.kappa, self.Omega0, self.sigma_x, self.cssq, self.Hg, self.Hs, self.G, fext_vals, impulse_val)
        
        sol_flat = solve(h_mat, bb_vec)
        return sol_flat.reshape(n, 3)
    

class two_fluid_integrator:
    def __init__(self, kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, cssq, Hg, Hs, G, fext_func=None, impulse_func=None):
        self.kx, self.kyc, self.Sigma_g, self.Sigma_s, self.kappa, self.Omega0, self.sigma_x, self.cssq, self.Hg, self.Hs, self.G = kx, kyc, Sigma_g, Sigma_s, kappa, Omega0, sigma_x, cssq, Hg, Hs, G
        self._fext_func = fext_func if fext_func else self.default_gaussian_pulse
        self._impulse_func = impulse_func if impulse_func else self.default_impulse

    @staticmethod
    def default_gaussian_pulse(t, inst, delta=0.05, amplitude=1.0):
        Kt = njit_Kfunc(t, inst.kx, inst.kyc, inst.kappa, inst.Omega0)
        if np.abs(Kt)<1e-3: return 0.0 
        pulse = amplitude * (np.exp(-(Kt**2 * delta**2) / 2.0) )/(4.0*(np.pi)**2)
        return pulse

    @staticmethod
    def default_impulse(inst, t_0, Sigma_impluse):
        Kt0 = njit_Kfunc(t_0, inst.kx, inst.kyc, inst.kappa, inst.Omega0)
        return 0.0 if np.abs(Kt0)<1e-3 else Sigma_impluse

    def get_stellar_surface_density(self, results, t_grid):
        psi_values = results[:, 2]
        Kts = njit_Kfunc_vec(t_grid[1:], self.kx, self.kyc, self.kappa, self.Omega0)
        return psi_values * (Kts *(1 + self.Hs*Kts) / (-2.0 * np.pi *self.G))
    
    def get_gas_surface_density(self, results, t_grid):
        psi_values = results[:, 0]
        Kts = njit_Kfunc_vec(t_grid[1:], self.kx, self.kyc, self.kappa, self.Omega0)
        return psi_values * (Kts*(1 + self.Hg*Kts) / (-2.0 * np.pi *self.G))

    def solve(self, t_grid, fext_params=None, impulse_params=None):
        fext_p = fext_params if fext_params else {}
        imp_p = impulse_params if impulse_params else {}
        n = len(t_grid) - 1
        
        fext_vals = np.array([self._fext_func(t, self, **fext_p) for t in t_grid])
        impulse_val = self._impulse_func(self, **imp_p)
        
        h_mat = build_h_matrix_two_fluid_njit(t_grid, self.kx, self.kyc, self.Sigma_g, self.Sigma_s, self.kappa, self.Omega0, self.sigma_x, self.cssq, self.Hg, self.Hs, self.G)
        bb_vec = build_bb_vec_two_fluid_combined_njit(t_grid, self.kx, self.kyc, self.Sigma_g, self.Sigma_s, self.kappa, self.Omega0, self.sigma_x, self.cssq, self.Hg, self.Hs, self.G, fext_vals, impulse_val)
        
        sol_flat = solve(h_mat, bb_vec)
        return sol_flat.reshape(n, 4)