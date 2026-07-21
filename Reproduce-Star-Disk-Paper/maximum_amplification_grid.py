import sys
import os
import matplotlib.pyplot as plt
import time
from numba import njit
from pathlib import Path
codedir = "../src/"

sys.path.append(os.path.abspath(codedir))

from Vlasov_integrators_thick_disk import *

from dimensionless_to_physical_units import *

from axisymmetric_stability import *

import argparse

#-------------------------------------------------------
kxval = 0.0
kappa_val=1
#-------------------------------------------------------
def Kfunc_kernel(t,ti, kyval, Sigma_s_val, sigma_x_val, Omega0_val, hs):
    return njit_K_kernel_func(t, ti, kxval, kyval, Sigma_s_val, kappa_val, Omega0_val, sigma_x_val, hs*sigma_x_val/kappa_val, G=1)


def Kfunc_kernel_max(tf,ti, kyval, Sigma_s_val, sigma_x_val, Omega0_val, hs, nvals = 500):
    tgrid= np.linspace(ti,tf, nvals)

    Kvals = np.abs(np.array([Kfunc_kernel(t,ti, kyval, Sigma_s_val, sigma_x_val, Omega0_val, hs) for t in tgrid]))

    return float(np.max(Kvals))


def get_self_gravity_response_max_gas_plus_stars(tf,ti, kyval, Rvals, Qgas, Qstars, hg, hs,  nvals = 500):
    _,_,Sigma_g_val, Sigma_s_val, sigma_x_val, cssq_val , Omega0_val, _ = \
    get_physical_densities_and_speed_of_sounds_from_rafikov_dimensionless_variables_for_paper(0.0,0.0,Qgas, Qstars, Rvals, kappa=kappa_val)
    sim = Gas_plus_stars_integrator(
        kx=kxval, kyc=kyval, Sigma_g= Sigma_g_val, 
        Sigma_s=Sigma_s_val,
        kappa = kappa_val, 
        Omega0 = Omega0_val, 
        sigma_x = sigma_x_val,
        cssq = cssq_val,
        Hg = hg*np.sqrt(cssq_val)/kappa_val,
        Hs = hs*sigma_x_val/kappa_val,
        G=1)

    t_grid = np.linspace(ti, tf, nvals)
    results = sim.solve(t_grid, fext_params={'delta': 0.0, 'amplitude' : 0}, impulse_params = {"t_0": ti, "Sigma_impluse" : 1})

    surface_densities_stellar = np.abs(sim.get_stellar_surface_density(results, t_grid))

    return np.max(surface_densities_stellar)

def get_plotting_arr(RvalQgQs,hg,hs, ksigmavals, tivals):
    plotting_arr = []

    for val in RvalQgQs:
        start = time.perf_counter()
        Rvals, Qgas, Qstars = val
        _,_,Sigma_g_val, Sigma_s_val, sigma_x_val, cssq_val , Omega0_val, _ = \
                get_physical_densities_and_speed_of_sounds_from_rafikov_dimensionless_variables_for_paper(0.0,0.0,Qgas, Qstars, Rval, kappa=kappa_val)
        maxvals = []
        print("-------")
        for j in range(len(ksigmavals)):
            maximum = 0.0
            maxnum = 0.0
            maxdenum = 0.0
            for ti in tivals:
                
                num = get_self_gravity_response_max_gas_plus_stars(20,ti, ksigmavals[j], Rvals, Qgas, Qstars, hg, hs, 100)
                
                denum = Kfunc_kernel_max(20, ti, ksigmavals[j], Sigma_s_val, sigma_x_val, Omega0_val, hs, 100)
                
                # maximum = max(maximum,  num/denum)
    
                maxdenum = max(maxdenum,  denum)
                maxnum = max(maxnum,  num)
    
                # if(denum<1e-10):
                #     print("ti = ",ti,"; ky = ",ksigmavals[j], "; max = ", maximum, "; num = ",num, "; denum = ", denum)
    
            maximum = maxnum/maxdenum
            maxvals.append(maximum)
    
        plotting_arr.append([val, maxvals])
    
        end = time.perf_counter()
    
        print(f"Elapsed time: {end - start:.6f} seconds", flush=True)

    return plotting_arr

def get_max_amplification(Rval, Qgas, Qstars,hg,hs, ksigmavals, tivals):
    start = time.perf_counter()
    _,_,Sigma_g_val, Sigma_s_val, sigma_x_val, cssq_val , Omega0_val, _ = \
                get_physical_densities_and_speed_of_sounds_from_rafikov_dimensionless_variables_for_paper(0.0,0.0,Qgas, Qstars, Rval, kappa=kappa_val)
    maxvals = []
    print("-------")
    for j in range(len(ksigmavals)):
        maximum = 0.0
        maxnum = 0.0
        maxdenum = 0.0
        for ti in tivals:
            
            num = get_self_gravity_response_max_gas_plus_stars(20,ti, ksigmavals[j], Rval, Qgas, Qstars, hg, hs, 100)
            
            denum = Kfunc_kernel_max(20, ti, ksigmavals[j], Sigma_s_val, sigma_x_val, Omega0_val, hs, 100)
            
            # maximum = max(maximum,  num/denum)

            maxdenum = max(maxdenum,  denum)
            maxnum = max(maxnum,  num)

            # if(denum<1e-10):
            #     print("ti = ",ti,"; ky = ",ksigmavals[j], "; max = ", maximum, "; num = ",num, "; denum = ", denum)

        maximum = maxnum/maxdenum
        maxvals.append(maximum)
    
    
    end = time.perf_counter()
    
    print(f"Elapsed time: {end - start:.6f} seconds", flush=True)

    return np.max(maxvals)


def get_max_amplification_data_and_plot(R, hg, hs):
    # -----------------------------------------------
    # Output directory
    # -----------------------------------------------
    output_dir = Path("./max_amplification_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------
    # Wave numbers and time to search
    # -----------------------------------------------
    ksigmavals = np.sort(1.0/np.linspace(0.5, 5, 100))
    tivals = np.linspace(-20, 18, 100)

    # -----------------------------------------------
    # Q grid
    # -----------------------------------------------
    inv_q_axis = np.linspace(0.01, 2.0, 100)

    Qg_vec = 1.0 / inv_q_axis
    Qs_vec = 1.0 / inv_q_axis

    max_amplification_arr = np.full(
        (len(inv_q_axis), len(inv_q_axis)),
        np.nan
    )

    for i, Qg in enumerate(Qg_vec):
        for j, Qs in enumerate(Qs_vec):

            stable = test_stability_thick_disks(R, Qg, Qs, hg, hs)

            if stable:

                print("i = ", i, "j = ", j, flush=True)
                
                max_amplification_arr[i, j] = get_max_amplification(
                    R, Qg, Qs,
                    hg, hs,
                    ksigmavals, tivals
                )

    # -----------------------------------------------
    # Save data
    # -----------------------------------------------
    output_path = output_dir / f"{R}_{hg}_{hs}.npz"

    np.savez(
        output_path,
        inv_q_axis=inv_q_axis,
        Qg_vec=Qg_vec,
        Qs_vec=Qs_vec,
        max_amplification_arr=max_amplification_arr,
        hg=hg,
        hs=hs,
        R=R
    )

    # -----------------------------------------------
    # Plot contour
    # -----------------------------------------------
    InvQg_mesh, InvQs_mesh = np.meshgrid(
        inv_q_axis,
        inv_q_axis,
        indexing="ij"
    )

    fig, ax = plt.subplots()

    cf = ax.contourf(
        InvQg_mesh,
        InvQs_mesh,
        max_amplification_arr,
        levels=30
    )

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"Maximum amplification")

    ax.set_xlabel(r"$1/Q_{\rm g}$")
    ax.set_ylabel(r"$1/Q_{\rm s}$")

    plt.tight_layout()
    plt.savefig(output_dir / f"{R}_{hg}_{hs}.pdf", dpi = 300)
    plt.close()

    return max_amplification_arr

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and plot maximum amplification for given R, hg, hs."
    )

    parser.add_argument(
        "--R",
        type=float,
        required=True,
        help="Dimensionless gas-to-stars ratio parameter R."
    )

    parser.add_argument(
        "--hg",
        type=float,
        required=True,
        help="Dimensionless gas scale height parameter."
    )

    parser.add_argument(
        "--hs",
        type=float,
        required=True,
        help="Dimensionless stellar scale height parameter."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Starting job ........", flush=True)
    print(f"R = {args.R}, hg = {args.hg}, hs = {args.hs}", flush=True)

    get_max_amplification_data_and_plot(args.R, args.hg, args.hs)

    print("Job finished", flush=True)
