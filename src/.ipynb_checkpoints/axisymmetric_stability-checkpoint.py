import matplotlib.pyplot as plt
from scipy.special import i0
import numpy as np
#======================================
# Axisymmetric stability from Rafikov 2001

def Rafikov_dispersion(kfactor, Qgas, Qstar, Rval):
    besselI0 = i0(kfactor**2)

    expfactor = np.exp(-kfactor**2)

    return (2.0/Qstar) * (1.0/kfactor) * (1 - expfactor*besselI0) + \
    (2.0/Qgas) * Rval * (kfactor/(1 + (kfactor*Rval)**2)) - 1

def Rafikov_dispersion_thick_disk(kfactor, Qg, Qs, R, Hg, Hs, aa=np.pi):
    besselI0 = i0(kfactor**2)

    expfactor = np.exp(-kfactor**2)

    q = kfactor

    oneminusexp = 1 - expfactor*besselI0

    Pi = np.pi

    return -1 + (2*oneminusexp*Pi)/(aa*q*Qs + aa*Hs*pow(q,2)*Qs) + (2*q*R)/(Qg*(1 + Hg*q*R)*(1 + pow(q,2)*pow(R,2)))

def test_stability(Rval, Qg, Qs, plot = 0):
    kvals = np.logspace(-2, 1, 200)
    disp = Rafikov_dispersion(kvals, Qg, Qs, Rval)

    if plot:
        plt.plot(kvals, disp)
    
    if np.any(disp > 0):
        return 0
    else:
        return 1

def test_stability_thick_disks(Rval, Qg, Qs, Hg, Hs, plot = 0):
    kvals = np.logspace(-2, 1, 200)
    disp = Rafikov_dispersion_thick_disk(kvals, Qg, Qs, Rval, Hg, Hs)

    if plot:
        plt.plot(kvals, disp)
    
    if np.any(disp > 0):
        return 0
    else:
        return 1

def get_stability_boundary(Rval, Qgas_range, Qstar_range):
    kvals = np.logspace(-3, 2, 200) 

    stability_grid = np.zeros((len(Qgas_range), len(Qstar_range)))

    for i, Qg in enumerate(Qgas_range):
        for j, Qs in enumerate(Qstar_range):
            disp = Rafikov_dispersion(kvals, Qg, Qs, Rval)
            if np.any(disp > 0):

                stability_grid[i, j] = 0

            else:

                stability_grid[i, j] = 1

    return stability_grid 


def get_stability_boundary_thick_disk(Rval, Qgas_range, Qstar_range, Hg = 0.874286, Hs = 0.396, aa = np.pi):
    kvals = np.logspace(-2, 1, 200) 

    stability_grid = np.zeros((len(Qgas_range), len(Qstar_range)))

    for i, Qg in enumerate(Qgas_range):
        for j, Qs in enumerate(Qstar_range):
            disp = Rafikov_dispersion_thick_disk(kvals, Qg, Qs, Rval, Hg, Hs, aa)
            if np.any(disp > 0):

                stability_grid[i, j] = 0

            else:

                stability_grid[i, j] = 1

    return stability_grid 