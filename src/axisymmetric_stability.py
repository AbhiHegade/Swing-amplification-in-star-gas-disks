import matplotlib.pyplot as plt
from scipy.special import i0
import numpy as np
#======================================
# Axisymmetric stability from Rafikov 2001
#------------------------------------------------------------------------------
def Rafikov_dispersion(kfactor, Qgas, Qstar, Rval):

    print("You are using rafikov dispersion relationship with his convention!" \
    "The paper uses a different convetion. Make sure that you really want to use this!!!!!!!")

    besselI0 = i0(kfactor**2)

    expfactor = np.exp(-kfactor**2)

    return (2.0/Qstar) * (1.0/kfactor) * (1 - expfactor*besselI0) + \
    (2.0/Qgas) * Rval * (kfactor/(1 + (kfactor*Rval)**2)) - 1

def test_stability(Rval, Qg, Qs, plot = 0):

    print("You are testing the axisymmetric stability rafikov dispersion relationship with his convention!" \
    "The paper uses a different convetion. Make sure that you really want to use this!!!!!!!")

    kvals = np.logspace(-2, 1, 200)
    disp = Rafikov_dispersion(kvals, Qg, Qs, Rval)

    if plot:
        plt.plot(kvals, disp)
    
    if np.any(disp > 0):
        return 0
    else:
        return 1

def get_stability_boundary(Rval, Qgas_range, Qstar_range):

    print("You are extracting the axisymmetric stability boundary for rafikov dispersion relationship with his convention!" \
    "The paper uses a different convetion. Make sure that you really want to use this!!!!!!!")

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

#------------------------------------------------------------------------------
# Thick disk and thin disk combined using our convention
def Rafikov_dispersion_thick_disk(kfactor, Qg, Qs, R, hg, hs, aa=3.36):
    besselI0 = i0(kfactor**2)

    expfactor = np.exp(-kfactor**2)

    q = kfactor

    oneminusexp = 1 - expfactor*besselI0

    Pi = np.pi

    return -1 + (2*oneminusexp*Pi)/(aa*q*Qs + aa*hs*pow(q,2)*Qs) + (2*q*R)/(Qg*(1 + hg*q*R)*(1 + pow(q,2)*pow(R,2)))


def test_stability_thick_disks(Rval, Qg, Qs, hg, hs, plot = 0, aa = 3.36):
    kvals = np.logspace(-2, 1, 200)
    disp = Rafikov_dispersion_thick_disk(kvals, Qg, Qs, Rval, hg, hs, aa)

    if plot:
        plt.plot(kvals, disp)
    
    if np.any(disp > 0):
        return 0
    else:
        return 1


def get_stability_boundary_thick_disk(Rval, Qgas_range, Qstar_range, hg = 0.87, hs = 0.4, aa = 3.36):
    kvals = np.logspace(-2, 1, 200) 

    stability_grid = np.zeros((len(Qgas_range), len(Qstar_range)))

    for i, Qg in enumerate(Qgas_range):
        for j, Qs in enumerate(Qstar_range):
            disp = Rafikov_dispersion_thick_disk(kvals, Qg, Qs, Rval, hg, hs, aa)
            if np.any(disp > 0):

                stability_grid[i, j] = 0

            else:

                stability_grid[i, j] = 1

    return stability_grid 

#------------------------------------------------------------------------------
# Thick disk and thin disk combined for two-fluid disk

import numpy as np
import matplotlib.pyplot as plt


def twofluid_dispersion_thick_disk(kfactor, Qg, Qs, R, hg, hs, aa=3.36):
    """
    Two-fluid axisymmetric stability function for a thick gas+stellar disk.

    Returns

        F(q) = -1 + gas response + stellar fluid response

    The disk is unstable if F(q) > 0 for any q.

    Conventions:
        q = k sigma_s / kappa
        R = sigma_g / sigma_s

        Qg = kappa sigma_g / (pi G Sigma_g)
        Qs = kappa sigma_s / (aa G Sigma_s), with aa=3.36 by default

    Thickness corrections:
        gas:    1 / (1 + hg q R)
        stars:  1 / (1 + hs q)
    """

    q = np.asarray(kfactor)

    gas_term = (
        2.0 * q * R
        / (Qg * (1.0 + hg * q * R) * (1.0 + q**2 * R**2))
    )

    star_fluid_term = (
        2.0 * np.pi * q
        / (aa * Qs * (1.0 + hs * q) * (1.0 + q**2))
    )

    return -1.0 + gas_term + star_fluid_term


def test_stability_twofluid_thick_disks(
    Rval,
    Qg,
    Qs,
    hg=0.87,
    hs=0.4,
    plot=0,
    aa=3.36,
):
    kvals = np.logspace(-2, 1, 200)

    disp = twofluid_dispersion_thick_disk(
        kvals, Qg, Qs, Rval, hg, hs, aa
    )

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(kvals, disp)
        plt.axhline(0.0, linestyle="--")
        plt.xscale("log")
        plt.xlabel(r"$q = k \sigma_s / \kappa$")
        plt.ylabel(r"$F(q)$")
        plt.title("Two-fluid thick-disk stability")
        plt.tight_layout()
        plt.show()

    if np.any(disp > 0):
        return 0
    else:
        return 1


def get_stability_boundary_twofluid_thick_disk(
    Rval,
    Qgas_range,
    Qstar_range,
    hg=0.87,
    hs=0.4,
    aa=3.36,
):
    kvals = np.logspace(-2, 1, 200)

    stability_grid = np.zeros((len(Qgas_range), len(Qstar_range)))

    for i, Qg in enumerate(Qgas_range):
        for j, Qs in enumerate(Qstar_range):

            disp = twofluid_dispersion_thick_disk(
                kvals, Qg, Qs, Rval, hg, hs, aa
            )

            if np.any(disp > 0):
                stability_grid[i, j] = 0
            else:
                stability_grid[i, j] = 1

    return stability_grid