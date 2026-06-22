import numpy as np

def get_physical_quantities_from_rafikov_dimensionless_variables(kx, ky,
    Qgas, Qstars, R, kappa=1, gamma = np.sqrt(2), kcrit = 1):
    '''
    G has been set to 1!
    '''

    print("You are using Rafikov's convention in these plots! These are not the same as in the paper! Make sure you want to use this!!!!")
    
    Pi = np.pi
    
    Sigma_g = (R*kappa*kappa)/(kcrit*Pi*Qgas)
    
    Sigma_s = (kappa*kappa)/(kcrit*Pi*Qstars)

    sigma_x = kappa/kcrit

    cssq = R*R*(sigma_x*sigma_x)

    Omega = 0.5*gamma*kappa

    A = Omega + (kappa*kappa)/(-4*Omega)


    return kx*kcrit, ky*kcrit, Sigma_g, Sigma_s, sigma_x, cssq , Omega, A


def get_physical_quantities_from_Binney_convention_for_stars_only(
    Qgas, Qstars, R, kappa=1, gamma = np.sqrt(2), kcrit = 1):
    '''
    G has been set to 1!
    '''

    Pi = np.pi
    
    Sigma_g = 0.0
    
    Sigma_s = (kappa*kappa)/(kcrit*Pi*2.0)

    sigma_x = 3.36*Qstars*kappa/(2.0*kcrit*Pi)

    cssq = 0.1 # Does not matter

    Omega = 0.5*gamma*kappa

    A = Omega + (kappa*kappa)/(-4*Omega)


    return Sigma_g, Sigma_s, sigma_x, cssq , Omega, A


def get_physical_densities_and_speed_of_sounds_from_rafikov_dimensionless_variables_for_paper(kx, ky, Qgas, Qstars, R, kappa=1, gamma = np.sqrt(2), kcrit = 1):
    '''
    G has been set to 1! Note that our convention for Qs differs from Rafikov
    '''
    Pi = np.pi
    aa=3.36 # Set to Pi for exact Rafikov convention
    
    Sigma_g = (R*(kappa*kappa))/(kcrit*Pi*Qgas)
    
    Sigma_s = (kappa*kappa)/(aa*kcrit*Qstars)

    sigma_x = kappa/kcrit

    cssq = (R*R*(kappa*kappa))/(kcrit*kcrit)

    Omega = 0.5*gamma*kappa

    A = Omega + (kappa*kappa)/(-4*Omega)


    return kx*kcrit, ky*kcrit, Sigma_g, Sigma_s, sigma_x, cssq , Omega, A
    