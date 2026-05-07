#alpha_from_octave_code_converted_to_python.py
# Converted Octave code to Python  

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os

# Function int (renamed to compute_integral to avoid conflict with Python's built-in int)
def compute_integral(beta, xi, Delta):
    """
    Computes the integrand for the given parameters.
    Equation derived from Jonas review.
    """
    return np.tanh(0.5 * beta * np.sqrt(xi**2 + Delta**2)) / np.sqrt(xi**2 + Delta**2)

# Function af

def af(Tlist, Tc, Tdb, w0, approximate_sigma):
    # Computes alpha
    ef = 0.1
    # Three columns correspond to 
    # Z = [transpose(ZThickLocal), transpose(ZExtremeAnom), transpose(ZThin)]

    Tlist = np.array(Tlist)
    Z1, _, _ = Zf(Tlist, Tc, Tdb, w0, approximate_sigma)
    Z2, _, _ = Zf(Tlist * (1.0 + ef), Tc, Tdb, w0, approximate_sigma)
    Rlist = np.real(Z1 + Z2) / 2

    # Transpose of Tlist
    Tl = Tlist.reshape(-1, 1)

    # Create Tmatrix with three identical columns
    Tmatrix = np.hstack((Tl, Tl, Tl))

    # Compute alpha
    a = (Tmatrix / Rlist) * ((Z2 - Z1) / (ef * Tmatrix))

    return a

# Note: The function Zf() should be defined elsewhere in the code, as it was in the original Octave code.
# Make sure to import or define Zf() before calling af().


# Function compute_alpha
def compute_alpha(constants, Tlist, approximate_sigma):
    """
    see Lindeman paper  Resonator-bolometer theory, microwave read out, and kinetic inductance bolometers 
    see Lindeman paper  Model of superconducting alternating current bolometers
    Computes alpha using the af function.
    The equations are based on Jonas review.
    """
    return af(Tlist, constants['Tc'], constants['Tdb'], constants['w0'], approximate_sigma)

# Function cross_check (renamed to avoid conflict with compute_alpha)
def cross_check_alpha(constants, Tlist):
    """
    Cross-checks alpha calculation using the af function.
    """
    return af(Tlist, constants['Tc'], constants['Tdb'], constants['w0'])

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import warnings

def delta_solver2(T, Tc, Tdb, plot=False):
    """
    Solves for Delta using a more robust approach.
    Based on Tinkham's equation 3.51.
    Now uses Brent's method for improved root finding.

    Parameters:
    - T: Temperature value.
    - Tc: Critical temperature value.
    - Tdb: Debye temperature value.
    - plot: Boolean flag to indicate whether to plot the function around the found root.

    Returns:
    - D_root: The root found by Brent's method.
    """
    # Define the function wrapper to solve
    def func_wrapper(Delta):
        return func(Delta, T, Tc, Tdb, False)
    
    # Initial bounds for the solver
    D_low = 1e-25
    D_high = 1e-15
    
    f1 = func_wrapper(D_low)
    f2 = func_wrapper(D_high)
    # Ensure bounds are valid, if not swap
    if f1 * f2 > 0:
        raise ValueError(f"Invalid initial bounds: The function does not have opposite signs at the bounds. {f1}  {f2}")
    
    # Use Brent's method to find the root
  
    D_root, results = brentq(func_wrapper, D_low, D_high, xtol=1e-40, rtol=1e-15, full_output=True )
     
    if not results.converged:
        print("WARNING: BAD CONVERGENCE")
        print(results)

    # If plot is True, plot the function in a neighborhood around the root
    if plot:
        ep = .0001
        D_min = (1-ep)* D_root
        D_max = (1+ep)* D_root
        D_vals = np.linspace(D_min, D_max, 500)
        f_vals = [func_wrapper(D) for D in D_vals]

        plt.figure(figsize=(10, 6))
        plt.plot(D_vals, f_vals, label='func(Delta)', color='b')
        plt.axhline(0, color='r', linestyle='--')
        plt.scatter(D_root, 0, color='g', marker='o', label=f'Root at Delta = {D_root:.5e}')
        plt.xlabel('Delta')
        plt.ylabel('func(Delta)')
        plt.title('Function Plot around the Found Root')
        plt.legend()
        plt.grid(True)
        plt.show()

    return D_root


def bcs_energy_gap_approx(T, Tc):
    """
    Computes the BCS energy gap (Delta) as a function of temperature (T) and critical temperature (Tc).
    
    Parameters:
    T : float
        The temperature in Kelvin.
    Tc : float
        The critical temperature in Kelvin.
    
    Returns:
    Delta : float
        The energy gap at temperature T.
    """
    if T > Tc:
        return 0.0  # Above Tc, the energy gap is zero in the superconducting state.
    
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    
    # Energy gap at zero temperature, Δ_0 ≈ 1.76 * k_B * Tc
    Delta_0 = 1.76 * k_B * Tc
    
    # Compute Δ(T) for T < Tc
    Delta_T = Delta_0 * np.tanh(1.74 * np.sqrt(Tc / T - 1))
    
    return Delta_T
 
 
 
 
 
# Function f0
def f0(beta, Delta, Ec, P, plt):
    """
    Computes the function f0.
    Uses the integral from Jonas review (eqn 26).
    """
    integral, err = quad(lambda xi: compute_integral(beta, xi, Delta), 0, Ec)
    f = integral - P
    if plt:
        print("Plotting is not implemented in this version.")
    return f

# Function fermi
def fermi(E, T):
    """
    Computes the Fermi function.
    Derived from Tinkham's equation 3.51.
    """
    kb = 1.3806488e-23
    return 1 / (np.exp(E / (kb * T)) + 1)

# Function func
def func(Delta, T, Tc, Tdb, plt):
    """
    Computes the function value used in the Delta solver.
    Based on Tinkham's equation 3.51.
    """
    kb = 1.3806488e-23
    beta = 1 / (kb * T)
    Ec = kb * Tdb
    P = np.log((1.13 * Ec) / (kb * Tc))
    return f0(beta, Delta, Ec, P, plt)

# Function g in Octave
def g1(E, T, Delta, w):
    """
    Computes the integrand to find sigma1.
    Derived from Jonas review equation 1.
    """
    hbar = 1.054571726e-34
    g1 = (2 / (hbar * w)) * ((E**2 + Delta**2 + hbar * w * E) / (np.sqrt(E**2 - Delta**2) * np.sqrt((E + hbar * w)**2 - Delta**2))) * (fermi(E, T) - fermi(E + hbar * w, T))
    return np.real(g1)

# Function g2
def g2(T, E, Delta, w):
    """
    Computes the integrand to find sigma2.
    Derived from Mazin thesis equation 2.4.
    """
    
    hbar = 1.054571726e-34
    
    x = (1 / (hbar * w)) * ((E**2 + Delta**2 + hbar * w * E) / (np.sqrt(Delta**2 - E**2) * np.sqrt((E + hbar * w)**2 - Delta**2))) * (1 - 2 * fermi(E, T))
    return np.real(x)

# Function th (hyperbolic tangent)
def th(x):
    """
    Computes the hyperbolic tangent.
    This form helps to avoid numberical errors
    """
    return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))

# Function s1 (computes sigma_1 for a list of temperatures)
def s1(Tlist, Delta_list, w0):
    """
    Computes sigma_1 for a list of temperatures.
    This is the real part of the complex conductivity
    Based on Jonas review equation 1.
    """
    s = []
    for i, T in enumerate(Tlist):
        Delta = Delta_list[i]
        Emin, Emax = Delta, 1.5 * Delta
        integral, err = quad(lambda E: g1(E, T, Delta, w0), Emin, Emax, epsrel=1e-15)
        s.append(integral)
    return np.array(s)

# Function s1approx (approximation for sigma_1)
def s1approx(Tlist, Delta_list, w0):
    """
    Computes an approximation for sigma_1.
    Derived from Jonas review (eqn 26).
    """
    hbar = 1.054571726e-34
    return (np.pi * Delta_list) / (hbar * w0) * (1 - 2 * fermi(Delta_list, Tlist))

# Function s2 (computes sigma_2 for a list of temperatures)
def s2(Tlist, Delta_list, w0, plt=False):
    ''' This computes sigma2, which is the *negative* of the imaginary part of complex conductivity
        sigma2 is positive so that conductivity has a negate imaginary part
    '''
    hbar = 1.054571726e-34
    num = 100000
    s = []
    for i, Tl in enumerate(Tlist):
        Delta = Delta_list[i]
        Emin = Delta - hbar * w0
        Emax = Delta

        if plt:
            dE = (Emax - Emin) / num
            E = np.arange(Emin + dE, Emax, dE)
            Int = g2(Tl, E, Delta, w0)  # Assuming function g2 is defined elsewhere
            estimate = np.trapz(Int, E)
            
            # Plotting
            plt.figure(4)
            plt.clf()
            plt.plot(E, Int)
            plt.xlabel('E')
            plt.ylabel('g2')
            plt.axis([Emin, Emax, None, None])
            plt.show()

        # Integration with tolerance
        Integral, err = quad(lambda E: g2(Tl, E, Delta, w0), Emin, Emax, epsrel=1e-15)
        s.append(Integral)

    return np.array(s)

# Function s2approx (approximation for sigma_2)
def s2approx(Tlist, Delta_list, w0):
    """
    Computes an approximation for sigma_2 from Jonas review (eqn 26).
    """
    hbar = 1.054571726e-34
    return (np.pi * Delta_list) / (hbar * w0) * (1 - 2 * fermi(Delta_list, Tlist))

# Function sf (surface impedance)
def sf(Tlist, Tc, Tdb, w0, approximate_sigma):
    """
    Computes surface impedance.
    Based on Jonas review equations.
    """
    Delta_list = np.array([delta_solver2(T, Tc, Tdb) for T in Tlist])
    if approximate_sigma:
        sigma1 = s1approx(Tlist, Delta_list, w0)
        sigma2 = s2approx(Tlist, Delta_list, w0)
    else:
        sigma1 = s1(Tlist, Delta_list, w0)
        sigma2 = s2(Tlist, Delta_list, w0)
    return sigma1 - 1j * sigma2, Delta_list

# Function Zf (computes impdedance)
def Zf(Tlist, Tc, Tdb, w0, approximate_sigma):
    """
    Compute a result that is proportional to the impedance of a TKID (KIB)
    Returns 3 columns of data corresponding to Z.
    Derived from Jonas review equations 9 and 11.
    """
    sigma, Delta_list = sf(Tlist, Tc, Tdb, w0, approximate_sigma)
    Z0 = 1j  # Set the constants Z(T=0)
    sigma0 = -1j  # Set the constants sigma(T=0)
    term = sigma / sigma0
    ZThickLocal = Z0 / np.sqrt(term)
    ZExtremeAnom = Z0 * (term) ** (-1 / 3)
    ZThin = Z0 / term
    return np.column_stack((ZThickLocal, ZExtremeAnom, ZThin)), Delta_list, sigma




# Main function to run the calculations and plot results
def main():
    # Set constants
    constants_lowT = {'Tc': 1, 'Tdb': 275, 'w0': 1.194e9 * 2 * np.pi, 'type': 'lowT'}
    # constants_TEST = {'Tc': 14, 'Tdb': 275, 'w0': 1.194e9 * 2 * np.pi, 'type': 'TEST'}
    constants_NBTiN = {'Tc': 14, 'Tdb': 275, 'w0': 1.194e9 * 2 * np.pi, 'type': 'NbTiN'}
    # # 
    constants = constants_NBTiN
    
    
    
    TArray = []
    AlphaArray = []
    DeltaArray = []
    SigmaArray = []

    # Set the temperatures to calculate
    temps = np.linspace(0.1 * constants['Tc'], .9*constants['Tc'], num=100)  # Temperatures from 0.1 Tc to Tc
     
    approximate_sigma = False
    
    # Calculate alpha values for new temperatures
    if len(temps) > 0:
        AlphaSet = compute_alpha(constants, temps, approximate_sigma)
        TArray = np.concatenate((TArray, temps))
        AlphaArray = np.vstack([AlphaArray, AlphaSet]) if AlphaArray != [] else AlphaSet
        _, DeltaArray, SigmaArray = Zf(temps, constants['Tc'], constants['Tdb'], constants['w0'], approximate_sigma)
    
    # Plot alpha results
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.xlabel("Temperature (K)")
    plt.ylabel("Alpha")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    #sort from low temperature to high
    sorted_indices = np.argsort(TArray)
    TArray_sorted = TArray[sorted_indices]
    AlphaArray_sorted = AlphaArray[sorted_indices]
    labels = ["Thick Local", "Extreme Anomalous", "Thin"]
    
    for i in range(3):
        plt.plot(TArray_sorted, np.real(AlphaArray_sorted[:, i]), label=f"{labels[i]} (Real)", linewidth=2)
        plt.plot(TArray_sorted, np.imag(AlphaArray_sorted[:, i]), label=f"{labels[i]} (Imag)", linestyle='--', linewidth=2)
    plt.semilogy()
    plt.legend(loc='best')
    plt.show()
    
    # Plot delta vs temperature
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(TArray_sorted, DeltaArray, label='Delta', color='b', linewidth=2)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Delta")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.show()
    
    # Plot real part of complex conductivity vs temperature
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(TArray_sorted, np.real(SigmaArray), label='Sigma (Real)', color='g', linewidth=2)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Real Part of Complex Conductivity (Sigma)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.show()
    
    # Plot imaginary part of complex conductivity vs temperature
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(TArray_sorted, np.imag(SigmaArray), label='Sigma (Imag)', color='r', linestyle='--', linewidth=2)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Imaginary Part of Complex Conductivity (Sigma)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    main()
