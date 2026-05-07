#alpha_from_octave_code_converted_to_python.py
# Converted Octave code to Python  

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
import sys

LOG_PATH = os.path.join(os.path.dirname(__file__), "alpha_progress.log")

def progress(msg):
    print(msg, file=sys.stderr, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass

# Function int (renamed to compute_integral to avoid conflict with Python's built-in int)
def compute_integral(beta, xi, Delta):
    """
    Computes the integrand for the given parameters.
    Equation derived from Jonas review.
    """
    return np.tanh(0.5 * beta * np.sqrt(xi**2 + Delta**2)) / np.sqrt(xi**2 + Delta**2)

# Function af

def af(Tlist, Tc, Tdb, w0, approximate_sigma=False, approximate_Delta=False):
    ''' Computes a list of alphas
    also returns a list of the related Deltas and an array of the corresponding complex conductivites sigma
    '''
    ef = 0.1
    # Three columns correspond to 
    # Z = [transpose(ZThickLocal), transpose(ZExtremeAnom), transpose(ZThin)]
 
    Tlist = np.array(Tlist)
    
    progress(f"[af] About to start alpha calculation for {len(Tlist)} temperature points")
    progress("[af] About to do step 1/4: compute Z1 at base temperatures")
    Z1, Delta_list, sigma = Zf(Tlist, Tc, Tdb, w0, approximate_sigma, approximate_Delta)
    progress("[af] About to do step 2/4: compute Z2 at perturbed temperatures")
    Z2, _, _ = Zf(Tlist * (1.0 + ef), Tc, Tdb, w0, approximate_sigma, approximate_Delta)
    progress("[af] About to do step 3/4: compute Rlist/Tmatrix")
    Rlist = np.real(Z1 + Z2) / 2
     
    # Transpose of Tlist
    Tl = Tlist.reshape(-1, 1)

    # Create Tmatrix with three identical columns
    Tmatrix = np.hstack((Tl, Tl, Tl))

    # Compute alpha
    progress("[af] About to do step 4/4: compute alpha")
    a = (Tmatrix / Rlist) * ((Z2 - Z1) / (ef * Tmatrix))
    progress("[af] Done")

    return np.array(a), np.array(Delta_list), sigma

# Note: The function Zf() should be defined elsewhere in the code, as it was in the original Octave code.
# Make sure to import or define Zf() before calling af().


# Function compute_alpha
def compute_alpha(constants, Tlist, approximate_sigma=False, approximate_Delta=False):
    """
    see Lindeman paper  Resonator-bolometer theory, microwave read out, and kinetic inductance bolometers 
    see Lindeman paper  Model of superconducting alternating current bolometers
    Computes alpha using the af function.
    The equations are based on Jonas review.
    """
    progress(
        "[compute_alpha] Start: "
        f"n={len(Tlist)}, Tc={constants['Tc']}, Tdb={constants['Tdb']}, "
        f"approx_sigma={approximate_sigma}, approx_Delta={approximate_Delta}"
    )
    out = af(Tlist, constants['Tc'], constants['Tdb'], constants['w0'], approximate_sigma, approximate_Delta)
    progress("[compute_alpha] Done")
    return out

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
    progress(f"[delta_solver2] About to solve Delta root for T={T:.6g}")
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

    progress(f"[delta_solver2] Done for T={T:.6g}")
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
    n = len(Tlist)
    progress(f"[s1] About to start sigma1 integrations for {n} point(s)")
    for i, T in enumerate(Tlist):
        progress(f"[s1] About to integrate point {i + 1}/{n} at T={T:.6g}")
        Delta = Delta_list[i]
        Emin, Emax = Delta, 1.5 * Delta
        integral, err = quad(lambda E: g1(E, T, Delta, w0), Emin, Emax, epsrel=1e-15)
        s.append(integral)
    progress("[s1] Done")
    return np.array(s)

# Function s1approx (approximation for sigma_1)
from scipy.special import kv

def s1approx(Tlist, Delta_list, Tc, w0):
    '''
    The approximate answer from Jonas review eq 16
    '''

    # Constants
    hbar = 1.054571726e-34  # Reduced Planck's constant (J.s)
    kb = 1.3806488e-23       # Boltzmann constant (J/K)

    # Calculate Delta0
    Delta0 = 1.764 * kb * Tc

    # Calculate the approximate s1 from Jonas review equation 16
    s = (4 * Delta_list / (hbar * w0) *
         np.exp(-Delta0 / (kb * Tlist)) *
         np.sinh(hbar * w0 / (2 * kb * Tlist)) *
         kv(0, hbar * w0 / (2 * kb * Tlist)))

    return s
    
    
    
    
    
    
# Function s2 (computes sigma_2 for a list of temperatures)
def s2(Tlist, Delta_list, w0, plt=False):
    ''' This computes sigma2, which is the *negative* of the imaginary part of complex conductivity
        sigma2 is positive so that conductivity has a negate imaginary part
    '''
    hbar = 1.054571726e-34
    num = 100000
    s = []
    n = len(Tlist)
    progress(f"[s2] About to start sigma2 integrations for {n} point(s)")
    for i, Tl in enumerate(Tlist):
        progress(f"[s2] About to integrate point {i + 1}/{n} at T={Tl:.6g}")
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

    progress("[s2] Done")
    return np.array(s)

# Function s2approx (approximation for sigma_2)
def s2approx(Tlist, Delta_list, w0):
    """
    Computes an approximation for sigma_2 from Jonas review (eqn 26).
    """
    hbar = 1.054571726e-34
    return (np.pi * Delta_list) / (hbar * w0) * (1 - 2 * fermi(Delta_list, Tlist))

# Function sf (surface impedance)
# Function sf (surface impedance)
def sf(Tlist, Tc, Tdb, w0, approximate_sigma=False, approximate_Delta=False):
    """
    Computes surface impedance.
    Based on Jonas review equations.
    """
    
    progress(
        "[sf] Start: "
        f"n={len(Tlist)}, approx_sigma={approximate_sigma}, approx_Delta={approximate_Delta}"
    )
    if approximate_Delta:
        progress("[sf] About to do step 1/3: compute Delta with approximation")
        Delta_list = np.array([bcs_energy_gap_approx(T, Tc) for T in Tlist])
    else:
        progress("[sf] About to do step 1/3: solve Delta roots (usually the slowest step)")
        Delta_list = np.array([delta_solver2(T, Tc, Tdb) for T in Tlist])

    if approximate_sigma:
        progress("[sf] About to do step 2/3: compute sigma1 approximation")
        sigma1 = s1approx(Tlist, Delta_list, Tc, w0)
        progress("[sf] About to do step 3/3: compute sigma2 approximation")
        sigma2 = s2approx(Tlist, Delta_list, w0)
    else:
        progress("[sf] About to do step 2/3: compute sigma1 by integration")
        sigma1 = s1(Tlist, Delta_list, w0)
        progress("[sf] About to do step 3/3: compute sigma2 by integration")
        sigma2 = s2(Tlist, Delta_list, w0)
    progress("[sf] Done")
    return sigma1 - 1j * sigma2, Delta_list

# Function Zf (computes impdedance)
def Zf(Tlist, Tc, Tdb, w0, approximate_sigma=False, approximate_Delta=False):
    """
    Compute a result that is proportional to the impedance of a TKID (KIB)
    Returns 3 columns of data corresponding to Z.
    Derived from Jonas review equations 9 and 11.
    """
    progress("[Zf] Start")
    sigma, Delta_list = sf(Tlist, Tc, Tdb, w0, approximate_sigma, approximate_Delta)
    Z0 = 1j  # Set the constants Z(T=0)
    sigma0 = -1j  # Set the constants sigma(T=0)
    term = sigma / sigma0
    ZThickLocal = Z0 / np.sqrt(term)
    ZExtremeAnom = Z0 * (term) ** (-1 / 3)
    ZThin = Z0 / term
    progress("[Zf] Done")
    return np.column_stack((ZThickLocal, ZExtremeAnom, ZThin)), Delta_list, sigma



 
import matplotlib.colors as mcolors

def lighten_color(color, amount=0.5):
    # Function to lighten color for plotting approximations
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    return [(1.0 - amount) * c[i] + amount for i in range(3)]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def lighten_color(color, amount=0.5):
    # Function to lighten color for plotting approximations
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    return [(1.0 - amount) * c[i] + amount for i in range(3)]

def main():
    '''
    Plots the parameters for Nb and its variants.
    The variants have approximations for Delta or Sigma.
    The plots show how closely the approximations match and also provide a reality check on the 
    more exact calculations.
    
    The approximations are reasonable for alpha for T>5 K, with Tc = 14 K
    '''
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass
    progress(f"[main] Starting alpha workflow from {__file__}")

    # Set constants
    constants_lowT = {'Tc': 2, 'Tdb': 275, 'w0': 1.194e9 * 2 * np.pi, 'type': 'lowT'}
    constants_NBTiN = {'Tc': 14, 'Tdb': 275, 'w0': 1e9 * 2 * np.pi, 'type': 'NbTiN'}
    constants = constants_lowT
    # constants = constants_NBTiN

    # Set the temperatures to calculate
    temps = np.linspace(0.05 * constants['Tc'], 0.9 * constants['Tc'], num=100)  # Temperatures from 0.3 Tc to 0.9 Tc
    TArray = np.array(temps)

    # Compute values without approximation
    approximate_sigma = False
    approximate_Delta = False
    progress("[main] About to compute exact alpha (no approximations)")
    AlphaSet, DeltaArray, SigmaArray = compute_alpha(constants, temps, approximate_sigma, approximate_Delta)

    # Compute values with approximation for Delta
    approximate_Delta = True
    progress("[main] About to compute alpha with Delta approximation only")
    AlphaSet_approx, DeltaArray_approx, SigmaArray_approx = compute_alpha(constants, temps, approximate_sigma, approximate_Delta)

    # Compute values with approximation for both Delta and Sigma
    approximate_sigma = True
    progress("[main] About to compute alpha with both Delta and Sigma approximations")
    AlphaSet_approx_both, DeltaArray_approx_both, SigmaArray_approx_both = compute_alpha(constants, temps, approximate_sigma, approximate_Delta)

    # Plot alpha results (Imaginary part)
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.xlabel("Temperature (K)")
    plt.ylabel("Imaginary Part of Alpha")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    labels = ["Thick Local", "Extreme Anomalous", "Thin"]
    colors = ['b', 'g', 'r']

    # Plot thicker curves first
    # Plot approximations with both Delta and Sigma approximated (lightest color)
    for i in range(3):
        color = colors[i]
        lightest_color = lighten_color(color, amount=0.75)
        # Plot approximations with both Delta and Sigma approximated
        plt.plot(TArray, np.imag(AlphaSet_approx_both[:, i]), label=f"{labels[i]} (Imag, Approx Both)", color=lightest_color, linestyle=':', linewidth=8)
    
    # Plot approximations with Delta approximated (lighter color)
    for i in range(3):
        color = colors[i]
        lighter_color = lighten_color(color, amount=0.5)
        # Plot approximations with Delta approximated
        plt.plot(TArray, np.imag(AlphaSet_approx[:, i]), label=f"{labels[i]} (Imag, Approx Delta)", color=lighter_color, linestyle=':', linewidth=4)
    
    for i in range(3):
        color = colors[i]
        # Plot original values (darker color)
        plt.plot(TArray, np.imag(AlphaSet[:, i]), label=f"{labels[i]} (Imag, Exact)", color=color, linestyle=':', linewidth=2)

    plt.title("Imaginary Part of Alpha vs Temperature")
    plt.semilogy()
    plt.legend(loc='best')
    plt.show()

    # Plot alpha results (Real part)
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.xlabel("Temperature (K)")
    plt.ylabel("Real Part of Alpha")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot thicker curves first
    # Plot approximations with both Delta and Sigma approximated (lightest color)
    for i in range(3):
        color = colors[i]
        lightest_color = lighten_color(color, amount=0.75)
        # Plot approximations with both Delta and Sigma approximated
        plt.plot(TArray, np.real(AlphaSet_approx_both[:, i]), label=f"{labels[i]} (Real, Approx Both)", color=lightest_color, linewidth=8)
        
    # Plot approximations with Delta approximated (lighter color)
    for i in range(3):
        color = colors[i]
        lighter_color = lighten_color(color, amount=0.5)
        # Plot approximations with Delta approximated
        plt.plot(TArray, np.real(AlphaSet_approx[:, i]), label=f"{labels[i]} (Real, Approx Delta)", color=lighter_color, linewidth=4)
        
    for i in range(3):
        color = colors[i]
        # Plot original values (darker color)
        plt.plot(TArray, np.real(AlphaSet[:, i]), label=f"{labels[i]} (Real, Exact)", color=color, linewidth=2)
        
    plt.title("Real Part of Alpha vs Temperature")
    plt.semilogy()
    plt.legend(loc='best')
    plt.show()

    # Plot delta vs temperature
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(TArray, DeltaArray_approx, label='Delta (Approx)', color=lighten_color('b', amount=0.5), linewidth=3, linestyle=':')
    plt.plot(TArray, DeltaArray, label='Delta (Exact)', color='b', linewidth=2)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Delta")
    plt.title("Delta vs Temperature")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.show()

    # Plot real part of complex conductivity vs temperature
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(TArray, np.real(SigmaArray_approx_both), label='Sigma (Real, Approx Both)', color=lighten_color('g', amount=0.75), linewidth=5, linestyle=':')
    plt.plot(TArray, np.real(SigmaArray_approx), label='Sigma (Real, Approx Delta)', color=lighten_color('g', amount=0.5), linewidth=3, linestyle=':')
    plt.plot(TArray, np.real(SigmaArray), label='Sigma (Real, Exact)', color='g', linewidth=2)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Real Part of Complex Conductivity (Sigma)")
    plt.title("Real Part of Sigma vs Temperature")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.show()

    # Plot imaginary part of complex conductivity vs temperature
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(TArray, np.imag(SigmaArray_approx_both), label='Sigma (Imag, Approx Both)', color=lighten_color('r', amount=0.75), linestyle='-.', linewidth=5)
    plt.plot(TArray, np.imag(SigmaArray_approx), label='Sigma (Imag, Approx Delta)', color=lighten_color('r', amount=0.5), linestyle='-.', linewidth=3)
    plt.plot(TArray, np.imag(SigmaArray), label='Sigma (Imag, Exact)', color='r', linestyle='--', linewidth=2)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Imaginary Part of Complex Conductivity (Sigma)")
    plt.title("Imaginary Part of Sigma vs Temperature")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    print("main is starting")
    main()

