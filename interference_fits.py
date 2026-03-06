# mekanik/fits.py
# Shrink- and press-fit calculations based directly on equations [31]–[38]
# Units: m, N, Pa, rad, °C

import numpy as np

# --- Frictional force at macro-sliding ---------------------------
def friction_force(mu, p=None, d=None, L=None, N=None):
    """
    Frictional (tangential) force for interference fits.

    Option 1 (pressure-based):
        F = μ * π * d * L * p
        p : interface pressure [Pa]
        d : interface diameter [m]
        L : contact length [m]

    Option 2 (normal-force-based):
        F = μ * N
        N : normal force [N]

    Exactly one of (p) or (N) must be provided.

    Returns: friction force [N]
    """

    if (p is None) == (N is None):
        raise ValueError("Provide exactly one of p (pressure) or N (normal force).")

    if p is not None:
        if d is None or L is None:
            raise ValueError("d and L must be provided when using pressure p.")
        return mu * np.pi * d * L * p

    return mu * N

# --- Resultant of friction force components ----------------------
def friction_force_resultant(F_axial, F_peripheral):
    """
    Resultant friction force components.
    F = sqrt(F_axial^2 + F_peripheral^2)
    """
    return np.sqrt(F_axial**2 + F_peripheral**2)


# --- External torque transmitted ---------------------------------
def torque_transmitted(F_peripheral, d):
    """
    Torque due to peripheral frictional force.
    M = (d / 2) * F_peripheral   [N·m]
    """
    return 0.5 * d * F_peripheral


# --- Interference and geometry relations ----------------------
def kappa(d, D):
    """
    Geometry ratio κ = d / D.
    """
    return d / D


def kappa0(d0, d):
    """
    Geometry ratio κ0 = d0 / d  (for hollow shaft cases).
    """
    return d0 / d


def interference_pressure(delta, d, d0, D, E_hub, E_shaft, nu_hub, nu_shaft):
    """  
    p = Δ / (d * [ ((1 - ν_hub^2)/E_hub) * ((1 + κ^2)/(1 - κ^2))
                + ((1 - ν_shaft^2)/E_shaft) ] )

    where:
        κ = d / D

    p        : interface pressure [Pa]
    d        : fit diameter (interface diameter) [m]
    d0       : shaft inner diameter [m]
    D        : hub outer diameter [m]
    E_hub    : Young's modulus of hub material [Pa]
    E_shaft  : Young's modulus of shaft material [Pa]
    nu_hub   : Poisson's ratio of hub material [-]
    nu_shaft : Poisson's ratio of shaft material [-]

    Returns: pressure [Pa]
    """
    
    k = kappa(d, D)
    k0 = kappa0(d0, d) 

    denom = d * ((1/E_hub) * ((1 + k**2) / (1 - k**2)) * nu_hub + (1/E_shaft) * ((1 + k0**2) / (1 - k0**2)) * nu_shaft)

    return delta / denom


def required_interference(p, d, d0, D, E_hub, E_shaft, nu_hub, nu_shaft):
    """  
    Δ = p * d * [ ((1 - ν_hub^2)/E_hub) * ((1 + κ^2)/(1 - κ^2))
                  + ((1 - ν_shaft^2)/E_shaft) ]

    where:
        κ = d / D

    p        : interface pressure [Pa]
    d        : fit diameter (interface diameter) [m]
    d0       : shaft inner diameter [m]
    D        : hub outer diameter [m]
    E_hub    : Young's modulus of hub material [Pa]
    E_shaft  : Young's modulus of shaft material [Pa]
    nu_hub   : Poisson's ratio of hub material [-]
    nu_shaft : Poisson's ratio of shaft material [-]

    Returns: diametral interference Δ [m]
    """
    
    k = kappa(d, D)
    k0 = kappa0(d0, d) 
    return p * d * ((1/E_hub) * ((1 + k**2) / (1 - k**2)) * nu_hub + (1/E_shaft) * ((1 + k0**2) / (1 - k0**2)) * nu_shaft)  


# --- Max equivalent stresses after press fit ---------------------
def stresses_after_fit(p, k, k0=None, material="hub", criterion="Tresca"):
    """
    Max effective stresses for hub, hollow shaft, or solid shaft.
    Inputs:
      p : contact pressure [Pa]
      k : d/D  for hub
      k0: d0/d for hollow shaft (if applicable)
      material: "hub", "hollow_shaft", or "solid_shaft"
      criterion: "Tresca" or "vonMises"
    Returns: σ_eq,max [Pa]
    """
    if material == "hub":
        if criterion.lower().startswith("t"):
            # Tresca
            sigma_eq = p * (2 * k**2) / (1 - k**2)
        else:
            # von Mises
            sigma_eq = p * (4 * k**2) / (3 * (1 - k**2)) * (1 + k**2)
    elif material == "hollow_shaft":
        if k0 is None:
            raise ValueError("Need k0 for hollow shaft.")
        if criterion.lower().startswith("t"):
            sigma_eq = p * (2 * k0**2) / (1 - k0**2)
        else:
            sigma_eq = p * (2 * k0**2) / (1 - k0**2)
    elif material == "solid_shaft":
        sigma_eq = p
    else:
        raise ValueError("material must be 'hub', 'hollow_shaft', or 'solid_shaft'.")
    return sigma_eq


# --- Required temperature difference for assembly ----------------
def required_temperature_difference(delta, d, alpha):
    """
    Required temperature difference ΔT for mounting by thermal expansion.
    ΔT > Δ / (α * d)
    """
    return delta / (alpha * d)