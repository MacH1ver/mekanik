# mekanik/springs.py
# Spring mechanics based on Formelsamling equations [39]–[50]
# Units: N, m, Pa, rad, J

import numpy as np

# --- Axial (helical) spring constant -----------------------------
def spring_constant_axial(G, d, D, n):
    """
    Axial spring constant for a cylindrical helical spring.
    c_a = (G * d^4) / (8 * n * D^3)
    G : shear modulus [Pa]
    d : wire diameter [m]
    D : mean coil diameter [m]
    n : number of active coils [-]
    Returns: axial stiffness [N/m]
    """
    return (G * d**4) / (8.0 * n * D**3)


# --- Torsional spring constant ----------------------------------
def spring_constant_torsional(E, d, D, n):
    """
    Torsional spring constant (rotational stiffness).
    c_v = (E * d^4) / (64 * n * D)
    E : Young's modulus [Pa]
    d : wire diameter [m]
    D : mean coil diameter [m]
    n : number of active coils [-]
    Returns: torsional stiffness [N·m/rad]
    """
    return (E * d**4) / (64.0 * n * D)


# --- Torsional shear stress -------------------------------------
def shear_stress_torsion(F, D, d):
    """
    Torsional shear stress in a helical spring.
    τ_v = (8 * F * D) / (π * d^3)
    F : spring force [N]
    D : mean coil diameter [m]
    d : wire diameter [m]
    Returns: shear stress [Pa]
    """
    return (8.0 * F * D) / (np.pi * d**3)


# --- Effective shear stress -------------------------------------
def shear_stress_effective(F, D, d, k):
    """
    Effective (corrected) shear stress in a helical spring.
    τ_e = k * (8 * F * D) / (π * d^3)
    F : spring force [N]
    D : mean coil diameter [m]
    d : wire diameter [m]
    k : correction factor (Wahl factor)
    Returns: effective shear stress [Pa]
    """
    return k * (8.0 * F * D) / (np.pi * d**3)


# --- Correction factor (Wahl factor) ----------------------------
def wahl_factor(D, d):
    """
    Correction for curvature and shear.
    k = 1 + 0.5*(d/D) + 0.615*(d/D)^2 ≈ 1 + (d/(2D))*(5/4 + 7/8*(d/D))
    More exact: k = 1 + 0.5*(d/D) + 0.615*(d/D)^2
    Returns: k [-]
    """
    c = D / d  # spring index
    return (4 * c - 1) / (4 * c - 4) + 0.615 / c


# --- Free spring length -----------------------------------------
def free_length(n, d, delta):
    """
    Minimum free spring length.
    l0 > 1.25*(n + 1)*d + δ
    n : number of active coils [-]
    d : wire diameter [m]
    δ : spring deflection under load [m]
    Returns: required free length [m]
    """
    return 1.25 * (n + 1) * d + delta


# --- Buckling condition -----------------------------------------
def check_buckling(l0, D):
    """
    Buckling criterion for compression springs.
    l0 < 2.6 * D  → stable
    Returns True if stable, False if buckling likely.
    """
    return l0 < 2.6 * D


# --- External work (linear spring) -------------------------------
def external_work_linear(F, delta):
    """
    External work (potential energy) for a linear spring.
    W = 0.5 * F * δ = F^2 / (2 * c)
    F : force [N]
    δ : deflection [m]
    Returns: work [J]
    """
    return 0.5 * F * delta


# --- External work (rotational spring) ---------------------------
def external_work_rotational(M, phi):
    """
    External work (potential energy) for rotational spring.
    W = 0.5 * M * φ
    M : torque [N·m]
    φ : angular deflection [rad]
    Returns: work [J]
    """
    return 0.5 * M * phi


# --- Max. storable energy (normal stress) ------------------------
def max_energy_density_normal(sigma_max, E, V):
    """
    Maximum energy density for normal stress.
    W_V = σ_max^2 / (2 * E)
    sigma_max : maximum normal stress [Pa]
    E : Young's modulus [Pa]
    Returns: energy [J]
    """
    return V* sigma_max**2 / (2.0 * E)


# --- Max. storable energy (shear stress) -------------------------
def max_energy_density_shear(tau_max, G, V):
    """
    Maximum energy density for shear stress.
    W_V = τ_max^2 / (2 * G)
    tau_max : maximum shear stress [Pa]
    G : shear modulus [Pa]
    Returns: energy [J]
    """
    return V * tau_max**2 / (2.0 * G)


# --- Utilization factor -----------------------------------------
def utilization_factor(W_actual, W_max):
    """
    Utilization degree of stored energy.
    η = W_actual / W_max   ,  0 < η < 1
    Returns: η [-]
    """
    return W_actual / W_max