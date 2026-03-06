# -------------------------------------------------------------
# Kollat 251214
# -------------------------------------------------------------

import numpy as np


# Screw force (bolt load)
def F_s(F0, FN, cs, ck):
    """
    Bolt force Fs = F0 + (cs / (cs + ck)) * FN
    F0 : preload [N]
    FN : external axial load [N]
    cs : bolt stiffness [N/m]
    ck : clamped parts (member) stiffness [N/m]
    """
    return F0 + (cs / (cs + ck)) * FN


# Clamping force (under the head)
def F_k(F0, FN, cs, ck):
    """
    Clamping (under-head) force Fk = F0 - (ck / (cs + ck)) * FN
    F0 : preload [N]
    FN : external axial load [N]
    """
    return F0 - (ck / (cs + ck)) * FN


# Total tightening torque (simplified)
def M_total(Fax, P, d2, r_m, mu_head, mu_thread):
    """
    Total tightening torque (simplified)
    M_total = (0.16 + 0.58 * μ_tot) * Fax * P * d2
    Fax     : axial bolt force [N]
    P       : thread pitch [m]
    d2      : pitch diameter [m]
    μ       : friction coefficient [-]
    """
    return (0.16 * P + 0.58 * mu_thread * d2 + mu_head * r_m) * Fax


# Friction torque under the head
def M_bearing(mu_b, Fb, rm):
    """
    Bearing friction torque:
    M_b = μ_b * Fb * r_m
    μ_b : bearing surface friction coefficient [-]
    Fb  : bearing force [N]
    r_m : mean friction radius [m]
    """
    return mu_b * Fb * rm


# Tightening torque (thread component)
def M_tightening(Fax, r_ax, phi, rho, P):
    """
    Thread tightening torque:
    M_tightening = (Fax * r_ax * tan(φ + ρ')) / (π * tan φ)
    φ  : lead angle [rad]
    ρ' : apparent friction angle [rad]
    """
    return Fax * (P * np.tan(phi + rho)) / (2 * np.pi * np.tan(phi))


# Loosening torque
def M_loosening(Fax, r_ax, phi, rho, P):
    """
    Loosening torque:
    M_loosening = (Fax * r_ax * tan(φ - ρ')) / (π * tan φ)
    """
    return -Fax * (P * np.tan(phi - rho)) / (2 * np.pi * np.tan(phi))


# Apparent friction angle
def rho_apparent(mu, alpha):
    """
    Apparent friction angle:
    ρ' = arctan(μ / cos(α))
    μ     : thread friction coefficient [-]
    α     : thread flank angle [rad] (ISO metric = 60°)
    """
    return np.arctan(mu / np.cos(alpha))


# Lead angle
def phi_lead(P, d2):
    """
    Lead angle of the thread:
    φ = arctan(P / (π * d2))
    P : thread pitch [m]
    d2: thread pitch diameter [m]
    """
    return np.arctan(P / (np.pi * d2))


# Thread stress area
def A_s(d1, d2):
    """
    Thread tensile stress area approximation:
    A_s ≈ (π/16) * (d1 + d2)^2
    d1 : minor diameter [m]
    d2 : pitch diameter [m]
    """
    return (np.pi / 16) * ((d1 + d2) ** 2)

# Metric thread dimensions
def M_dimensions(size):
    """
    Return metric thread dimensions for a given M size.

    Parameters
    ----------
    size : float or int
        Nominal thread size (e.g. 1, 1.6, 6, 10)

    Returns
    -------
    dict
        Dictionary with thread dimensions
    """

    THREAD_TABLE = {
        1:   {"P": 0.25, "d": 1.0,  "d2": 0.838, "d1": 0.729, "dh_fine": 1.1, "dh_medium": 1.2, "dh_coarse": 1.3},
        1.2: {"P": 0.25, "d": 1.2,  "d2": 1.038, "d1": 0.929, "dh_fine": 1.3, "dh_medium": 1.4, "dh_coarse": 1.5},
        1.4: {"P": 0.30, "d": 1.4,  "d2": 1.205, "d1": 1.075, "dh_fine": 1.5, "dh_medium": 1.6, "dh_coarse": 1.8},
        1.6: {"P": 0.35, "d": 1.6,  "d2": 1.373, "d1": 1.221, "dh_fine": 1.7, "dh_medium": 1.8, "dh_coarse": 2.0},
        1.8: {"P": 0.35, "d": 1.8,  "d2": 1.573, "d1": 1.421, "dh_fine": 2.0, "dh_medium": 2.1, "dh_coarse": 2.2},
        2:   {"P": 0.40, "d": 2.0,  "d2": 1.740, "d1": 1.567, "dh_fine": 2.2, "dh_medium": 2.4, "dh_coarse": 2.6},
        2.5: {"P": 0.45, "d": 2.5,  "d2": 2.208, "d1": 2.013, "dh_fine": 2.7, "dh_medium": 2.9, "dh_coarse": 3.1},
        3:   {"P": 0.50, "d": 3.0,  "d2": 2.675, "d1": 2.459, "dh_fine": 3.2, "dh_medium": 3.4, "dh_coarse": 3.6},
        4:   {"P": 0.70, "d": 4.0,  "d2": 3.545, "d1": 3.242, "dh_fine": 4.3, "dh_medium": 4.5, "dh_coarse": 4.8},
        5:   {"P": 0.80, "d": 5.0,  "d2": 4.480, "d1": 4.134, "dh_fine": 5.3, "dh_medium": 5.5, "dh_coarse": 5.8},
        6:   {"P": 1.00, "d": 6.0,  "d2": 5.350, "d1": 4.917, "dh_fine": 6.4, "dh_medium": 6.6, "dh_coarse": 7.0},
        8:   {"P": 1.25, "d": 8.0,  "d2": 7.188, "d1": 6.647, "dh_fine": 8.4, "dh_medium": 9.0, "dh_coarse": 10.0},
        10:  {"P": 1.50, "d": 10.0, "d2": 9.026, "d1": 8.376, "dh_fine": 10.5, "dh_medium": 11.0, "dh_coarse": 12.0},
        12:  {"P": 1.75, "d": 12.0, "d2": 10.863, "d1": 10.106, "dh_fine": 13.0, "dh_medium": 13.5, "dh_coarse": 14.5},
        16:  {"P": 2.00, "d": 16.0, "d2": 14.701, "d1": 13.835, "dh_fine": 17.0, "dh_medium": 17.5, "dh_coarse": 18.5},
        20:  {"P": 2.50, "d": 20.0, "d2": 18.376, "d1": 17.294, "dh_fine": 21.0, "dh_medium": 22.0, "dh_coarse": 24.0},
        24:  {"P": 3.00, "d": 24.0, "d2": 22.051, "d1": 20.752, "dh_fine": 25.0, "dh_medium": 26.0, "dh_coarse": 28.0},
        30:  {"P": 3.50, "d": 30.0, "d2": 27.727, "d1": 26.211, "dh_fine": 31.0, "dh_medium": 33.0, "dh_coarse": 35.0},
        36:  {"P": 4.00, "d": 36.0, "d2": 33.402, "d1": 31.670, "dh_fine": 37.0, "dh_medium": 39.0, "dh_coarse": 42.0},
        42:  {"P": 4.50, "d": 42.0, "d2": 39.077, "d1": 37.129, "dh_fine": 43.0, "dh_medium": 45.0, "dh_coarse": 48.0},
        48:  {"P": 5.00, "d": 48.0, "d2": 44.752, "d1": 42.587, "dh_fine": 50.0, "dh_medium": 52.0, "dh_coarse": 56.0},
        56:  {"P": 5.50, "d": 56.0, "d2": 52.428, "d1": 50.046, "dh_fine": 58.0, "dh_medium": 62.0, "dh_coarse": 66.0},
        64:  {"P": 6.00, "d": 64.0, "d2": 60.103, "d1": 57.505, "dh_fine": 66.0, "dh_medium": 70.0, "dh_coarse": 74.0},
    }

    try:
        return THREAD_TABLE[size]
    except KeyError:
        raise ValueError(f"M{size} is not available in the thread table")