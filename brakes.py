# ------------------------------------------------------------
# Brakes
# ------------------------------------------------------------

def archard_wear_law(K, W, L, H):
    """
    Archard's wear law (Wikipedia / classical form).

        Q = (K * W * L) / H

    where:
        Q : total wear volume [m^3]
        K : dimensionless wear coefficient [-]
        W : normal load [N]
        L : sliding distance [m]
        H : hardness of the softer material [Pa]

    Returns:
        Q : total wear volume [m^3]
    """
    return (K * W * L) / H




def disc_brake_new_pads(mu, F_E, R_o, R_i):
    """
    Disc brake torque for new pads (linearly translating pad, per pad).
        M = (2/3) * mu * F * (R_o^3 - R_i^3) / (R_o^2 - R_i^2)
    where:
        F_E  : normal force per pad [N]
        R_o: outer radius [m]
        R_i: inner radius [m]
        mu : friction coefficient [-]
    Returns:
        Torque M [N·m]
    """
    numerator = R_o**3 - R_i**3
    denominator = R_o**2 - R_i**2
    return (2 / 3) * mu * F_E * (numerator / denominator)


def disc_brake_worn_pads(mu, F_W, R_o, R_i):
    """
    Disc brake with worn pads (linearly translating pad, per pad).
        W = mu * F_W * (R_o^2 - R_i^2) / 2
    where:
        mu : friction coefficient [-]
        F_W : normal force per pad [N]
        R_o: outer radius [m]
        R_i: inner radius [m]
    Returns:
        Effective area or wear term [m^2]
    """
    return mu * F_W * (R_o - R_i) / 2