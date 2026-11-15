"""Explicitly phenomenological short-range pair potentials for sensitivity work.

Neither function supplies measured BaTiO3/ethanol parameters or a molecular
polymer-brush model. Fit amplitudes to forces/adsorption before predictions.
"""

import math
import numpy as np


def solvation_kbt(separation_m, contact_amplitude_kbt, decay_length_m):
    """Positive monotonic pair-energy envelope B exp(-h/lambda), in kBT.

    This tests an assumed repulsive branch, omitting molecular oscillations,
    phase, attractive branches and chemically specific interactions.
    """
    h = np.asarray(separation_m, dtype=float)
    if (not math.isfinite(contact_amplitude_kbt) or contact_amplitude_kbt < 0
            or not math.isfinite(decay_length_m) or decay_length_m <= 0
            or np.any(~np.isfinite(h)) or np.any(h < 0)):
        raise ValueError("require finite h>=0, amplitude>=0 and decay length>0")
    return contact_amplitude_kbt*np.exp(-h/decay_length_m)


def corona_overlap_m3(radius_m, layer_m, separation_m):
    """Intersection volume of two equal OUTER spheres, not shell-shell volume.

    U=w*V_overlap is a coarse pair-energy penalty for overlapping coronas.
    It includes core exclusion implicitly in the fitted w. It is not a
    microscopic osmotic-pressure formula or an Alexander-de Gennes brush law.
    """
    h = np.asarray(separation_m, dtype=float)
    if (not math.isfinite(radius_m) or radius_m <= 0 or not math.isfinite(layer_m)
            or layer_m < 0 or np.any(~np.isfinite(h)) or np.any(h < 0)):
        raise ValueError("require finite radius>0, layer>=0, h>=0")
    outer = radius_m+layer_m
    center = 2*radius_m+h
    overlap = np.maximum(0, 2*outer-center)
    return np.pi*(4*outer+center)*overlap**2/12


def corona_kbt(radius_m, layer_m, separation_m, penalty_j_m3, thermal_energy_j):
    """Return w*overlap_volume/kBT; w has units J/m3 (=Pa), is a fitted penalty."""
    if (not math.isfinite(penalty_j_m3) or penalty_j_m3 < 0
            or not math.isfinite(thermal_energy_j) or thermal_energy_j <= 0):
        raise ValueError("require finite penalty>=0 and thermal energy>0")
    return penalty_j_m3*corona_overlap_m3(radius_m, layer_m, separation_m)/thermal_energy_j
