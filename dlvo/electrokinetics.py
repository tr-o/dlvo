"""Conditional mapping of shear-plane potential to surface potential.

Assumes zeta equals isolated radial PB potential at a specified shear plane.
This is not a complete electrokinetic mobility or Henry-function inversion.
"""

import math

from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import numpy as np

from . import E_CHARGE, Parameters
from .regulation import isolated_sphere_charge


def surface_potential_for_zeta(params: Parameters, zeta_v, shear_offset_m):
    if not math.isfinite(zeta_v) or not math.isfinite(shear_offset_m) or shear_offset_m < 0:
        raise ValueError("require finite zeta and nonnegative shear offset")
    if shear_offset_m == 0 or zeta_v == 0:
        return zeta_v
    voltage = params.thermal_energy_j/E_CHARGE
    k = params.radius_m/params.debye_length_m
    target = abs(zeta_v)/voltage
    endpoint = 1+shear_offset_m/params.radius_m

    def residual(surface):
        q = isolated_sphere_charge(k, surface)
        solution = solve_ivp(lambda x,y: [y[1], k*k*np.sinh(y[0])-2*y[1]/x],
                             (1, endpoint), [surface, -q], rtol=1e-8, atol=1e-10)
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution.y[0, -1]-target

    upper = max(target*1.5, 1)
    while residual(upper) < 0 and upper < 10:
        upper = min(10, upper*1.4)
    if residual(upper) < 0:
        raise ValueError("Requested zeta/offset requires surface potential outside the explored range")
    root = brentq(residual, target, upper, xtol=1e-7)
    return float(np.sign(zeta_v)*root*voltage)
