"""Nonlinear, constant-potential planar PB with sphere Derjaguin integration.

Symmetric monovalent free ions only. Does not remove curvature restrictions.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp

from . import E_CHARGE, Parameters


def dimensionless_edl(surface_potential, gaps, tolerance=1e-6):
    """Return integral_x^infinity (t-x) [cosh(phi_mid(t))-1] dt.

    surface_potential is e*psi/kBT; gaps are increasing kappa*h, strictly
    positive. Last gap must be >= 25 to make the omitted far-field tail tiny.
    Uses half-gap coordinate s=2*z/h, phi'(0)=0 and phi(1)=surface_potential.
    """
    gaps = np.asarray(gaps, dtype=float)
    if (gaps.ndim != 1 or gaps.size < 3 or not np.all(np.isfinite(gaps))
            or gaps[0] <= 0 or not np.all(np.diff(gaps) > 0) or gaps[-1] < 25):
        raise ValueError("gaps must increase from a positive value to >=25")
    if not np.isfinite(surface_potential) or abs(surface_potential) > 6:
        raise ValueError("require finite |e psi/kBT| <= 6")
    surface_potential = abs(surface_potential)
    if surface_potential == 0:
        return np.zeros_like(gaps)
    coordinate = np.linspace(0, 1, 80)
    pressures = []
    for gap in gaps:
        half = gap / 2
        guess = np.vstack((surface_potential * np.cosh(half * coordinate) / np.cosh(half),
                           surface_potential * half * np.sinh(half * coordinate) / np.cosh(half)))
        solution = solve_bvp(
            lambda s, y: np.vstack((y[1], half**2 * np.sinh(y[0]))),
            lambda left, right: np.array([left[1], right[0] - surface_potential]),
            coordinate, guess, tol=tolerance, max_nodes=5000,
        )
        if not solution.success:
            raise RuntimeError(f"PB solver failed at kappa*h={gap}: {solution.message}")
        # 2*sinh(phi/2)^2 avoids cancellation at tiny midplane potential.
        pressures.append(2 * np.sinh(solution.y[0, 0] / 2)**2)
    force = -cumulative_trapezoid(np.asarray(pressures)[::-1], gaps[::-1], initial=0)[::-1]
    energy = -cumulative_trapezoid(force[::-1], gaps[::-1], initial=0)[::-1]
    return energy


def energies(params: Parameters, gaps, dimensionless_energy):
    """Return h (m), U_vdw (J), U_edl (J), U_total (J) arrays.

    Nonretarded vdW and both sphere interactions use Derjaguin approximation.
    """
    h = np.asarray(gaps) * params.debye_length_m
    vdw = -params.hamaker_j * params.radius_m / (12 * h)
    thermal_voltage = params.thermal_energy_j / E_CHARGE
    edl = np.pi * params.radius_m * params.permittivity * thermal_voltage**2 * np.asarray(dimensionless_energy)
    return h, vdw, edl, vdw + edl
