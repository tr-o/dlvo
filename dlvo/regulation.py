"""Local charge regulation on curved dielectric spheres, coupled to bulk PB.

Free surface charge is q=q_infinity-C_inner*(u_surface-u_infinity).
The dielectric displacement inside each sphere is represented by its exact
harmonic Legendre Dirichlet-to-Neumann map (truncated angular expansion).
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.integrate import simpson, solve_bvp
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from . import E_CHARGE, Parameters
from .spheres import SphereGeometry


def isolated_sphere_charge(kappa_a, potential, linear=False):
    """Return dimensionless free charge q=-du/dr at r/a=1 for isolated sphere."""
    if not np.isfinite(kappa_a) or kappa_a <= 0 or not np.isfinite(potential):
        raise ValueError("require finite kappa*a>0 and finite potential")
    if linear:
        return (1+kappa_a)*potential
    r = np.r_[1, 1+np.geomspace(1e-6, 18/kappa_a, 450)]
    guess = potential*np.exp(-kappa_a*(r-1))/r
    solution = solve_bvp(
        lambda x, y: np.vstack((y[1], kappa_a**2*np.sinh(y[0])-2*y[1]/x)),
        lambda left, right: np.array([left[0]-potential, right[0]]), r,
        np.vstack((guess, -(kappa_a+1/r)*guess)), tol=1e-7, max_nodes=8000)
    if not solution.success:
        raise RuntimeError(solution.message)
    return float(-solution.y[1, 0])


def isolated_regulation_data(params, linear=False):
    """Calibrate surface charge and differential diffuse capacitance at infinity."""
    k = params.radius_m/params.debye_length_m
    u = abs(params.potential_v)*E_CHARGE/params.thermal_energy_j
    q = isolated_sphere_charge(k, u, linear=linear)
    delta = 1e-3
    capacitance = (isolated_sphere_charge(k, u+delta, linear=linear)
                   - isolated_sphere_charge(k, u-delta, linear=linear))/(2*delta)
    return u, q, capacitance


@dataclass
class RegulatedSolution:
    force_n: float
    scaled_residual: float
    iterations: int
    angles: np.ndarray
    surface_potential_v: np.ndarray
    free_charge_m2_c: np.ndarray
    maximum_surface_potential_v: float


class RegulatedGeometry:
    """Add local Robin surface conditions and dielectric interior to SphereGeometry.

    The normal derivative is reconstructed by weighted cubic fits to nearby
    exterior values, constrained by the unknown potential at each surface node.
    These surface unknowns are solved simultaneously with the nonlinear PB field.
    """

    def __init__(self, geometry: SphereGeometry):
        self.geometry = geometry
        count = int(np.ceil(np.pi/geometry.step))+1
        mu, weight = leggauss(count)
        mu, weight = mu[::-1], weight[::-1]
        self.angles = np.arccos(mu)
        vander = legvander(mu, count-1)
        modes = np.arange(count)
        self.interior_dtn = (vander*modes) @ ((2*modes+1)[:, None]/2*vander.T*weight)
        n = geometry.matrix.shape[0]
        # Interpolate potential at every exact sphere / grid-edge intersection.
        rows, cols, data = [], [], []
        for row, coefficient, angle in geometry.surface_links:
            upper = np.searchsorted(self.angles, angle)
            if upper == 0 or upper == count:
                rows.append(row)
                cols.append(0 if upper == 0 else count-1)
                data.append(coefficient)
            else:
                lower = upper-1
                fraction = (angle-self.angles[lower])/(self.angles[upper]-self.angles[lower])
                rows.extend([row, row])
                cols.extend([lower, upper])
                data.extend([coefficient*(1-fraction), coefficient*fraction])
        self.interpolation = coo_matrix((data, (rows, cols)), shape=(n, count)).tocsr()

        rr, zz = np.meshgrid(geometry.r, geometry.z, indexing="ij")
        points = np.c_[rr[geometry.fluid], zz[geometry.fluid]-geometry.center]
        ids = np.arange(n)
        reflected = points[:, 0] > 0
        extended = np.r_[points, points[reflected]*[-1, 1]]
        extended_ids = np.r_[ids, ids[reflected]]
        tree = cKDTree(extended)
        rows, cols, data, sums = [], [], [], []
        for row, angle in enumerate(self.angles):
            normal = np.array([np.sin(angle), np.cos(angle)])
            tangent = np.array([np.cos(angle), -np.sin(angle)])
            indices = tree.query_ball_point(normal, 4.5*geometry.step)
            relative = (extended[indices]-normal)/geometry.step
            normal_distance, tangent_distance = relative@normal, relative@tangent
            design = np.c_[normal_distance, tangent_distance, normal_distance**2,
                           normal_distance*tangent_distance, tangent_distance**2,
                           normal_distance**3, normal_distance**2*tangent_distance,
                           normal_distance*tangent_distance**2, tangent_distance**3]
            weights = np.exp(-np.sum(relative**2, axis=1)/8)
            operator = np.linalg.pinv(design*weights[:, None], rcond=1e-10)[0]*weights/geometry.step
            if np.linalg.matrix_rank(design) < 9:
                raise RuntimeError("Insufficient points for surface-normal derivative")
            for col, value in zip(extended_ids[indices], operator):
                rows.append(row)
                cols.append(col)
                data.append(value)
            sums.append(float(np.sum(operator)))
        self.exterior_derivative = coo_matrix((data, (rows, cols)), shape=(count, n)).tocsr()
        self.derivative_surface_sum = np.asarray(sums)

    def solve(self, params: Parameters, regulation=0.5, particle_permittivity=150,
              calibration=None, linear=False, tolerance=1e-9):
        if not 0 <= regulation <= 1 or not np.isfinite(particle_permittivity) or particle_permittivity < 0:
            raise ValueError("require regulation in [0,1], finite particle permittivity>=0")
        g = self.geometry
        if not np.isclose(params.radius_m, g.radius_m, rtol=1e-10, atol=0) or not np.isclose(
                params.radius_m/params.debye_length_m, g.k, rtol=1e-10):
            raise ValueError("Parameters do not match geometry")
        voltage = params.thermal_energy_j/E_CHARGE
        uinf, qinf, cd = calibration or isolated_regulation_data(params, linear=linear)
        if regulation == 0:
            matrix = bmat([[g.matrix, -self.interpolation],
                           [None, diags(np.ones(len(self.angles)))]], format="csr")
            rhs_surface = np.full(len(self.angles), uinf)
        else:
            cin = cd*(1-regulation)/regulation
            surface_block = (diags(self.derivative_surface_sum+cin)
                             + csr_matrix(particle_permittivity/params.relative_permittivity*self.interior_dtn))
            matrix = bmat([[g.matrix, -self.interpolation],
                           [-self.exterior_derivative, surface_block]], format="csr")
            rhs_surface = np.full(len(self.angles), qinf+cin*uinf)
        n = g.matrix.shape[0]
        mass = np.r_[np.full(n, g.k**2), np.zeros(len(self.angles))]
        rhs = np.r_[np.zeros(n), rhs_surface]
        scale = np.asarray(abs(matrix).sum(axis=1)).ravel()+mass
        u = spsolve(matrix+diags(mass), rhs)
        iterations, residual = 0, 0.0
        if not linear:
            for iterations in range(40):
                nonlinear = np.r_[g.k**2*np.sinh(u[:n]), np.zeros(len(self.angles))]
                res = matrix@u+nonlinear-rhs
                residual = float(np.max(abs(res)/scale))
                if residual < tolerance:
                    break
                derivative = np.r_[g.k**2*np.cosh(u[:n]), np.zeros(len(self.angles))]
                step = spsolve(matrix+diags(derivative), -res)
                for power in range(16):
                    trial = u+step*0.5**power
                    if np.max(abs(trial[:n])) > 20:
                        continue
                    trial_res = matrix@trial+np.r_[g.k**2*np.sinh(trial[:n]), np.zeros(len(self.angles))]-rhs
                    if np.max(abs(trial_res)/scale) < residual:
                        u = trial
                        break
                else:
                    raise RuntimeError("Regulated PB Newton line search failed")
            else:
                raise RuntimeError("Regulated PB Newton did not converge")
        potential = np.zeros(g.ids.shape)
        potential[g.fluid] = u[:n]
        mid = potential[:, 0]
        gradient = np.gradient(mid, g.r, edge_order=2)
        gradient[0] = 0
        pressure = g.k**2*(mid**2/2 if linear else 2*np.sinh(mid/2)**2)
        force = 2*np.pi*params.permittivity*voltage**2*simpson(g.r*(pressure+gradient**2/2), x=g.r)
        surface = u[n:]
        if regulation == 0:
            charge = self.derivative_surface_sum*surface-self.exterior_derivative@u[:n]
        else:
            charge = qinf-cd*(1-regulation)/regulation*(surface-uinf)
        sign = np.sign(params.potential_v)
        return RegulatedSolution(float(force), residual, iterations, self.angles,
                                 surface*voltage*sign,
                                 charge*params.permittivity*voltage/params.radius_m*sign,
                                 float(np.max(abs(surface))*voltage))
