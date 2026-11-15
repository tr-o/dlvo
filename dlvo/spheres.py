"""Axisymmetric nonlinear PB around two equal constant-potential spheres.

Lengths in the numerical PDE are scaled by sphere radius, potential by kBT/e.
An irregular-boundary finite-difference stencil locates sphere intersections
analytically; surfaces are not replaced by flat plates or voxel boundaries.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import simpson
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from . import E_CHARGE, Parameters


def _graded_interval(left, right, step, grow=1.10, both=False):
    if right <= left:
        return np.array([left])
    if both:
        mid = (left + right) / 2
        one = _graded_interval(left, mid, step, grow)
        return np.r_[one, left + right - one[-2::-1]]
    points = [left]
    delta = step
    while points[-1] + delta < right:
        points.append(points[-1] + delta)
        delta *= grow
    # Avoid an arbitrarily tiny final interval.
    if len(points) > 1 and right - points[-1] < 0.4 * delta:
        points.pop()
    points.append(right)
    return np.array(points)


@dataclass
class SphereSolution:
    force_n: float
    scaled_residual: float
    iterations: int
    unknowns: int
    r_m: np.ndarray
    z_m: np.ndarray
    potential_v: np.ndarray


class SphereGeometry:
    """Discretize half of the two-sphere domain with mirror plane z=0.

    Dirichlet: sphere surfaces at a common potential, outer box at zero.
    Neumann symmetry: axis r=0 and mirror plane z=0.
    resolution is points per radius around the sphere, NOT a global grid size.
    padding is the number of Debye lengths to the outer boundaries.
    """

    def __init__(self, radius_m, debye_length_m, separation_m, resolution=20, padding=12, gap_points=0):
        if any(not math.isfinite(v) or v <= 0 for v in (radius_m, debye_length_m, separation_m, padding)):
            raise ValueError("geometry lengths and padding must be finite and positive")
        if resolution < 8:
            raise ValueError("resolution must be >=8")
        self.radius_m = radius_m
        self.k = radius_m / debye_length_m
        self.gap = separation_m / radius_m
        center = 1 + self.gap / 2
        step = min(1 / resolution, 1 / (self.k * 8))
        self.center, self.step = center, step
        local_r = np.linspace(0, 1.5, int(np.ceil(1.5 / step)) + 1)
        r = np.r_[local_r, _graded_interval(1.5, 1.5 + padding / self.k, step)[1:]]
        low, high = max(0, center - 1.5), center + 1.5
        local_z = np.linspace(low, high, int(np.ceil((high - low) / step)) + 1)
        below = _graded_interval(0, low, step, both=True)[:-1] if low > 0 else np.array([])
        above = _graded_interval(high, high + padding / self.k, step)[1:]
        z = np.r_[below, local_z, above]
        if gap_points and np.count_nonzero(z < self.gap/2) < gap_points:
            # Surface-normal charge conditions need several fluid nodes even
            # when the axial half-gap is smaller than the ordinary core step.
            z = np.r_[np.linspace(0, self.gap/2, gap_points+1), z[z > self.gap/2+1e-12]]
        rr, zz = np.meshgrid(r, z, indexing="ij")
        solid = rr**2 + (zz - center)**2 <= 1 + 1e-13
        fluid = ~solid
        fluid[-1, :] = False
        fluid[:, -1] = False
        ids = np.full(rr.shape, -1, dtype=int)
        ids[fluid] = np.arange(np.count_nonzero(fluid))
        ii, jj = np.nonzero(fluid)
        n = len(ii)
        diagonal = np.zeros(n)
        boundary = np.zeros(n)
        rows, cols, values = [], [], []
        surface_links = []

        for row, (i, j) in enumerate(zip(ii, jj)):
            for axis in (0, 1):
                coord, pos = (r, i) if axis == 0 else (z, j)
                # Two neighbours, with boundary distances shortened where an
                # edge intersects the exact spherical surface.
                distances, neighbours = [], []
                directions = (1,) if pos == 0 else (-1, 1)
                for direction in directions:
                    ni, nj = (i + direction, j) if axis == 0 else (i, j + direction)
                    distance = abs(coord[pos + direction] - coord[pos])
                    surface = solid[ni, nj]
                    if surface:
                        if axis == 0:
                            intersection = math.sqrt(max(0, 1 - (z[j] - center)**2))
                            distance = abs(r[i] - intersection)
                        else:
                            half_height = math.sqrt(max(0, 1 - r[i]**2))
                            intersection = center - half_height if direction == 1 else center + half_height
                            distance = abs(z[j] - intersection)
                    if distance <= 1e-14:
                        raise RuntimeError("Degenerate interface intersection")
                    distances.append(distance)
                    neighbours.append((ni, nj, surface))
                if pos == 0:
                    coefficients = [(4 if axis == 0 else 2) / distances[0]**2]
                else:
                    dm, dp = distances
                    coefficients = [2 / (dm * (dm + dp)), 2 / (dp * (dm + dp))]
                    if axis == 0:
                        coefficients[0] -= dp / (r[i] * dm * (dm + dp))
                        coefficients[1] += dm / (r[i] * dp * (dm + dp))
                for coefficient, (ni, nj, surface), distance, direction in zip(
                        coefficients, neighbours, distances, directions):
                    diagonal[row] += coefficient
                    if surface:
                        boundary[row] += coefficient
                        sr = r[i] + distance*direction if axis == 0 else r[i]
                        sz = z[j] + distance*direction if axis == 1 else z[j]
                        surface_links.append((row, coefficient, math.atan2(sr, sz-center)))
                    elif ids[ni, nj] >= 0:
                        rows.append(row)
                        cols.append(ids[ni, nj])
                        values.append(-coefficient)
        rows.extend(range(n))
        cols.extend(range(n))
        values.extend(diagonal)
        self.matrix = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
        self.boundary = boundary
        self.diagonal = diagonal
        self.r, self.z, self.ids, self.solid, self.fluid = r, z, ids, solid, fluid
        self.surface_links = surface_links

    def solve(self, params: Parameters, linear=False, tolerance=1e-9):
        if not math.isclose(params.radius_m, self.radius_m, rel_tol=1e-10) or not math.isclose(
                params.radius_m / params.debye_length_m, self.k, rel_tol=1e-10):
            raise ValueError("Parameters do not match this geometry")
        voltage = params.thermal_energy_j / E_CHARGE
        surface = abs(params.potential_v) / voltage
        if surface > 6:
            raise ValueError("require |e psi/kBT| <=6")
        k2 = self.k**2
        rhs = self.boundary * surface
        u = spsolve(self.matrix + diags(np.full(len(rhs), k2)), rhs)
        residual = 0.0
        iterations = 0
        if not linear:
            for iterations in range(30):
                res = self.matrix @ u + k2 * np.sinh(u) - rhs
                residual = float(np.max(np.abs(res) / (self.diagonal + k2)))
                if residual < tolerance:
                    break
                update = spsolve(self.matrix + diags(k2 * np.cosh(u)), -res)
                # Damped Newton: require a decrease in the scaled residual.
                for power in range(15):
                    trial = u + update * 0.5**power
                    trial_res = self.matrix @ trial + k2 * np.sinh(trial) - rhs
                    if np.max(np.abs(trial_res) / (self.diagonal + k2)) < residual:
                        u = trial
                        break
                else:
                    raise RuntimeError("PB Newton line search failed")
            else:
                raise RuntimeError("PB Newton iterations did not converge")
        potential = np.zeros(self.ids.shape)
        potential[self.solid] = surface
        potential[self.fluid] = u
        mid = potential[:, 0]
        derivative = np.gradient(mid, self.r, edge_order=2)
        derivative[0] = 0
        pressure = k2 * (mid**2 / 2 if linear else 2 * np.sinh(mid / 2)**2)
        force = 2 * np.pi * params.permittivity * voltage**2 * simpson(
            self.r * (pressure + derivative**2 / 2), x=self.r)
        return SphereSolution(float(force), residual, iterations, len(u),
                              self.r * self.radius_m, self.z * self.radius_m,
                              potential * voltage * np.sign(params.potential_v))


def sphere_vdw_j(radius_m, separation_m, hamaker_j):
    """Exact nonretarded pairwise-additive Hamaker energy of identical spheres.

    Uses a convergent far-field series to avoid cancellation for h>>a.
    This is geometrically exact within the continuum Hamaker approximation.
    """
    h = np.asarray(separation_m, dtype=float)
    if (not math.isfinite(radius_m) or not math.isfinite(hamaker_j)
            or radius_m <= 0 or hamaker_j <= 0 or np.any(~np.isfinite(h)) or np.any(h <= 0)):
        raise ValueError("radius, Hamaker constant and separations must be positive and finite")
    t = 2 * radius_m**2 / (h + 2 * radius_m)**2
    near = t >= 0.05
    bracket = np.zeros_like(t)
    tn = t[near]
    bracket[near] = tn / (1 - 2 * tn) + tn + np.log1p(-2 * tn)
    tf = t[~near]
    for n in range(3, 30):
        bracket[~near] += 2**(n-1) * (1 - 2 / n) * tf**n
    return -hamaker_j / 6 * bracket
