import importlib.util
import math
import unittest

import numpy as np

if importlib.util.find_spec("scipy") is None:
    raise unittest.SkipTest("Install .[research] to test spherical PB")

from dlvo import E_CHARGE, Parameters
from dlvo.spheres import SphereGeometry, sphere_vdw_j


class SphereTests(unittest.TestCase):
    def test_published_nonlinear_two_sphere_force(self):
        # Qiao, Li & Tang (2006), section 5.1/table 2:
        # kappa*a=5, kappa*h=0.5, e*psi/kBT=2; F*=15.535.
        ref = Parameters(radius_m=5e-9, relative_permittivity=24.3)
        strength = ref.ionic_strength_mol_l * (ref.debye_length_m / 1e-9)**2
        p = Parameters(radius_m=5e-9, relative_permittivity=24.3,
                       ionic_strength_mol_l=strength,
                       potential_v=2 * ref.thermal_energy_j / E_CHARGE)
        s = SphereGeometry(p.radius_m, p.debye_length_m, .5e-9, resolution=20).solve(p)
        scaled = s.force_n / (math.pi * p.permittivity * (p.thermal_energy_j / E_CHARGE)**2)
        self.assertLess(abs(scaled / 15.535 - 1), 0.005)

    def test_grid_and_outer_boundary_convergence(self):
        p = Parameters(radius_m=5e-9, potential_v=.034, relative_permittivity=24.3)
        coarse = SphereGeometry(p.radius_m, p.debye_length_m, 1e-9, resolution=12).solve(p)
        fine = SphereGeometry(p.radius_m, p.debye_length_m, 1e-9, resolution=24).solve(p)
        wide = SphereGeometry(p.radius_m, p.debye_length_m, 1e-9, resolution=24, padding=16).solve(p)
        self.assertLess(abs(coarse.force_n / fine.force_n - 1), 0.01)
        self.assertLess(abs(fine.force_n / wide.force_n - 1), 0.002)
        self.assertLess(fine.scaled_residual, 1e-9)

    def test_linear_far_field_and_weak_potential_limit(self):
        p = Parameters(radius_m=5e-9, potential_v=.001, relative_permittivity=24.3,
                       ionic_strength_mol_l=1e-4)
        h = 100e-9
        g = SphereGeometry(p.radius_m, p.debye_length_m, h, resolution=16)
        linear = g.solve(p, linear=True)
        nonlinear = g.solve(p)
        d, k = 2 * p.radius_m + h, 1 / p.debye_length_m
        far = 4 * np.pi * p.permittivity * p.potential_v**2 * p.radius_m**2 * np.exp(-k*h) * (k/d + 1/d**2)
        self.assertLess(abs(linear.force_n / far - 1), 0.03)
        self.assertLess(abs(linear.force_n / nonlinear.force_n - 1), 0.002)

    def test_zero_potential_and_sign_symmetry(self):
        p = Parameters(radius_m=5e-9, potential_v=0, relative_permittivity=24.3)
        g = SphereGeometry(p.radius_m, p.debye_length_m, 1e-9, resolution=12)
        self.assertEqual(g.solve(p).force_n, 0)
        p1 = Parameters(radius_m=5e-9, potential_v=.034, relative_permittivity=24.3)
        p2 = Parameters(radius_m=5e-9, potential_v=-.034, relative_permittivity=24.3)
        self.assertEqual(g.solve(p1).force_n, g.solve(p2).force_n)

    def test_hamaker_near_and_far_limits(self):
        a, ah = 5e-9, 8e-20
        h = a * 1e-6
        self.assertLess(abs(float(sphere_vdw_j(a, h, ah)) / (-ah*a/(12*h)) - 1), 1e-4)
        d = 1e4 * a
        exact = float(sphere_vdw_j(a, d - 2*a, ah))
        far = -16 * ah * a**6 / (9*d**6)
        self.assertLess(abs(exact / far - 1), 1e-6)
