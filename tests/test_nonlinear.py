import unittest
import importlib.util

import numpy as np

if importlib.util.find_spec("scipy") is None:
    raise unittest.SkipTest("Install .[research] to test nonlinear PB")

from dlvo import E_CHARGE, Parameters
from dlvo.nonlinear import dimensionless_edl, energies


class NonlinearTests(unittest.TestCase):
    def test_linear_limit_and_grid_convergence(self):
        p = Parameters(potential_v=0.001)
        y = p.potential_v * E_CHARGE / p.thermal_energy_j
        coarse = np.geomspace(1e-3, 30, 250)
        fine = np.geomspace(1e-3, 30, 700)
        u_coarse = dimensionless_edl(y, coarse)
        u_fine = dimensionless_edl(y, fine)
        mask = fine < 5
        analytic = 2 * y**2 * np.log1p(np.exp(-fine[mask]))
        np.testing.assert_allclose(u_fine[mask], analytic, rtol=0.003)
        np.testing.assert_allclose(np.interp(fine[mask], coarse, u_coarse), u_fine[mask], rtol=0.025)

    def test_nonlinear_grid_convergence_and_symmetry(self):
        coarse = np.geomspace(1e-4, 30, 350)
        fine = np.geomspace(1e-4, 30, 700)
        c = dimensionless_edl(3, coarse)
        f = dimensionless_edl(-3, fine)
        mask = fine < 4
        np.testing.assert_allclose(np.interp(fine[mask], coarse, c), f[mask], rtol=0.015)
        far = (fine > 12) & (fine < 15)
        asymptote = 32 * np.tanh(3 / 4)**2 * np.exp(-fine[far])
        np.testing.assert_allclose(f[far], asymptote, rtol=0.02)
        self.assertTrue(np.all(f >= 0))
        self.assertTrue(np.all(np.diff(f) <= 0))

    def test_units_zero_and_validation(self):
        gaps = np.geomspace(0.01, 30, 50)
        p = Parameters()
        h, vdw, edl, total = energies(p, gaps, dimensionless_edl(0, gaps))
        np.testing.assert_allclose(h, gaps * p.debye_length_m)
        np.testing.assert_allclose(total, vdw)
        np.testing.assert_array_equal(edl, 0)
        for invalid in ([0, 1, 30], [1, 30, 2], [1, 2, 3]):
            with self.assertRaises(ValueError):
                dimensionless_edl(1, invalid)
