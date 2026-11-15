import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec("scipy") is None:
    raise unittest.SkipTest("Install .[research] to test charge regulation")

from dlvo import E_CHARGE, Parameters
from dlvo.regulation import RegulatedGeometry, isolated_regulation_data, isolated_sphere_charge
from dlvo.spheres import SphereGeometry
from dlvo.short_range import corona_kbt, corona_overlap_m3, solvation_kbt
from dlvo.electrokinetics import surface_potential_for_zeta
from dlvo.speciation import free_ionic_strength_m


class RegulationTests(unittest.TestCase):
    def setUp(self):
        self.p = Parameters(radius_m=5e-9, potential_v=.034, relative_permittivity=24.3)

    def test_isolated_linear_limit_and_zero_charge(self):
        k = self.p.radius_m/self.p.debye_length_m
        self.assertEqual(isolated_sphere_charge(k, 0), 0)
        self.assertLess(abs(isolated_sphere_charge(k, .01)/((1+k)*.01)-1), 1e-5)

    def test_constant_potential_limit_and_regulation_order(self):
        g = SphereGeometry(self.p.radius_m, self.p.debye_length_m, 1e-9, resolution=20, gap_points=8)
        r = RegulatedGeometry(g)
        data = isolated_regulation_data(self.p)
        cp = r.solve(self.p, regulation=0)
        cr = r.solve(self.p, regulation=.5, calibration=data)
        cc = r.solve(self.p, regulation=1, calibration=data)
        self.assertLess(abs(cp.force_n/g.solve(self.p).force_n-1), 1e-8)
        self.assertLess(cp.force_n, cr.force_n)
        self.assertLess(cr.force_n, cc.force_n)
        np.testing.assert_allclose(cc.free_charge_m2_c, cc.free_charge_m2_c[0], atol=1e-15)
        self.assertGreater(cc.maximum_surface_potential_v, self.p.potential_v)

    def test_normal_derivative_and_interior_dtn(self):
        g = SphereGeometry(self.p.radius_m, self.p.debye_length_m, 1e-9, resolution=20, gap_points=8)
        r = RegulatedGeometry(g)
        rr, zz = np.meshgrid(g.r, g.z-g.center, indexing="ij")
        distance = np.sqrt(rr**2+zz**2)[g.fluid]
        analytic = np.exp(-g.k*(distance-1))/distance
        reconstructed = r.derivative_surface_sum-r.exterior_derivative@analytic
        np.testing.assert_allclose(reconstructed, 1+g.k, rtol=.004)
        np.testing.assert_allclose(r.interior_dtn@np.ones(len(r.angles)), 0, atol=1e-8)
        # The l=2 solid harmonic has normal derivative 2*P2 on the unit sphere.
        p2 = (3*np.cos(r.angles)**2-1)/2
        np.testing.assert_allclose(r.interior_dtn@p2, 2*p2, atol=1e-8)

    def test_neumann_far_field_and_refinement(self):
        data = isolated_regulation_data(self.p)
        g = SphereGeometry(self.p.radius_m, self.p.debye_length_m, 100e-9, resolution=20, gap_points=8)
        far = RegulatedGeometry(g).solve(self.p, regulation=1, calibration=data)
        np.testing.assert_allclose(far.surface_potential_v, self.p.potential_v, rtol=.005)
        forces = []
        for resolution in (20, 32):
            g = SphereGeometry(self.p.radius_m, self.p.debye_length_m, 1e-9,
                               resolution=resolution, gap_points=8)
            forces.append(RegulatedGeometry(g).solve(self.p, regulation=1, calibration=data).force_n)
        self.assertLess(abs(forces[0]/forces[1]-1), .01)


class ShortRangeTests(unittest.TestCase):
    def test_corona_intersection_geometry_and_scale(self):
        radius, layer = 5e-9, 2e-9
        h = np.array([0, 1e-9, 4e-9, 5e-9])
        v = corona_overlap_m3(radius, layer, h)
        self.assertTrue(np.all(np.diff(v) <= 0))
        np.testing.assert_array_equal(v[2:], 0)
        self.assertAlmostEqual(v[0]/(np.pi*(4*7+10)*4**2/12*1e-27), 1)
        np.testing.assert_allclose(corona_kbt(radius, layer, h, 2e6, 4e-21),
                                   2*corona_kbt(radius, layer, h, 1e6, 4e-21))

    def test_solvation_limits_and_validation(self):
        np.testing.assert_allclose(solvation_kbt([0, 1e-9], 10, 1e-9), [10, 10/np.e])
        np.testing.assert_array_equal(solvation_kbt([0, 1e-9], 0, 1e-9), 0)
        with self.assertRaises(ValueError):
            solvation_kbt([1e-9], -1, 1e-9)
        with self.assertRaises(ValueError):
            corona_overlap_m3(5e-9, -1e-9, 1e-9)


class ElectrokineticAndSpeciationTests(unittest.TestCase):
    def test_weak_potential_shear_plane_mapping(self):
        p = Parameters(radius_m=5e-9, potential_v=.001, relative_permittivity=24.3)
        offset = .5e-9
        expected = .001*(1+offset/p.radius_m)*np.exp(offset/p.debye_length_m)
        actual = surface_potential_for_zeta(p, .001, offset)
        self.assertLess(abs(actual/expected-1), .001)
        self.assertEqual(surface_potential_for_zeta(p, .034, 0), .034)
        self.assertAlmostEqual(surface_potential_for_zeta(p, -.001, offset), -actual)

    def test_ion_pair_mass_balance(self):
        for total in (0, 1e-6, 1e-3):
            for k in (0, 100, 1e6):
                free = free_ionic_strength_m(total, k)
                self.assertAlmostEqual(free+k*free**2, total, places=12)
                self.assertLessEqual(free, total)
