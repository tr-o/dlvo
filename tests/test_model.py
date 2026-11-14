import math
import unittest

from dlvo import Parameters


class ModelTests(unittest.TestCase):
    def test_debye_length_reference_water_one_mm(self):
        self.assertAlmostEqual(Parameters().debye_length_m * 1e9, 9.6198, places=3)

    def test_debye_concentration_scaling(self):
        self.assertAlmostEqual(Parameters(ionic_strength_mol_l=0.1).debye_length_m,
                               Parameters().debye_length_m / 10)

    def test_attraction_analytic_value(self):
        attraction, _, _ = Parameters().energy_j(1e-9)
        self.assertAlmostEqual(attraction / (-8.333333333333333e-20), 1)

    def test_zero_potential_and_sum(self):
        attraction, repulsion, total = Parameters(potential_v=0).energy_j(1e-9)
        self.assertEqual(repulsion, 0)
        self.assertEqual(total, attraction)

    def test_double_layer_contact_limit(self):
        p = Parameters()
        expected = 2 * math.pi * p.permittivity * p.radius_m * p.potential_v**2 * math.log(2)
        self.assertAlmostEqual(p.energy_j(1e-20)[1] / expected, 1)

    def test_salt_screens_repulsion(self):
        self.assertLess(Parameters(ionic_strength_mol_l=0.1).energy_j(5e-9)[1],
                        Parameters().energy_j(5e-9)[1])

    def test_sign_symmetry_and_infinite_separation(self):
        self.assertEqual(Parameters(potential_v=-0.02).energy_j(1e-9),
                         Parameters(potential_v=0.02).energy_j(1e-9))
        self.assertLess(abs(Parameters().energy_j(1e6)[2]), 1e-30)

    def test_invalid_inputs(self):
        for field in ("radius_m", "ionic_strength_mol_l", "hamaker_j",
                      "temperature_k", "relative_permittivity"):
            for value in (0, -1, math.inf, math.nan):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    Parameters(**{field: value})
        for value in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                Parameters().energy_j(value)


if __name__ == "__main__":
    unittest.main()
