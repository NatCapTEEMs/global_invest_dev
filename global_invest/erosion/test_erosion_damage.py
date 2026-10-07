"""Small physical and accounting invariants; no raster runtime required.

Run from this directory: python -m unittest test_erosion_damage
"""
import unittest
import numpy as np
from erosion_damage import productivity_level, stressed_soil_loss, annual_damage_change, summarize_damage_areas


class DamageTests(unittest.TestCase):
    def test_raster_threshold_before_aggregation_and_missing_coverage(self):
        rows=summarize_damage_areas([10.,12.,np.nan,11.], [30.,20.,30.,11.],
                                   [1.,.5,1.,1.],10.,[1,1,1,2]).set_index('zone_id')
        self.assertEqual(rows.loc[1,'valid_crop_ha'],15.)
        self.assertEqual(rows.loc[1,'severe_crop_ha'],5.)
        self.assertEqual(rows.loc[1,'excluded_crop_ha'],10.)
        self.assertAlmostEqual(rows.loc[1,'level'],-8/3)
        self.assertEqual(rows.loc[1,'stressed_level'],-8.)
        self.assertEqual(rows.loc[2,'level'],0.)

    def test_none_all_and_half_severe(self):
        np.testing.assert_allclose(productivity_level([0, 50, 100], 100), [0, -4, -8])

    def test_deterioration_and_unchanged(self):
        base = productivity_level(10, 100)
        self.assertAlmostEqual(float(productivity_level(30, 100) - base), -1.6)
        self.assertEqual(float(base - base), 0)

    def test_missing_is_not_zero_damage(self):
        self.assertTrue(np.isnan(productivity_level(0, 0)))
        self.assertTrue(np.isnan(productivity_level(np.nan, 100)))

    def test_impossible_areas_rejected(self):
        for severe, total in ((101, 100), (-1, 100), (1, -100)):
            with self.assertRaises(ValueError):
                productivity_level(severe, total)

    def test_stress_threshold_and_zero_prevention(self):
        # First field crosses >11 after losing 20% of its 20 t/ha prevention.
        # Second has no prevention to lose. Third is exactly at the threshold.
        soil_loss = stressed_soil_loss([10, 10, 11], [30, 10, 11])
        np.testing.assert_allclose(soil_loss, [14, 10, 11])
        np.testing.assert_array_equal(soil_loss > 11, [True, False, False])

    def test_stress_monotonic_and_endpoints(self):
        actual, potential = np.array([0, 3, 12]), np.array([0, 20, 40])
        np.testing.assert_allclose(stressed_soil_loss(actual, potential, 0), actual)
        np.testing.assert_allclose(stressed_soil_loss(actual, potential, 1), potential)
        self.assertTrue(np.all(stressed_soil_loss(actual, potential) >= actual))
        self.assertTrue(np.isnan(stressed_soil_loss(np.nan, 20)))

    def test_invalid_physical_inputs_rejected(self):
        with self.assertRaises(ValueError):
            stressed_soil_loss(20, 10)
        with self.assertRaises(ValueError):
            stressed_soil_loss(1, 10, 1.2)

    def test_annual_stress_has_no_anticipation_or_repeated_cut(self):
        years = np.arange(2023, 2051)
        ordinary = annual_damage_change(-1, [2030, 2040, 2050], [-2, -2, -2], years)
        stress = annual_damage_change(-1, [2030, 2040, 2050], [-2, -2, -2], years,
                                      stressed_levels=[-3, -3, -3])
        np.testing.assert_array_equal(stress[years < 2030], ordinary[years < 2030])
        np.testing.assert_allclose(stress[years >= 2030], -2)
        self.assertEqual(stress[0], 0)
        factors = 1 + stress / 100
        annual = factors[1:] / factors[:-1]
        np.testing.assert_allclose(np.cumprod(annual), factors[1:])

    def test_stress_improvement_is_rejected(self):
        with self.assertRaises(ValueError):
            annual_damage_change(-1, [2030], [-2], [2023, 2030], stressed_levels=[-1])


if __name__ == '__main__':
    unittest.main()
