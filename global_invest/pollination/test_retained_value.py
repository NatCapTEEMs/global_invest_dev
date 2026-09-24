import unittest
import numpy as np
from retained_value import retained_value_change, output_share_pct, annual_retained_rows


class RetainedValueTests(unittest.TestCase):
    def test_half_retained_does_not_receive_whole_cell_value(self):
        # $100 crop value; $20 dependent; half retained; sufficiency .4 -> .6.
        delta = retained_value_change(20, .5, .4, .6)
        self.assertAlmostEqual(float(delta), 2.)
        self.assertAlmostEqual(output_share_pct(delta, 100), 2.)

    def test_turnover_alone_is_not_habitat_productivity_change(self):
        np.testing.assert_allclose(retained_value_change([20, 20], [0, .5], [.4, .4], [.4, .4]), 0)

    def test_habitat_loss_never_produces_gain(self):
        delta = retained_value_change([20, 10], [.5, 1], [.8, .5], [.2, .4])
        self.assertTrue(np.all(delta < 0))

    def test_sector_denominator_includes_all_baseline_crop_value(self):
        delta = retained_value_change([20, 10], [.5, 0], [.4, np.nan], [.6, np.nan])
        self.assertAlmostEqual(output_share_pct(delta, [100, 100]), 1.)

    def test_inconsistent_coverage_and_missing_aggregation_refused(self):
        with self.assertRaises(ValueError):
            retained_value_change(20, .5, .4, np.nan)
        with self.assertRaises(ValueError):
            output_share_pct([1, np.nan], [100, 100])

    def test_common_value_unit_conversion_cancels(self):
        self.assertAlmostEqual(output_share_pct([2, -1], [100, 50]),
                               output_share_pct([2000, -1000], [100000, 50000]))

    def test_annual_dollars_allow_zero_baseline_and_preserve_anchor_change(self):
        import pandas as pd
        index=pd.MultiIndex.from_tuples([('AEZ1','USA')])
        zero=pd.Series([0.],index=index); future=pd.Series([2.],index=index)
        rows=annual_retained_rows({2030:zero},{2030:future},zero,'current_policies',2023,['V_F'])
        self.assertEqual(rows.iloc[0].delta_pollination_usd,0.)
        self.assertEqual(rows.iloc[-1].delta_pollination_usd,2.)
        np.testing.assert_allclose(rows.delta_pollination_usd,np.linspace(0,2,8))


if __name__ == '__main__':
    unittest.main()
