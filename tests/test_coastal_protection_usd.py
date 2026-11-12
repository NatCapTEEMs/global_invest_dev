"""Unit tests for coastal-protection USD-year harmonization (no service data needed)."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd

MODULE_PATH = (Path(__file__).resolve().parents[1] / "global_invest" /
               "coastal_protection" / "coastal_protection_functions.py")
spec = importlib.util.spec_from_file_location("coastal_usd_functions", MODULE_PATH)
functions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(functions)


class USDConversionTests(unittest.TestCase):
    def test_us_inflation_factor_uses_us_not_country_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "deflator.csv"
            pd.DataFrame({
                "Country Code": ["USA", "CAN"],
                "2011": [100.0, 100.0],
                "2019": [120.0, 140.0],
            }).to_csv(path, index=False)
            self.assertAlmostEqual(
                functions.get_us_inflation_factor(str(path), 2011, 2019), 1.2
            )

    def test_zeros_and_missing_are_preserved(self):
        values = pd.Series([0.0, 100.0, float("nan")])
        result = functions.convert_usd_to_target_year(values, 1.2)
        self.assertEqual(result.iloc[0], 0.0)
        self.assertAlmostEqual(result.iloc[1], 120.0)
        self.assertTrue(pd.isna(result.iloc[2]))

    def test_invalid_inflation_factor_is_rejected(self):
        for factor in (0, -1, math.inf, math.nan):
            with self.subTest(factor=factor):
                with self.assertRaises(ValueError):
                    functions.convert_usd_to_target_year(pd.Series([100]), factor)

    def test_no_valid_us_deflator_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "deflator.csv"
            pd.DataFrame({"Country Code": ["CAN"], "2011": [100],
                          "2019": [130]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                functions.get_us_inflation_factor(str(path), 2011, 2019)


if __name__ == "__main__":
    unittest.main()
