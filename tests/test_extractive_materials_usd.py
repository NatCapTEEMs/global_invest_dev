"""Offline tests for extractive-material GEP in constant 2019 US dollars."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import pandas as pd

# The currency helpers do not use geospatial dependencies; stub hazelbean for
# lightweight offline tests in environments without the full project stack.
sys.modules.setdefault("hazelbean", types.ModuleType("hazelbean"))
MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "global_invest"
    / "extractive_materials_provision" / "extractive_materials_provision_functions.py"
)
spec = importlib.util.spec_from_file_location("extractive_usd_functions", MODULE_PATH)
functions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(functions)


class ExtractiveMaterialsUSDTests(unittest.TestCase):
    def test_2019_factor_is_one_and_older_year_uses_us_only(self):
        deflators = pd.DataFrame({
            "Country Code": ["USA", "USA", "CAN", "CAN"],
            "year": [2011, 2019, 2011, 2019],
            "GDP_deflator": [100.0, 120.0, 100.0, 160.0],
        })
        factors = functions.get_us_deflator_factors(deflators, 2019)
        result = dict(zip(factors["year"], factors["us_deflator_factor"]))
        self.assertAlmostEqual(result[2011], 1.2)
        self.assertAlmostEqual(result[2019], 1.0)

    def test_formula_uses_current_usd_gdp_and_preserves_missing(self):
        data = pd.DataFrame({
            "mineral_rent_percent": [10.0, 10.0, 0.0, float("nan")],
            "GDP_current_USD": [1000.0, 1000.0, 1000.0, 1000.0],
            "us_deflator_factor": [1.0, 1.2, 1.0, 1.0],
        })
        result = functions.calculate_mineral_gep_2019_usd(data)
        self.assertAlmostEqual(result.iloc[0], 49.0)
        self.assertAlmostEqual(result.iloc[1], 58.8)
        self.assertEqual(result.iloc[2], 0.0)
        self.assertTrue(pd.isna(result.iloc[3]))

    def test_missing_us_year_deflator_keeps_nan(self):
        factors = functions.get_us_deflator_factors(pd.DataFrame({
            "Country Code": ["USA", "USA"],
            "year": [2018, 2019],
            "GDP_deflator": [float("nan"), 120],
        }), 2019)
        self.assertTrue(pd.isna(factors.loc[factors["year"].eq(2018),
                                          "us_deflator_factor"].iloc[0]))

    def test_missing_us_base_year_is_rejected(self):
        with self.assertRaises(ValueError):
            functions.get_us_deflator_factors(pd.DataFrame({
                "Country Code": ["USA"], "year": [2018], "GDP_deflator": [100],
            }), 2019)

    def test_gdp_loader_labels_direct_usd(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gdp.csv"
            pd.DataFrame({"Country Code": ["USA"], "2019": [100.0]}).to_csv(
                path, index=False
            )
            values = functions.read_GDP_values(str(path))
            self.assertEqual(values.loc[0, "GDP_current_USD"], 100.0)
            self.assertNotIn("GDP_current_LCU", values.columns)


if __name__ == "__main__":
    unittest.main()
