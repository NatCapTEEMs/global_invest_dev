"""Numerical regression of the production counting expression without GIS imports."""
import ast
import math
from pathlib import Path
from typing import Tuple
import unittest
import numpy as np
from scipy.ndimage import convolve

SOURCE = Path(__file__).with_name('pollination_tasks.py')
TREE = ast.parse(SOURCE.read_text())
NS = dict(np=np, math=math, Tuple=Tuple, convolve=convolve)
HELPERS = {'_compute_radii_pixels', '_make_elliptical_kernel'}
nodes = [n for n in TREE.body if
         isinstance(n, ast.FunctionDef) and n.name in HELPERS or
         isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
         t.id in {'_COS_LAT_FLOOR', '_RADIUS_METERS', '_METERS_PER_DEG_LAT', '_MAX_RY', '_MAX_RX'} for t in n.targets)]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), NS)
worker = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == '_process_tile')
count = next(n for n in ast.walk(worker) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'counts' for t in n.targets))
EXPR = compile(ast.Expression(count.value), str(SOURCE), 'eval')

def production(mask, kernel):
    return eval(EXPR, dict(NS, nat_mask=mask, kernel=kernel))

class HabitatCounts(unittest.TestCase):
    def test_center_matches_exact_neighbor_sum(self):
        for latitude in (0, 30, 50, 55, 60, 70):
            with self.subTest(latitude=latitude):
                ry, rx = NS['_compute_radii_pixels'](latitude, 1/360, 1/360)
                kernel = NS['_make_elliptical_kernel'](ry, rx)
                mask = kernel.copy()
                mask[ry, rx] = 0  # focal pixel is cropland
                self.assertEqual(int(production(mask, kernel)[ry, rx]), int(mask.sum()))

    def test_habitat_removal_cannot_increase_sufficiency(self):
        kernel = NS['_make_elliptical_kernel'](7, 12)
        mask = kernel.copy()
        mask[7, 12] = 0
        previous = 1.0
        for row, col in np.argwhere(mask):
            mask[row, col] = 0
            count = int(production(mask, kernel)[7, 12])
            sufficiency = min(count / int(kernel.sum()) / .3, 1.)
            self.assertLessEqual(sufficiency, previous)
            previous = sufficiency
        self.assertEqual(previous, 0.)

if __name__ == '__main__':
    unittest.main()
