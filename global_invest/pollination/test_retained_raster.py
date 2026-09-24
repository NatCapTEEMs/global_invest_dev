import unittest
import numpy as np
from retained_raster import _overlap


class OverlapTests(unittest.TestCase):
    def test_coincident_edges_do_not_create_neighbour_slivers(self):
        weights=_overlap(2,0,1000.+1e-12,1.,0.,1.,1003).toarray()
        self.assertEqual(np.count_nonzero(weights),2)
        np.testing.assert_array_equal(weights[1000:1002],np.eye(2))

    def test_real_small_overlap_is_preserved(self):
        weights=_overlap(1,0,1000.+1e-5,1.,0.,1.,1002).toarray()
        self.assertEqual(np.count_nonzero(weights),2)
        self.assertAlmostEqual(weights[1001,0],1e-5,places=10)
        self.assertAlmostEqual(weights.sum(),1.)


if __name__=='__main__': unittest.main()
