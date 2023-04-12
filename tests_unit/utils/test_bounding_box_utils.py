import unittest
from kappadata.utils.bounding_box_utils import intersection_area_ijkl, intersection_area_ijhw

class TestBoundingBoxUtils(unittest.TestCase):
    def _test_intersection_area_ijkl(self, expected, i0, j0, k0, l0, i1, j1, k1, l1):
        area_ijkl = intersection_area_ijkl(i0=i0, j0=j0, k0=k0, l0=l0, i1=i1, j1=j1, k1=k1, l1=l1)
        h0 = k0 - i0
        w0 = l0 - j0
        h1 = k1 - i1
        w1 = l1 - j1
        area_ijhw = intersection_area_ijhw(i0=i0, j0=j0, h0=h0, w0=w0, i1=i1, j1=j1, h1=h1, w1=w1)
        self.assertEqual(expected, area_ijkl)
        self.assertEqual(area_ijhw, area_ijkl)
        # swap order of bounding boxes
        area_ijkl = intersection_area_ijkl(i0=i1, j0=j1, k0=k1, l0=l1, i1=i0, j1=j0, k1=k0, l1=l0)
        area_ijhw = intersection_area_ijhw(i0=i1, j0=j1, h0=h1, w0=w1, i1=i0, j1=j0, h1=h0, w1=w0)
        self.assertEqual(expected, area_ijkl)
        self.assertEqual(area_ijhw, area_ijkl)

    def test_IntersectionArea_PartialOverlap(self):
        self._test_intersection_area_ijkl(
            expected=6,
            i0=1, j0=2, k0=6, l0=7,
            i1=3, j1=5, k1=7, l1=11,
        )

    def test_IntersectionArea_NoOverlap(self):
        self._test_intersection_area_ijkl(
            expected=0,
            i0=1, j0=2, k0=6, l0=7,
            i1=10, j1=5, k1=14, l1=11,
        )

    def test_IntersectionArea_FullOverlap0(self):
        self._test_intersection_area_ijkl(
            expected=25,
            i0=1, j0=2, k0=6, l0=7,
            i1=1, j1=2, k1=6, l1=7,
        )

    def test_IntersectionArea_FullOverlap1(self):
        self._test_intersection_area_ijkl(
            expected=25,
            i0=1, j0=2, k0=6, l0=7,
            i1=1, j1=1, k1=9, l1=7,
        )
