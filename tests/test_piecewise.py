# std
import unittest

# 3p
import numpy as np

# prj
from piecewise import piecewise
from piecewise.plotter import piecewise_plot


class TestPiecewise(unittest.TestCase):

    def test_wrs_data(self):

        t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        v = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            11,
            11,
            11,
            11,
            11
        ]

        '''
        t = np.arange(10)
        v = np.array(
            [2*i for i in range(5)] +
            [10-i for i in range(5, 10)]
        ) + np.random.normal(0, 1, 10)
        '''
        # piecewise_plot(t, v, .001)
        model = piecewise(t, v)
        np.testing.assert_equal(len(model.segments), 2)
        seg = model.segments[0]
        np.testing.assert_equal(seg.start_t, 1)
        np.testing.assert_equal(seg.end_t, 12)
        np.testing.assert_equal(model.segments[1].start_t, 12)
        np.testing.assert_equal(model.segments[1].end_t, 16)
        # np.testing.assert_almost_equal(seg.coeffs[0], intercept, decimal=0)
        # np.testing.assert_almost_equal(seg.coeffs[1], slope, decimal=0)


    def test_wrs_data(self):

        t = [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        v = [
            4672.92, # 55
            5037.72, # 56
            5402.4, # 57
            5675.88, # 58
            5957.04, # 59
            6230.64, # 60
            6504.12, # 61
            6777.72, # 62
            7051.20, # 63
            7324.80, # 64
            7598.28, # 65 first point of second segment, last point of first segment
            7598.28, # 66
            7598.28, # 67
            7598.28, # 68
            7598.28, # 69
            7598.28 # 70
        ]

        '''
        t = np.arange(10)
        v = np.array(
            [2*i for i in range(5)] +
            [10-i for i in range(5, 10)]
        ) + np.random.normal(0, 1, 10)
        '''
        # piecewise_plot(t, v, .001)
        model = piecewise(t, v, .01)
        np.testing.assert_equal(len(model.segments), 2)
        seg = model.segments[0]
        np.testing.assert_equal(seg.start_t, 55)
        np.testing.assert_equal(seg.end_t, 66)
        np.testing.assert_equal(model.segments[1].start_t, 66)
        np.testing.assert_equal(model.segments[1].end_t, 70)
        # np.testing.assert_almost_equal(seg.coeffs[0], intercept, decimal=0)
        # np.testing.assert_almost_equal(seg.coeffs[1], slope, decimal=0)


    def test_single_line(self):
        """ When the data follows a single linear path with Gaussian noise, then
        only one segment should be found.
        """
        # Generate some data.
        np.random.seed(1)
        intercept = -45.0
        slope = 0.7
        t = np.arange(2000)
        v = intercept + slope*t + np.random.normal(0, 1, 2000)
        # Fit the piecewise regression.
        model = piecewise(t, v)
        # A single segment should be found, encompassing the whole domain with
        # coefficients approximately equal to those used to generate the data.
        np.testing.assert_equal(len(model.segments), 1)
        seg = model.segments[0]
        np.testing.assert_equal(seg.start_t, 0)
        np.testing.assert_equal(seg.end_t, 1999)
        np.testing.assert_almost_equal(seg.coeffs[0], intercept, decimal=0)
        np.testing.assert_almost_equal(seg.coeffs[1], slope, decimal=0)

    def test_single_line_with_nans(self):
        """ Some nans in the data shouldn't break the regression, and leading and
        trailing nans should lead to exclusion of the corresponding t values from
        the segment domain.
        """
        # Generate some data, and introduce nans.
        np.random.seed(1)
        intercept = -45.0
        slope = 0.7
        t = np.arange(2000)
        v = intercept + slope*t + np.random.normal(0, 1, 2000)
        v[[0, 24, 400, 401, 402, 1000, 1999]] = np.nan
        # Fit the piecewise regression.
        model = piecewise(t, v)
        # A single segment should be found, encompassing the whole domain (excluding
        # the leading and trailing nans) with coefficients approximately equal to
        # those used to generate the data.
        np.testing.assert_equal(len(model.segments), 1)
        seg = model.segments[0]
        np.testing.assert_equal(seg.start_t, 1)
        np.testing.assert_equal(seg.end_t, 1998)
        np.testing.assert_almost_equal(seg.coeffs[0], intercept, decimal=0)
        np.testing.assert_almost_equal(seg.coeffs[1], slope, decimal=0)

    def test_five_segments(self):
        """ If there are multiple distinct segments, piecewise() should be able to
        find the proper breakpoints between them.
        """
        # Generate some data.
        t = np.arange(1900, 2000)
        v = t % 20
        # Fit the piecewise regression.
        model = piecewise(t, v)
        # There should be five segments, each with a slope of 1.
        np.testing.assert_equal(len(model.segments), 5)
        for segment in model.segments:
            np.testing.assert_almost_equal(segment.coeffs[1], 1.0)
        # The segments should be in time order and each should cover 20 units of the
        # domain.
        np.testing.assert_equal(model.segments[0].start_t, 1900)
        np.testing.assert_equal(model.segments[1].start_t, 1920)
        np.testing.assert_equal(model.segments[2].start_t, 1940)
        np.testing.assert_equal(model.segments[3].start_t, 1960)
        np.testing.assert_equal(model.segments[4].start_t, 1980)

    def test_messy_ts(self):
        """ Unevenly-spaced, out-of-order, float t-values should work.
        """
        # Generate some step-function data.
        t = [1.0, 0.2, 0.5, 0.4, 2.3, 1.1]
        v = [5, 0, 0, 0, 5, 5]
        # Fit the piecewise regression.
        model = piecewise(t, v)
        # There should be two constant-valued segments.
        np.testing.assert_equal(len(model.segments), 2)
        seg1, seg2 = model.segments

        np.testing.assert_equal(seg1.start_t, 0.2)
        np.testing.assert_equal(seg1.end_t, 1.0)
        np.testing.assert_almost_equal(seg1.coeffs[0], 0)
        np.testing.assert_almost_equal(seg1.coeffs[1], 0)

        np.testing.assert_equal(seg2.start_t, 1.0)
        np.testing.assert_equal(seg2.end_t, 2.3)
        np.testing.assert_almost_equal(seg2.coeffs[0], 5)
        np.testing.assert_almost_equal(seg2.coeffs[1], 0)

    def test_non_unique_ts(self):
        """ A dataset with multiple values with the same t should not break the
        code, and all points with the same t should be assigned to the same
        segment.
        """
        # Generate some data.
        t1 = [t for t in range(100)]
        v1 = [v for v in np.random.normal(3, 1, 100)]
        t2 = [t for t in range(99, 199)]
        v2 = [v for v in np.random.normal(20, 1, 100)]
        t = t1 + t2
        v = v1 + v2
        # Fit the piecewise regression.
        model = piecewise(t, v)
        # There should be two segments, and the split shouldn't be in the middle
        # of t=99.
        np.testing.assert_equal(len(model.segments), 2)
        seg1, seg2 = model.segments
        assert seg1.end_t == seg2.start_t
