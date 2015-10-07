import unittest
import numpy as np

from deap.tools._hypervolume.hype import *

class TestHypE(unittest.TestCase):
    def testSimple(self):
        front = np.array([[1., 9.], [5.,5.], [9.,1.]])
        ref   = np.array([10.,10])

        h0 = hypeIndicatorNaive(front, ref)
        h1 = hypeIndicatorExact(front, 1, ref)
        h2 = hypeIndicatorSampled(front, 1, ref)

        self.assertFalse( np.any((h0-h1)/h0>1e-3) )
        self.assertFalse( np.any((h2-h1)/h1>0.1) )

        h1 = hypeIndicatorExact(front, 2, ref)
        h2 = hypeIndicatorSampled(front, 2, ref)
        self.assertFalse( np.any((h2-h1)/h1>0.1) )


if __name__=='__main__':
    unittest.main()

