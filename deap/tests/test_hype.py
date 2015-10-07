import unittest
import numpy as np
from StringIO import StringIO
import time

from deap.tools._hypervolume.hype import *
from deap.tools.emo import sortLogNondominated

# {{{ test data
def _mkRandDTLZ2Front(N,M,seed=None):
    '''Return N random points on M-dimensional unit circle
    '''
    if not seed is None: np.random.seed(seed)

    d = np.random.rand(N,M)
    norm = np.sqrt(np.sum(d*d,axis=1))
    return d/np.tile(np.reshape(norm,(N,1)), M)

_pfDTLZ2_2 = _mkRandDTLZ2Front(20,2,seed=1991)
# exact hypE values calculated by the matlab code from
#    http://www.tik.ee.ethz.ch/sop/download/supplementary/hype/files/hypeIndicatorSampled.m
_hDTLZ2_2_1 = np.array([1.3214e-03, 2.6738e-03, 1.0704e-03, 7.5823e-04, 1.6548e-04,
                        4.7371e-04, 5.0512e-04, 6.7670e-06, 1.7871e-04, 6.5548e-04,
                        2.2569e-04, 1.8362e-04, 7.5619e-04, 1.8221e-04, 3.1833e-03,
                        1.2722e-03, 3.6897e-04, 1.5016e-02, 2.9470e-04, 2.4649e-04])    # k=1
_hDTLZ2_2_10= np.array([1.9861e-03, 6.4552e-03, 1.6355e-03, 2.0903e-03, 4.7848e-04,
                        6.7974e-04, 8.6291e-04, 8.0897e-06, 9.9438e-04, 7.2217e-04,
                        5.1125e-04, 5.7328e-04, 4.4757e-03, 4.7685e-04, 3.8436e-03,
                        1.9042e-03, 8.1494e-04, 1.6278e-02, 7.0114e-04, 7.8694e-04])    # k=10

_pfDTLZ2_3 = _mkRandDTLZ2Front(20,3,seed=1991)
_hDTLZ2_3_1 = np.array([7.1870e-03, 1.5411e-02, 5.6988e-03, 1.0970e-03, 2.1144e-03,
                        1.2064e-04, 2.4963e-03, 6.1732e-04, 3.2817e-04, 1.5216e-03,
                        2.8451e-03, 9.6825e-04, 2.7465e-03, 7.3540e-04, 8.6717e-03,
                        1.7250e-03, 3.3287e-03, 1.0862e-03, 6.8070e-04, 1.1422e-03])    # k=1
_hDTLZ2_3_10= np.array([0.0086941, 0.0164106, 0.0076240, 0.0032628, 0.0023249,
                        0.0010343, 0.0049303, 0.0024448, 0.0012616, 0.0023313,
                        0.0059034, 0.0017749, 0.0044725, 0.0013967, 0.0113239,
                        0.0025447, 0.0043495, 0.0024424, 0.0030368, 0.0020010])         # k=10

_pfDTLZ2_4 = _mkRandDTLZ2Front(20,4,seed=1991)
_hDTLZ2_4_1 = np.array([1.5449e-02, 8.8251e-03, 1.2184e-04, 1.7749e-02, 3.3640e-04,
                        3.2368e-04, 5.2649e-03, 1.1672e-02, 1.2604e-03, 5.4516e-04,
                        1.1219e-03, 9.9191e-04, 7.1390e-03, 1.6260e-04, 1.0333e-03,
                        3.7659e-03, 2.7365e-03, 1.3309e-02, 5.5573e-03, 1.0525e-02])    # k=1
_hDTLZ2_4_10= np.array([1.8962e-02, 1.0251e-02, 7.3226e-04, 2.0666e-02, 1.6535e-03,
                        1.2955e-03, 7.6983e-03, 1.3198e-02, 2.0914e-03, 1.1155e-03,
                        1.9595e-03, 2.0044e-03, 1.0203e-02, 1.3917e-03, 1.8054e-03,
                        5.0980e-03, 4.4994e-03, 1.6719e-02, 6.7450e-03, 1.3790e-02])    # k=10
# }}}

class TestHypE(unittest.TestCase):
    def testHypeSimple(self):
        front = np.array([[1., 9.], [5.,5.], [9.,1.]])
        #ref   = np.array([10.,10])
        ref = None

        h0 = hypeIndicatorNaive(front, ref)
        h1 = hypeIndicatorExact(front, 1, ref)
        h2 = hypeIndicatorSampled(front, 1, ref)

        self.assertTrue( np.allclose(h0,h1,rtol=1e-3,atol=1e-5))
        self.assertTrue( np.allclose(h1,h2,rtol=0.1,atol=1e-4))

        h1 = hypeIndicatorExact(front, 2, ref)
        h2 = hypeIndicatorSampled(front, 2, ref)
        self.assertTrue( np.allclose(h1,h2,rtol=0.1,atol=1e-4))

    def testDTLZ2FrontKnown(self):
        for front,k,h0 in [(_pfDTLZ2_2, 1,  _hDTLZ2_2_1),
                          (_pfDTLZ2_2, 10, _hDTLZ2_2_10),
                          (_pfDTLZ2_3,  1, _hDTLZ2_3_1),
                          (_pfDTLZ2_3, 10, _hDTLZ2_3_10),
                          (_pfDTLZ2_4,  1, _hDTLZ2_4_1),
                          (_pfDTLZ2_4, 10, _hDTLZ2_4_10),
                         ]:
            N,M = np.shape(front)
            ref   = np.ones((M,))
            h1 = hypeIndicatorExact(front, k, ref)
            h2 = hypeIndicatorSampled(front, k, ref, 100000)

            self.assertTrue(np.allclose(h0,h1,rtol=1e-3,atol=1e-5))
            self.assertTrue(np.allclose(h0,h2,rtol=1e-3,atol=1e-3))

    def testDTLZ2FrontRandom(self):
        for M,N in [(2,20), (2, 100), (3,20), (3, 100), (4, 20), (5, 20)]:
            for k in [1,2,10]:
                front = _mkRandDTLZ2Front(N,M)
                ref   = np.ones((M,))

                h1 = hypeIndicatorExact(front, k, ref)
                h2 = hypeIndicatorSampled(front, k, ref, 100000)

                self.assertTrue(np.allclose(h1,h2,rtol=1e-3,atol=1e-3))

    def testSampledSpeed(self):

        for M,N in [(2,20), (2, 100), (3,100), (5, 100), (10, 100), (20, 100), (30, 100), (50, 100)]:
            for k in [1,2,10]:
                front = _mkRandDTLZ2Front(N,M)
                ref   = np.ones((M,))
                t0 = time.time()
                h2 = hypeIndicatorSampled(front, k, ref, 100000)
                #print 'M , N , k = ', M, N, k, time.time() - t0

if __name__=='__main__':
    unittest.main()

