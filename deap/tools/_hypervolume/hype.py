''' This module contains implementation of the hyper-volume estimation routines,
as described in [Bader2011]_ .

    .. [Bader2011] Bader, Johannes, and Eckart Zitzler. "HypE: An algorithm
    for fast hypervolume-based many-objective optimization." Evolutionary
    computation 19.1 (2011): 45-76.
'''

import numpy as np
from functools import reduce

try:
    # try importing the C version
    import hv
except ImportError:
    # fallback on python version
    import pyhv as hv

def hypeIndicatorNaive(points, ref=None):
    ''' Calculates naively the exact HypE fitness for k=1.

    :param  points:     coordinates of points in a pareto front.
    :type   points:     np.ndarray(N,M)
    :param  ref:        reference point
    :type   ref:        np.ndarray(M,)
    :returns:           hypervolume indicator Ih_1(a, P, R) for each point a in the front P, w.r.t reference point R.
    :rtype:             np.ndarray(N,)
    '''
    nrP, dim = np.shape(points)
    F = np.zeros((nrP,))
    if ref is None:
        ref = np.max(points, 0)
    for i,pt in enumerate(points):
        F[i] = hv.hypervolume(np.concatenate((points[:i], points[i+1:])), ref)
    return hv.hypervolume(points,ref) - F

def hypeIndicatorSampled(points, k, ref=None, nrOfSamples=10000):
    ''' Calculates the HypE fitness using Monte-Carlo integration.

    Adapted from the Matlab implementation available at: 
    http://www.tik.ee.ethz.ch/sop/download/supplementary/hype/files/hypeIndicatorSampled.m

    :param  points:     coordinates of points in a pareto front.
    :type   points:     np.ndarray(N,M)
    :param  k:          number of least important points to be removed from the front
    :type   k:          int
    :param  ref:        reference point
    :type   ref:        np.ndarray(M,)
    :param  nrOfSamples: number of Monte-Carlo samples.
    :type   nrOfSamples: int
    :returns:           hypervolume indicator Ih_k(a, P, R) for each point a in the front P, w.r.t reference point R.
    :rtype:             np.ndarray(N,)
    '''

    nrP, dim = np.shape(points)
    F = np.zeros((nrP,))

    alpha = np.zeros((nrP,))
    alpha[0] = 1.0
    for i in xrange(1,k):
        alpha[i] = alpha[i-1] * (k-i)/(nrP-i)
    alpha = alpha/range(1,nrP+1)

    # lower bound of each objectives
    BoxL = np.min(points, 0)
    # upper bound of each objectives
    if ref is None: BoxU = np.max(points, 0)
    else:           BoxU = ref

    # generate a random sample uniformly distributed between BoxL and ref
    S = np.random.random((nrOfSamples,dim)) * (ref-BoxL) + BoxL

    # number of points in the front that dominate each sample point
    dominated = np.zeros((nrOfSamples,),dtype='int')
    for j in xrange(nrP):
        B = S - points[j,:]
        idx = np.sum(B >=0 ,1) == dim
        dominated[idx] += 1

    # do integration
    for j in xrange(nrP):
        B = S - points[j,:]
        idx = np.sum(B >=0 ,1) == dim
        x = dominated[idx]
        F[j] = np.sum( alpha[x-1] )

    return F* np.prod(BoxU-BoxL)/nrOfSamples

def hypeIndicatorExact(points, k, ref=None):
    ''' Calculates the HypE fitness using exact algorithm

    Adapted from the Matlab implementation available at: 
    http://www.tik.ee.ethz.ch/sop/download/supplementary/hype/files/hypeIndicatorExact.m

    :param  points:     coordinates of points in a pareto front.
    :type   points:     np.ndarray(N,M)
    :param  k:          number of least important points to be removed from the front
    :type   k:          int
    :param  ref:        reference point
    :type   ref:        np.ndarray(M,)
    :param  nrOfSamples: number of Monte-Carlo samples.
    :type   nrOfSamples: int
    :returns:           hypervolume indicator Ih_k(a, P, R) for each point a in the front P, w.r.t reference point R.
    :rtype:             np.ndarray(N,)
    '''

    nrP, dim = np.shape(points)

    alpha = np.zeros((nrP,))
    alpha[0] = 1.0
    for i in xrange(1,k):
        alpha[i] = alpha[i-1] * (k-i)/(nrP-i)
    alpha = alpha/range(1,nrP+1)

    def hypeSub(A, actDim, pvec):
        h = np.zeros((nrP,))
        i = A[:,actDim].argsort()
        S = A[i]
        pvec = pvec[i]

        nrS = len(S)
        for i in xrange(nrS):
            if i<nrS-1:
                extrusion = S[i+1,actDim] - S[i,actDim]
            else:
                extrusion = ref[actDim] - S[i,actDim]
            if actDim==0:
                if i+1 <= k:
                    h[pvec[:i+1]] += extrusion * alpha[i]
            elif extrusion > 0:
                h += extrusion* hypeSub(S[:i+1,:], actDim-1, pvec[:i+1])
        return h

    return hypeSub(points, dim-1, np.array(range(nrP), dtype=int))

__all__=['hypeIndicatorNaive', 'hypeIndicatorSampled', 'hypeIndicatorExact']
