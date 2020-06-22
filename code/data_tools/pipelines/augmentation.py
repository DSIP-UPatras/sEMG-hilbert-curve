import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TrfJitter(BaseEstimator, TransformerMixin):
    def __init__(self, snrdb, p=1, verbose=0):
        self.snrdb = snrdb
        self.snr = 10 ** (self.snrdb/10)
        self.p = p
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        output = np.copy(X)
        q = np.random.rand(X.shape[0]) < self.p
        if self.verbose:
            print('Jitter: ', q)
        for i in range(X.shape[0]):
            if q[i]:
                output[i] = self.jitter(X[i])
        if y is not None:
            return output, y
        else:
            return output
    
    def jitter(self, x):
        Xp = np.sum(x**2, axis=0, keepdims=True) / x.shape[0]
        Np = Xp / self.snr
        n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)
        return x + n

class TrfMagWarp(BaseEstimator, TransformerMixin):
    def __init__(self, sigma, p=1, verbose=0):
        self.sigma = sigma
        self.p = p
        self.verbose = verbose
        self.knot = 4

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        output = np.copy(X)
        q = np.random.rand(X.shape[0]) < self.p
        if self.verbose:
            print('Warp: ', q)
        for i in range(X.shape[0]):
            if q[i]:
                output[i] = self.mag_warp(X[i])
        if y is not None:
            return output, y
        else:
            return output
    
    def mag_warp(self, x):
        def _generate_random_curve(x, sigma=0.2, knot=4):
            # knot = max(0, min(knot, x.shape[0]-2))
            xx = np.arange(0, x.shape[0], (x.shape[0]-1)//(knot+1)).transpose()
            yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2,))
            x_range = np.arange(x.shape[0])
            cs = CubicSpline(xx[:], yy[:])
            return np.array(cs(x_range)).transpose()

        output = np.zeros(x.shape)
        for i in range(x.shape[1]):
            rc = _generate_random_curve(x[:,i], self.sigma, self.knot)
            output[:,i] =  x[:,i] * rc
        return output
