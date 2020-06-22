import numpy as np
import math
from sklearn.base import BaseEstimator, TransformerMixin
from data_tools import spfc_utils

class TrfSpaceFillCurve(BaseEstimator, TransformerMixin):
    def __init__(self, curve, axis):
        self.curve = curve
        self.axis = axis

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        output = []
        self.init_curve(X[0].shape)
        for i in range(len(X)):
            img = self.apply_curve(X[i])
            output.append(img)
        output = spfc_utils.crop_zeros(np.array(output))
        return np.array(output)

    def init_curve(self, input_shape):
        L = input_shape[self.axis]
        M = np.prod(input_shape) // L
        if self.curve == 'PeanoCurve':
            N = int(np.power(3,np.ceil(math.log(np.sqrt(L), 3))))
            nord = int(math.log(N, 3))
        else:
            N = int(np.power(2,np.ceil(np.log2(np.sqrt(L)))))
            nord = int(math.log(N, 2))
        
        t = eval('spfc_utils.{}({})'.format(self.curve, nord))
        coords = np.array([t.coordinates_from_distance(int(i)) for i in range(L)])
        self.coords_i = coords[:,1]
        self.coords_j = coords[:,0]
        self.output_shape = (N, N, M)

    def apply_curve(self, x):
        x = np.swapaxes(x, 0, self.axis)
        y = np.zeros(self.output_shape)
        y[self.coords_i, self.coords_j, :] = x
        return y


