import numpy as np
import math


class SpaceFillingCurve():
    def __init__(self, order):
        if order <= 0:
            raise ValueError('order must be > 0')
        self.order = order

    def distance_from_coordinates(self, x, y):
        pass

    def coordinates_from_distance(self, d):
        pass


class HilbertCurveND(SpaceFillingCurve):
    """
    This is a module to convert between one dimensional distance along a
    `Hilbert curve`_, :math:`h`, and N-dimensional coordinates,
    :math:`(x_0, x_1, ... x_N)`.  The two important parameters are :math:`N`
    (the number of dimensions, must be > 0) and :math:`p` (the number of
    iterations used in constructing the Hilbert curve, must be > 0).

    We consider an N-dimensional `hypercube`_ of side length :math:`2^p`.
    This hypercube contains :math:`2^{N p}` unit hypercubes (:math:`2^p` along
    each dimension).  The number of unit hypercubes determine the possible
    discrete distances along the Hilbert curve (indexed from :math:`0` to
    :math:`2^{N p} - 1`).
    """
    def __init__(self, order, n=2):
        """Initialize a hilbert curve with,

        Args:
            p (int): iterations to use in the hilbert curve
            n (int): number of dimensions
        """
        if order <= 0:
            raise ValueError('order must be > 0')
        if n <= 0:
            raise ValueError('n must be > 0')
        self.p = order
        self.n = n

        # maximum distance along curve
        self.max_h = 2**(self.p * self.n) - 1

        # maximum coordinate value in any dimension
        self.max_x = 2**self.p - 1

        self._binary_repr = lambda n,w: format(n, 'b').zfill(w)

    def _hilbert_integer_to_transpose(self, h):
        """Store a hilbert integer (`h`) as its transpose (`x`).

        Args:
            h (int): integer distance along hilbert curve

        Returns:
            x (list): transpose of h
                      (n components with values between 0 and 2**p-1)
        """
        h_bit_str = self._binary_repr(h, self.p*self.n)
        x = [int(h_bit_str[i::self.n], 2) for i in range(self.n)]
        return x

    def _transpose_to_hilbert_integer(self, x):
        """Restore a hilbert integer (`h`) from its transpose (`x`).

        Args:
            x (list): transpose of h
                      (n components with values between 0 and 2**p-1)

        Returns:
            h (int): integer distance along hilbert curve
        """
        x_bit_str = [self._binary_repr(x[i], self.p) for i in range(self.n)]
        h = int(''.join([y[i] for i in range(self.p) for y in x_bit_str]), 2)
        return h

    def coordinates_from_distance(self, h):
        """Return the coordinates for a given hilbert distance.

        Args:
            h (int): integer distance along hilbert curve

        Returns:
            x (list): transpose of h
                      (n components with values between 0 and 2**p-1)
        """
        if h > self.max_h:
            raise ValueError('h={} is greater than 2**(p*N)-1={}'.format(h, self.max_h))
        if h < 0:
            raise ValueError('h={} but must be > 0'.format(h))

        x = self._hilbert_integer_to_transpose(h)
        Z = 2 << (self.p-1)

        # Gray decode by H ^ (H/2)
        t = x[self.n-1] >> 1
        for i in range(self.n-1, 0, -1):
            x[i] ^= x[i-1]
        x[0] ^= t

        # Undo excess work
        Q = 2
        while Q != Z:
            P = Q - 1
            for i in range(self.n-1, -1, -1):
                if x[i] & Q:
                    # invert
                    x[0] ^= P
                else:
                    # exchange
                    t = (x[0] ^ x[i]) & P
                    x[0] ^= t
                    x[i] ^= t
            Q <<= 1

        # done
        return x

    def distance_from_coordinates(self, x_in):
        """Return the hilbert distance for a given set of coordinates.

        Args:
            x_in (list): transpose of h
                         (n components with values between 0 and 2**p-1)

        Returns:
            h (int): integer distance along hilbert curve
        """
        x = list(x_in)
        if len(x) != self.n:
            raise ValueError('x={} must have N={} dimensions'.format(x, self.n))

        if any(elx > self.max_x for elx in x):
            raise ValueError(
                'invalid coordinate input x={}.  one or more dimensions have a '
                'value greater than 2**p-1={}'.format(x, self.max_x))

        if any(elx < 0 for elx in x):
            raise ValueError(
                'invalid coordinate input x={}.  one or more dimensions have a '
                'value less than 0'.format(x))

        M = 1 << (self.p - 1)

        # Inverse undo excess work
        Q = M
        while Q > 1:
            P = Q - 1
            for i in range(self.n):
                if x[i] & Q:
                    x[0] ^= P
                else:
                    t = (x[0] ^ x[i]) & P
                    x[0] ^= t
                    x[i] ^= t
            Q >>= 1

        # Gray encode
        for i in range(1, self.n):
            x[i] ^= x[i-1]
        t = 0
        Q = M
        while Q > 1:
            if x[self.n-1] & Q:
                t ^= Q - 1
            Q >>= 1
        for i in range(self.n):
            x[i] ^= t

        h = self._transpose_to_hilbert_integer(x)
        return h


class HilbertCurve(SpaceFillingCurve):
    """ To be removed. Use HilbertCurveND instead. """
    def __init__(self, order):
        super(HilbertCurve, self).__init__(order)
        self.n = 2**order
        self.max_d = self.n**2 - 1
        self.max_c = self.n - 1

    def distance_from_coordinates(self, x, y):
        d=0
        s=self.n//2
        while s>0:
            rx = (x & s) > 0
            ry = (y & s) > 0
            d += s * s * ((3 * rx) ^ ry)
            x,y = self._rot(s, x, y, rx, ry)
            s = s//2
        
        return d

    def coordinates_from_distance(self, d):
        t = d
        x, y = 0, 0
        s = 1
        while s < self.n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = self._rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s = s * 2
        return x, y

    def _rot(self, p, x, y, rx, ry):
        if (ry == 0):
            if (rx == 1):
                x = p - 1 - x
                y = p - 1 - y

            # Swap x and y
            t = x
            x = y
            y = t
        return x, y


class PeanoCurve(SpaceFillingCurve):
    """
    Implementation based on http://www.davidsalomon.name/DC2advertis/AppendC.pdf
    """
    def __init__(self, order):
        super(PeanoCurve, self).__init__(order)
        self.n = 3**order
        self.max_d = self.n**2 - 1
        self.max_c = self.n - 1

    def distance_from_coordinates(self, x, y):
        x_str = str(x).zfill(2 * self.order)
        y_str = str(y).zfill(2 * self.order)
        xrgc_str = self._rgc(x_str)
        yrgc_str = self._rgc(y_str)
        drgc_str = "".join(i for j in zip(yrgc_str, xrgc_str) for i in j)
        d_str = self._rgc(drgc_str)
        return int(d_str, 3)

    def coordinates_from_distance(self, d):
        d_str = np.base_repr(d, 3).zfill(2 * self.order)
        drgc_str = self._rgc(d_str)
        xrgc_str = drgc_str[1::2]
        yrgc_str = drgc_str[0::2]
        x_str = self._rgc(xrgc_str)
        y_str = self._rgc(yrgc_str)
        x = int(x_str, 3)
        y = int(y_str, 3)
        return x, y

    def _rgc(self, a):
        b = a[0]
        for i in range(2, len(a) + 1):
            p_i = np.sum([int(j) for j in a[:i - 1]]) % 2
            if p_i == 0:
                b = b + a[i - 1]
            else:
                b = b + str(2 - int(a[i - 1]))
        return b


class ZOrderCurve(SpaceFillingCurve):
    """
    Implementation based on https://en.wikipedia.org/wiki/Z-order_curve
    """
    def __init__(self, order):
        super(ZOrderCurve, self).__init__(order)
        self.n = 2**order
        self.max_d = self.n**2 - 1
        self.max_c = self.n - 1

    def distance_from_coordinates(self, x, y):
        x_str = str(x).zfill(2 * self.order)
        y_str = str(y).zfill(2 * self.order)
        d_str = "".join(i for j in zip(y_str, x_str) for i in j)
        return int(d_str, 2)

    def coordinates_from_distance(self, d):
        d_str = np.base_repr(d, 2).zfill(2 * self.order)
        x_str = d_str[1::2]
        y_str = d_str[0::2]
        x = int(x_str, 2)
        y = int(y_str, 2)
        return x, y


def signal_to_image(x, curve):
    """
    Arguments:
        x -- array, of size [L,M]. SpaceFillingCurve applied along the axis=0
        transform -- transformation class
    Returns:
        y -- array, of size [N,N,M] where N=np.sqrt(L)
    """
    assert curve in ['HilbertCurve', 'PeanoCurve', 'ZOrderCurve']
    L, M = x.shape

    if curve == 'PeanoCurve':
        N = int(np.power(3,np.ceil(math.log(np.sqrt(L), 3))))
        nord = int(math.log(N, 3))
    else:
        N = int(np.power(2,np.ceil(np.log2(np.sqrt(L)))))
        nord = int(math.log(N, 2))
    
    t = eval('{}({})'.format(curve, nord))
    coords = [t.coordinates_from_distance(int(i)) for i in range(L)]
    y = np.zeros((N, N, M))
    for j in range(M):
        # img array coordinates are (y,x)
        for k in range(L):
            y[coords[k][1], coords[k][0], j] = x[k, j]
    return y


def image_to_signal(y, curve):
    """
    Arguments:
        y -- array, of size [N,N,M]. SpaceFillingCurve applied along the axis=2
        transform -- transformation class
    Returns:
        x -- array, of size [L,M] where L=N*N
    """
    assert curve in ['HilbertCurve', 'PeanoCurve', 'ZOrderCurve']
    N1, N2, M = y.shape
    L = N1*N2
    if curve == 'PeanoCurve':
        N = int(np.power(3,np.ceil(math.log(np.sqrt(L), 3))))
        nord = int(math.log(N, 3))
    else:
        N = int(np.power(2,np.ceil(np.log2(np.sqrt(L)))))
        nord = int(math.log(N, 2))
    x = np.zeros((L, M))
    t = eval('{}({})'.format(curve, nord))
    coords = [t.coordinates_from_distance(int(i)) for i in range(L)]
    for j in range(M):
        # img array coordinates are (y,x)
        for k in range(L):
            x[k, j] = y[coords[k][1], coords[k][0], j]
    return x
