# -*- coding: utf-8 -*-#

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class ProjectionRobustWasserstein:

    def __init__(self, X, Y, a, b, algo, k):
        """
        X    : (number_points_1, dimension) matrix of atoms for the first measure
        Y    : (number_points_2, dimension) matrix of atoms for the second measure
        a    : (number_points_1,) vector of weights for the first measure
        b    : (number_points_2,) vector of weights for the second measure
        algo : algorithm to compute the SRW distance (instance of class 'ProjectedGradientAscent' or 'FrankWolfe')
        k    : dimension parameter (can be of type 'int', 'list' or 'set' in order to compute SRW for several paremeters 'k').
        """

        # Check shapes
        d = X.shape[1]
        n = X.shape[0]
        m = Y.shape[0]
        assert d == Y.shape[1]
        assert n == a.shape[0]
        assert m == b.shape[0]

        if isinstance(k, int):
            assert k <= d
            assert k == int(k)
            assert 1 <= k
        elif isinstance(k, list) or isinstance(k, set):
            assert len(k) > 0
            k = list(set(k))
            k.sort(reverse=True)
            assert k[0] <= d
            assert k[-1] >= 1
            for l in k:
                assert l == int(l)
        else:
            raise TypeError("Parameter 'k' should be of type 'int' or 'list' or 'set'.")

        # Measures
        self.X = X
        self.Y = Y
        self.a = a
        self.b = b
        self.d = d

        # Algorithm
        self.algo = algo
        self.k = k
        self.Omega = np.identity(self.d)
        self.pi = None
        self.maxmin_values = []
        self.minmax_values = []
        self.num_iter=0

    def run(self, tp, param_dict):
        """Run algorithm algo on the data."""
        self.algo.initialize(self.a, self.b, self.X, self.Y, None, self.k)
        if tp == 0:
            self.Omega, self.maxmin_values, self.num_iter = self.algo.run_riemanngrad(self.a, self.b, self.X, self.Y, self.k, **param_dict)
        elif tp==1:
            self.Omega, self.maxmin_values, self.num_iter = self.algo.run_riemannadap(self.a, self.b, self.X, self.Y, self.k, **param_dict)
        else:
            self.Omega, self.maxmin_values, self.num_iter = self.algo.run_stiefel(self.a, self.b, self.X, self.Y, self.k, param_dict)
        return self.Omega, self.algo.pi, self.maxmin_values
    

    def get_Omega(self):
        return self.Omega

    def get_pi(self):
        return self.pi
    def get_num_iter(self):
        return self.num_iter

    def get_maxmin_values(self):
        """Get the values of the maxmin problem along the iterations."""
        return self.maxmin_values

    def get_value(self):
        """Return the SRW distance."""
        return np.max(self.maxmin_values)

    def plot_transport_plan(self, path=None):
        pi = self.algo.pi

        for i in range(self.X.shape[0]):
            print(i, self.X.shape[0])
            for j in range(self.Y.shape[0]):
                if pi[i, j] > 0.:
                    plt.plot([self.X[i, 0], self.Y[j, 0]], [self.X[i, 1], self.Y[j, 1]], c='k', lw=30 * pi[i, j])

        plt.scatter(self.X[:, 0], self.X[:, 1], s=self.X.shape[0] * 20 * self.a, c='r', zorder=10, alpha=0.7)
        plt.scatter(self.Y[:, 0], self.Y[:, 1], s=self.Y.shape[0] * 20 * self.b, c='b', zorder=10, alpha=0.7)
        plt.title('Optimal PRW transport plan (n=%d)' % (self.X.shape[0],), fontsize=15)
        plt.axis('equal')
        if path is not None:
            plt.savefig(path)
        # plt.show()
        plt.close()

    def get_projected_pushforwards(self):
        """Return the projection of words to 2D plane."""
        U = self.algo.U
        proj_X = self.X.dot(U)
        proj_Y = self.Y.dot(U)
        return proj_X, proj_Y
