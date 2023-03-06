# -*- coding: utf-8 -*-

import numpy as np
from .algorithm import Algorithm


class RiemmanAdaptive(Algorithm):

    def __init__(self, reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu,
                 verbose=False):
        assert reg >= 0
        step_size_0 = None
        super().__init__(reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu, verbose)
        self.addon_vars = None

    @staticmethod
    def proj(X, B):
        bx = B.T.dot(X)
        sym = (bx + bx.T) / 2
        B_proj = B - X.dot(sym)
        return B_proj

    @staticmethod
    def polar_retraction(tan_vec):  # tan_vec, p-by-n, p <= n
        n, p = tan_vec.shape
        U, S, V = np.linalg.svd(tan_vec.T)
        # print(U.shape, tan_vec.shape, V.shape)
        V_trun = V[:, :p]
        return V_trun.dot(U)

    @staticmethod
    def qr_retraction(tan_vec):  # tan_vec, p-by-n, p <= n
        """refer to DPCP_RieSub qr function"""
        q, r = np.linalg.qr(tan_vec)
        d = np.diag(r, 0)
        ph = np.sign(d)  # d.sign()
        ph_2 = np.expand_dims(ph, axis=0)
        ph_3 = np.repeat(ph_2, q.shape[0], axis=0)
        q *= ph_3
        return q

    def initialize(self, a, b, X, Y, Omega, k):
        """Initialize Omega with the projection onto the subspace spanned by top-k eigenvectors of V_pi*, where pi* (=OT_plan) is the (classical) optimal transport plan."""
        d = X.shape[1]
        U_0 = np.eye(k)
        U = np.zeros((d, k))
        U[:k, :] = U_0

        self.U = U
        self.Omega = U.dot(U.T)



    # Riemann Grad with Regularized PRW, if reg>0
    def run_riemanngrad(self, a, b, X, Y, k, lr=0.1, beta=None):

        maxmin_values = []
        U_t = self.U
        Omega = self.Omega

        for t in range(self.max_iter):

            # Optimal transport computation (Sinkhorn)
            C = self.Mahalanobis(X, Y, Omega)
            OT_val, OT_plan = self.OT(a, b, C)
            maxmin_values.append(OT_val)
            self.pi = OT_plan

            # Second-order moment of the displacements
            V = self.Vpi(X, Y, a, b, OT_plan)  # d x d

            # G_t
            epsilon = self.proj(U_t, 2 * V.dot(U_t))  # dxk
            gepsilon = U_t + lr * epsilon
            U_t = self.qr_retraction(gepsilon)

            # stopping criteria
            gap = np.linalg.norm(self.U-U_t) / np.linalg.norm(self.U)
            if gap < self.threshold:
                # break
                pass
            self.U = U_t
        
            # update Omega
            Omega = U_t.dot(U_t.T)
            # assert np.linalg.norm(np.transpose(U_t)@U_t-np.eye(k))<1e-5

        # print(t)
        # print('Method = riemanngrad   Num iter = '+str(t)+'  OT val = '+str(OT_val))
        return Omega, maxmin_values, t

    # Riemann Adaptive
    def run_riemannadap(self, a, b, X, Y, k, lr=0.6, beta=0.6):
        # must be regOT
        assert self.reg > 0

        n = X.shape[0]
        d = X.shape[1]
        m = Y.shape[0]

        c_0 = np.zeros((k,))
        cc_0 = np.zeros((k,)) + 1e-6
        r_0 = np.zeros((d,))
        rr_0 = np.zeros((d,)) + 1e-6
        self.addon_vars = [c_0, r_0, cc_0, rr_0]

        maxmin_values = []
        U_t = self.U
        Omega = self.Omega


        for t in range(self.max_iter):

            # Optimal transport computation (Sinkhorn)
            C = self.Mahalanobis(X, Y, Omega)
            OT_val, OT_plan = self.OT(a, b, C)

            maxmin_values.append(OT_val)
            self.pi = OT_plan

            # Second-order moment of the displacements
            V = self.Vpi(X, Y, a, b, OT_plan)  # d x d

            # G_t
            g_proj = self.proj(U_t, 2 * V.dot(U_t))  # dxk
            c_t, r_t, cc_t, rr_t = self.addon_vars

            ggt = g_proj.dot(g_proj.T)
            gtg = g_proj.T.dot(g_proj)
            r_t1 = beta * r_t + (1 - beta) * np.diag(ggt) / k
            rr_t1 = np.maximum(rr_t, r_t1)
            c_t1 = beta * c_t + (1 - beta) * np.diag(gtg) / d
            cc_t1 = np.maximum(cc_t, c_t1)

            self.addon_vars = [c_t1, r_t1, cc_t1, rr_t1]

            # Algorithm 1, Line 8
            rr_t4 = np.diag(np.power(rr_t1, -0.25))
            cc_t4 = np.diag(np.power(cc_t1, -0.25))
            DGD = rr_t4.dot(g_proj.dot(cc_t4))

            step_size = lr / np.sqrt(t + 1)

            DGD_proj = U_t + step_size * self.proj(U_t, DGD)
            U_t = self.qr_retraction(DGD_proj)
            # U_t = self.polar_retraction(DGD_proj)

            # stopping criteria
            gap = np.linalg.norm(self.U-U_t) / np.linalg.norm(self.U)
            if gap < self.threshold:
                # break
                pass
            self.U = U_t

            # update Omega
            Omega = U_t.dot(U_t.T)
            assert np.linalg.norm(np.transpose(U_t)@U_t-np.eye(k))<1e-5
        # print('Num iter = '+str(t)+'  OT val = '+str(OT_val))
        return Omega, maxmin_values, t