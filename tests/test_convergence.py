"""Do the cyclic projections (ITML, ITML+ with and without the correction) reach
the optimum found by cvxpy on small random problems?"""
import os
import sys

import cvxpy as cp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from itmlplus.metric import _itml_core, _itmlplus_core


def dld(A):
    return np.trace(A) - np.linalg.slogdet(A)[1] - A.shape[0]


def ell(a, b):
    return a / b - np.log(a / b) - 1.0


def make_problem(seed, n=40, h=8, g=8, m=30):
    rng = np.random.RandomState(seed)
    lab = rng.randint(0, 4, n)
    X = 0.8 * rng.randn(4, h)[lab] + 0.7 * rng.randn(n, h)
    Z = 1.0 * rng.randn(4, g)[lab] + 0.4 * rng.randn(n, g)
    pairs = np.array([rng.choice(n, 2, replace=False) for _ in range(m)])
    y = np.where(lab[pairs[:, 0]] == lab[pairs[:, 1]], 1, -1)
    V = X[pairs[:, 0]] - X[pairs[:, 1]]
    W = Z[pairs[:, 0]] - Z[pairs[:, 1]]
    d = (V ** 2).sum(1)
    u, l = np.percentile(d, 10), np.percentile(d, 90)
    xi0 = np.where(y == 1, u, l)
    return X, Z, pairs, y, V, W, xi0


def itmlplus_objective(M, P, V, W, xi0, gamma, lam):
    s = np.einsum('ij,jk,ik->i', W, P, W)
    return dld(M) + lam * dld(P) + gamma * ell(s, xi0).sum()


def solve_cvx_itmlplus(V, W, y, xi0, gamma, lam):
    h, g = V.shape[1], W.shape[1]
    M = cp.Variable((h, h), PSD=True)
    P = cp.Variable((g, g), PSD=True)
    dM = cp.hstack([cp.quad_form(v, M, assume_PSD=True) for v in V])
    dP = cp.hstack([cp.quad_form(w, P, assume_PSD=True) for w in W])
    obj = (cp.trace(M) - cp.log_det(M) + lam * (cp.trace(P) - cp.log_det(P))
           + gamma * cp.sum(cp.multiply(1.0 / xi0, dP) - cp.log(dP)))
    prob = cp.Problem(cp.Minimize(obj), [cp.multiply(y, dM) <= cp.multiply(y, dP)])
    prob.solve(solver=cp.SCS, eps=1e-9, max_iters=200000)
    return M.value, P.value


def solve_cvx_itml(V, y, xi0, gamma):
    h = V.shape[1]
    M = cp.Variable((h, h), PSD=True)
    xi = cp.Variable(len(y))
    dM = cp.hstack([cp.quad_form(v, M, assume_PSD=True) for v in V])
    obj = cp.trace(M) - cp.log_det(M) + gamma * cp.sum(cp.multiply(1.0 / xi0, xi) - cp.log(xi))
    prob = cp.Problem(cp.Minimize(obj), [cp.multiply(y, dM) <= cp.multiply(y, xi)])
    prob.solve(solver=cp.CLARABEL)
    return M.value, xi.value


def main():
    print('problem  gamma  lambda | F* (cvxpy) | F corrected (gap) | F literal Lemma 2 (gap) | max viol corr / lit')
    for seed in range(3):
        X, Z, pairs, y, V, W, xi0 = make_problem(seed)
        for gamma, lam in [(0.1, 1.0), (1.0, 0.3), (0.01, 3.0)]:
            Mc, Pc = solve_cvx_itmlplus(V, W, y, xi0, gamma, lam)
            fstar = itmlplus_objective(Mc, Pc, V, W, xi0, gamma, lam)
            res = []
            for corr in [True, False]:
                M, P = np.eye(V.shape[1]), np.eye(W.shape[1])
                _itmlplus_core(X, Z, pairs[:, 0].copy(), pairs[:, 1].copy(), y.astype(np.int64),
                               xi0.astype(float), gamma, lam, M, P,
                               20000 * len(y), 1e-12, corr)
                f = itmlplus_objective(M, P, V, W, xi0, gamma, lam)
                dm = np.einsum('ij,jk,ik->i', V, M, V)
                dp = np.einsum('ij,jk,ik->i', W, P, W)
                res.append((f, np.max(y * (dm - dp))))
            print(f'  {seed}     {gamma:5g}  {lam:5g} | {fstar:10.5f} | {res[0][0]:10.5f} ({res[0][0] - fstar:+.1e}) '
                  f'| {res[1][0]:10.5f} ({res[1][0] - fstar:+.1e}) | {res[0][1]:+.1e} / {res[1][1]:+.1e}')
    print('ITML: F* (cvxpy) vs ItmlAlg.m port')
    for seed in range(3):
        X, Z, pairs, y, V, W, xi0 = make_problem(seed)
        for gamma in [0.1, 1.0]:
            Mc, xic = solve_cvx_itml(V, y, xi0, gamma)
            fstar = dld(Mc) + gamma * ell(xic, xi0).sum()
            A = np.eye(V.shape[1])
            # ItmlAlg.m's slack update keeps bhat implicitly; recover xi from duals
            _itml_core(X, pairs[:, 0].copy(), pairs[:, 1].copy(), y.astype(np.int64),
                       xi0.astype(float), gamma, A, 20000 * len(y), 1e-12, True)
            dm = np.einsum('ij,jk,ik->i', V, A, V)
            # optimal slack given M: xi = argmin gamma*ell(xi, xi0) s.t. y dM <= y xi
            xi = np.where(y == 1, np.maximum(dm, xi0), np.minimum(dm, xi0))
            f = dld(A) + gamma * ell(xi, xi0).sum()
            print(f'  {seed} gamma={gamma:g}: F*={fstar:.5f}  ITML={f:.5f}  gap={f - fstar:+.1e}')


if __name__ == '__main__':
    main()
