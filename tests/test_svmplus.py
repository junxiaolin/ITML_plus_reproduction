"""SVM+ solvers (own interior-point method, Clarabel) vs. cvxpy on the primal."""
import os
import sys

import cvxpy as cp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from itmlplus.baselines import SVMPlus


def primal(Xa, Za, y, C, gamma):
    w = cp.Variable(Xa.shape[1])
    v = cp.Variable(Za.shape[1])
    xi = Za @ v
    obj = 0.5 * cp.sum_squares(w) + 0.5 * gamma * cp.sum_squares(v) + C * cp.sum(xi)
    cons = [cp.multiply(y, Xa @ w) >= 1 - xi, xi >= 0]
    p = cp.Problem(cp.Minimize(obj), cons)
    p.solve(solver=cp.CLARABEL)
    return w.value, v.value, p.value


def objective(w, v, Za, C, gamma):
    return 0.5 * w @ w + 0.5 * gamma * v @ v + C * (Za @ v).sum()


def problems(rng, n=600, dx=20, dz=15):
    lab = rng.randint(0, 2, n) * 2 - 1
    # (a) weak signal, uninformative privileged features (degenerate optimum w = 0)
    X = rng.randn(n, dx) + 0.3 * lab[:, None] * rng.randn(1, dx)
    Z = rng.randn(n, dz) + 0.6 * lab[:, None] * rng.randn(1, dz)
    yield 'uninformative Z', X, Z, lab
    # (b) per-sample noise level that the privileged features reveal (SVM+ helps)
    sig = rng.gamma(2.0, 0.5, n)
    X = 1.5 * lab[:, None] * np.ones((1, dx)) / np.sqrt(dx) + sig[:, None] * rng.randn(n, dx)
    Z = np.c_[sig, sig ** 2, rng.randn(n, dz - 2)]
    yield 'informative Z', X, Z, lab


if __name__ == '__main__':
    rng = np.random.RandomState(0)
    for name, X, Z, lab in problems(rng):
        Xa, Za = np.hstack([X, np.ones((len(X), 1))]), np.hstack([Z, np.ones((len(Z), 1))])
        for C, gamma in [(0.1, 1.0), (1.0, 0.01), (10.0, 100.0), (100.0, 0.01)]:
            wc, vc, fc = primal(Xa, Za, lab.astype(float), C, gamma)
            line = f'[{name}] C={C:g} gamma={gamma:g}: cvxpy {fc:.6f}'
            for solver in ['ipm', 'qp']:
                m = SVMPlus(C=C, gamma=gamma, solver=solver).fit(X, Z, lab)
                f = objective(m.w_, m.v_, Za, C, gamma)
                viol = max(0, np.max(1 - Za @ m.v_ - lab * (Xa @ m.w_)), np.max(-(Za @ m.v_)))
                line += (f' | {solver} {f:.6f} (rel gap {(f - fc) / abs(fc):+.1e}, viol {viol:.0e}, '
                         f'|w-w*| {np.abs(m.w_ - wc).max():.0e}, it {m.iters_})')
            print(line)
