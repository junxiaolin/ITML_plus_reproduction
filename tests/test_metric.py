"""Checks of the ITML / ITML+ solvers: ITML against metric-learn's loop, one
ITML+ step against a numerical solver, PSD and feasibility after many sweeps."""
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from itmlplus.metric import (ITML, ITMLPlus, _itml_core, _itmlplus_core,
                             compute_bounds, pair_distances)


def check_against_metric_learn():
    rng = np.random.RandomState(0)
    n, d = 60, 5
    X = rng.randn(n, d)
    lab = rng.randint(0, 4, n)
    pairs, y = [], []
    for _ in range(300):
        i, j = rng.choice(n, 2, replace=False)
        pairs.append((i, j))
        y.append(1 if lab[i] == lab[j] else -1)
    pairs, y = np.array(pairs), np.array(y)
    order = np.r_[np.flatnonzero(y == 1), np.flatnonzero(y == -1)]
    pairs, y = pairs[order], y[order]
    bounds = (0.5, 6.0)
    for gamma in [1.0, 0.01]:
        sweeps = 7
        A_ref = itml_reference(X, pairs, y, bounds, gamma, sweeps)
        A = np.eye(d)
        xi0 = np.where(y == 1, bounds[0], bounds[1]).astype(float)
        _itml_core(X, pairs[:, 0].copy(), pairs[:, 1].copy(), y.astype(np.int64),
                   xi0, gamma, A, sweeps * len(y), 1e-12, True)
        err = np.abs(A - A_ref).max()
        print(f'[ITML numba vs numpy reference] gamma={gamma}: max |dM| = {err:.2e}')
        assert err < 1e-8


def itml_reference(X, pairs, y, bounds, gamma, sweeps):
    """Plain-numpy copy of metric-learn's ITML._fit loop."""
    pos, neg = pairs[y == 1], pairs[y == -1]
    num_pos = len(pos)
    lam = np.zeros(len(pairs))
    gamma_proj = gamma / (gamma + 1.)
    pos_bhat = np.zeros(num_pos) + bounds[0]
    neg_bhat = np.zeros(len(neg)) + bounds[1]
    pos_vv = X[pos[:, 0]] - X[pos[:, 1]]
    neg_vv = X[neg[:, 0]] - X[neg[:, 1]]
    A = np.eye(X.shape[1])
    for _ in range(sweeps):
        for i, v in enumerate(pos_vv):
            wtw = v.dot(A).dot(v)
            alpha = min(lam[i], gamma_proj * (1. / wtw - 1. / pos_bhat[i]))
            lam[i] -= alpha
            beta = alpha / (1 - alpha * wtw)
            pos_bhat[i] = 1. / ((1 / pos_bhat[i]) + (alpha / gamma))
            Av = A.dot(v)
            A += np.outer(Av, Av * beta)
        for i, v in enumerate(neg_vv):
            wtw = v.dot(A).dot(v)
            alpha = min(lam[i + num_pos], gamma_proj * (1. / neg_bhat[i] - 1. / wtw))
            lam[i + num_pos] -= alpha
            beta = -alpha / (1 + alpha * wtw)
            neg_bhat[i] = 1. / ((1 / neg_bhat[i]) - (alpha / gamma))
            Av = A.dot(v)
            A += np.outer(Av, Av * beta)
    return A


def dld(A, B):
    C = A @ np.linalg.inv(B)
    return np.trace(C) - np.log(np.linalg.det(C)) - A.shape[0]


def check_single_projection():
    """Closed form (11)-(13), (23)-(24) vs. numerical solution of (8)-(10)."""
    rng = np.random.RandomState(1)
    h, g = 2, 2
    for trial in range(6):
        Mt = np.cov(rng.randn(h, 10)) + 0.3 * np.eye(h)
        Pt = np.cov(rng.randn(g, 10)) + 0.3 * np.eye(g)
        v = rng.randn(h)
        w = rng.randn(g)
        y = 1 if trial % 2 == 0 else -1
        gamma, lam = [(0.5, 2.0), (0.1, 0.3), (2.0, 1.0)][trial % 3]
        r, s = v @ Mt @ v, w @ Pt @ w
        # make the constraint violated half of the time
        xit = (0.5 * r if y == 1 else 2.0 * r) if trial < 4 else (2.0 * r if y == 1 else 0.5 * r)
        X = np.stack([v, np.zeros(h)])
        Z = np.stack([w, np.zeros(g)])
        M, P = Mt.copy(), Pt.copy()
        _, _, xi = _itmlplus_core(X, Z, np.array([0]), np.array([1]),
                                  np.array([y]), np.array([xit]),
                                  np.array([True]), gamma, lam, M, P, 1, 0.0, False)
        # numerical: variables = cholesky params of M, P (PSD by construction)
        def unpack(t):
            L1 = np.array([[np.exp(t[0]), 0], [t[1], np.exp(t[2])]])
            L2 = np.array([[np.exp(t[3]), 0], [t[4], np.exp(t[5])]])
            return L1 @ L1.T, L2 @ L2.T

        def obj(t):
            Mm, Pp = unpack(t)
            x = w @ Pp @ w  # xi = d_P(z_i, z_j) (equality constraint (10))
            return dld(Mm, Mt) + lam * dld(Pp, Pt) + gamma * (x / xit - np.log(x / xit) - 1)

        def con(t):
            Mm, Pp = unpack(t)
            return y * (w @ Pp @ w) - y * (v @ Mm @ v)  # >= 0  (constraint (9))

        def init(A):
            L = np.linalg.cholesky(A)
            return [np.log(L[0, 0]), L[1, 0], np.log(L[1, 1])]
        t0 = np.r_[init(Mt), init(Pt)]
        best = None
        for start in [t0, t0 + 0.01 * rng.randn(6)]:
            res = minimize(obj, start, constraints=[{'type': 'ineq', 'fun': con}],
                           method='SLSQP', options={'ftol': 1e-14, 'maxiter': 2000})
            if best is None or res.fun < best.fun:
                best = res
        Mn, Pn = unpack(best.x)
        closed = dld(M, Mt) + lam * dld(P, Pt) + gamma * (xi[0] / xit - np.log(xi[0] / xit) - 1)
        print(f'[proj] y={y:+d} viol={"yes" if trial < 4 else "no "} obj closed={closed:.6f} '
              f'numeric={best.fun:.6f}  |dM|={np.abs(M - Mn).max():.1e} |dP|={np.abs(P - Pn).max():.1e} '
              f'xi=d_P? {abs(xi[0] - w @ P @ w):.1e}  r_new={v @ M @ v:.4f} xi_new={xi[0]:.4f}')
        assert abs(closed - best.fun) < 1e-5
        assert abs(xi[0] - w @ P @ w) < 1e-9


def check_full_runs():
    # 3000 pairs vs 55 + 36 parameters: ITML+ shrinks M and P towards 0 here
    # (homogeneous constraints, see the README's implementation notes)
    rng = np.random.RandomState(2)
    n, h, g = 200, 10, 8
    lab = rng.randint(0, 10, n)
    centers_x = rng.randn(10, h)
    centers_z = rng.randn(10, g) * 2
    X = centers_x[lab] + 1.5 * rng.randn(n, h)
    Z = centers_z[lab] + 0.5 * rng.randn(n, g)
    pairs = np.array([rng.choice(n, 2, replace=False) for _ in range(3000)])
    y = np.where(lab[pairs[:, 0]] == lab[pairs[:, 1]], 1, -1)
    u, l = compute_bounds(X)
    for gamma in [1e-3, 1e-1, 1.0]:
        it = ITML(gamma=gamma, bounds=(u, l), max_sweeps=50).fit(X, pairs, y)
        ip = ITMLPlus(gamma=gamma, lam=1.0, bounds=(u, l), max_sweeps=50).fit(X, Z, pairs, y)
        for name, Mx in [('ITML', it.M_), ('ITML+', ip.M_)]:
            d = pair_distances(Mx, X, pairs)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, -d)
            print(f'[run] gamma={gamma:g} {name:6s} min eig={np.linalg.eigvalsh(Mx).min():.2e} '
                  f'train AUC={auc:.4f} iters={it.n_iter_ if name == "ITML" else ip.n_iter_}')
        dp = pair_distances(ip.P_, Z, pairs)
        dm = pair_distances(ip.M_, X, pairs)
        viol = np.mean(y * dm > y * dp + 1e-6)
        print(f'       ITML+ min eig P={np.linalg.eigvalsh(ip.P_).min():.2e} '
              f'fraction violating y d_M <= y d_P: {viol:.3f}; conv={ip.conv_:.2e}')


if __name__ == '__main__':
    check_against_metric_learn()
    check_single_projection()
    check_full_runs()
    print('all checks passed')
