"""ITML and ITML+ solved with cyclic Bregman projections.
X: RGB features, Z: depth features, y = +1 (similar) or -1 (dissimilar)."""
import numpy as np
from numba import njit


def compute_bounds(X, lo_pct=3.0, hi_pct=97.0, max_samples=2000, seed=0):
    """Targets (u, l): lo_pct / hi_pct percentiles of squared L2 distances
    between training samples."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    if n > max_samples:
        X = X[rng.choice(n, max_samples, replace=False)]
    sq = (X ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    d = np.maximum(D[iu], 0.0)
    return float(np.percentile(d, lo_pct)), float(np.percentile(d, hi_pct))


@njit(cache=True)
def _matvec(A, v, out):
    d = v.shape[0]
    q = 0.0
    for p in range(d):
        s = 0.0
        for k in range(d):
            s += A[p, k] * v[k]
        out[p] = s
        q += v[p] * s
    return q


@njit(cache=True)
def _rank1(A, coef, a):
    d = a.shape[0]
    for p in range(d):
        cp = coef * a[p]
        for k in range(d):
            A[p, k] += cp * a[k]


@njit(cache=True)
def _itml_core(X, i1, i2, y, xi0, gamma, A, max_iters, tol, correction):
    """ITML cyclic projections, ported from ItmlAlg.m. correction=False only
    projects onto violated constraints, like Lemma 2."""
    c = i1.shape[0]
    d = X.shape[1]
    lam = np.zeros(c)
    lam_old = np.zeros(c)
    bhat = xi0.copy()
    if np.isinf(gamma):
        gamma_proj = 1.0
    else:
        gamma_proj = gamma / (gamma + 1.0)
    v = np.empty(d)
    Av = np.empty(d)
    it = 0
    i = 0
    conv = np.inf
    while True:
        for k in range(d):
            v[k] = X[i1[i], k] - X[i2[i], k]
        wtw = _matvec(A, v, Av)
        if y[i] == 1:
            alpha = gamma_proj * (1.0 / wtw - 1.0 / bhat[i])
            alpha = min(lam[i], alpha) if correction else min(0.0, alpha)
            lam[i] -= alpha
            beta = alpha / (1.0 - alpha * wtw)
            if not np.isinf(gamma):
                bhat[i] = 1.0 / (1.0 / bhat[i] + alpha / gamma)
        else:
            alpha = gamma_proj * (1.0 / bhat[i] - 1.0 / wtw)
            alpha = min(lam[i], alpha) if correction else min(0.0, alpha)
            lam[i] -= alpha
            beta = -alpha / (1.0 + alpha * wtw)
            if not np.isinf(gamma):
                bhat[i] = 1.0 / (1.0 / bhat[i] - alpha / gamma)
        if beta != 0.0:
            _rank1(A, beta, Av)
        it += 1
        if i == c - 1:
            normsum = np.sqrt((lam ** 2).sum()) + np.sqrt((lam_old ** 2).sum())
            if normsum == 0.0:
                break
            conv = np.abs(lam_old - lam).sum() / normsum
            if conv < tol or it >= max_iters:
                break
            lam_old[:] = lam
        i = (i + 1) % c
    return it, conv


@njit(cache=True)
def _itmlplus_core(X, Z, i1, i2, y, xi0, gamma, lam, M, P,
                   max_iters, tol, correction):
    """Algorithm 1 (ITML+). correction=False is Lemma 2 as printed; correction=True adds the Hildreth
    correction alpha = max(-Lambda_ij, alpha_eq), with Lambda_ij the sum of past alphas."""
    c = i1.shape[0]
    dx = X.shape[1]
    dz = Z.shape[1]
    xi = xi0.copy()
    cum_a = np.zeros(c)
    cum_b = np.zeros(c)
    old_a = np.zeros(c)
    old_b = np.zeros(c)
    v = np.empty(dx)
    Mv = np.empty(dx)
    w = np.empty(dz)
    Pw = np.empty(dz)
    it = 0
    i = 0
    conv = np.inf
    while True:
        yi = float(y[i])
        for k in range(dx):
            v[k] = X[i1[i], k] - X[i2[i], k]
        r = _matvec(M, v, Mv)
        for k in range(dz):
            w[k] = Z[i1[i], k] - Z[i2[i], k]
        s = _matvec(P, w, Pw)
        f = gamma / xi[i] + lam / s - (lam + gamma) / r
        alpha = max(-cum_a[i] if correction else 0.0,
                    f / (yi * (lam + gamma + 1.0)))
        beta = lam / (lam + gamma) * (gamma / s - gamma / xi[i] + yi * alpha)
        if alpha != 0.0:
            _rank1(M, -yi * alpha / (1.0 + yi * alpha * r), Mv)
        if beta != 0.0:
            _rank1(P, beta / (lam - beta * s), Pw)
        xi[i] = lam * s / (lam - s * beta)
        cum_a[i] += alpha
        cum_b[i] += beta
        it += 1
        if i == c - 1:
            normsum = (np.abs(cum_a).sum() + np.abs(cum_b).sum()
                       + np.abs(old_a).sum() + np.abs(old_b).sum())
            if normsum == 0.0:
                break
            conv = (np.abs(cum_a - old_a).sum() + np.abs(cum_b - old_b).sum()) / normsum
            if conv < tol or it >= max_iters:
                break
            old_a[:] = cum_a
            old_b[:] = cum_b
        i = (i + 1) % c
    return it, conv, xi


def _prepare_pairs(X, pairs, y, seed, Z=None):
    pairs = np.asarray(pairs, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    diff = X[pairs[:, 0]] - X[pairs[:, 1]]
    keep = np.linalg.norm(diff, axis=1) > 1e-9  # ItmlAlg.m drops identical vectors
    if Z is not None:
        keep &= np.linalg.norm(Z[pairs[:, 0]] - Z[pairs[:, 1]], axis=1) > 1e-9
    pairs, y = pairs[keep], y[keep]
    order = np.random.RandomState(seed).permutation(len(y))
    return pairs[order], y[order], np.flatnonzero(keep)[order]


class ITML:
    """Information-theoretic metric learning on pairwise constraints."""

    def __init__(self, gamma=1.0, bounds=None, max_sweeps=10, tol=1e-3,
                 correction=True, seed=0):
        self.gamma = gamma
        self.bounds = bounds
        self.max_sweeps = max_sweeps
        self.tol = tol
        self.correction = correction
        self.seed = seed

    def fit(self, X, pairs, y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        u, l = self.bounds if self.bounds is not None else compute_bounds(X)
        pairs, y, _ = _prepare_pairs(X, pairs, y, self.seed)
        xi0 = np.where(y == 1, u, l).astype(np.float64)
        A = np.eye(X.shape[1])
        self.n_iter_, self.conv_ = _itml_core(
            X, pairs[:, 0].copy(), pairs[:, 1].copy(), y, xi0, float(self.gamma),
            A, int(self.max_sweeps * len(y)), float(self.tol), bool(self.correction))
        self.M_ = A
        return self


class ITMLPlus:
    """ITML+: ITML with the slack of each pair replaced by its depth distance."""

    def __init__(self, gamma=1e-2, lam=1.0, bounds=None, max_sweeps=10,
                 tol=1e-3, correction=True, seed=0):
        self.gamma = gamma
        self.lam = lam
        self.bounds = bounds
        self.max_sweeps = max_sweeps
        self.tol = tol
        self.correction = correction
        self.seed = seed

    def fit(self, X, Z, pairs, y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        Z = np.ascontiguousarray(Z, dtype=np.float64)
        u, l = self.bounds if self.bounds is not None else compute_bounds(X)
        n_pairs = len(y)
        pairs, y, idx = _prepare_pairs(X, pairs, y, self.seed, Z)
        xi0 = np.where(y == 1, u, l).astype(np.float64)
        M = np.eye(X.shape[1])
        P = np.eye(Z.shape[1])
        self.n_iter_, self.conv_, xi = _itmlplus_core(
            X, Z, pairs[:, 0].copy(), pairs[:, 1].copy(), y, xi0,
            float(self.gamma), float(self.lam), M, P,
            int(self.max_sweeps * len(y)), float(self.tol), bool(self.correction))
        self.M_, self.P_ = M, P
        self.xi_ = np.full(n_pairs, np.nan)  # final slack per input pair
        self.xi_[idx] = xi
        return self


def pair_distances(M, X, pairs):
    D = X[pairs[:, 0]] - X[pairs[:, 1]]
    return np.einsum('ij,jk,ik->i', D, M, D)


def metric_factor(M):
    """L with M = L L' (M is PSD up to round-off)."""
    e, V = np.linalg.eigh((M + M.T) / 2.0)
    return V * np.sqrt(np.maximum(e, 0.0))


def cross_distances(M, A, B):
    """Squared Mahalanobis distances between rows of A and rows of B (M=None: L2)."""
    if M is not None:
        L = metric_factor(M)
        A, B = A @ L, B @ L
    sa = (A ** 2).sum(1)
    sb = (B ** 2).sum(1)
    return np.maximum(sa[:, None] + sb[None, :] - 2.0 * A @ B.T, 0.0)
