"""Baselines of Sec. V-A: LMNN, SVM, SVM+ (pair-feature classifiers), ITML-S."""
import numpy as np
from numba import njit
from sklearn.svm import LinearSVC

from .metric import ITML, compute_bounds, pair_distances


# ----------------------------------------------------------------------------
# LMNN (Weinberger & Saul, JMLR 2009) -- uses class labels of training samples
# ----------------------------------------------------------------------------
def lmnn(X, labels, mu=0.5, k=5, n_iter=500, lr=1e-2, device=None, seed=0):
    """LMNN: learn M = L'L with k same-class target neighbours by full-batch
    Adam on L, starting from the identity."""
    import torch
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float64, device=device)
    y = torch.tensor(np.unique(labels, return_inverse=True)[1], device=device)
    n, d = Xt.shape
    D0 = torch.cdist(Xt, Xt) ** 2
    same = y[:, None] == y[None, :]
    D0s = D0.clone()
    D0s[~same] = float('inf')
    D0s.fill_diagonal_(float('inf'))
    kk = min(k, int(same.sum(1).min().item()) - 1)
    tn = D0s.topk(kk, largest=False).indices  # (n, k) target neighbours
    diff_mask = (~same).double()
    L = torch.eye(d, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([L], lr=lr)
    for _ in range(n_iter):
        Y = Xt @ L.T
        D = torch.cdist(Y, Y) ** 2
        dt = D.gather(1, tn)  # (n, k)
        pull = dt.sum()
        hinge = torch.relu(1.0 + dt[:, :, None] - D[:, None, :]) * diff_mask[:, None, :]
        loss = (1 - mu) * pull + mu * hinge.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    Lf = L.detach().cpu().numpy()
    return Lf.T @ Lf


# ----------------------------------------------------------------------------
# Pair features of Kumar et al. (ICCV 2009), used for SVM / SVM+
# ----------------------------------------------------------------------------
def pair_features(F, pairs):
    """[|f_i - f_j| o g ; f_i o f_j o g], g = N(0.5(f_i + f_j); 0, 1) elementwise."""
    a, b = F[pairs[:, 0]], F[pairs[:, 1]]
    g = np.exp(-0.5 * (0.5 * (a + b)) ** 2) / np.sqrt(2 * np.pi)
    return np.hstack([np.abs(a - b) * g, a * b * g])


class Standardizer:
    def fit(self, F):
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-12
        return self

    def __call__(self, F):
        return (F - self.mu) / self.sd


def svm_fit(Fx, pairs, y, C):
    clf = LinearSVC(C=C, loss='hinge', dual=True, max_iter=20000, tol=1e-4)
    clf.fit(pair_features(Fx, pairs), y)
    return clf


@njit(cache=True)
def _svmplus_dcd(X, Z, y, C, gamma, n_epochs, tol, seed):
    """Linear SVM+ (Vapnik & Vashist 2009) by dual coordinate ascent; biases are
    absorbed into constant features."""
    n, dx = X.shape
    dz = Z.shape[1]
    a = np.zeros(n)
    b = np.full(n, C)
    w = np.zeros(dx)
    u = np.zeros(dz)  # sum (a + b - C) z  (v = u / gamma)
    qx = (X ** 2).sum(1)
    qz = (Z ** 2).sum(1)
    w_old = np.zeros(dx)
    u_old = np.zeros(dz)
    np.random.seed(seed)
    for ep in range(n_epochs):
        order = np.random.permutation(n)
        for t in range(n):
            i = order[t]
            wx = 0.0
            for k in range(dx):
                wx += w[k] * X[i, k]
            uz = 0.0
            for k in range(dz):
                uz += u[k] * Z[i, k]
            grad = 1.0 - y[i] * wx - uz / gamma
            na = max(0.0, a[i] + grad / (qx[i] + qz[i] / gamma))
            da = na - a[i]
            if da != 0.0:
                a[i] = na
                for k in range(dx):
                    w[k] += da * y[i] * X[i, k]
                for k in range(dz):
                    u[k] += da * Z[i, k]
                uz += da * qz[i]
            nb = max(0.0, b[i] - uz / qz[i])
            db = nb - b[i]
            if db != 0.0:
                b[i] = nb
                for k in range(dz):
                    u[k] += db * Z[i, k]
        # stop when the primal solution (w, v = u / gamma) no longer moves
        dw = 0.0
        nw = 0.0
        for k in range(dx):
            dw += (w[k] - w_old[k]) ** 2
            nw += w[k] ** 2
            w_old[k] = w[k]
        du = 0.0
        nu = 0.0
        for k in range(dz):
            du += (u[k] - u_old[k]) ** 2
            nu += u[k] ** 2
            u_old[k] = u[k]
        if ep > 0 and dw <= tol ** 2 * max(nw, 1e-12) and du <= tol ** 2 * max(nu, 1e-12):
            break
    return w, u / gamma, a, b, ep + 1


def _svmplus_clarabel(Xa, Za, y, C, gamma):
    """The same primal problem solved by Clarabel; variables u = [w, v]."""
    import clarabel
    import scipy.sparse as sp
    n, dx = Xa.shape
    dz = Za.shape[1]
    P = sp.diags(np.r_[np.ones(dx), gamma * np.ones(dz)]).tocsc()
    q = np.r_[np.zeros(dx), C * Za.sum(0)]
    # -y_i x_i'w - z_i'v <= -1   and   -z_i'v <= 0
    A = sp.vstack([sp.hstack([sp.csr_matrix(-(y[:, None] * Xa)), sp.csr_matrix(-Za)]),
                   sp.hstack([sp.csr_matrix((n, dx)), sp.csr_matrix(-Za)])]).tocsc()
    b = np.r_[-np.ones(n), np.zeros(n)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    sol = clarabel.DefaultSolver(P, q, A, b, [clarabel.NonnegativeConeT(2 * n)], settings).solve()
    u = np.array(sol.x)
    return u[:dx], u[dx:], str(sol.status), sol.iterations


def _svmplus_ipm(Xa, Za, y, C, gamma, tol=1e-9, max_iter=100):
    """The same primal problem solved by a Mehrotra predictor-corrector
    interior-point method; G'WG is built block-wise to keep memory O(n (dx + dz))."""
    n, dx = Xa.shape
    dz = Za.shape[1]
    m = 2 * n
    p = np.r_[np.ones(dx), gamma * np.ones(dz)]
    q = np.r_[np.zeros(dx), C * Za.sum(0)]
    h = np.r_[-np.ones(n), np.zeros(n)]
    yX = y[:, None] * Xa

    def G(u):
        zv = Za @ u[dx:]
        return np.r_[-(yX @ u[:dx]) - zv, -zv]

    def Gt(a):
        return np.r_[-(yX.T @ a[:n]), -(Za.T @ (a[:n] + a[n:]))]

    u = np.zeros(dx + dz)
    s = np.ones(m)
    lam = np.ones(m)
    for it in range(max_iter):
        rd = p * u + q + Gt(lam)
        rp = G(u) + s - h
        mu = s @ lam / m
        if (np.abs(rp).max() <= tol * (1 + np.abs(h).max())
                and np.abs(rd).max() <= tol * (1 + np.abs(q).max()) and mu <= tol):
            break
        w = lam / s
        w1, w2 = w[:n], w[n:]
        H = np.empty((dx + dz, dx + dz))
        H[:dx, :dx] = Xa.T @ (w1[:, None] * Xa)
        H[:dx, dx:] = yX.T @ (w1[:, None] * Za)
        H[dx:, :dx] = H[:dx, dx:].T
        H[dx:, dx:] = Za.T @ ((w1 + w2)[:, None] * Za)
        H[np.diag_indices_from(H)] += p
        L = np.linalg.cholesky(H)

        def solve(rc):
            rhs = -rd - Gt((rc + lam * rp) / s)
            du = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
            ds = -rp - G(du)
            dl = (rc - lam * ds) / s
            return du, ds, dl

        def step(x, dx_):
            neg = dx_ < 0
            return min(1.0, np.min(-x[neg] / dx_[neg])) if neg.any() else 1.0
        du, ds, dl = solve(-s * lam)                       # predictor
        a = min(step(s, ds), step(lam, dl))
        sigma = ((s + a * ds) @ (lam + a * dl) / m / mu) ** 3
        du, ds, dl = solve(-s * lam - ds * dl + sigma * mu)  # corrector
        a = 0.99 * min(step(s, ds), step(lam, dl))
        u, s, lam = u + a * du, s + a * ds, lam + a * dl
    return u[:dx], u[dx:], 'solved' if it < max_iter - 1 else 'max_iter', it


class SVMPlus:
    """Linear SVM+ on pair features; solver='ipm' (default, exact and
    memory-light), 'qp' (Clarabel) or 'cd' (dual coordinate ascent)."""

    def __init__(self, C=1.0, gamma=1.0, solver='ipm', n_epochs=3000, tol=1e-4, seed=0):
        self.C, self.gamma, self.solver = C, gamma, solver
        self.n_epochs, self.tol, self.seed = n_epochs, tol, seed

    def fit(self, X, Z, y):
        Xa = np.hstack([X, np.ones((len(X), 1))])
        Za = np.hstack([Z, np.ones((len(Z), 1))])
        y = y.astype(np.float64)
        if self.solver == 'ipm':
            self.w_, self.v_, self.status_, self.iters_ = _svmplus_ipm(
                Xa, Za, y, float(self.C), float(self.gamma))
        elif self.solver == 'qp':
            self.w_, self.v_, self.status_, self.iters_ = _svmplus_clarabel(
                Xa, Za, y, float(self.C), float(self.gamma))
        else:
            self.w_, self.v_, self.a_, self.b_, self.iters_ = _svmplus_dcd(
                Xa, Za, y, float(self.C), float(self.gamma), int(self.n_epochs),
                float(self.tol), int(self.seed))
        return self

    def decision_function(self, X):
        return X @ self.w_[:-1] + self.w_[-1]


# ----------------------------------------------------------------------------
# ITML-S (Fouad et al., TNNLS 2013): two-step use of privileged information
# ----------------------------------------------------------------------------
def itml_s(X, Z, pairs, y, gamma, bounds_x, bounds_z, max_sweeps=10, seed=0):
    """ITML-S: ITML on depth, drop the training pairs that the depth metric gets
    wrong at the best S/D threshold, then ITML on RGB with the remaining pairs."""
    P = ITML(gamma=gamma, bounds=bounds_z, max_sweeps=max_sweeps, seed=seed).fit(Z, pairs, y).M_
    dz = pair_distances(P, Z, pairs)
    order = np.argsort(dz)
    ys = y[order]
    # threshold after position t: errors = #dissimilar among first t + #similar after t
    err = np.r_[0, np.cumsum(ys == -1)] + np.r_[np.cumsum((ys == 1)[::-1])[::-1], 0]
    t = int(np.argmin(err))
    thr = dz[order[t - 1]] if t > 0 else -np.inf
    keep = np.where(y == 1, dz <= thr, dz > thr)
    M = ITML(gamma=gamma, bounds=bounds_x, max_sweeps=max_sweeps, seed=seed).fit(
        X, pairs[keep], y[keep]).M_
    return M, keep
