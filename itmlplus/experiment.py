"""Training, model selection on the validation set, and testing for one round."""
import os
import time
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed, parallel_config

from .baselines import SVMPlus, itml_s, lmnn, pair_features, svm_fit
from .metric import ITML, ITMLPlus, cross_distances, metric_factor
from .protocol import rank1, verification_scores

GAMMAS = [10.0 ** e for e in np.arange(-4, 0.01, 0.5)]   # ITML, ITML-S, ITML+
LAMBDAS = [10.0 ** e for e in np.arange(-2, 2.01, 0.5)]  # ITML+
CS = [10.0 ** e for e in range(-2, 3)]                   # SVM, SVM+
SVMP_GAMMAS = [10.0 ** e for e in range(-2, 3)]          # SVM+
MUS = [round(0.1 * k, 1) for k in range(1, 11)]          # LMNN

ALL_METHODS = ['L2', 'SVM', 'ITML', 'LMNN', 'ITML-S', 'SVM+', 'ITML+', 'ITML+ (Lemma 2 verbatim)']


def default_grid(method):
    if method == 'L2':
        return [{}]
    if method in ('ITML', 'ITML-S'):
        return [{'gamma': g} for g in GAMMAS]
    if method.startswith('ITML+'):
        return [{'gamma': g, 'lam': l} for g in GAMMAS for l in LAMBDAS]
    if method == 'LMNN':
        return [{'mu': m} for m in MUS]
    if method == 'SVM':
        return [{'C': c} for c in CS]
    if method == 'SVM+':
        return [{'C': c, 'gamma': g} for c in CS for g in SVMP_GAMMAS]
    raise ValueError(method)


@dataclass
class Task:
    """One round of one experiment, already preprocessed (PCA'd features)."""
    Xtr: np.ndarray            # main features of training images
    Ztr: np.ndarray            # privileged features of training images
    ytr_labels: np.ndarray     # identities of training images (LMNN only)
    pairs: np.ndarray          # training pairs (indices into Xtr)
    y: np.ndarray              # +1 similar / -1 dissimilar
    bounds_x: tuple
    bounds_z: tuple
    has_priv: np.ndarray = None
    # verification: dict name -> (X, pairs, y); re-id: name -> (Xp, lp, Xg, lg)
    evals: dict = field(default_factory=dict)
    kind: str = 'verification'
    max_sweeps: int = 10


class Identity:
    def __call__(self, F):
        return F


class Scorer:
    """Wraps a learned model so that it can score pairs / probe-gallery sets."""

    def __init__(self, M=None, svm=None, std=None):
        self.M, self.svm, self.std = M, svm, std
        self.L = metric_factor(M) if M is not None else None

    def pair_scores(self, X, pairs, chunk=20000):
        if self.svm is not None:
            # pair features are built in chunks: all probe-gallery pairs of a
            # re-id test set (250 x 4750) would need several GB at once
            Fx = self.std(X)
            return np.concatenate([self.svm.decision_function(pair_features(Fx, pairs[s:s + chunk]))
                                   for s in range(0, len(pairs), chunk)])
        Y = X @ self.L if self.L is not None else X
        D = Y[pairs[:, 0]] - Y[pairs[:, 1]]
        return -(D ** 2).sum(1)

    def probe_gallery_dist(self, Xp, Xg):
        if self.svm is not None:
            F = np.vstack([Xp, Xg])
            n_p, n_g = len(Xp), len(Xg)
            ii, jj = np.meshgrid(np.arange(n_p), n_p + np.arange(n_g), indexing='ij')
            pairs = np.stack([ii.ravel(), jj.ravel()], 1)
            return -self.pair_scores(F, pairs).reshape(n_p, n_g)
        return cross_distances(self.M, Xp, Xg)


def fit_method(task, method, hp, seed=0):
    t0 = time.perf_counter()
    if method == 'L2':
        sc = Scorer()
    elif method == 'ITML':
        sc = Scorer(ITML(gamma=hp['gamma'], bounds=task.bounds_x, max_sweeps=task.max_sweeps,
                         seed=seed).fit(task.Xtr, task.pairs, task.y).M_)
    elif method == 'ITML-S':
        M, _ = itml_s(task.Xtr, task.Ztr, task.pairs, task.y, hp['gamma'], task.bounds_x,
                      task.bounds_z, max_sweeps=task.max_sweeps, seed=seed)
        sc = Scorer(M)
    elif method.startswith('ITML+'):
        corr = 'verbatim' not in method
        model = ITMLPlus(gamma=hp['gamma'], lam=hp['lam'], bounds=task.bounds_x,
                         max_sweeps=task.max_sweeps, correction=corr, seed=seed)
        model.fit(task.Xtr, task.Ztr, task.pairs, task.y, has_priv=task.has_priv)
        sc = Scorer(model.M_)
        sc.P = model.P_
    elif method == 'LMNN':
        sc = Scorer(lmnn(task.Xtr, task.ytr_labels, mu=hp['mu'], k=5, seed=seed))
    elif method == 'SVM':
        # no per-dimension standardisation of the pair features: it breaks ranking
        sc = Scorer(svm=svm_fit(task.Xtr, task.pairs, task.y, hp['C']), std=Identity())
    elif method == 'SVM+':
        Fx = pair_features(task.Xtr, task.pairs)
        Fz = pair_features(task.Ztr, task.pairs)
        sc = Scorer(svm=SVMPlus(C=hp['C'], gamma=hp['gamma'], seed=seed).fit(Fx, Fz, task.y),
                    std=Identity())
    else:
        raise ValueError(method)
    return sc, time.perf_counter() - t0


def evaluate(task, sc):
    out = {}
    for name, ev in task.evals.items():
        if task.kind == 'verification':
            X, pairs, y = ev
            ap, auc = verification_scores(y, sc.pair_scores(X, pairs))
            out[name] = {'AP': ap, 'AUC': auc}
        elif len(ev) == 2:  # leave-one-out: every image is a probe against all others
            X, lab = ev
            D = sc.probe_gallery_dist(X, X)
            np.fill_diagonal(D, np.inf)
            out[name] = {'rank1': rank1(D, lab, lab)}
        else:
            Xp, lp, Xg, lg = ev
            out[name] = {'rank1': rank1(sc.probe_gallery_dist(Xp, Xg), lp, lg)}
    return out


def _job(task, method, hp, seed):
    sc, t = fit_method(task, method, hp, seed)
    return {'method': method, 'hp': hp, 'time': t, 'scores': evaluate(task, sc)}


N_JOBS = int(os.environ.get('ITMLPLUS_JOBS', '8'))  # parallel workers; each needs < 1 GB


def run_task(task, methods, select_on, n_jobs=None, seed=0, grids=None, verbose=True):
    """Fit every (method, hyper-parameter) setting of one round and select on
    `select_on`, e.g. ('val', 'AP'). Returns {method: {'hp', 'time', 'scores', 'all'}}."""
    grids = grids or {}
    jobs = [(m, hp) for m in methods for hp in grids.get(m, default_grid(m))]
    cpu = [(m, hp) for m, hp in jobs if m != 'LMNN']
    gpu = [(m, hp) for m, hp in jobs if m == 'LMNN']
    res = []
    if cpu:
        with parallel_config(backend='loky', inner_max_num_threads=1):
            res += Parallel(n_jobs=n_jobs or N_JOBS)(delayed(_job)(task, m, hp, seed) for m, hp in cpu)
    res += [_job(task, m, hp, seed) for m, hp in gpu]
    out = {}
    vname, measure = select_on
    for m in methods:
        rows = [r for r in res if r['method'] == m]
        best = max(rows, key=lambda r: r['scores'][vname][measure])
        out[m] = {'hp': best['hp'], 'time': best['time'], 'scores': best['scores'], 'all': rows}
        if verbose:
            s = ', '.join(f'{k}:' + '/'.join(f'{v * 100:.2f}' for v in d.values())
                          for k, d in best['scores'].items())
            print(f'    {m:26s} hp={best["hp"]}  {s}', flush=True)
    return out
