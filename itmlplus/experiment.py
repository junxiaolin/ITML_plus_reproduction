"""Training, model selection on the validation set, and testing for one round."""
import os
import time
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed, parallel_config

from .metric import ITML, ITMLPlus, cross_distances, metric_factor
from .protocol import rank1, verification_scores

GAMMAS = [10.0 ** e for e in np.arange(-4, 0.01, 0.5)]   # ITML, ITML+
LAMBDAS = [10.0 ** e for e in np.arange(-2, 2.01, 0.5)]  # ITML+

ALL_METHODS = ['L2', 'ITML', 'ITML+']


def default_grid(method):
    if method == 'L2':
        return [{}]
    if method == 'ITML':
        return [{'gamma': g} for g in GAMMAS]
    if method == 'ITML+':
        return [{'gamma': g, 'lam': l} for g in GAMMAS for l in LAMBDAS]
    raise ValueError(method)


@dataclass
class Task:
    """One round of one experiment, already preprocessed (PCA'd features)."""
    Xtr: np.ndarray            # main features of training images
    Ztr: np.ndarray            # privileged features of training images
    pairs: np.ndarray          # training pairs (indices into Xtr)
    y: np.ndarray              # +1 similar / -1 dissimilar
    bounds_x: tuple
    # verification: dict name -> (X, pairs, y); re-id: name -> (Xp, lp, Xg, lg)
    evals: dict = field(default_factory=dict)
    kind: str = 'verification'
    max_sweeps: int = 10


class Scorer:
    """Wraps a learned model so that it can score pairs / probe-gallery sets."""

    def __init__(self, M=None):
        self.M = M
        self.L = metric_factor(M) if M is not None else None

    def pair_scores(self, X, pairs):
        Y = X @ self.L if self.L is not None else X
        D = Y[pairs[:, 0]] - Y[pairs[:, 1]]
        return -(D ** 2).sum(1)

    def probe_gallery_dist(self, Xp, Xg):
        return cross_distances(self.M, Xp, Xg)


def fit_method(task, method, hp, seed=0):
    t0 = time.perf_counter()
    if method == 'L2':
        sc = Scorer()
    elif method == 'ITML':
        sc = Scorer(ITML(gamma=hp['gamma'], bounds=task.bounds_x, max_sweeps=task.max_sweeps,
                         seed=seed).fit(task.Xtr, task.pairs, task.y).M_)
    elif method == 'ITML+':
        model = ITMLPlus(gamma=hp['gamma'], lam=hp['lam'], bounds=task.bounds_x,
                         max_sweeps=task.max_sweeps, seed=seed)
        model.fit(task.Xtr, task.Ztr, task.pairs, task.y)
        sc = Scorer(model.M_)
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
    with parallel_config(backend='loky', inner_max_num_threads=1):
        res = Parallel(n_jobs=n_jobs or N_JOBS)(delayed(_job)(task, m, hp, seed) for m, hp in jobs)
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
