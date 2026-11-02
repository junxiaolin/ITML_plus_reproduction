"""ITML on RGB, on depth, and on RGB + depth (depth also at test time), as an
upper bound for privileged-information methods."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from itmlplus.experiment import GAMMAS, Task, run_task
from itmlplus.metric import compute_bounds
from itmlplus.protocol import Preprocess, positive_pairs, sample_pairs
from run_verif import eval_pairs, split_subjects


def tasks_for_round(D, r, protocol):
    """Same subjects/pairs as run_verif.build_task; three feature views."""
    subj = D['subject']
    tr, va, te, rng = split_subjects(subj, r)
    itr, iva, ite = (np.flatnonzero(np.isin(subj, s)) for s in (tr, va, te))
    pr, pd = Preprocess(150).fit(D['rgb'][itr]), Preprocess(150).fit(D['depth'][itr])
    ltr = subj[itr]
    if protocol == 'eurocom':
        pairs, y = sample_pairs(ltr, None, 10000 - len(positive_pairs(ltr)), rng)
    else:
        pairs, y = sample_pairs(ltr, 15000, 15000, rng)
    ev = {name: eval_pairs(subj[idx], protocol, rng) for name, idx in [('val', iva), ('test', ite)]}
    views = {'RGB': lambda idx: pr(D['rgb'][idx]),
             'depth': lambda idx: pd(D['depth'][idx]),
             'RGB+depth': lambda idx: np.hstack([pr(D['rgb'][idx]), pd(D['depth'][idx])]) / np.sqrt(2)}
    out = {}
    for name, f in views.items():
        X = f(itr)
        out[name] = Task(Xtr=X, Ztr=X, ytr_labels=ltr, pairs=pairs, y=y, bounds_x=compute_bounds(X),
                         bounds_z=compute_bounds(X),
                         evals={k: (f(idx), *ev[k]) for k, idx in [('val', iva), ('test', ite)]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--jobs', type=int, default=2)
    args = ap.parse_args()
    D = dict(np.load(args.data, allow_pickle=True))
    res = {}
    for r in range(args.rounds):
        for name, task in tasks_for_round(D, r, args.protocol).items():
            out = run_task(task, ['L2', 'ITML'], ('val', 'AP'), seed=r, n_jobs=args.jobs, verbose=False,
                           grids={'ITML': [{'gamma': g} for g in GAMMAS]})
            for m in ['L2', 'ITML']:
                res.setdefault((name, m), []).append([out[m]['scores']['test']['AP'],
                                                      out[m]['scores']['test']['AUC']])
        print(f'round {r} done', flush=True)
    for (name, m), v in res.items():
        v = np.array(v) * 100
        print(f'{m:5s} on {name:10s}: AP {v[:, 0].mean():.2f}±{v[:, 0].std(ddof=1):.2f}  '
              f'AUC {v[:, 1].mean():.2f}±{v[:, 1].std(ddof=1):.2f}')


if __name__ == '__main__':
    main()
