"""Face verification: --mode main (Tables I/II) or trainsize (Fig. 3).
--protocol eurocom: all similar pairs + dissimilar up to 10000 (train) / 5000 (val, test);
--protocol curtin: 15000 + 15000."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from itmlplus.experiment import ALL_METHODS, Task, run_task
from itmlplus.metric import compute_bounds
from itmlplus.protocol import Preprocess, positive_pairs, sample_pairs

RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def split_subjects(subjects, r):
    rng = np.random.RandomState(2000 + r)
    u = np.unique(subjects)
    perm = rng.permutation(u)
    n_tr = len(u) // 2
    n_va = (len(u) - n_tr) // 2
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:], rng


def eval_pairs(labels, protocol, rng):
    if protocol == 'eurocom':
        n_pos = len(positive_pairs(labels))
        return sample_pairs(labels, None, max(5000 - n_pos, 0), rng)
    return sample_pairs(labels, 15000, 15000, rng)


def build_task(D, r, protocol, train_frac=1.0):
    subj = D['subject']
    tr, va, te, rng = split_subjects(subj, r)
    itr, iva, ite = (np.flatnonzero(np.isin(subj, s)) for s in (tr, va, te))
    px, pz = Preprocess(150).fit(D['rgb'][itr]), Preprocess(150).fit(D['depth'][itr])
    Xtr, Ztr = px(D['rgb'][itr]), pz(D['depth'][itr])
    ltr = subj[itr]
    if protocol == 'eurocom':
        n_pos = len(positive_pairs(ltr))
        pairs, y = sample_pairs(ltr, None, 10000 - n_pos, rng)
    else:
        pairs, y = sample_pairs(ltr, 15000, 15000, rng)
    evals = {}
    for name, idx in [('val', iva), ('test', ite)]:
        p, yy = eval_pairs(subj[idx], protocol, rng)
        evals[name] = (px(D['rgb'][idx]), p, yy)
    if train_frac < 1.0:  # Fig. 3: a random subset with the same class proportions
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == -1)
        keep = np.r_[rng.choice(pos, int(round(train_frac * len(pos))), replace=False),
                     rng.choice(neg, int(round(train_frac * len(neg))), replace=False)]
        pairs, y = pairs[np.sort(keep)], y[np.sort(keep)]
    return Task(Xtr=Xtr, Ztr=Ztr, pairs=pairs, y=y, bounds_x=compute_bounds(Xtr),
                evals=evals, kind='verification')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--protocol', choices=['eurocom', 'curtin'], required=True)
    ap.add_argument('--mode', choices=['main', 'trainsize'], default='main')
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--methods', nargs='+', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    D = dict(np.load(args.data, allow_pickle=True))
    print(f'{len(D["subject"])} images, {len(np.unique(D["subject"]))} persons, '
          f'rgb {D["rgb"].shape[1]}-d, depth {D["depth"].shape[1]}-d')
    if args.mode == 'main':
        settings = [{'methods': args.methods or ALL_METHODS}]
    else:
        settings = [{'methods': args.methods or ['ITML', 'ITML+'], 'train_frac': f}
                    for f in [0.2, 0.4, 0.6, 0.8, 1.0]]
    results = []
    for st in settings:
        for r in range(args.rounds):
            t0 = time.time()
            task = build_task(D, r, args.protocol, st.get('train_frac', 1.0))
            print(f'[{st}] round {r}: {len(task.y)} train pairs ({(task.y == 1).sum()} similar), '
                  f'test pairs {len(task.evals["test"][2])}', flush=True)
            out = run_task(task, st['methods'], ('val', 'AP'), seed=r)
            rec = {'setting': {k: v for k, v in st.items() if k != 'methods'}, 'round': r,
                   'results': {m: {k: v for k, v in d.items() if k != 'all'} for m, d in out.items()},
                   'all': {m: d['all'] for m, d in out.items()}}
            results.append(rec)
            with open(os.path.join(RES, args.out), 'w') as f:
                json.dump(results, f, indent=1, default=float)
            print(f'  done in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
