"""ITML, ITML-S and ITML+ with gamma from 1e-4 to 1e2 (wider than the paper's grid)."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from itmlplus.experiment import LAMBDAS, run_task

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
GAMMAS_EXT = [10.0 ** e for e in np.arange(-4, 2.01, 0.5)]
METHODS = ['ITML', 'ITML-S', 'ITML+']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['biwi_a', 'biwi_b', 'uw'], required=True)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--data', default=None, help='face .npz (default: ../data/processed/<dataset>_full_glbp.npz)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    grids = {'ITML': [{'gamma': g} for g in GAMMAS_EXT],
             'ITML-S': [{'gamma': g} for g in GAMMAS_EXT],
             'ITML+': [{'gamma': g, 'lam': l} for g in GAMMAS_EXT for l in LAMBDAS]}
    results = []
    if args.dataset == 'uw':
        from run_reid import build_task, make_round
        from itmlplus.uw_rgbd import KDESExtractor, index
        inst, ext = index(), KDESExtractor()
        tasks = (build_task(ext, *make_round(inst, r), seed=r) for r in range(args.rounds))
        select = ('val', 'rank1')
    else:
        from run_verif import build_task
        D = dict(np.load(args.data or os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                                                   'processed', f'{args.dataset}_full_glbp.npz'),
                         allow_pickle=True))
        proto = 'eurocom' if args.dataset == 'biwi_a' else 'curtin'
        tasks = (build_task(D, r, proto) for r in range(args.rounds))
        select = ('val', 'AP')
    for r, task in enumerate(tasks):
        print(f'[{args.dataset}, extended gamma grid] round {r}', flush=True)
        out = run_task(task, METHODS, select, seed=r, grids=grids)
        results.append({'setting': {'gamma_grid': 'extended'}, 'round': r,
                        'results': {m: {k: v for k, v in d.items() if k != 'all'} for m, d in out.items()},
                        'all': {m: d['all'] for m, d in out.items()}})
        with open(os.path.join(RES, args.out), 'w') as f:
            json.dump(results, f, indent=1, default=float)
        print(f'round {r} done', flush=True)


if __name__ == '__main__':
    main()
