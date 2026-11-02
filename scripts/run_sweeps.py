"""ITML and ITML+ with 30 sweeps instead of 10."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from itmlplus.experiment import run_task
from run_verif import build_task

RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--sweeps', type=int, default=30)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    D = dict(np.load(args.data, allow_pickle=True))
    results = []
    for r in range(args.rounds):
        task = build_task(D, r, args.protocol)
        task.max_sweeps = args.sweeps
        out = run_task(task, ['ITML', 'ITML-S', 'ITML+'], ('val', 'AP'), seed=r)
        results.append({'setting': {'sweeps': args.sweeps}, 'round': r,
                        'results': {m: {k: v for k, v in d.items() if k != 'all'} for m, d in out.items()},
                        'all': {m: d['all'] for m, d in out.items()}})
        with open(os.path.join(RES, args.out), 'w') as f:
            json.dump(results, f, indent=1, default=float)
        print(f'round {r} done', flush=True)


if __name__ == '__main__':
    main()
