"""Rounds 5-19 of a face experiment (rounds 0-4 are the main experiment)."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from itmlplus.experiment import ALL_METHODS, run_task
from run_verif import build_task

RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--start', type=int, default=5)
    ap.add_argument('--end', type=int, default=20)
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    D = dict(np.load(args.data, allow_pickle=True))
    results = []
    for r in range(args.start, args.end):
        task = build_task(D, r, args.protocol)
        out = run_task(task, args.methods, ('val', 'AP'), seed=r, verbose=False)
        results.append({'setting': {}, 'round': r,
                        'results': {m: {k: v for k, v in d.items() if k != 'all'} for m, d in out.items()},
                        'all': {m: d['all'] for m, d in out.items()}})
        with open(os.path.join(RES, args.out), 'w') as f:
            json.dump(results, f, indent=1, default=float)
        print(f'round {r} done: ' + ', '.join(f'{m} {out[m]["scores"]["test"]["AP"] * 100:.2f}'
                                              for m in args.methods), flush=True)


if __name__ == '__main__':
    main()
