"""Table IV analogue: training time of ITML and ITML+ on one CPU thread,
 with the hyper-parameters selected in the main experiment."""
import argparse
import json
import os
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from itmlplus.experiment import fit_method
from run_verif import build_task

RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--protocol', default='eurocom')
    ap.add_argument('--out', default='timing.json')
    args = ap.parse_args()
    D = dict(np.load(args.data, allow_pickle=True))
    with open(os.path.join(RES, args.results)) as f:
        recs = [r for r in json.load(f) if r['setting'] == {}]
    times = {m: [] for m in ['ITML', 'ITML+']}
    with threadpool_limits(1):
        for rec in recs:
            task = build_task(D, rec['round'], args.protocol)
            for m in times:
                hp = rec['results'][m]['hp']
                fit_method(task, m, hp, seed=rec['round'])  # warm-up (numba JIT cache)
                _, t = fit_method(task, m, hp, seed=rec['round'])
                times[m].append(t)
                print(f'round {rec["round"]} {m:7s} hp={hp} {t:.2f}s', flush=True)
    summary = {m: (float(np.mean(v)), float(np.std(v, ddof=1))) for m, v in times.items()}
    print(' | '.join(f'{m}: {a:.2f}±{s:.2f}s' for m, (a, s) in summary.items()))
    with open(os.path.join(RES, args.out), 'w') as f:
        json.dump({'times': times, 'summary': summary}, f, indent=1)


if __name__ == '__main__':
    main()
