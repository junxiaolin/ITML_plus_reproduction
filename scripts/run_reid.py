"""Table III analogue: re-identification on Washington RGB-D Object (KDES + EMK).
Test sets: Seq (same video) and Cross (other camera elevations)."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from itmlplus.experiment import ALL_METHODS, Task, run_task
from itmlplus.metric import compute_bounds
from itmlplus.protocol import Preprocess, sample_pairs
from itmlplus.uw_rgbd import KDESExtractor, index

RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def uniform_shots(frames, k):
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    return [frames[i][2] for i in idx]


def make_round(inst, r, n_shots=20):
    rng = np.random.RandomState(1000 + r)
    names = np.array(list(inst))
    perm = rng.permutation(len(names))
    tr, va, te = names[perm[:25]], names[perm[25:50]], names[perm[50:]]
    split = {}
    split['train'] = [(p, i) for i, n in enumerate(tr) for p in uniform_shots(inst[n], n_shots)]
    split['val'] = [(p, i) for i, n in enumerate(va) for p in uniform_shots(inst[n], n_shots)]
    seq, cross = [], []
    for i, n in enumerate(te):
        fr = inst[n]
        vids = sorted({v for v, _, _ in fr})
        v = vids[rng.randint(len(vids))]
        fv = [f for f in fr if f[0] == v]
        s = rng.randint(len(fv))
        seq += [(fv[(s + k) % len(fv)][2], i) for k in range(n_shots)]
        f1 = [f for f in fr if f[0] == 1]
        f24 = [f for f in fr if f[0] != 1]
        cross.append((f1[rng.randint(len(f1))][2], i))
        cross += [(p, i) for p in uniform_shots(f24, n_shots - 1)]
    split['test_seq'], split['test_cross'] = seq, cross
    return split, rng


def probe_gallery(X, labels, n_shots=20):
    """First image of every identity = probe, the remaining ones = gallery."""
    first = np.arange(0, len(labels), n_shots)
    rest = np.setdiff1d(np.arange(len(labels)), first)
    return X[first], labels[first], X[rest], labels[rest]


def build_task(ext, split, rng, seed):
    ptr = [p for p, _ in split['train']]
    ltr = np.array([l for _, l in split['train']])
    feats = {}
    for name in split:
        paths = [p for p, _ in split[name]]
        feats[name] = ext.emk_features(paths, 'rgb', ptr, seed=seed)
    dep = ext.emk_features(ptr, 'depth', ptr, seed=seed)
    px, pz = Preprocess(150).fit(feats['train']), Preprocess(150).fit(dep)
    Xtr, Ztr = px(feats['train']), pz(dep)
    pairs, y = sample_pairs(ltr, None, 4750, rng)
    evals = {'val': (px(feats['val']), np.array([l for _, l in split['val']]))}
    for name in ['test_seq', 'test_cross']:
        lab = np.array([l for _, l in split[name]])
        evals[name] = probe_gallery(px(feats[name]), lab)
    return Task(Xtr=Xtr, Ztr=Ztr, ytr_labels=ltr, pairs=pairs, y=y,
                bounds_x=compute_bounds(Xtr), bounds_z=compute_bounds(Ztr),
                evals=evals, kind='reid')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS)
    ap.add_argument('--out', default='reid_uw.json')
    ap.add_argument('--relit', action='store_true',
                    help='controlled experiment: RGB crops re-lit with random lights (depth unchanged)')
    ap.add_argument('--start', type=int, default=0)
    args = ap.parse_args()
    inst = index()
    print(f'{len(inst)} instances, {sum(len(v) for v in inst.values())} frames')
    ext = KDESExtractor(relight=args.relit)
    results = []
    for r in range(args.start, args.rounds):
        t0 = time.time()
        split, rng = make_round(inst, r)
        task = build_task(ext, split, rng, seed=r)
        print(f'round {r}: features {time.time() - t0:.1f}s, {len(task.y)} pairs, '
              f'u,l = {task.bounds_x[0]:.3f},{task.bounds_x[1]:.3f}', flush=True)
        out = run_task(task, args.methods, ('val', 'rank1'), seed=r)
        results.append({'setting': {}, 'round': r,
                        'results': {m: {k: v for k, v in d.items() if k != 'all'} for m, d in out.items()},
                        'all': {m: d['all'] for m, d in out.items()}})
        with open(os.path.join(RES, args.out), 'w') as f:
            json.dump(results, f, indent=1, default=float)
        print(f'round {r} done in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
