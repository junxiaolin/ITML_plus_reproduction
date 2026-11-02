"""Fig. 4 analogue: distances of similar and dissimilar pairs under the ITML and
ITML+ metrics (re-ID data, round 0)."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from itmlplus.metric import ITML, ITMLPlus, pair_distances
from itmlplus.protocol import sample_pairs
from itmlplus.uw_rgbd import KDESExtractor, index
from run_reid import build_task, make_round

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG = os.path.join(os.path.dirname(__file__), '..', 'figures')


def overlap_stats(d, y):
    """Pairs between min(dissimilar distance) and max(similar distance)."""
    dp, dn = d[y == 1], d[y == -1]
    lo, hi = dn.min(), dp.max()
    return int(((d > lo) & (d < hi)).sum()), lo, hi


def main():
    with open(os.path.join(RES, 'reid_uw.json')) as f:
        recs = json.load(f)
    hp_itml = recs[0]['results']['ITML']['hp']
    hp_plus = recs[0]['results']['ITML+']['hp']
    inst = index()
    split, rng = make_round(inst, 0)
    task = build_task(KDESExtractor(), split, rng, seed=0)
    u, l = task.bounds_x
    Xv, lv = task.evals['val']
    vpairs, vy = sample_pairs(lv, None, 4750, np.random.RandomState(7))
    pos = np.flatnonzero(task.y == 1)[:200]
    neg = np.flatnonzero(task.y == -1)[:200]
    tsel = np.r_[pos, neg]
    vsel = np.r_[np.flatnonzero(vy == 1)[:200], np.flatnonzero(vy == -1)[:200]]
    g0 = hp_itml['gamma']
    configs = [
        (f'ITML (gamma={g0:g}, selected)', ITML(gamma=g0, bounds=(u, l))),
        (f'ITML+ (gamma={hp_plus["gamma"]:g}, lambda={hp_plus["lam"]:g}, selected)',
         ITMLPlus(gamma=hp_plus['gamma'], lam=hp_plus['lam'], bounds=(u, l))),
        (f'ITML+ (gamma={hp_plus["gamma"]:g}, lambda=100)', ITMLPlus(gamma=hp_plus['gamma'], lam=100.0, bounds=(u, l))),
        ('ITML (gamma=0.01)', ITML(gamma=0.01, bounds=(u, l))),
        ('ITML+ (gamma=0.01, lambda=100)', ITMLPlus(gamma=0.01, lam=100.0, bounds=(u, l))),
    ]
    fig, axs = plt.subplots(2, 1, figsize=(5.2, 6.4))
    fig2, axs2 = plt.subplots(2, 1, figsize=(5.2, 6.4))
    panels = {0: axs[0], 1: axs[1], 3: axs2[0], 4: axs2[1]}  # paper layout + loose-slack setting
    lines = ['| metric | training pairs between dashed lines (of 400) | training AUC | '
             'validation pairs between dashed lines (of 400) | validation AUC |', '|---|---|---|---|---|']
    for k, (name, model) in enumerate(configs):
        if isinstance(model, ITMLPlus):
            M = model.fit(task.Xtr, task.Ztr, task.pairs, task.y).M_
        else:
            M = model.fit(task.Xtr, task.pairs, task.y).M_
        dt = pair_distances(M, task.Xtr, task.pairs[tsel])
        dv = pair_distances(M, Xv, vpairs[vsel])
        nt, lo, hi = overlap_stats(dt, task.y[tsel])
        nv, _, _ = overlap_stats(dv, vy[vsel])
        auc_t = roc_auc_score(task.y == 1, -pair_distances(M, task.Xtr, task.pairs))
        auc_v = roc_auc_score(vy == 1, -pair_distances(M, Xv, vpairs))
        lines.append(f'| {name} | {nt} | {auc_t * 100:.2f} | {nv} | {auc_v * 100:.2f} |')
        if k in panels:
            ax = panels[k]
            ax.scatter(np.arange(1, 201), dt[:200], marker='*', s=14, c='red', label='positive pair')
            ax.scatter(np.arange(201, 401), dt[200:], marker='o', s=10, facecolors='none',
                       edgecolors='blue', label='negative pair')
            for yv in (u, l):
                ax.axhline(yv, color='k', lw=1)
            for yv in (lo, hi):
                ax.axhline(yv, color='k', lw=1, ls='--')
            ax.set_title(f'({"abab"[min(k, 3) if k < 2 else k - 3]}) {name}', fontsize=8)
            ax.set_xlabel('Index of the training pairs')
            ax.set_ylabel('Distance')
    for f_, a_, fn in [(fig, axs, 'fig4_distances.png'), (fig2, axs2, 'fig4_distances_gamma0.01.png')]:
        a_[0].legend(loc='upper left', fontsize=8)
        f_.tight_layout()
        f_.savefig(os.path.join(FIG, fn), dpi=150)
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(RES, 'fig4_distances.md'), 'w') as f:
        f.write(text + '\n')


if __name__ == '__main__':
    main()
