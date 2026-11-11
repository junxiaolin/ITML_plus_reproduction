"""Turn results/*.json into Markdown tables and figures (paper layout)."""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from itmlplus.experiment import ALL_METHODS
from itmlplus.protocol import significance_marks

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG = os.path.join(os.path.dirname(__file__), '..', 'figures')


def load(name):
    """name may join several result files with '+' (e.g. 5 main + 15 extra rounds)."""
    recs = []
    for part in name.split('+'):
        with open(os.path.join(RES, part)) as f:
            recs += json.load(f)
    return recs


def per_round(recs, evalset, measure, setting=None):
    out = {}
    for rec in recs:
        if setting is not None and rec['setting'] != setting:
            continue
        for m, d in rec['results'].items():
            out.setdefault(m, []).append(d['scores'][evalset][measure])
    return out


def fmt_table(rows, cols, title):
    """rows: list of (label, {method: values}). Bold = best mean or not
    significantly worse (paired t-test)."""
    lines = [f'**{title}**', '', '| | ' + ' | '.join(cols) + ' |', '|---' * (len(cols) + 1) + '|']
    for label, table in rows:
        table = {k: np.asarray(v) * 100 for k, v in table.items() if k in cols}
        best, tied = significance_marks(table)
        cells = []
        for m in cols:
            if m not in table:
                cells.append('-')
                continue
            s = f'{table[m].mean():.2f}±{table[m].std(ddof=1):.2f}'
            cells.append(f'**{s}**' if m in tied else s)
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def selected_hparams(recs):
    out = {}
    for rec in recs:
        for m, d in rec['results'].items():
            out.setdefault(m, []).append(d['hp'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verif', nargs='*', default=[])      # name=file.json
    ap.add_argument('--reid', default=None)
    ap.add_argument('--trainsize', default=None)
    ap.add_argument('--grid', nargs='*', default=[])      # name=file.json:evalset:measure
    ap.add_argument('--extended', nargs='*', default=[])  # name=file.json:eval:meas[:eval:meas]
    ap.add_argument('--timing', default=None)
    ap.add_argument('--out', default='summary.md', help="file in results/ ('' = do not write)")
    ap.add_argument('--figtag', default='', help='suffix for the Fig. 2/3 file names')
    args = ap.parse_args()
    md = []
    for spec in args.verif:
        name, fn = spec.split('=')
        recs = load(fn)
        cols = [m for m in ALL_METHODS if m in recs[0]['results']]
        md.append(fmt_table([('AP', per_round(recs, 'test', 'AP')),
                             ('AUC', per_round(recs, 'test', 'AUC'))], cols,
                            f'{name}: mean AP / AUC (%) ± std over {len(recs)} rounds'))
        hp = selected_hparams(recs)
        md.append('Selected hyper-parameters per round: ' + '; '.join(
            f'{m}: ' + ', '.join(str({k: float(f'{v:.3g}') for k, v in h.items()}) for h in hp[m])
            for m in cols if m != 'L2') + '\n')
    if args.reid:
        recs = load(args.reid)
        cols = [m for m in ALL_METHODS if m in recs[0]['results']]
        md.append(fmt_table([('Cross (Walking analogue)', per_round(recs, 'test_cross', 'rank1')),
                             ('Seq (Still analogue)', per_round(recs, 'test_seq', 'rank1'))], cols,
                            f'Re-identification: mean rank-1 (%) ± std over {len(recs)} rounds'))
    if args.trainsize:
        recs = load(args.trainsize)
        fr = sorted({r['setting']['train_frac'] for r in recs})
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        lines = []
        for ax, meas in zip(axs, ['AP', 'AUC']):
            for m, c, mk in [('ITML+', 'tab:red', 'o'), ('ITML', 'tab:blue', '^')]:
                vals = [np.array(per_round(recs, 'test', meas, {'train_frac': f})[m]) for f in fr]
                ax.errorbar(fr, [v.mean() for v in vals], [v.std(ddof=1) for v in vals], marker=mk,
                            color=c, capsize=3, label=m)
                lines.append(f'| {meas} {m} | ' + ' | '.join(
                    f'{v.mean() * 100:.2f}±{v.std(ddof=1) * 100:.2f}' for v in vals) + ' |')
            ax.set_xlabel('The ratio of training pairs')
            ax.set_ylabel(meas)
            ax.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f'fig3_trainsize{args.figtag}.png'), dpi=150)
        md.append('**ITML vs ITML+ with fewer training pairs (Fig. 3 analogue)**\n\n| | '
                  + ' | '.join(f'{f:g}' for f in fr) + ' |\n' + '|---' * (len(fr) + 1) + '|\n'
                  + '\n'.join(lines) + '\n')
    for spec in args.grid:
        # name=file.json:evalset:measure -> heat map of ITML+ over (gamma, lambda)
        name, rest = spec.split('=')
        fn, evalset, meas = rest.split(':')
        md.append(grid_figure(load(fn), name, evalset, meas))
    for spec in args.extended:
        name, rest = spec.split('=')
        fn, *pairs = rest.split(':')
        recs = load(fn)
        cols = [m for m in ALL_METHODS if m in recs[0]['results']]
        rows = [(f'{e} {m}', per_round(recs, e, m)) for e, m in zip(pairs[::2], pairs[1::2])]
        md.append(fmt_table(rows, cols, f'{name}, extended gamma grid (1e-4 ... 1e2): mean (%) ± std'))
        hp = selected_hparams(recs)
        md.append('Selected hyper-parameters per round: ' + '; '.join(
            f'{m}: ' + ', '.join(str({k: float(f'{v:.3g}') for k, v in h.items()}) for h in hp[m])
            for m in cols) + '\n')
    if args.timing:
        with open(os.path.join(RES, args.timing)) as f:
            T = json.load(f)
        md.append('**Training time (s), single CPU thread (Table IV analogue)**\n\n| | '
                  + ' | '.join(T['summary']) + ' |\n' + '|---' * (len(T['summary']) + 1) + '|\n| Time | '
                  + ' | '.join(f'{a:.2f}±{s:.2f}' for a, s in T['summary'].values()) + ' |\n')
    text = '\n'.join(md)
    print(text)
    if args.out:
        with open(os.path.join(RES, args.out), 'w') as f:
            f.write(text)


def grid_means(recs, method, evalset, meas):
    acc = {}
    for rec in recs:
        for row in rec['all'][method]:
            key = tuple(sorted(row['hp'].items()))
            acc.setdefault(key, []).append(row['scores'][evalset][meas])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def grid_figure(recs, name, evalset, meas):
    """Mean test score of ITML+ over the (gamma, lambda) grid, and ITML+ vs
    ITML for every gamma (lambda chosen per gamma on validation)."""
    plus = grid_means(recs, 'ITML+', evalset, meas)
    itml = grid_means(recs, 'ITML', evalset, meas)
    G = sorted({dict(k)['gamma'] for k in plus})
    L = sorted({dict(k)['lam'] for k in plus})
    A = np.array([[plus[tuple(sorted({'gamma': g, 'lam': l}.items()))] for l in L] for g in G]) * 100
    # per gamma: lambda selected on the validation set, round by round
    vname = 'val'
    vmeas = 'rank1' if meas == 'rank1' else 'AP'
    sel = {g: [] for g in G}
    for rec in recs:
        for g in G:
            rows = [r for r in rec['all']['ITML+'] if r['hp']['gamma'] == g]
            best = max(rows, key=lambda r: r['scores'][vname][vmeas])
            sel[g].append(best['scores'][evalset][meas])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    im = axs[0].imshow(A, origin='lower', aspect='auto', cmap='viridis')
    axs[0].set_xticks(range(len(L)), [f'{l:.0e}' for l in L], rotation=45, fontsize=7)
    axs[0].set_yticks(range(len(G)), [f'{g:.0e}' for g in G], fontsize=7)
    axs[0].set_xlabel('lambda')
    axs[0].set_ylabel('gamma')
    axs[0].set_title(f'ITML+ mean test {meas} (%)', fontsize=9)
    fig.colorbar(im, ax=axs[0])
    axs[1].plot(G, [itml[(('gamma', g),)] * 100 for g in G], 'b^-', label='ITML')
    axs[1].plot(G, [np.mean(sel[g]) * 100 for g in G], 'ro-', label='ITML+ (lambda by validation)')
    axs[1].plot(G, A[:, -1], 'r--', alpha=0.6, label=f'ITML+ (lambda={L[-1]:g})')
    axs[1].set_xscale('log')
    axs[1].set_xlabel('gamma')
    axs[1].set_ylabel(f'test {meas} (%)')
    axs[1].legend(fontsize=8)
    fig.suptitle(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f'grid_{name}.png'), dpi=150)
    lines = [f'**{name}: test {meas} (%) per gamma, mean over rounds**', '',
             '| gamma | ' + ' | '.join(f'{g:.0e}' for g in G) + ' |', '|---' * (len(G) + 1) + '|',
             '| ITML | ' + ' | '.join(f'{itml[(("gamma", g),)] * 100:.2f}' for g in G) + ' |',
             '| ITML+ (lambda by val.) | ' + ' | '.join(f'{np.mean(sel[g]) * 100:.2f}' for g in G) + ' |',
             f'| ITML+ (lambda={L[-1]:g}) | ' + ' | '.join(f'{a:.2f}' for a in A[:, -1]) + ' |']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
