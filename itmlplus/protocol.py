"""Pair sampling, preprocessing and evaluation measures (Sec. V)."""
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score


def positive_pairs(labels):
    labels = np.asarray(labels)
    out = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        iu = np.triu_indices(len(idx), 1)
        out.append(np.stack([idx[iu[0]], idx[iu[1]]], 1))
    return np.concatenate(out)


def sample_pairs(labels, n_pos, n_neg, rng):
    """All (or n_pos random) within-class pairs and n_neg random
    between-class pairs (no duplicates)."""
    labels = np.asarray(labels)
    pos = positive_pairs(labels)
    if n_pos is not None and n_pos < len(pos):
        pos = pos[rng.choice(len(pos), n_pos, replace=False)]
    n = len(labels)
    seen, neg = set(), []
    while len(neg) < n_neg:
        i, j = rng.randint(0, n, 2)
        if labels[i] == labels[j]:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        neg.append(key)
    pairs = np.concatenate([pos, np.array(neg, dtype=np.int64)])
    y = np.r_[np.ones(len(pos), np.int64), -np.ones(n_neg, np.int64)]
    return pairs, y


class Preprocess:
    """PCA fitted on the training images, then scaled so that the mean squared
    distance between training samples is 2 (RGB and depth on the same scale)."""

    def __init__(self, dim=150):
        self.dim = dim

    def fit(self, F):
        self.pca = PCA(n_components=min(self.dim, F.shape[0] - 1, F.shape[1]), random_state=0).fit(F)
        T = self.pca.transform(F)
        self.scale = np.sqrt((T.var(0)).sum())
        return self

    def __call__(self, F):
        return self.pca.transform(F) / self.scale


def verification_scores(y, score):
    """AP and AUC with similar pairs as positives (score: higher = more similar)."""
    return average_precision_score(y == 1, score), roc_auc_score(y == 1, score)


def rank1(dist, probe_labels, gallery_labels):
    nn = np.argmin(dist, axis=1)
    return float(np.mean(gallery_labels[nn] == probe_labels))


def cmc(dist, probe_labels, gallery_labels, topk=20):
    order = np.argsort(dist, axis=1)
    hits = gallery_labels[order] == probe_labels[:, None]
    first = np.argmax(hits, axis=1)
    return np.array([np.mean(first < k) for k in range(1, topk + 1)])


def significance_marks(table, higher_is_better=True, alpha=0.05):
    """Best method by mean, and the methods not significantly worse than it
    (paired t-test)."""
    means = {k: np.mean(v) for k, v in table.items()}
    best = max(means, key=means.get) if higher_is_better else min(means, key=means.get)
    tied = {best}
    for k, v in table.items():
        if k == best:
            continue
        diff = np.asarray(table[best]) - np.asarray(v)
        if np.allclose(diff, 0):
            tied.add(k)
            continue
        p = stats.ttest_rel(table[best], v).pvalue
        if not (p < alpha):
            tied.add(k)
    return best, tied
