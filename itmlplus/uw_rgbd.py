"""Washington RGB-D Object data (eval subset) for re-identification; object
instances play the role of persons."""
import glob
import os
import re
import zlib
from collections import defaultdict

import cv2
import numpy as np
import torch

from .features import EMK, GradientKDES

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'uw_rgbd',
                    'extracted', 'rgbd-dataset_eval')
_PAT = re.compile(r'(.+)_(\d+)_(\d+)_(\d+)_crop\.png$')


def index(root=ROOT):
    """instance name -> list of (video, frame, rgb path) sorted by video/frame."""
    inst = defaultdict(list)
    for f in glob.glob(os.path.join(root, '*', '*', '*_crop.png')):
        m = _PAT.match(os.path.basename(f))
        cat, num, vid, frm = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        inst[f'{cat}_{num}'].append((vid, frm, f))
    return {k: sorted(v) for k, v in sorted(inst.items())}


def load_gray(path, max_side=300):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return _limit(im, max_side)


def load_depth(path, max_side=300, rng_mm=300.0):
    """Depth crop as a [0, 1] image: 0.5 at the median object depth, nearer = brighter."""
    d = cv2.imread(path.replace('_crop.png', '_depthcrop.png'), cv2.IMREAD_UNCHANGED).astype(np.float32)
    valid = d > 0
    out = np.zeros_like(d)
    if valid.any():
        med = np.median(d[valid])
        out[valid] = np.clip(0.5 - (d[valid] - med) / rng_mm, 0.0, 1.0)
    return _limit(out, max_side)


def load_gray_relit(path, max_side=300, ambient=0.25, f_px=570.0):
    """RGB crop re-lit by a random light (seeded by the file name), with Lambertian
    shading from the depth normals."""
    rgb = cv2.imread(path).astype(np.float32) / 255.0
    d = cv2.imread(path.replace('_crop.png', '_depthcrop.png'), cv2.IMREAD_UNCHANGED).astype(np.float32)
    valid = d > 0
    if valid.sum() > 10:
        med = np.median(d[valid])
        d[~valid] = med
        z = -cv2.GaussianBlur(d, (0, 0), 2.0)  # nearer = larger
        mm_per_px = med / f_px
        gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8.0 / mm_per_px
        gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / 8.0 / mm_per_px
        n = np.dstack([-gx, -gy, np.ones_like(z)])
        n /= np.linalg.norm(n, axis=2, keepdims=True)
    else:
        n = np.dstack([np.zeros_like(d), np.zeros_like(d), np.ones_like(d)])
    rng = np.random.RandomState(zlib.crc32(os.path.basename(path).encode()))
    theta, phi = np.radians(rng.uniform(25, 70)), rng.uniform(0, 2 * np.pi)
    L = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    shade = ambient + (1 - ambient) * np.clip(n @ L, 0, None)
    ref = ambient + (1 - ambient) * np.clip(n[..., 2], 0, None)
    gain = rng.uniform(0.6, 1.4)
    gamma = np.exp(rng.uniform(np.log(0.6), np.log(1.6)))
    out = np.clip(rgb * np.clip(shade / ref, 0.2, 2.0)[..., None] * gain, 0, 1) ** gamma
    gray = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return _limit(gray, max_side)


def _limit(im, max_side):
    h, w = im.shape
    s = max_side / max(h, w)
    if s < 1:
        im = cv2.resize(im, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return im


class KDESExtractor:
    """Patch-level gradient KDES with an in-memory cache, EMK pooling on demand."""

    def __init__(self, device=None, relight=False):
        self.kdes = GradientKDES(device=device)
        self.cache = {}
        self.relight = relight

    def patches(self, path, modality):
        key = (path, modality)
        if key not in self.cache:
            if modality == 'rgb':
                img = load_gray_relit(path) if self.relight else load_gray(path)
            else:
                img = load_depth(path)
            f, c = self.kdes(img)
            self.cache[key] = (f.half(), c.half())
        f, c = self.cache[key]
        return f.float(), c.float()

    def emk_features(self, paths, modality, codebook_paths, n_words=1000, seed=0):
        levels = (1, 2, 4) if modality == 'rgb' else (1, 2, 3)
        pool = torch.cat([self.patches(p, modality)[0] for p in codebook_paths])
        emk = EMK(n_words=n_words, levels=levels, seed=seed).fit(pool)
        out = []
        for p in paths:
            f, c = self.patches(p, modality)
            v = emk.transform(f, c)
            out.append((v / (v.norm() + 1e-12)).cpu().numpy())
        return np.stack(out)
