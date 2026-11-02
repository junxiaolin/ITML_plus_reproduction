"""Re-light every RGB face crop with a random light computed from its depth
normals (depth unchanged), and write a control copy without re-lighting."""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from itmlplus.biwi import FACE_W_MM, OUT_W
from itmlplus.features import glbp

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
FIG = os.path.join(os.path.dirname(__file__), '..', 'figures')
MM_PER_PX = FACE_W_MM / OUT_W  # crop pixel size in mm
MM_PER_LEVEL = 1.0             # depth_to_gray(): 1 gray level per mm


def normals(d8):
    z = cv2.GaussianBlur(d8.astype(np.float32), (0, 0), 5.0) * MM_PER_LEVEL  # nearer = larger; Kinect noise
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8.0 / MM_PER_PX
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / 8.0 / MM_PER_PX
    n = np.dstack([-gx, -gy, np.ones_like(z)])
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def relight(rgb, d8, rng, ambient=0.25):
    n = normals(d8)
    theta = np.radians(rng.uniform(25, 70))
    phi = rng.uniform(0, 2 * np.pi)
    L = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    shade = ambient + (1 - ambient) * np.clip(n @ L, 0, None)
    ref = ambient + (1 - ambient) * np.clip(n[..., 2], 0, None)  # roughly frontal room light
    gain = rng.uniform(0.6, 1.4)
    gamma = np.exp(rng.uniform(np.log(0.6), np.log(1.6)))
    ratio = np.clip(shade / ref, 0.2, 2.0)
    out = rgb.astype(np.float32) / 255.0 * ratio[..., None] * gain
    return (np.clip(out, 0, 1) ** gamma * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='e.g. biwi_a_full')
    args = ap.parse_args()
    C = np.load(os.path.join(DATA, f'{args.name}_crops.npz'))
    G = dict(np.load(os.path.join(DATA, f'{args.name}_glbp.npz'), allow_pickle=True))
    rng = np.random.RandomState(0)
    relit = np.stack([relight(c, d, rng) for c, d in zip(C['rgb'], C['depth'])])
    for tag, imgs in [('relit', relit), ('nolight', C['rgb'])]:
        F = np.array([glbp(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) for im in imgs], np.float32)
        out = dict(G)
        out['rgb'] = F
        np.savez_compressed(os.path.join(DATA, f'{args.name}_{tag}_glbp.npz'), **out)
        print(tag, F.shape)
    subj = G['subject']
    tiles = []
    for s in sorted(set(subj))[:3]:
        ii = np.flatnonzero(subj == s)[np.linspace(0, (subj == s).sum() - 1, 8).round().astype(int)]
        tiles.append(np.hstack([C['rgb'][i] for i in ii]))
        tiles.append(np.hstack([relit[i] for i in ii]))
    cv2.imwrite(os.path.join(FIG, f'{args.name}_relit_samples.png'), np.vstack(tiles))


if __name__ == '__main__':
    main()
