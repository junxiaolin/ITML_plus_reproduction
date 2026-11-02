"""Build BIWI-A and BIWI-B (face crops of RGB and depth) and extract G-LBP features."""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from itmlplus.biwi import OUT_H, OUT_W, SEQ2SUBJ, BiwiZip, crop, depth_to_gray, fill_depth, project
from itmlplus.features import glbp

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
FACE_OFFSET_MM = 70.0  # head centre -> face centre, along the face normal


def face_crops(bz, seq, fr, cal):
    (Kr, Rr, tr), (Kd, _, _) = cal
    R, c = bz.pose(seq, fr)
    # face centre (depth-camera frame); the pose.txt rotation maps camera -> head
    # coordinates, so the face normal in camera coordinates is R' [0, 0, -1]
    cf = c + R.T @ np.array([0.0, 0.0, -FACE_OFFSET_MM])
    rgb = bz.rgb(seq, fr)
    dep = bz.depth(seq, fr).astype(np.float32)
    Xr = Rr @ cf + tr
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    g = crop(gray, project(Kr, Xr), Kr[0, 0], Xr[2], cv2.INTER_AREA)
    dpt = crop(fill_depth(dep), project(Kd, cf), Kd[0, 0], cf[2], cv2.INTER_NEAREST)
    centre = dpt[OUT_H // 4:3 * OUT_H // 4, OUT_W // 4:3 * OUT_W // 4]
    d8 = depth_to_gray(dpt, np.median(centre))
    col = crop(rgb, project(Kr, Xr), Kr[0, 0], Xr[2], cv2.INTER_AREA)
    return g, d8, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default=os.path.join(DATA, 'raw', 'biwi_headpose', 'BIWI', 'data', 'faces_0.zip'))
    ap.add_argument('--ann', default=os.path.join(DATA, 'raw', 'biwi_headpose', 'extracted', 'db_annotations'))
    ap.add_argument('--tag', default='', help='suffix of the output files, e.g. "full"')
    args = ap.parse_args()
    bz = BiwiZip(args.zip)
    rows = []  # (subject, seq, frame, pitch, yaw, roll)
    for seq in sorted(bz.frames):
        for fr in bz.frames[seq]:
            fn = os.path.join(args.ann, seq, f'frame_{fr:05d}_pose.bin')
            if not os.path.exists(fn):
                continue
            a = np.fromfile(fn, dtype=np.float32)
            rows.append((SEQ2SUBJ[seq], seq, fr, a[3], a[4], a[5]))
    print(len(rows), 'annotated frames')
    # (max |yaw|, max |pitch|, images per person)
    configs = {'biwi_a': (30, 30, 22), 'biwi_b': (60, 45, 95)}
    cals = {seq: bz.cal(seq) for seq in bz.frames}
    tag = f'_{args.tag}' if args.tag else ''
    for name, (ymax, pmax, n_img) in configs.items():
        sel = []
        for s in sorted(set(SEQ2SUBJ.values())):
            el = [r for r in rows if r[0] == s and abs(r[4]) <= ymax and abs(r[3]) <= pmax]
            idx = np.unique(np.linspace(0, len(el) - 1, n_img).round().astype(int))
            sel += [el[i] for i in idx]
            print(f'  {name} {s}: {len(el)} eligible frames, {len(idx)} used')
        F_rgb, F_dep, crops = [], [], []
        for (s, seq, fr, p, y, r) in sel:
            g, d8, col = face_crops(bz, seq, fr, cals[seq])
            F_rgb.append(glbp(g))
            F_dep.append(glbp(d8))
            crops.append((col, d8))
        meta = np.array(sel, dtype=object)
        np.savez_compressed(os.path.join(DATA, 'processed', f'{name}{tag}_glbp.npz'),
                            rgb=np.array(F_rgb, np.float32), depth=np.array(F_dep, np.float32),
                            subject=meta[:, 0].astype(str), seq=meta[:, 1].astype(str),
                            frame=meta[:, 2].astype(int), pitch=meta[:, 3].astype(float),
                            yaw=meta[:, 4].astype(float), roll=meta[:, 5].astype(float))
        np.savez_compressed(os.path.join(DATA, 'processed', f'{name}{tag}_crops.npz'),
                            rgb=np.array([c for c, _ in crops]), depth=np.array([d for _, d in crops]))
        print(name, len(sel), 'images saved')


if __name__ == '__main__':
    main()
