"""Reading the Biwi Kinect Head Pose database and cropping RGB and depth faces
at a fixed physical size."""
import re
import zipfile

import cv2
import numpy as np

# sequence -> person (from the database README / HF loader)
SEQ2SUBJ = {'01': 'F01', '02': 'F02', '03': 'F03', '04': 'F04', '05': 'F05', '06': 'F06',
            '07': 'M01', '08': 'M02', '09': 'M03', '10': 'M04', '11': 'M05', '12': 'M06',
            '13': 'M07', '14': 'M08', '15': 'F03', '16': 'M09', '17': 'M10', '18': 'F05',
            '19': 'M11', '20': 'M12', '21': 'F02', '22': 'M01', '23': 'M13', '24': 'M14'}

FACE_W_MM, FACE_H_MM = 150.0, 171.4   # 105:120 aspect ratio, as the paper's crops
OUT_W, OUT_H = 105, 120


def read_depth_bin(buf):
    """Run-length encoded depth frame (int16 mm, 0 = no measurement)."""
    w, h = np.frombuffer(buf[:8], dtype=np.int32)
    out = np.zeros(w * h, dtype=np.int16)
    p, off = 0, 8
    while p < w * h:
        empty, full = np.frombuffer(buf[off:off + 8], dtype=np.int32)
        off += 8
        p += empty
        out[p:p + full] = np.frombuffer(buf[off:off + 2 * full], dtype=np.int16)
        off += 2 * full
        p += full
    return out.reshape(h, w)


def read_cal(text):
    rows = [r.split() for r in text.strip().splitlines()]
    K = np.array(rows[0:3], dtype=float)
    R = np.array(rows[6:9], dtype=float)
    t = np.array(rows[10], dtype=float)
    return K, R, t


def project(K, X):
    return np.array([K[0, 0] * X[0] / X[2] + K[0, 2], K[1, 1] * X[1] / X[2] + K[1, 2]])


def crop(img, center, f, z, interp):
    """Crop a FACE_W_MM x FACE_H_MM box at distance z around pixel `center`."""
    w = f * FACE_W_MM / z
    h = f * FACE_H_MM / z
    x0, y0 = center[0] - w / 2, center[1] - h / 2
    A = np.array([[OUT_W / w, 0, -x0 * OUT_W / w], [0, OUT_H / h, -y0 * OUT_H / h]])
    return cv2.warpAffine(img, A, (OUT_W, OUT_H), flags=interp, borderMode=cv2.BORDER_REPLICATE)


def fill_depth(d):
    """Fill missing depth (0) by inpainting."""
    d = d.astype(np.float32)
    miss = (d <= 0).astype(np.uint8)
    if miss.any() and (~miss.astype(bool)).any():
        lo, hi = d[d > 0].min(), d[d > 0].max()
        s = (d - lo) / max(hi - lo, 1.0)
        s[miss.astype(bool)] = 0
        s = cv2.inpaint((s * 65535).astype(np.uint16), miss, 3, cv2.INPAINT_NS).astype(np.float32) / 65535
        d = s * (hi - lo) + lo
    return d


def depth_to_gray(d, z_ref, mm_per_level=1.0):
    """Depth (mm) -> uint8 image, 128 at z_ref, nearer = brighter."""
    return np.clip(128.0 + (z_ref - d) / mm_per_level, 0, 255).astype(np.uint8)


class BiwiZip:
    """Reads frames directly from faces_0.zip (no extraction needed)."""

    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
        self.frames, self.prefix = {}, ''
        names = set(self.z.namelist())
        for n in names:
            m = re.match(r'(.*?)(\d\d)/frame_(\d+)_rgb\.png$', n)
            if m:
                pre, seq, fr = m.group(1), m.group(2), int(m.group(3))
                need = [f'{pre}{seq}/frame_{fr:05d}_{s}' for s in ('rgb.png', 'depth.bin', 'pose.txt')]
                if all(x in names for x in need):
                    self.prefix = pre
                    self.frames.setdefault(seq, []).append(fr)
        for k in self.frames:
            self.frames[k].sort()

    def _find(self, seq, suffix):
        return f'{self.prefix}{seq}/{suffix}'

    def cal(self, seq):
        return (read_cal(self.z.read(self._find(seq, 'rgb.cal')).decode()),
                read_cal(self.z.read(self._find(seq, 'depth.cal')).decode()))

    def rgb(self, seq, fr):
        buf = np.frombuffer(self.z.read(self._find(seq, f'frame_{fr:05d}_rgb.png')), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def depth(self, seq, fr):
        return read_depth_bin(self.z.read(self._find(seq, f'frame_{fr:05d}_depth.bin')))

    def pose(self, seq, fr):
        txt = self.z.read(self._find(seq, f'frame_{fr:05d}_pose.txt')).decode().split()
        v = np.array(txt, dtype=float)
        return v[:9].reshape(3, 3), v[9:12]


def euler_from_R(R):
    """(pitch, yaw, roll) in degrees, convention of the Biwi annotation files."""
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1)))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll
