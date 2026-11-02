"""G-LBP (faces, 8x7 cells -> 6888 dims) and gradient KDES + EMK (objects)."""
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# G-LBP
# ----------------------------------------------------------------------------
def _u2_table(P=8):
    table = np.zeros(2 ** P, dtype=np.int64)
    nxt = 0
    for code in range(2 ** P):
        bits = [(code >> k) & 1 for k in range(P)]
        transitions = sum(bits[k] != bits[(k + 1) % P] for k in range(P))
        if transitions <= 2:
            table[code] = nxt
            nxt += 1
        else:
            table[code] = -1
    table[table < 0] = nxt  # all non-uniform patterns share the last bin
    return table, nxt + 1


_U2, _N_U2 = _u2_table()
# neighbours of LBP_{8,1} in orientation order (right, counter-clockwise)
_OFFSETS = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]


def glbp(img, grid=(8, 7)):
    """G-LBP descriptor of a uint8-valued image (grayscale or scaled depth)."""
    I = np.asarray(img, dtype=np.int32)
    H, W = I.shape
    pad = np.pad(I, 1, mode='edge')
    c = pad[1:H + 1, 1:W + 1]
    code = np.zeros((H, W), dtype=np.int32)
    odd = []
    for p, (dy, dx) in enumerate(_OFFSETS):
        g = pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        code |= ((g - c) >= 0).astype(np.int32) << p
        if p < 4:  # the other 4 orientations are the negatives of these
            odd.append(np.clip(g - c, -8, 7) + 8)
    lab = _U2[code]
    gh, gw = grid
    ch, cw = H // gh, W // gw
    feats = []
    for r in range(gh):
        for q in range(gw):
            sl = (slice(r * ch, (r + 1) * ch), slice(q * cw, (q + 1) * cw))
            n = float(ch * cw)
            feats.append(np.bincount(lab[sl].ravel(), minlength=_N_U2) / n)
            for o in odd:
                feats.append(np.bincount(o[sl].ravel(), minlength=16) / n)
    return np.concatenate(feats)


# ----------------------------------------------------------------------------
# Gradient kernel descriptors + EMK
# ----------------------------------------------------------------------------
class GradientKDES:
    """Dense gradient kernel descriptors on 16x16 patches (25 orientation x 5x5
    position basis vectors, kernel PCA to 200 dims)."""

    def __init__(self, patch=16, stride=8, n_orient=25, n_pos=5, gamma_o=5.0,
                 gamma_p=3.0, dim=200, eps_g=0.8, device=None):
        self.patch, self.stride, self.eps_g = patch, stride, eps_g
        self.gamma_o = gamma_o
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        ang = 2 * np.pi * np.arange(n_orient) / n_orient
        Bo = np.stack([np.sin(ang), np.cos(ang)], 1)
        t = np.linspace(0, 1, n_pos)
        Bp = np.stack(np.meshgrid(t, t, indexing='ij'), -1).reshape(-1, 2)
        Ko = np.exp(-gamma_o * ((Bo[:, None] - Bo[None]) ** 2).sum(-1))
        Kp = np.exp(-gamma_p * ((Bp[:, None] - Bp[None]) ** 2).sum(-1))
        K = np.kron(Ko, Kp)  # index a * n_pos^2 + b
        e, V = np.linalg.eigh(K)
        idx = np.argsort(e)[::-1][:dim]
        G = V[:, idx] / np.sqrt(e[idx])
        pos = np.arange(patch) / (patch - 1.0)
        P = np.stack(np.meshgrid(pos, pos, indexing='ij'), -1).reshape(-1, 2)
        kp_pix = np.exp(-gamma_p * ((P[:, None] - Bp[None]) ** 2).sum(-1))  # (patch^2, 25)
        n_p = Bp.shape[0]
        w = kp_pix.T.reshape(n_p, 1, patch, patch)
        dev = self.device
        self.Bo = torch.tensor(Bo, dtype=torch.float32, device=dev)
        self.G = torch.tensor(G, dtype=torch.float32, device=dev)
        # grouped conv: input channel a (orientation) -> outputs a*n_p + b
        self.wpos = torch.tensor(np.tile(w, (Bo.shape[0], 1, 1, 1)), dtype=torch.float32, device=dev)
        self.ones = torch.ones(1, 1, patch, patch, device=dev)
        self.n_o = Bo.shape[0]

    @torch.no_grad()
    def __call__(self, img):
        """img: HxW float array in [0, 1]. Returns patch descriptors and patch centres."""
        I = torch.tensor(img, dtype=torch.float32, device=self.device)
        H, W = I.shape
        if H < self.patch or W < self.patch:
            I = F.pad(I[None, None], (0, max(0, self.patch - W), 0, max(0, self.patch - H)),
                      mode='replicate')[0, 0]
            H, W = I.shape
        Ip = F.pad(I[None, None], (1, 1, 1, 1), mode='replicate')[0, 0]
        gx = (Ip[1:-1, 2:] - Ip[1:-1, :-2]) / 2.0
        gy = (Ip[2:, 1:-1] - Ip[:-2, 1:-1]) / 2.0
        mag = torch.sqrt(gx ** 2 + gy ** 2)
        th = torch.atan2(gy, gx)
        o = torch.stack([torch.sin(th), torch.cos(th)], 0)  # (2,H,W)
        ko = torch.exp(-self.gamma_o * ((o[None] - self.Bo[:, :, None, None]) ** 2).sum(1))  # (25,H,W)
        inp = (mag[None] * ko)[None]
        S = F.conv2d(inp, self.wpos, stride=self.stride, groups=self.n_o)[0]  # (625,h,w)
        nrm = torch.sqrt(F.conv2d((mag ** 2)[None, None], self.ones, stride=self.stride)[0, 0] + self.eps_g)
        S = S / nrm[None]
        Fd = torch.einsum('chw,cd->hwd', S, self.G)
        h, w = Fd.shape[:2]
        cy = (torch.arange(h, device=self.device) * self.stride + self.patch / 2.0) / H
        cx = (torch.arange(w, device=self.device) * self.stride + self.patch / 2.0) / W
        centers = torch.stack(torch.meshgrid(cy, cx, indexing='ij'), -1).reshape(-1, 2)
        return Fd.reshape(-1, Fd.shape[-1]), centers.clamp(max=0.999)


def kmeans_torch(X, k, n_iter=50, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(n_iter):
        d = torch.cdist(X, C)
        a = d.argmin(1)
        newC = torch.zeros_like(C).index_add_(0, a, X)
        cnt = torch.bincount(a, minlength=k).float()
        empty = cnt == 0
        newC = newC / cnt.clamp(min=1)[:, None]
        if empty.any():  # re-seed empty clusters with random points
            newC[empty] = X[torch.randint(0, X.shape[0], (int(empty.sum()),), generator=g,
                                          device=X.device)]
        C = newC
    return C


class EMK:
    """Efficient match kernel pooling over a k-means codebook + spatial pyramid."""

    def __init__(self, n_words=1000, levels=(1, 2, 4), seed=0):
        self.n_words, self.levels, self.seed = n_words, levels, seed

    @torch.no_grad()
    def fit(self, patch_feats):
        X = patch_feats
        self.C = kmeans_torch(X, self.n_words, seed=self.seed)
        d2 = torch.cdist(X[:20000], self.C) ** 2
        self.gamma = 1.0 / d2.median().item()
        K = torch.exp(-self.gamma * torch.cdist(self.C, self.C) ** 2).double()
        e, V = torch.linalg.eigh(K)
        e = e.clamp(min=1e-4 * e.max())
        self.Gm = (V / torch.sqrt(e)).float() @ V.T.float()  # K^{-1/2}
        return self

    @torch.no_grad()
    def transform(self, feats, centers):
        phi = torch.exp(-self.gamma * torch.cdist(feats, self.C) ** 2) @ self.Gm
        out = []
        for L in self.levels:
            cell = (centers[:, 0] * L).long() * L + (centers[:, 1] * L).long()
            pooled = torch.zeros(L * L, phi.shape[1], device=phi.device).index_add_(0, cell, phi)
            cnt = torch.bincount(cell, minlength=L * L).float().clamp(min=1)
            out.append((pooled / cnt[:, None]).reshape(-1))
        return torch.cat(out)
