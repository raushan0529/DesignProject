"""Vision module: landmarks -> EAR / MAR -> eye-state CNN -> window features -> vision score.

The CNN is written in plain NumPy (no deep-learning framework is installed here).
"""
import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view as swv


# ---------- geometry ----------------------------------------------------------
def ear(eye):
    """Eye Aspect Ratio from six landmarks p1..p6 (Soukupova and Cech, 2016)."""
    d = lambda a, b: np.linalg.norm(eye[a] - eye[b])
    return (d(1, 5) + d(2, 4)) / (2.0 * d(0, 3) + 1e-9)


def mar(m):
    """Mouth Aspect Ratio from four landmarks: left, top, right, bottom."""
    return np.linalg.norm(m[1] - m[3]) / (np.linalg.norm(m[0] - m[2]) + 1e-9)


# ---------- eye image rendering (training data for the CNN) --------------------
def render_eye(ear_v, rng, size=24, noise=True):
    """Draw a grey-scale eye patch whose lid opening follows EAR, with lighting
    changes, blur, sensor noise and optional glare / frame lines (safety glasses)."""
    S = 96
    img = np.full((S, S), rng.uniform(150, 215), np.uint8)               # skin
    cv2.GaussianBlur(img, (0, 0), 6, dst=img)
    cx, cy = S // 2 + rng.integers(-5, 6), S // 2 + rng.integers(-4, 5)
    w = int(rng.uniform(30, 38))
    h = max(1, int(ear_v * 2 * w * 0.8 + rng.normal(0, 1)))               # half-opening in px
    cv2.ellipse(img, (cx, cy), (w, max(h, 1)), 0, 0, 360, int(rng.uniform(235, 255)), -1)   # sclera
    if h > 5:
        r = min(h - 1, int(w * .45))
        cv2.circle(img, (cx + rng.integers(-4, 5), cy), r, int(rng.uniform(25, 70)), -1)    # iris
        cv2.circle(img, (cx, cy), max(2, r // 2), 10, -1)                                    # pupil
    cv2.line(img, (cx - w, cy), (cx + w, cy), int(rng.uniform(40, 90)), 3)                   # lid line
    cv2.ellipse(img, (cx, cy - h - 6), (w + 4, 10), 0, 180, 360, int(rng.uniform(60, 110)), 3)  # brow
    if rng.random() < .25:                                                                   # glasses glare
        x0 = rng.integers(0, S - 30)
        cv2.line(img, (x0, 0), (x0 + 25, S), int(rng.uniform(230, 255)), int(rng.integers(3, 9)))
    if rng.random() < .15:                                                                   # frame bar
        y0 = rng.integers(S // 2 + 12, S - 4)
        cv2.line(img, (0, y0), (S, y0), 30, 4)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(.8, 2.2))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6)
    if noise:
        img = img + rng.normal(0, .12, img.shape)                                           # sensor noise
    return img


# ---------- a small CNN in NumPy ------------------------------------------------
def _conv(x, w, b):
    cols = swv(x, w.shape[2:], axis=(2, 3))                       # N,C,H',W',k,k
    return np.einsum("nchwij,fcij->nfhw", cols, w, optimize=True) + b[None, :, None, None], cols


def _conv_back(dout, cols, w):
    dw = np.einsum("nchwij,nfhw->fcij", cols, dout, optimize=True)
    k = w.shape[2]
    dp = np.pad(dout, ((0, 0), (0, 0), (k - 1, k - 1), (k - 1, k - 1)))
    dcols = swv(dp, (k, k), axis=(2, 3))
    dx = np.einsum("nfhwij,fcij->nchw", dcols, w[:, :, ::-1, ::-1], optimize=True)
    return dx, dw, dout.sum((0, 2, 3))


def _pool(x):
    n, c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    r = x[:, :, :h2 * 2, :w2 * 2].reshape(n, c, h2, 2, w2, 2)
    out = r.max((3, 5))
    mask = (r == out[:, :, :, None, :, None])
    return out, mask


def _pool_back(d, mask, shape):
    n, c, h2, _, w2, _ = mask.shape
    m = mask / mask.sum((3, 5), keepdims=True)
    dx = np.zeros(shape, np.float32)
    dx[:, :, :h2 * 2, :w2 * 2] = (m * d[:, :, :, None, :, None]).reshape(n, c, h2 * 2, w2 * 2)
    return dx


class EyeCNN:
    """conv3x3(8)-ReLU-pool -> conv3x3(16)-ReLU-pool -> dense(2). About 5k parameters."""

    def __init__(self, seed=0):
        r = np.random.default_rng(seed)
        self.p = {"w1": r.normal(0, np.sqrt(2 / 9), (8, 1, 3, 3)).astype(np.float32), "b1": np.zeros(8, np.float32),
                  "w2": r.normal(0, np.sqrt(2 / 72), (16, 8, 3, 3)).astype(np.float32), "b2": np.zeros(16, np.float32),
                  "w3": r.normal(0, np.sqrt(2 / 256), (256, 2)).astype(np.float32), "b3": np.zeros(2, np.float32)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x, keep=False):
        p = self.p
        a1, c1 = _conv(x, p["w1"], p["b1"]); r1 = np.maximum(a1, 0); q1, m1 = _pool(r1)
        a2, c2 = _conv(q1, p["w2"], p["b2"]); r2 = np.maximum(a2, 0); q2, m2 = _pool(r2)
        f = q2.reshape(len(x), -1)
        z = f @ p["w3"] + p["b3"]
        if keep:
            self.cache = (x, c1, a1, m1, q1, c2, a2, m2, q2, f)
        return z

    def predict_proba(self, x):
        out = []
        for i in range(0, len(x), 512):
            z = self.forward(x[i:i + 512, None])
            z = z - z.max(1, keepdims=True)
            e = np.exp(z)
            out.append((e / e.sum(1, keepdims=True))[:, 1])        # P(eye closed)
        return np.concatenate(out)

    def step(self, x, y, lr=3e-3):
        n = len(x)
        z = self.forward(x[:, None], keep=True)
        z = z - z.max(1, keepdims=True)
        pr = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        loss = -np.log(pr[np.arange(n), y] + 1e-9).mean()
        dz = pr.copy(); dz[np.arange(n), y] -= 1; dz /= n
        x_, c1, a1, m1, q1, c2, a2, m2, q2, f = self.cache
        g = {"w3": f.T @ dz, "b3": dz.sum(0)}
        df = (dz @ self.p["w3"].T).reshape(q2.shape)
        dr2 = _pool_back(df, m2, a2.shape) * (a2 > 0)
        dq1, g["w2"], g["b2"] = _conv_back(dr2, c2, self.p["w2"])
        dr1 = _pool_back(dq1, m1, a1.shape) * (a1 > 0)
        _, g["w1"], g["b1"] = _conv_back(dr1, c1, self.p["w1"])
        self.t += 1
        for k in self.p:                                        # Adam
            self.m[k] = .9 * self.m[k] + .1 * g[k]
            self.v[k] = .999 * self.v[k] + .001 * g[k] ** 2
            mh = self.m[k] / (1 - .9 ** self.t); vh = self.v[k] / (1 - .999 ** self.t)
            self.p[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        return loss

    def fit(self, X, y, epochs=6, bs=64, seed=0, log=print):
        r = np.random.default_rng(seed)
        for ep in range(epochs):
            idx = r.permutation(len(X)); tot = 0
            for i in range(0, len(X) - bs + 1, bs):
                b = idx[i:i + bs]
                tot += self.step(X[b], y[b], lr=3e-3 * (0.6 ** ep))
            log(f"  CNN epoch {ep + 1}/{epochs} loss {tot / (len(X) // bs):.4f}")

    def save(self, path):
        np.savez(path, **self.p)

    def load(self, path):
        d = np.load(path)
        for k in self.p:
            self.p[k] = d[k]


# ---------- window features -> vision evidence ------------------------------------
def vision_features(closed_p, mar_v, fps=10):
    """Features for one window (10 s) of per-frame eye-closed probability and MAR."""
    closed = closed_p > .5
    perclos = closed.mean()
    starts = np.flatnonzero(closed[1:] & ~closed[:-1])
    blink_rate = len(starts) * 60.0 / (len(closed) / fps)
    runs, run = [], 0
    for c in closed:
        if c: run += 1
        elif run: runs.append(run); run = 0
    if run: runs.append(run)
    longest = (max(runs) if runs else 0) / fps
    yawn = (mar_v > .45).mean()
    return np.array([perclos, blink_rate, longest, yawn, mar_v.mean(), closed_p.mean()])
