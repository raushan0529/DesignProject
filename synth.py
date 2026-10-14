"""Simulated worker sessions (the stand-in for real sensors until hardware arrives).

A hidden fatigue level f(t) in [0,1] drives everything a real camera and wristband
would measure. Each simulated worker has personal baselines, so the models must
generalise to unseen workers, as in the Mohapatra et al. paper.
"""
import numpy as np

FPS = 10          # camera frames per second (landmark rate)
IMU_HZ = 50       # accelerometer rate
WIN_S = 10        # window length in seconds
FATIGUE_T = 0.55  # latent fatigue above this is "fatigued" (the label)


def fatigue_curve(n_s, rng, kind=None):
    """Latent fatigue over a session: rise, plateau, recover, with breaks."""
    t = np.arange(n_s)
    kind = kind or rng.choice(["ramp", "late", "fast", "fresh"], p=[.4, .25, .2, .15])
    if kind == "fresh":
        f = 0.12 + 0.08 * np.sin(t / 60.0)
    elif kind == "fast":
        f = 1 / (1 + np.exp(-(t - n_s * .3) / (n_s * .07)))
    elif kind == "late":
        f = 1 / (1 + np.exp(-(t - n_s * .65) / (n_s * .06)))
    else:
        f = np.clip(t / (n_s * .75), 0, 1) ** 1.3
    f = f * rng.uniform(.85, 1.0) + rng.uniform(0, .08)
    # a short break near the end lets fatigue drop
    b = int(n_s * rng.uniform(.82, .9))
    f[b:] *= np.linspace(1, .25, n_s - b)
    f = np.convolve(f, np.ones(15) / 15, mode="same")
    return np.clip(f, 0, 1)


def make_worker(seed, n_s=480, occlusions=True, f_override=None, face_override=None):
    """One worker's session. Returns dict of aligned raw streams + labels."""
    rng = np.random.default_rng(seed)
    f = fatigue_curve(n_s, rng)
    if f_override is not None:
        f = np.asarray(f_override, float)
    # personal baselines
    ear_open = rng.uniform(.22, .34)          # some workers naturally have narrower eyes
    hr0 = rng.uniform(62, 80)
    hr_gain = rng.uniform(5, 34)               # workers differ a lot in how fatigue shows in heart rate
    imu_gain = rng.uniform(.3, 1.0)             # ... and in how much their movement rhythm degrades
    spo0 = rng.uniform(97, 99)
    n_f = n_s * FPS
    ff = np.repeat(f, FPS)
    # --- eyes: blink process, longer closures with fatigue
    ear = np.zeros(n_f)
    i = 0
    while i < n_f:
        fl = ff[min(i, n_f - 1)]
        gap = int(rng.uniform(15, 45) * (1 + .3 * fl) * 1)
        ear[i:i + gap] = ear_open * (1 - .22 * fl) + rng.normal(0, .008, min(gap, n_f - i))
        i += gap
        dur = int(rng.integers(2, 4) + fl * rng.integers(2, 9))      # blink / micro-sleep length
        depth = rng.uniform(.04, .09)
        ear[i:i + dur] = depth + rng.normal(0, .006, min(dur, max(n_f - i, 0)))
        i += dur
    ear = ear[:n_f]
    # --- yawns: rate grows with fatigue
    mar = rng.uniform(.06, .12) + rng.normal(0, .01, n_f)
    t = 0
    while t < n_f:
        fl = ff[min(t, n_f - 1)]
        t += int(rng.exponential(FPS * (95 - 80 * fl)))
        d = int(rng.uniform(2.5, 5) * FPS)
        if t < n_f and rng.random() < .15 + .85 * fl:
            ramp = np.sin(np.linspace(0, np.pi, min(d, n_f - t)))
            mar[t:t + d] += ramp * rng.uniform(.5, .7)
        t += d
    # --- landmark detector output: 6 eye points per eye + 4 mouth points, with jitter
    face = np.ones(n_f, bool)
    if occlusions:   # safety glasses / helmet / turned head: face lost for stretches
        for _ in range(rng.integers(2, 5)):
            s = rng.integers(0, n_f - 400)
            face[s:s + rng.integers(150, 500)] = False
    if face_override is not None:
        face = np.repeat(np.asarray(face_override, bool), FPS)
    # --- wearable
    hr = hr0 + hr_gain * ff[::FPS] ** 1.1 + np.convolve(rng.normal(0, 2.2, n_s), np.ones(5) / 5, "same")
    hr += 6 * np.sin(np.arange(n_s) / 45.0 + rng.uniform(0, 6))      # exertion bumps unrelated to fatigue
    spo = spo0 - 3.2 * f + rng.normal(0, .35, n_s)
    n_i = n_s * IMU_HZ
    fi = np.repeat(f, IMU_HZ)
    ti = np.arange(n_i) / IMU_HZ
    cadence = rng.uniform(1.2, 1.5) - .5 * fi + .08 * np.sin(ti / 7)
    phase = 2 * np.pi * np.cumsum(cadence) / IMU_HZ
    # exertion bursts (heavy lifting): raise heart rate and make movement erratic whatever the fatigue level
    exert = np.zeros(n_s)
    for _ in range(rng.integers(2, 5)):
        s0 = rng.integers(0, n_s - 70); exert[s0:s0 + rng.integers(30, 70)] = rng.uniform(.4, 1.0)
    exert = np.convolve(exert, np.ones(9) / 9, "same")
    hr = hr + 22 * exert
    ex_i = np.repeat(exert, IMU_HZ)
    steady = np.clip(1 - 1.15 * imu_gain * fi - .5 * ex_i, .05, 1)
    acc = steady * np.sin(phase) + .3 * steady * np.sin(2 * phase) + (1 - steady) * rng.normal(0, 1.2, n_i) \
        + rng.normal(0, .08, n_i)
    return dict(label_bias=rng.normal(0, .07), f=f, ear=ear, mar=mar, face=face, hr=hr, spo=spo, acc=acc,
                ear_open=ear_open, seed=seed)


def landmarks_from(ear_v, mar_v, rng):
    """Turn a (EAR, MAR) pair into the 6+6+4 landmark coordinates a detector would output."""
    def eye(cx, cy, e):
        w = 30.0
        h = e * w                       # vertical gap so that EAR = 2h/(2w)
        pts = np.array([[cx - w / 2, cy], [cx - w / 6, cy - h / 2], [cx + w / 6, cy - h / 2],
                        [cx + w / 2, cy], [cx + w / 6, cy + h / 2], [cx - w / 6, cy + h / 2]])
        return pts + rng.normal(0, .35, pts.shape)
    def mouth(cx, cy, m):
        w = 50.0
        h = m * w
        pts = np.array([[cx - w / 2, cy], [cx, cy - h / 2], [cx + w / 2, cy], [cx, cy + h / 2]])
        return pts + rng.normal(0, .5, pts.shape)
    return eye(70, 50, ear_v), eye(150, 50, ear_v), mouth(110, 110, mar_v)


def demo_session(seed=9001, n_s=300):
    """Scripted fatigue path for the demo video, on a worker the models have never seen:
    fresh -> tiring -> fatigued -> face hidden by safety glasses -> face back -> break and recovery."""
    t = np.arange(n_s)
    f = np.interp(t, [0, 45, 120, 240, 262, 299], [.10, .12, .82, .88, .22, .15])
    f = np.convolve(f, np.ones(9) / 9, mode="same")
    face = np.ones(n_s, bool)
    face[150:215] = False            # safety glasses / helmet visor hide the face for 65 s
    w = make_worker(seed, n_s=n_s, occlusions=False, f_override=f, face_override=face)
    w["label_bias"] = 0.0
    return w
