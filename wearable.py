"""Wearable module: heart rate, SpO2 and IMU -> PSD / spectral features -> fatigue probability."""
import numpy as np
from scipy.signal import welch

IMU_HZ = 50
BANDS = [(0.1, 0.8), (0.8, 2.0), (2.0, 5.0), (5.0, 12.0)]


def imu_psd(acc, fs=IMU_HZ):
    """Power spectral density of an accelerometer window (Welch, 5.12 s segments)."""
    f, p = welch(acc - np.mean(acc), fs=fs, nperseg=256, noverlap=128)
    return f, p


def wearable_features(hr, spo, acc, fs=IMU_HZ):
    """Features for one 10 s window: heart rate, SpO2 and movement-spectrum features."""
    f, p = imu_psd(acc, fs)
    tot = p.sum() + 1e-12
    band = [p[(f >= a) & (f < b)].sum() / tot for a, b in BANDS]
    sel = (f >= .3) & (f <= 6)
    dom = f[sel][np.argmax(p[sel])]
    q = p[sel] / (p[sel].sum() + 1e-12)
    ent = -(q * np.log(q + 1e-12)).sum() / np.log(len(q))          # spectral entropy, 0 steady .. 1 erratic
    peak = p[sel].max() / (p[sel].sum() + 1e-12)
    t = np.arange(len(hr))
    slope = np.polyfit(t, hr, 1)[0] if len(hr) > 2 else 0.0
    return np.array([hr.mean(), hr.std(), slope, spo.mean(), *band, dom, ent, peak, np.sqrt(np.mean(acc ** 2))])


WEAR_NAMES = ["hr_mean", "hr_std", "hr_slope", "spo2_mean", "band_0.1-0.8Hz", "band_0.8-2Hz", "band_2-5Hz",
              "band_5-12Hz", "dominant_Hz", "spectral_entropy", "peak_ratio", "imu_rms"]
