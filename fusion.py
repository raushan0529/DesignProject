"""Decision-level fusion and alerting."""
import numpy as np
from sklearn.linear_model import LogisticRegression

ALERT_ON, ALERT_OFF = 0.55, 0.45     # hysteresis so the alert does not flicker


def fusion_inputs(pv, pw, face):
    """pv: vision prob, pw: wearable prob, face: bool. When the face is lost the vision input is neutral."""
    pv = np.where(face, pv, 0.5)
    lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
    return np.column_stack([lg(pv) * face, lg(pw), face.astype(float)])


class Fusion:
    """Learned fusion: logistic regression on both modality scores plus a face-visible flag.
    Missed fatigue is penalised more than a false alarm (asymmetric weights, as in Mohapatra et al.)."""

    def __init__(self, miss_weight=3.0):
        self.w = miss_weight
        self.m = LogisticRegression(C=1.0, max_iter=500)

    def fit(self, pv, pw, face, y):
        X = fusion_inputs(pv, pw, face)
        self.m.fit(X, y, sample_weight=np.where(y == 1, self.w, 1.0))
        return self

    def prob(self, pv, pw, face):
        return self.m.predict_proba(fusion_inputs(np.atleast_1d(pv), np.atleast_1d(pw), np.atleast_1d(face)))[:, 1]


class AlertState:
    """Raises an alert at ALERT_ON, clears it only below ALERT_OFF, requires 2 consecutive windows."""

    def __init__(self):
        self.on = False
        self.streak = 0

    def update(self, risk):
        if not self.on:
            self.streak = self.streak + 1 if risk >= ALERT_ON else 0
            if self.streak >= 2:
                self.on = True
                return "ALERT_ON"
        elif risk < ALERT_OFF:
            self.on = False
            self.streak = 0
            return "ALERT_OFF"
        return None
