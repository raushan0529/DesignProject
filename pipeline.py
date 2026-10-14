"""Build the dataset, train the CNN / wearable model / fusion, evaluate on UNSEEN workers."""
import sys, json, pickle, time
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
import xgboost as xgb
import synth, vision, wearable, fusion

N_WORKERS, N_TRAIN = 48, 32
WIN, STEP = 10, 5
OUT = "artifacts"


def log(*a):
    print(*a, flush=True)


def train_cnn():
    rng = np.random.default_rng(123)
    N = 14000
    e = np.concatenate([rng.uniform(.03, .19, N // 2), rng.uniform(.21, .38, N // 2)])
    X = np.stack([vision.render_eye(v, rng) for v in e]).astype(np.float32)
    y = (e < .2).astype(int)
    p = rng.permutation(N); X, y = X[p], y[p]
    cnn = vision.EyeCNN(0)
    cnn.fit(X[:11500], y[:11500], epochs=7, log=log)
    pr = cnn.predict_proba(X[11500:])
    acc = float(((pr > .5) == y[11500:]).mean())
    log(f"  eye-state CNN held-out accuracy: {acc:.4f}")
    cnn.save(f"{OUT}/eye_cnn.npz")
    return cnn, acc


def windows_for(w, cnn, rng):
    """Cut one worker's session into windows and compute both modalities' features."""
    n_s = len(w["f"]); nf = len(w["ear"])
    # per-frame landmark geometry (EAR/MAR from landmarks) and CNN closed-probability on the eye patch
    ear_lm = np.empty(nf); mar_lm = np.empty(nf)
    for i in range(nf):
        L, R, M = synth.landmarks_from(w["ear"][i], w["mar"][i], rng)
        ear_lm[i] = (vision.ear(L) + vision.ear(R)) / 2; mar_lm[i] = vision.mar(M)
    patches = np.stack([vision.render_eye(v, rng) for v in w["ear"]]).astype(np.float32)
    closed_p = cnn.predict_proba(patches)
    rows = []
    for s in range(0, n_s - WIN + 1, STEP):
        a, b = s * synth.FPS, (s + WIN) * synth.FPS
        vis = w["face"][a:b]
        face = vis.mean() >= .8
        if face:
            vf = vision.vision_features(closed_p[a:b][vis], mar_lm[a:b][vis])
            vf = np.append(vf, (ear_lm[a:b][vis] < .2).mean())
        else:
            vf = np.zeros(7)
        wf = wearable.wearable_features(w["hr"][s:s + WIN], w["spo"][s:s + WIN],
                                        w["acc"][s * 50:(s + WIN) * 50])
        y = int(w["f"][s:s + WIN].mean() + w["label_bias"] > synth.FATIGUE_T)   # self-report style label offset
        rows.append((vf, wf, face, y, w["f"][s:s + WIN].mean()))
    return rows


def build(cnn):
    V, W, F, Y, G, FL = [], [], [], [], [], []
    t = time.time()
    for k in range(N_WORKERS):
        w = synth.make_worker(1000 + k)
        for vf, wf, face, y, fl in windows_for(w, cnn, np.random.default_rng(k)):
            V.append(vf); W.append(wf); F.append(face); Y.append(y); G.append(k); FL.append(fl)
        if k % 8 == 7:
            log(f"  worker {k + 1}/{N_WORKERS} done ({time.time() - t:.0f}s)")
    return (np.array(V), np.array(W), np.array(F, bool), np.array(Y), np.array(G), np.array(FL))


def gbm():
    return xgb.XGBClassifier(n_estimators=160, max_depth=3, learning_rate=.08, subsample=.9,
                             colsample_bytree=.9, eval_metric="logloss", n_jobs=2)


def metrics(y, p, thr=.5):
    yh = (p >= thr).astype(int)
    out = dict(accuracy=accuracy_score(y, yh), recall=recall_score(y, yh, zero_division=0),
               precision=precision_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0))
    out["auc"] = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    out["missed_fatigue_rate"] = 1 - out["recall"]
    return {k: round(float(v), 4) for k, v in out.items()}


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    log("1/4 training eye-state CNN"); cnn, cnn_acc = train_cnn()
    log("2/4 building windowed dataset from 48 simulated workers"); V, W, F, Y, G, FL = build(cnn)
    tr, te = G < N_TRAIN, G >= N_TRAIN
    log(f"  windows: train {tr.sum()}, test {te.sum()} (test workers never seen in training); "
        f"fatigued share {Y.mean():.2f}; face lost in {1 - F.mean():.0%} of windows")
    log("3/4 training vision model, wearable model and fusion (out-of-fold stacking)")
    # out-of-fold base-model scores on training workers, so the fusion never sees over-fitted scores
    oof_v = np.full(len(Y), .5); oof_w = np.zeros(len(Y))
    for a, b in GroupKFold(5).split(V[tr], Y[tr], G[tr]):
        ia, ib = np.flatnonzero(tr)[a], np.flatnonzero(tr)[b]
        mv = gbm().fit(V[ia][F[ia]], Y[ia][F[ia]]); oof_v[ib] = mv.predict_proba(V[ib])[:, 1]
        mw = gbm().fit(W[ia], Y[ia]); oof_w[ib] = mw.predict_proba(W[ib])[:, 1]
    mv = gbm().fit(V[tr & F], Y[tr & F]); mw = gbm().fit(W[tr], Y[tr])
    fus = fusion.Fusion(miss_weight=3.0).fit(oof_v[tr], oof_w[tr], F[tr], Y[tr])
    pv = mv.predict_proba(V)[:, 1]; pw = mw.predict_proba(W)[:, 1]
    pf = fus.prob(pv, pw, F)
    log("4/4 evaluating on the 16 unseen test workers")
    avg = np.where(F, (pv + pw) / 2, pw)
    vis_only = np.where(F, pv, 0.0)          # camera cannot say anything when the face is lost
    res = {"cnn_heldout_accuracy": round(cnn_acc, 4),
           "n_workers_train": N_TRAIN, "n_workers_test": N_WORKERS - N_TRAIN,
           "windows_test": int(te.sum()), "face_lost_share_test": round(float(1 - F[te].mean()), 3),
           "all_windows": {"vision_only": metrics(Y[te], vis_only[te]), "wearable_only": metrics(Y[te], pw[te]),
                           "simple_average": metrics(Y[te], avg[te]), "learned_fusion": metrics(Y[te], pf[te])},
           "face_lost_windows": {"vision_only": metrics(Y[te & ~F], vis_only[te & ~F]),
                                 "wearable_only": metrics(Y[te & ~F], pw[te & ~F]),
                                 "learned_fusion": metrics(Y[te & ~F], pf[te & ~F])}}
    # unseen-worker gap (Mohapatra et al. reported ~21% error rise): compare train-worker vs test-worker accuracy
    res["wearable_train_vs_test_accuracy"] = [round(float(accuracy_score(Y[tr], mw.predict(W[tr]))), 3),
                                              round(float(accuracy_score(Y[te], mw.predict(W[te]))), 3)]
    json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=1)
    pickle.dump(dict(mv=mv, mw=mw, fus=fus), open(f"{OUT}/models.pkl", "wb"))
    np.savez(f"{OUT}/dataset.npz", V=V, W=W, F=F, Y=Y, G=G, FL=FL, pv=pv, pw=pw, pf=pf)
    imp = dict(zip(wearable.WEAR_NAMES, [round(float(x), 4) for x in mw.feature_importances_]))
    json.dump(imp, open(f"{OUT}/wearable_importance.json", "w"), indent=1)
    log(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
