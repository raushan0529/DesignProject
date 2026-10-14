"""Edge device + supervisor dashboard server.

The edge loop runs the REAL pipeline second by second on a simulated worker:
landmarks -> EAR/MAR -> eye-state CNN -> vision features -> vision model,
wearable stream -> PSD features -> wearable model, learned fusion, alert logic.
Alerts are delivered to the supervisor dashboard over HTTP (POST /api/alert).
Run:  python3 server.py   then open http://127.0.0.1:5055
"""
import base64, io, json, pickle, threading, time, datetime
import numpy as np
import cv2, requests
from flask import Flask, jsonify, request, send_from_directory
import synth, vision, wearable, fusion

PORT = 5055
ART = "artifacts"
app = Flask(__name__, static_folder="static")
S = dict(ticks=[], alerts=[], running=False, run_id=0, done=False)
LOCK = threading.Lock()

cnn = vision.EyeCNN(); cnn.load(f"{ART}/eye_cnn.npz")
M = pickle.load(open(f"{ART}/models.pkl", "rb"))


def png64(img, scale=6):
    g = np.clip(img * 40 + 128, 0, 255).astype(np.uint8)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", g)[1]).decode()


def edge_loop(run_id, speed, seed):
    w = synth.demo_session(seed)
    n_s = len(w["f"])
    rng = np.random.default_rng(seed)
    ear_buf, mar_buf, cp_buf, vis_buf, acc_buf = [], [], [], [], []
    alert = fusion.AlertState()
    pv = pw = risk = 0.0
    face_ok = True
    t0 = time.time()
    for s in range(n_s):
        if S["run_id"] != run_id:
            return
        a, b = s * synth.FPS, (s + 1) * synth.FPS
        frames, patches = [], []
        for i in range(a, b):
            L, R, Mo = synth.landmarks_from(w["ear"][i], w["mar"][i], rng)
            e = (vision.ear(L) + vision.ear(R)) / 2
            m = vision.mar(Mo)
            frames.append([round(float(e), 4), round(float(m), 4)])
            patches.append(vision.render_eye(w["ear"][i], rng))
            ear_buf.append(e); mar_buf.append(m); vis_buf.append(bool(w["face"][i]))
        cps = cnn.predict_proba(np.stack(patches).astype(np.float32))
        cp_buf.extend(cps.tolist())
        acc = w["acc"][s * 50:(s + 1) * 50]
        acc_buf.extend(acc.tolist())
        hr, spo = float(w["hr"][s]), float(w["spo"][s])
        tick = dict(t=s, frames=frames, face=bool(w["face"][a:b].mean() >= .5), hr=round(hr, 1), spo=round(spo, 1),
                    acc=[round(float(x), 3) for x in acc[::2]], cp=round(float(cps[-1]), 3),
                    patch=png64(patches[-1]), true_f=round(float(w["f"][s]), 3))
        # ---- window decision every 5 s, using the last 10 s
        if s >= 9 and (s - 9) % 5 == 0:
            wa, wb = len(ear_buf) - 100, len(ear_buf)
            vis = np.array(vis_buf[wa:wb])
            face_ok = vis.mean() >= .8
            vf = None
            if face_ok:
                vf = np.append(vision.vision_features(np.array(cp_buf[wa:wb])[vis], np.array(mar_buf[wa:wb])[vis]),
                               (np.array(ear_buf[wa:wb])[vis] < .2).mean())
                pv = float(M["mv"].predict_proba(vf[None])[0, 1])
            hrw = w["hr"][s - 9:s + 1]; spw = w["spo"][s - 9:s + 1]; ac = np.array(acc_buf[-500:])
            wf = wearable.wearable_features(hrw, spw, ac)
            pw = float(M["mw"].predict_proba(wf[None])[0, 1])
            risk = float(M["fus"].prob(pv if face_ok else .5, pw, face_ok)[0])
            ev = alert.update(risk)
            f_, p_ = wearable.imu_psd(ac)
            sel = (f_ > .2) & (f_ <= 8)
            tick["win"] = dict(pv=round(pv, 3) if face_ok else None, pw=round(pw, 3), risk=round(risk, 3),
                               face_ok=bool(face_ok), alert=alert.on, event=ev,
                               vf=None if vf is None else dict(perclos=round(float(vf[0]), 2), blinks=round(float(vf[1])),
                                                              longest=round(float(vf[2]), 1), yawn=round(float(vf[3]), 2)),
                               ent=round(float(wf[9]), 2), dom=round(float(wf[8]), 2), hr_slope=round(float(wf[2]), 2),
                               psd_f=[round(float(x), 2) for x in f_[sel]], psd=[float(x) for x in (p_[sel] / (p_[sel].max() + 1e-12))])
            if ev == "ALERT_ON":
                msg = dict(topic="factory/line3/alerts", worker="W-17", sim_time_s=s,
                           risk_percent=round(risk * 100), state="FATIGUED",
                           vision_score=round(pv * 100) if face_ok else None, wearable_score=round(pw * 100),
                           mode="camera + wearable" if face_ok else "wearable only, camera blocked",
                           actions=["buzz wristband", "notify supervisor"],
                           sent_at=datetime.datetime.now().strftime("%H:%M:%S"))
                try:
                    requests.post(f"http://127.0.0.1:{PORT}/api/alert", json=msg, timeout=2)
                except Exception as ex:
                    print("alert delivery failed", ex)
            elif ev == "ALERT_OFF":
                try:
                    requests.post(f"http://127.0.0.1:{PORT}/api/alert", json=dict(topic="factory/line3/alerts", worker="W-17",
                        sim_time_s=s, state="CLEARED", risk_percent=round(risk * 100),
                        sent_at=datetime.datetime.now().strftime("%H:%M:%S")), timeout=2)
                except Exception as ex:
                    print("alert delivery failed", ex)
        with LOCK:
            S["ticks"].append(tick)
        delay = t0 + (s + 1) / speed - time.time()
        if delay > 0:
            time.sleep(delay)
    S["done"] = True
    S["running"] = False


@app.route("/")
def index():
    return send_from_directory("static", "dashboard.html")


@app.route("/api/start")
def start():
    speed = float(request.args.get("speed", 2.5))
    seed = int(request.args.get("seed", 9001))
    with LOCK:
        S["run_id"] += 1
        S.update(ticks=[], alerts=[], running=True, done=False)
        rid = S["run_id"]
    threading.Thread(target=edge_loop, args=(rid, speed, seed), daemon=True).start()
    return jsonify(ok=True, run_id=rid)


@app.route("/api/ticks")
def ticks():
    since = int(request.args.get("since", 0))
    with LOCK:
        return jsonify(ticks=S["ticks"][since:], alerts=S["alerts"], done=S["done"], run_id=S["run_id"])


@app.route("/api/alert", methods=["POST"])
def alert():
    msg = request.get_json(force=True)
    with LOCK:
        S["alerts"].append(msg)
    with open("alerts.jsonl", "a") as fh:
        fh.write(json.dumps(msg) + "\n")
    return jsonify(ok=True)


@app.route("/api/metrics")
def metrics():
    return send_from_directory(ART, "metrics.json")


@app.route("/api/importance")
def importance():
    return send_from_directory(ART, "wearable_importance.json")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, threaded=True)
