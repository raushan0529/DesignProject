# Intelligent Industrial Worker Safety and Fatigue Monitoring System: working implementation

E1CSA313 Design Project-I, Alliance School of Advanced Computing, Bengaluru.

A camera module and a wristband module are fused on an "edge device" to detect fatigue and alert the worker and supervisor.

## What is real and what is simulated
- **Real code:** EAR/MAR from landmarks, an eye-state CNN written in NumPy, window features (PERCLOS, blink rate, yawns), IMU power spectrum (Welch PSD, spectral entropy), two XGBoost models, learned asymmetric fusion, alert logic with hysteresis, HTTP delivery of alerts to a supervisor dashboard.
- **Simulated:** all sensor input. No real camera or wristband is connected. `synth.py` generates workers with a hidden fatigue level that drives eyes, yawns, heart rate, SpO2 and accelerometer data. Workers differ in how fatigue shows up, and heavy-lifting bursts raise heart rate and make movement erratic without fatigue.
- Eye images for the CNN are drawn with OpenCV (`vision.render_eye`), not taken from MRL Eye or YawDD.
- **Results below are on simulated data. They test the pipeline. They are not real-world accuracy.**

## Files
| File | Purpose |
|---|---|
| `synth.py` | simulated workers and the scripted demo session |
| `vision.py` | EAR, MAR, eye-patch renderer, NumPy CNN, window features |
| `wearable.py` | PSD and wearable feature extraction |
| `fusion.py` | learned fusion (miss weight 3x) and alert state machine |
| `pipeline.py` | trains the CNN and models, evaluates on unseen workers |
| `server.py` + `static/dashboard.html` | live edge pipeline and supervisor dashboard |
| `record.py` | records the demo video with Playwright |
| `artifacts/` | trained models, `metrics.json`, feature importances |

## Run
```
pip install numpy scipy scikit-learn xgboost opencv-python flask requests matplotlib
python3 pipeline.py        # trains and evaluates (about 6 minutes on 2 CPU cores)
python3 server.py          # open http://127.0.0.1:5055 and press the page's start (or run startDemo(2.5) in the console)
```

## Results (16 workers never seen in training, 1520 ten-second windows, face hidden in 22%)
| Method | Accuracy | Missed fatigue | Alert precision |
|---|---|---|---|
| Vision only | 86.5% | 26.0% | 87.4% |
| Wearable only | 94.6% | 3.2% | 89.5% |
| Simple average | 94.8% | 3.9% | 90.5% |
| Learned fusion | 91.5% | 1.3% | 81.9% |

Eye-state CNN accuracy on held-out drawn eye images: 97.1%.
When the face is hidden: vision-only misses 100% of fatigue; learned fusion misses 3.9%.

**Reading these honestly:** fusion is not more accurate than the wearable alone here. It misses the fewest tired workers and keeps working when the camera is blind, at the cost of more false alarms (the 3x miss penalty). The wearable is strong partly because the simulated movement cue is clean; literature on real data reports 76 to 92%.

## Next steps
Train the CNN on MRL Eye and YawDD, train the wearable model on the Mohapatra et al. dataset, run on a Raspberry Pi and measure frames per second and latency, then a pilot with volunteers.

## Kit for the remaining items (added later)
| Item | File | Status |
|---|---|---|
| Train CNN on MRL Eye | `kit/train_cnn_real.py` | tested on stand-in images, needs the real dataset |
| Train wearable model on real data | `kit/train_wearable_real.py` | tested on stand-in CSVs, column names set by options |
| Time on a Raspberry Pi | `kit/bench_edge.py` | run on a laptop: about 21 ms per decision; needs a Pi |
| Wristband and camera unit | `HARDWARE.md`, `kit/firmware/wristband.ino`, `kit/edge_runner.py` | edge runner tested with `--sim`; camera, serial and firmware untested on hardware |
| Pilot study | `pilot/` (protocol, consent, log sheet, `analyze_pilot.py`) | analysis tested on simulated logs |
