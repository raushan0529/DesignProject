# Hardware build guide: wristband and camera unit

Everything here is a plan with code. None of it has been assembled or run on real parts in this build.

## Parts list
| Part | Used for | Notes |
|---|---|---|
| Raspberry Pi 4 or 5 (2 GB or more) + power supply + microSD | Edge device: runs `edge_runner.py` | Pi 4 is enough; the pipeline needs only a few ms per decision on a laptop |
| USB webcam or Raspberry Pi Camera Module | Face video, 10 frames per second is enough | Mount at the workstation facing the worker, or on a helmet |
| ESP32 development board | Wristband brain | Any ESP32 DevKit with USB |
| MAX30102 breakout | Heart rate and SpO2 | I2C, 3.3 V |
| MPU6050 breakout | Accelerometer (IMU) | I2C, share the bus with the MAX30102 |
| Coin vibration motor + NPN transistor (2N2222) + 1N4148 diode + 330 ohm resistor | Buzz alert on the wrist | The motor must not be driven directly from a pin |
| LiPo 3.7 V battery + charger board, strap, small enclosure | Wearable power and housing | |
| Jumper wires, breadboard or perfboard | Wiring | |
Prices vary by seller, so get quotes before submitting a budget.

## Wiring (ESP32)
- MAX30102: VIN to 3V3, GND to GND, SDA to GPIO21, SCL to GPIO22.
- MPU6050: VCC to 3V3, GND to GND, SDA to GPIO21, SCL to GPIO22 (same two wires as above).
- Motor: GPIO25 through 330 ohm to the transistor base; motor between battery plus and the collector; emitter to GND; diode across the motor (stripe toward plus).

## Steps
1. Arduino IDE: install the ESP32 board package and the two libraries named at the top of `firmware/wristband.ino`; upload it.
2. Open the serial monitor at 115200: you should see one JSON line per second with `hr`, `spo2` and 50 `acc` values. Fingertip on the MAX30102 gives heart rate; `0` means no reading yet.
3. On the Pi: `pip install opencv-python mediapipe pyserial requests numpy scipy scikit-learn xgboost`, copy the `fatigue_system` folder, plug the ESP32 into USB.
4. Run `python3 kit/edge_runner.py --camera 0 --serial /dev/ttyUSB0 --log session.jsonl --dashboard http://<laptop-ip>:5055/api/alert`. On the laptop run `python3 server.py` so the supervisor dashboard receives alerts.
5. Run `python3 kit/bench_edge.py` on the Pi and keep `bench_result.json` as the timing evidence.

## Important limit
The models in `artifacts/` were trained on simulated data. Real sensors give different value ranges, so retrain with
`kit/train_cnn_real.py` and `kit/train_wearable_real.py` on the real datasets (and on your own pilot recordings) before
trusting any alert from hardware.
