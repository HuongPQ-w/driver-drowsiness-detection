# Driver Drowsiness Detection

A real-time driver drowsiness monitoring demo built with **MediaPipe Face Landmarks** and **OpenCV**. It combines several fatigue signals instead of relying on a single "eyes closed" check, and includes a watchdog for automatic recovery and CSV event logging.

## Features

- **Multi-cue detection**
  - **EAR** (eye aspect ratio): continuous eye closure, with short blinks ignored.
  - **PERCLOS**: percentage of time the eyes are closed in a sliding 30 s window, which is more stable than counting continuous closure and reduces false alarms.
  - **MAR** (mouth aspect ratio): yawning detection.
  - **Head pose**: head-down and nodding detection (angle, velocity and nod frequency).
  - **Facial expression cues**: eyebrow raise/furrow and mouth-corner droop.
- **Per-person baselines**: thresholds adapt to each user via exponential moving averages.
- **Watchdog thread**: detects a frozen processing loop (camera drop, driver error) and automatically releases and re-initializes the camera and model.
- **Event logging**: every session start/stop, drowsy alert (with duration), yawn, head-down, freeze and recovery is written to `logs/drowsiness_events.csv` with a session ID.
- **Pluggable backend**: landmark extraction is isolated in a backend class (CPU MediaPipe by default), with a GPU memory budget check, so a heavier model can be added later.
- **MediaPipe API compatibility**: automatically falls back between the legacy `solutions` API and the new `tasks` API.

## Tech stack

Python · MediaPipe · OpenCV · NumPy

## Getting started

Python 3.10 or 3.11 is recommended (tested with 3.11).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
python -m pip install -r requirements.txt
python drowsiness_monitor.py --source 0
```

- `--source 0` uses the default webcam. Pass a video path (e.g. `--source test.mp4`) to analyze a recorded video.
- Press **ESC** to quit.
- A `logs/` folder is created on the first run (and `models/` if the new MediaPipe `tasks` API is used).

### Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'mediapipe'` | Activate the virtual environment first, then run `python -m pip install -r requirements.txt`. |
| No matching `mediapipe` version found | Use Python 3.10/3.11 (see the notes in `requirements.txt`). |
| Webcam does not open | Try `--source 1` or `--source 2`, and close apps using the camera (Zoom, Teams). |

## Limitations & future work

- Thresholds (EAR, MAR, PERCLOS, head angle) are hand-tuned; calibrating them on a labeled dataset would improve reliability.
- Accuracy depends on lighting and camera position; glasses and extreme head angles can affect landmarks.
- This is a desktop demo, not a validated safety product.
