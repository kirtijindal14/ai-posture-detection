# 🧍 AI Posture Detection System

A real-time computer vision application that detects user posture using pose estimation and computes a dynamic posture score.

---

## 🚀 Features

- Real-time body landmark detection using MediaPipe Pose
- Neck angle computation using geometric vector analysis
- Slouch detection based on angular threshold
- Live posture score percentage overlay
- macOS-compatible camera backend

---

## 🧠 How It Works

1. Detects 33 body landmarks using MediaPipe Pose.
2. Extracts nose and shoulder coordinates.
3. Computes neck angle relative to a vertical reference.
4. Classifies posture as Good or Slouching.
5. Displays real-time feedback overlay.

---

## 🛠 Tech Stack

- Python 3.11
- OpenCV
- MediaPipe
- NumPy

---

## ▶️ How To Run

```bash
python posture_detector.py
```
Press `Q` or `Esc` to exit.
