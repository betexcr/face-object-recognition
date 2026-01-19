# 👁️ Real-Time AI-Powered Face & Object Detection System

This project is a high-performance, real-time AI vision application that uses advanced computer vision and deep learning to detect, identify, and analyze human faces and physical objects from a live video stream.

Designed for efficiency and speed, it utilizes **GPU-accelerated** inference with **YOLOv8** and **DeepFace**, while maintaining a clean, multithreaded architecture. The system contextualizes human presence, estimates personal attributes, and identifies interactions with objects—such as held items.

---

## 🔍 Features

### 👤 Human Detection & Analysis
- 🧍 Real-time **person detection** using YOLOv8.
- 🎯 Facial localization via `face_recognition`.
- 🧠 Attribute estimation via DeepFace:
  - Age
  - Gender
  - Race
  - Emotion
- 📊 **Mood Tracking**: Averages facial emotion expressions over 1 minute to infer dominant mood.

### 📦 Object Detection
- ⚡ Fast, lightweight detection via **YOLOv8 nano**.
- 🟥 Highlights **held objects** in **red**.
- ⚫ Ignores unheld/background objects for speed.

### 🧠 Performance-Oriented Architecture
- 🚀 Multithreaded pipeline (capture, inference, display).
- 🧪 Frame resizing to 640×360 for fast inference.
- 🔁 Attribute inference every 5 seconds per person.
- 🧠 Emotion tracking using incremental frequency count.

---

## 🖥️ Visual Output

- 🟩 **Faces** — green boxes
- 🟨 **Persons** — yellow boxes
- 🟥 **Held Objects** — red boxes with label
- 🧾 **Info above person**: Age: 34, Gender: Male, Race: Latino, Mood: Happy

---

## 🛠️ Tech Stack

| Component          | Technology                |
|-------------------|---------------------------|
| Video Input        | OpenCV                    |
| Object Detection   | YOLOv8 (Ultralytics)      |
| Face Detection     | face_recognition          |
| Attribute Analysis | DeepFace                  |
| Deep Learning      | PyTorch (GPU Accelerated) |
| UI / Display       | OpenCV GUI (cv2)          |
| Threading          | Python multithreading     |

---

## 🚀 Getting Started

### 📋 Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA (recommended for best performance)
- Python packages:

```bash
pip install opencv-python torch torchvision torchaudio deepface face_recognition ultralytics
```

### ▶️ Run the App

```bash
python main.py
```

---

## Build a macOS app

PyInstaller builds are platform-specific, so run this on a Mac.

```bash
python build_mac.py
```

The app bundle will be created at `dist/FaceRecognition.app`.
