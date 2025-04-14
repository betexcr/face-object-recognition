⸻

👁️ Real-Time AI-Powered Face & Object Detection System

This project is a high-performance, real-time AI vision application that uses advanced computer vision and deep learning to detect, identify, and analyze human faces and physical objects from a live video stream.

Designed for efficiency and speed, it utilizes GPU-accelerated inference with YOLOv8 and DeepFace, while maintaining a clean, multithreaded architecture. The system is capable of contextualizing human presence, estimating personal attributes, and identifying interactions with objects, such as held items.

⸻

🔍 Features

👤 Human Detection & Analysis
	•	Real-time person tracking using YOLOv8.
	•	Facial recognition and localization via face_recognition.
	•	Attribute Estimation using DeepFace: Age, Gender, Race
	•	Emotional Disposition: Averages facial emotion detections every minute to assess dominant mood.

📦 Object Detection
	•	Fast and lightweight YOLOv8 nano model.
	•	Highlights objects that are being held by people in red.
	•	Ignores unheld (background) objects to improve performance.

🧠 Performance-Oriented Architecture
	•	Multithreaded processing: Frame capture, inference, and rendering are fully decoupled for low latency.
	•	Optimized resolution pipeline: YOLO runs on 640×360 scaled frames for speed without compromising effectiveness.
	•	Attribute caching: Facial attributes are updated every 5 seconds to avoid redundant computation.

⸻

🖥️ Visual Output
	•	Faces → Highlighted in green
	•	People → Bounding box in yellow
	•	Held Objects → Bounding box in red, with object label
	•	Attributes → Clear label above each detected person (Ex: Age: 34, Gender: Male, Race: Latino, Mood: Happy)



⸻

🛠️ Tech Stack
 
Video Input	OpenCV
Object Detection	Ultralytics YOLOv8 (nano)
Face Detection	face_recognition
Attribute Analysis	DeepFace
Deep Learning	PyTorch + GPU acceleration 



⸻

🚀 Getting Started

Prerequisites
	•	Python 3.8+
	•	NVIDIA GPU with CUDA (optional but recommended)
	•	Dependencies:

pip install opencv-python torch torchvision torchaudio deepface face_recognition ultralytics

Running

python main.py

Press Q to exit the application.

⸻

📌 Notes
	•	The application only draws objects held by people to minimize visual noise and maximize relevance.
	•	This is a desktop application intended for real-time use. Web and PWA variants are planned.

⸻