import cv2
import torch
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import face_recognition
from collections import defaultdict
import time
from threading import Thread
import queue

# Load YOLOv8 nano model
yolo_model = YOLO('yolov8n.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model.to(device)

# Capture video
video_capture = cv2.VideoCapture(0)
frame_queue = queue.Queue(maxsize=5)
display_queue = queue.Queue(maxsize=1)

# Memory
emotion_counts = defaultdict(lambda: defaultdict(int))
last_attr_time = {}
last_attrs = {}

# Helper: check if object is inside person box
def is_inside(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return bx1 < ax1 < bx2 and by1 < ay1 < by2

def process_frames():
    global emotion_counts, last_attr_time, last_attrs

    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        frame_small = cv2.resize(frame, (640, 360))  # Speed boost
        results = yolo_model(frame_small)
        persons = []
        objects = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for box, class_id in zip(boxes, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = result.names[int(class_id)]
                scale_x = frame.shape[1] / 640
                scale_y = frame.shape[0] / 360
                box_scaled = [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]
                if label == 'person':
                    persons.append(box_scaled)
                else:
                    objects.append((*box_scaled, label))

        current_time = time.time()
        for (px1, py1, px2, py2) in persons:
            pid = (px1, py1, px2, py2)
            person_crop = frame[py1:py2, px1:px2]
            face_locs = face_recognition.face_locations(person_crop)

            for top, right, bottom, left in face_locs:
                ft, fr, fb, fl = py1 + top, px1 + right, py1 + bottom, px1 + left
                face_crop = frame[ft:fb, fl:fr]

                try:
                    if pid not in last_attr_time or current_time - last_attr_time[pid] > 5:
                        analysis = DeepFace.analyze(face_crop, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
                        age = analysis[0]['age']
                        gender = analysis[0]['dominant_gender']
                        race = analysis[0]['dominant_race']
                        emotion = analysis[0]['dominant_emotion']
                        last_attrs[pid] = (age, gender, race, emotion)
                        last_attr_time[pid] = current_time
                    else:
                        age, gender, race, emotion = last_attrs[pid]

                    emotion_counts[pid][emotion] += 1
                    dominant_emotion = max(emotion_counts[pid], key=emotion_counts[pid].get)

                except:
                    age = gender = race = dominant_emotion = "N/A"

                cv2.rectangle(frame, (fl, ft), (fr, fb), (0, 255, 0), 2)

            # Draw person box
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
            text = f'Age: {age}, Gender: {gender}, Race: {race}, Mood: {dominant_emotion}'
            cv2.putText(frame, text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Only draw held objects
        for (ox1, oy1, ox2, oy2, label) in objects:
            for person_box in persons:
                if is_inside((ox1, oy1, ox2, oy2), person_box):
                    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (ox1, oy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    break

        if not display_queue.full():
            display_queue.put(frame)
        frame_queue.task_done()

# Start the background processing
Thread(target=process_frames, daemon=True).start()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    if not frame_queue.full():
        frame_queue.put(frame)

    if not display_queue.empty():
        output = display_queue.get()
        cv2.imshow('AI Detection (Fast & Clean)', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
frame_queue.put(None)