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
import tkinter as tk
from tkinter import simpledialog, messagebox

def select_camera_gui():
    """
    GUI dialog for camera selection using tkinter
    Returns the camera index selected by user
    Exits if no cameras are detected
    """
    print("\n" + "="*60)
    print("DETECTING CAMERAS...")
    print("="*60)
    
    # Detect available cameras
    available_cameras = []
    camera_info = {}
    
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available_cameras.append(i)
                camera_info[i] = f"Camera {i} ({width}x{height})"
                print(f"✓ Found: {camera_info[i]}")
            cap.release()
    
    if not available_cameras:
        print("\n" + "="*60)
        print("✗ CRITICAL ERROR: NO CAMERAS DETECTED!")
        print("="*60)
        print("\nThe application requires at least one camera to run.")
        print("Please:")
        print("  1. Connect a camera to your computer")
        print("  2. Ensure the camera is not in use by another application")
        print("  3. Check device manager for camera driver issues")
        print("  4. Restart the application")
        print("="*60 + "\n")
        
        # Show error dialog
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "No Camera Found",
                "⚠ No cameras detected!\n\n"
                "This application requires at least one camera connected.\n\n"
                "Please:\n"
                "  • Connect a camera to your computer\n"
                "  • Ensure camera is not in use by other apps\n"
                "  • Check device drivers\n\n"
                "Then restart the application."
            )
            root.destroy()
        except Exception as e:
            print(f"Could not show dialog: {e}")
        
        # Exit the application
        import sys
        sys.exit(1)
    
    # Single or multiple cameras - show confirmation dialog
    print(f"\n✓ Found {len(available_cameras)} camera(s). Opening camera selector dialog...")
    
    try:
        root = tk.Tk()
        root.title("Camera Selection")
        root.geometry("450x350")
        root.resizable(False, False)
        
        # Set window to always on top
        root.attributes('-topmost', True)
        
        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (225)
        y = (root.winfo_screenheight() // 2) - (175)
        root.geometry(f"+{x}+{y}")
        
        selected_camera = tk.IntVar(value=available_cameras[0])
        
        # Title
        title_label = tk.Label(root, text="📷 Select Camera", font=("Arial", 16, "bold"), fg="#333")
        title_label.pack(pady=15)
        
        # Description
        if len(available_cameras) == 1:
            desc_text = "One camera found. Ready to start:"
        else:
            desc_text = "Multiple cameras found. Please select one:"
        desc_label = tk.Label(root, text=desc_text, font=("Arial", 11), fg="#666")
        desc_label.pack(pady=5)
        
        # Radio buttons frame with scrolling support
        frame = tk.Frame(root, bg="white")
        frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        for cam_id in available_cameras:
            radio = tk.Radiobutton(
                frame,
                text=camera_info[cam_id],
                variable=selected_camera,
                value=cam_id,
                font=("Arial", 12),
                pady=12,
                bg="white",
                activebackground="#e3f2fd",
                selectcolor="#2196F3"
            )
            radio.pack(anchor=tk.W, fill=tk.X, padx=10)
        
        # Buttons
        button_frame = tk.Frame(root, bg="#f5f5f5")
        button_frame.pack(pady=15, fill=tk.X, padx=20)
        
        selected = [None]
        
        def on_ok():
            selected[0] = selected_camera.get()
            print(f"✓ Selected camera {selected[0]}")
            root.destroy()
        
        def on_cancel():
            print("✗ Application cancelled by user")
            import sys
            sys.exit(0)
        
        ok_btn = tk.Button(
            button_frame, 
            text="✓ Start", 
            command=on_ok, 
            font=("Arial", 11, "bold"),
            bg="#4CAF50", 
            fg="white", 
            padx=25, 
            pady=10,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        )
        ok_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(
            button_frame, 
            text="Cancel", 
            command=on_cancel, 
            font=("Arial", 11),
            bg="#f44336", 
            fg="white", 
            padx=25, 
            pady=10,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Focus and show
        root.focus_force()
        root.lift()
        
        print("\n⚠ Camera selection dialog is open. Please confirm selection.")
        print("="*60)
        
        root.mainloop()
        
        return selected[0] if selected[0] is not None else available_cameras[0]
        
    except Exception as e:
        print(f"✗ Dialog error: {e}")
        print(f"✓ Using default camera {available_cameras[0]}")
        return available_cameras[0]

# Select camera via GUI before loading model
print("\n" + "="*60)
print("FACE RECOGNITION - INITIALIZING")
print("="*60)

selected_camera = select_camera_gui()

print(f"\nLoading YOLOv8 model...")
# Load YOLOv8 nano model
yolo_model = YOLO('yolov8n.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model.to(device)
print(f"✓ Model loaded on {device}")

# Capture video with selected camera
print(f"Opening camera {selected_camera}...")
video_capture = cv2.VideoCapture(selected_camera)
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

            # Default values to avoid UnboundLocalError
            age = gender = race = dominant_emotion = "N/A"

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

            # Draw person box and label
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