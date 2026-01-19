"""
Build script to create a Windows .exe for the Face Recognition application
"""
import subprocess
import sys
import os

def build_exe():
    """Build the Windows executable using PyInstaller"""
    
    # Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(script_dir, "main.py")
    yolo_model = os.path.join(script_dir, "yolov8n.pt")
    
    # PyInstaller command
    # --onefile: creates a single executable file
    # --windowed: creates a GUI app (no console window)
    # --add-data: includes the YOLO model file
    # --collect-all: collects all packages needed
    # --name: specifies the output executable name
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--add-data", f"{yolo_model}{os.pathsep}.",
        "--collect-all", "cv2",
        "--collect-all", "torch",
        "--collect-all", "deepface",
        "--collect-all", "ultralytics",
        "--collect-all", "face_recognition",
        "--name", "FaceRecognition",
        "--distpath", os.path.join(script_dir, "dist"),
        "--workpath", os.path.join(script_dir, "build"),
        "--specpath", os.path.join(script_dir, "build"),
        main_py
    ]
    
    print("Building Windows executable...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        exe_path = os.path.join(script_dir, "dist", "FaceRecognition.exe")
        print(f"\n✅ Build successful!")
        print(f"📦 Executable created at: {exe_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_exe()
