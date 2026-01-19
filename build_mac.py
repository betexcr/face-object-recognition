"""
Build script to create a macOS .app (and CLI binary) for the Face Recognition application.
Run this on macOS (PyInstaller builds are platform-specific).
"""
import os
import subprocess
import sys


def build_app():
    """Build the macOS app bundle using PyInstaller."""
    if sys.platform != "darwin":
        print("This script must be run on macOS to produce a macOS executable.")
        sys.exit(1)

    # Ensure PyInstaller is installed
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(script_dir, "main.py")
    yolo_model = os.path.join(script_dir, "yolov8n.pt")

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--windowed",
        "--add-data",
        f"{yolo_model}{os.pathsep}.",
        "--collect-all",
        "cv2",
        "--collect-all",
        "torch",
        "--collect-all",
        "deepface",
        "--collect-all",
        "ultralytics",
        "--collect-all",
        "face_recognition",
        "--name",
        "FaceRecognition",
        "--distpath",
        os.path.join(script_dir, "dist"),
        "--workpath",
        os.path.join(script_dir, "build"),
        "--specpath",
        os.path.join(script_dir, "build"),
        main_py,
    ]

    print("Building macOS app...")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.check_call(cmd)
        app_path = os.path.join(script_dir, "dist", "FaceRecognition.app")
        print("\nBuild successful!")
        print(f"App bundle created at: {app_path}")
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    build_app()
