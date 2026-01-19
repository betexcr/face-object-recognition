#!/usr/bin/env python
"""
Face Recognition Application Entry Point
Optimized for PyInstaller bundling
"""
import sys
import os

# Add the current directory to the path for loading the model
if hasattr(sys, '_MEIPASS'):
    # Running as PyInstaller bundle
    os.chdir(sys._MEIPASS)
else:
    # Running as script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main application
if __name__ == "__main__":
    from main import *
