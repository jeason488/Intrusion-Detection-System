import os
import sys
print(f"Python executable: {sys.executable}")
print(f"Current directory: {os.getcwd()}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
try:
    from PyQt5.QtWidgets import QApplication
    print("PyQt5 imported successfully")
except ImportError as e:
    print(f"PyQt5 import error: {e}")
try:
    from api.server import create_app
    print("API server imported successfully")
except ImportError as e:
    print(f"API server import error: {e}")
print("\nTest completed successfully!")
