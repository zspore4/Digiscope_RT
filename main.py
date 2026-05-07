"""
Digiscope Python - ECG Signal Analysis Application
Entry point: run this file to launch the application.
  python main.py
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from ui.app import DigiScopeApp


def main():
    app = DigiScopeApp()
    app.run()


if __name__ == "__main__":
    main()
