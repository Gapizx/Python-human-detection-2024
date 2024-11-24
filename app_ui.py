# app_ui.py

import tkinter as tk
from pose_tracker import start_tracking, stop_tracking

def create_app_ui():
    root = tk.Tk()
    root.title("Bicep Curl Counter Application")

    start_button = tk.Button(root, text="Start Tracking", command=start_tracking)
    start_button.pack(pady=10)

    stop_button = tk.Button(root, text="Stop Tracking", command=stop_tracking)
    stop_button.pack(pady=10)

    root.protocol("WM_DELETE_WINDOW", lambda: (stop_tracking(), root.quit()))
    root.mainloop()
