import tkinter as tk
from face_tracker import start_face_mesh_tracking, stop_face_mesh_tracking, set_secondary_frame_callback as set_face_mesh_callback
from pose_tracker import start_pose_tracking, stop_pose_tracking
from body_tracker import start_body_tracking, stop_body_tracking, set_secondary_frame_callback as set_body_callback
import threading
import cv2
from PIL import Image, ImageTk


class MultiTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Tracker Application")
        self.tracking_threads = {
            "pose": None,
            "face_mesh": None,
            "body": None,
        }
        self.secondary_window = None
        self.canvas = None
        self.secondary_image = None

        # Create UI components
        self.create_widgets()

        # Set callbacks for secondary frame updates
        set_face_mesh_callback(self.update_secondary_display)
        set_body_callback(self.update_secondary_display)

    def create_widgets(self):
        # Pose Tracker Section
        pose_label = tk.Label(self.root, text="Pose Tracker", font=("Arial", 12, "bold"))
        pose_label.pack(pady=5)

        self.start_pose_button = tk.Button(
            self.root, text="Start Pose Tracking", command=self.start_pose_tracking, bg="green", fg="white"
        )
        self.start_pose_button.pack(pady=5)

        self.stop_pose_button = tk.Button(
            self.root, text="Stop Pose Tracking", command=self.stop_pose_tracking, bg="red", fg="white"
        )
        self.stop_pose_button.pack(pady=5)

        # Face Mesh Tracker Section
        face_mesh_label = tk.Label(self.root, text="Face Mesh Tracker", font=("Arial", 12, "bold"))
        face_mesh_label.pack(pady=10)

        self.start_face_mesh_button = tk.Button(
            self.root, text="Start Face Mesh Tracking", command=self.start_face_mesh_tracking, bg="blue", fg="white"
        )
        self.start_face_mesh_button.pack(pady=5)

        self.stop_face_mesh_button = tk.Button(
            self.root, text="Stop Face Mesh Tracking", command=self.stop_face_mesh_tracking, bg="orange", fg="white"
        )
        self.stop_face_mesh_button.pack(pady=5)

        # Whole-Body Tracker Section
        body_label = tk.Label(self.root, text="Whole-Body Tracker", font=("Arial", 12, "bold"))
        body_label.pack(pady=10)

        self.start_body_button = tk.Button(
            self.root, text="Start Body Tracking", command=self.start_body_tracking, bg="purple", fg="white"
        )
        self.start_body_button.pack(pady=5)

        self.stop_body_button = tk.Button(
            self.root, text="Stop Body Tracking", command=self.stop_body_tracking, bg="brown", fg="white"
        )
        self.stop_body_button.pack(pady=5)

        # Exit Button
        exit_button = tk.Button(self.root, text="Exit", command=self.exit_app, bg="gray", fg="white")
        exit_button.pack(pady=20)

    def create_secondary_window(self):
        """Create a secondary window for the blank frame display."""
        if self.secondary_window is None or not self.secondary_window.winfo_exists():
            self.secondary_window = tk.Toplevel(self.root)
            self.secondary_window.title("Secondary Display")
            self.canvas = tk.Canvas(self.secondary_window, width=640, height=480, bg="black")  # Set default size and background
            self.canvas.pack()

    def update_secondary_display(self, frame):
        """Update the secondary window with the processed frame."""
        self.create_secondary_window()

        # Convert OpenCV frame to a Tkinter-compatible format
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_image)
        self.secondary_image = ImageTk.PhotoImage(image=img)

        # Efficiently update the Canvas with the new image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.secondary_image)

    def start_pose_tracking(self):
        """Start Pose Tracking in a separate thread."""
        if self.tracking_threads["pose"] is None or not self.tracking_threads["pose"].is_alive():
            self.tracking_threads["pose"] = threading.Thread(target=start_pose_tracking)
            self.tracking_threads["pose"].start()

    def stop_pose_tracking(self):
        """Stop Pose Tracking."""
        stop_pose_tracking()

    def start_face_mesh_tracking(self):
        """Start Face Mesh Tracking in a separate thread."""
        if self.tracking_threads["face_mesh"] is None or not self.tracking_threads["face_mesh"].is_alive():
            self.tracking_threads["face_mesh"] = threading.Thread(target=start_face_mesh_tracking)
            self.tracking_threads["face_mesh"].start()

    def stop_face_mesh_tracking(self):
        """Stop Face Mesh Tracking."""
        stop_face_mesh_tracking()

    def start_body_tracking(self):
        """Start Body Tracking in a separate thread."""
        if self.tracking_threads["body"] is None or not self.tracking_threads["body"].is_alive():
            self.tracking_threads["body"] = threading.Thread(target=start_body_tracking)
            self.tracking_threads["body"].start()

    def stop_body_tracking(self):
        """Stop Body Tracking."""
        stop_body_tracking()

    def exit_app(self):
        """Stop all trackers and exit the application."""
        self.stop_pose_tracking()
        self.stop_face_mesh_tracking()
        self.stop_body_tracking()
        self.root.quit()