import cv2
import mediapipe as mp

# Initialize Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables
tracking = False
status_callback = None  # Callback for sending status updates
secondary_frame_callback = None  # Callback for sending a secondary frame (e.g., landmarks only)

def set_status_callback(callback):
    """Set a callback function to send tracking updates."""
    global status_callback
    status_callback = callback

def set_secondary_frame_callback(callback):
    """Set a callback function to send secondary frames."""
    global secondary_frame_callback
    secondary_frame_callback = callback

def track_face():
    global tracking
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        if status_callback:
            status_callback("Error: Camera not accessible.")
        return

    try:
        while tracking:
            ret, frame = cap.read()
            if not ret:
                if status_callback:
                    status_callback("Error: Failed to read frame from camera.")
                break

            # Convert image color to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # Create a blank black image for secondary display
            blank_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blank_frame[:] = 0
            blank_frame = cv2.cvtColor(blank_frame, cv2.COLOR_GRAY2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw landmarks on the primary frame (camera feed)
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                    # Draw landmarks on the blank frame (secondary display)
                    mp_drawing.draw_landmarks(
                        image=blank_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                if status_callback:
                    status_callback("Face detected and landmarks drawn.")
            else:
                if status_callback:
                    status_callback("No face detected.")

            # Send secondary frame for external display
            if secondary_frame_callback:
                secondary_frame_callback(blank_frame)

            # Show the primary frame
            cv2.imshow("Face Mesh Tracker - Camera View", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    except Exception as e:
        if status_callback:
            status_callback(f"Error: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if status_callback:
            status_callback("Tracking stopped, resources released.")

def start_face_mesh_tracking():
    global tracking
    if not tracking:
        tracking = True
        if status_callback:
            status_callback("Starting face mesh tracking...")
        track_face()

def stop_face_mesh_tracking():
    global tracking
    tracking = False
    if status_callback:
        status_callback("Stopping face mesh tracking...")

# Example usage:
if __name__ == "__main__":
    start_face_mesh_tracking()
