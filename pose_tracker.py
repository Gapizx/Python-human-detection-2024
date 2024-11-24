# pose_tracker.py

import cv2
import mediapipe as mp
import numpy as np

# Initialize BlazePose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Global variables
tracking = False
arm_position = "down"  # Track if arm is in "up" or "down" position
feedback = "Start"

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)  # Shoulder
    b = np.array(b)  # Elbow
    c = np.array(c)  # Wrist
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def track_body():
    global tracking, arm_position, feedback
    cap = cv2.VideoCapture(0)
    
    while tracking and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert image color to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Define the color for the arm
            arm_color = (255, 255, 255)  # Default to white

            # Extract landmarks for shoulder, elbow, and wrist (right side for bicep curl)
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle between shoulder, elbow, and wrist
            angle = calculate_angle(shoulder, elbow, wrist)

            # Bicep curl logic:
            if angle < 45:  # Arm is fully curled
                feedback = "Nice"
                arm_color = (0, 255, 0)  # Green for good curl
                arm_position = "up"
            elif angle > 160:  # Arm is fully extended
                feedback = "Start"
                arm_color = (255, 255, 255)  # White when arm is extended
                arm_position = "down"

            # Draw only the arm landmarks
            shoulder_point = (int(shoulder[0] * frame.shape[1]), int(shoulder[1] * frame.shape[0]))
            elbow_point = (int(elbow[0] * frame.shape[1]), int(elbow[1] * frame.shape[0]))
            wrist_point = (int(wrist[0] * frame.shape[1]), int(wrist[1] * frame.shape[0]))

            # Draw lines for the arm with the updated arm color
            cv2.line(frame, shoulder_point, elbow_point, arm_color, 4)
            cv2.line(frame, elbow_point, wrist_point, arm_color, 4)

            # Draw circles at each key point
            cv2.circle(frame, shoulder_point, 5, arm_color, -1)
            cv2.circle(frame, elbow_point, 5, arm_color, -1)
            cv2.circle(frame, wrist_point, 5, arm_color, -1)

        cv2.imshow("Bicep Curl Arm Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_tracking():
    global tracking
    if not tracking:
        tracking = True
        track_body()

def stop_tracking():
    global tracking
    tracking = False