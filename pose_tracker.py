import cv2
import mediapipe as mp
import numpy as np

# Initialize BlazePose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Global variables
tracking = False
right_arm_position = "down"  # Track if the right arm is in "up" or "down" position
left_arm_position = "down"  # Track if the left arm is in "up" or "down" position
right_curl_count = 0  # Counter for right arm curls
left_curl_count = 0  # Counter for left arm curls


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
    global tracking, right_arm_position, left_arm_position, right_curl_count, left_curl_count
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    try:
        while tracking:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.")
                break

            # Convert image color to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Right arm landmarks
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Left arm landmarks
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angles
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                # Right arm curl logic
                if right_angle < 45 and right_arm_position == "down":
                    right_arm_position = "up"
                    right_curl_count += 1
                elif right_angle > 160 and right_arm_position == "up":
                    right_arm_position = "down"

                # Left arm curl logic
                if left_angle < 45 and left_arm_position == "down":
                    left_arm_position = "up"
                    left_curl_count += 1
                elif left_angle > 160 and left_arm_position == "up":
                    left_arm_position = "down"

                # Skeleton color logic
                skeleton_color = (0, 255, 0) if right_arm_position == "up" or left_arm_position == "up" else (255, 255, 255)

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=2),
                )

                # Display feedback
                right_feedback = "Nice Curl!" if right_arm_position == "up" else "Starting Point"
                left_feedback = "Nice Curl!" if left_arm_position == "up" else "Starting Point"

                # Display feedback and counters
                cv2.putText(frame, f"Right Curls: {right_curl_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2)
                cv2.putText(frame, f"Left Curls: {left_curl_count}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2)
                cv2.putText(frame, f"Right Arm: {right_feedback}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0) if right_arm_position == "up" else (0, 0, 255), 2)
                cv2.putText(frame, f"Left Arm: {left_feedback}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0) if left_arm_position == "up" else (0, 0, 255), 2)

            cv2.imshow("Bicep Curl Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Tracking stopped, resources released.")


def start_pose_tracking():
    global tracking, right_curl_count, left_curl_count
    if not tracking:
        right_curl_count = 0  # Reset right curl count
        left_curl_count = 0  # Reset left curl count
        tracking = True
        print("Starting tracking...")
        track_body()


def stop_pose_tracking():
    global tracking
    tracking = False
    print("Stopping tracking...")