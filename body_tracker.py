import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and Hands models
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Global variables
tracking = False
secondary_frame_callback = None  # Callback for sending a secondary frame

def set_secondary_frame_callback(callback):
    """Set a callback function to send secondary frames."""
    global secondary_frame_callback
    secondary_frame_callback = callback

def track_body_and_hands():
    global tracking
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

            # Process Pose and Hands
            pose_results = pose.process(rgb_frame)
            hands_results = hands.process(rgb_frame)

            # Create a blank black image for secondary display
            blank_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blank_frame[:] = 0
            blank_frame = cv2.cvtColor(blank_frame, cv2.COLOR_GRAY2BGR)

            # Draw pose landmarks on the primary frame (camera feed)
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=pose_results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

                # Draw pose landmarks on the blank frame (secondary display)
                mp_drawing.draw_landmarks(
                    image=blank_frame,
                    landmark_list=pose_results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

            # Draw hand landmarks on both frames
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                    )

                    mp_drawing.draw_landmarks(
                        image=blank_frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                    )

            # Send secondary frame to the external display
            if secondary_frame_callback:
                secondary_frame_callback(blank_frame)

            # Show the primary frame
            cv2.imshow("Whole-Body and Hands Tracker - Camera View", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

def start_body_tracking():
    global tracking
    if not tracking:
        tracking = True
        print("Starting whole-body and hands tracking...")
        track_body_and_hands()

def stop_body_tracking():
    global tracking
    tracking = False
    print("Stopping whole-body and hands tracking...")
