import time
import cv2
import mediapipe as mp
import math

import numpy as np

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to calculate the angle between two points and the vertical axis
def calculate_vertical_angle(x1, y1, x2, y2):
    vertical_distance = abs(y2 - y1)
    horizontal_distance = abs(x2 - x1)
    return math.degrees(math.atan(vertical_distance / horizontal_distance))

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # Use the correct device index (0 or 1) for your webcam

prev_posture = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame with MediaPipe Pose
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks is not None:
        # Calculate hip, right shoulder, left shoulder, and spine coordinates.
        image_height, image_width, _ = frame.shape
        hip_x = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x +
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x) / 2 * image_width
        hip_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y +
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y) / 2 * image_height

        right_shoulder_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
        right_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height

        left_shoulder_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
        left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height

        # Calculate the vertical angle between hip and shoulders
        angle = calculate_vertical_angle(left_shoulder_x, left_shoulder_y, hip_x, hip_y)

        if angle < 75:
            posture = "Bad"
            time.sleep(0.1)
        else:
            # Check if the head leans forward
            head_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
            head_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height
            neck_x = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x +
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x) / 2 * image_width
            neck_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y +
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y) / 2 * image_height

            head_angle = calculate_vertical_angle(neck_x, neck_y, head_x, head_y)

            if head_angle < 20:
                posture = "Bad (Head Lean)"
                time.sleep(0.1)
            else:
                posture = "Good"

        if prev_posture is None or posture != prev_posture:
            prev_posture = posture
            print(f"Posture: {posture}")

    # Draw landmarks and annotations on the frame
    annotated_image = frame.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    cv2.imshow('Posture Detection', annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()