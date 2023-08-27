import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Function to calculate the angle between three points using the cosine rule
def calculate_angle(x1, y1, x2, y2, x3, y3):
    a = euclidean_distance(x2, y2, x3, y3)
    b = euclidean_distance(x1, y1, x3, y3)
    c = euclidean_distance(x1, y1, x2, y2)
    return math.degrees(math.acos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))


# Function to process and display upper body pose landmarks and detect posture
def process_and_detect_posture(image, idx):
    # Convert the BGR image to RGB and process it with MediaPipe Pose.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Check if the posture is upright, hunched in, or hunched over
    # posture_message = "Good Posture!"
    if results.pose_landmarks is not None:
        # Calculate hip, right shoulder, left shoulder, and spine coordinates.
        image_height, image_width, _ = image.shape
        hip_x = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x +
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x) / 2 * image_width
        hip_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y +
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y) / 2 * image_height

        right_shoulder_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
        right_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height

        left_shoulder_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
        left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height

        # Calculate the angle between the shoulders and the hip line
        shoulder_hip_angle_degrees = calculate_angle(right_shoulder_x, right_shoulder_y, hip_x, hip_y,
                                                     left_shoulder_x, left_shoulder_y)

        # Draw pose landmarks (optional)
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow('Upper Body Pose Landmarks', annotated_image)
        cv2.imwrite("C:/Users/ac913/PycharmProjects/appChallenge/unlabeled_data/folder_" + str(m) + "/Frame_" + str(idx//25) + ".jpg",
                    annotated_image)


# Initialize MediaPipe Pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # Use the correct device index (0 or 1) for your Logitech webcam

import os

index = 0
m = 0
if os.path.exists("C:/Users/ac913/PycharmProjects/appChallenge/unlabeled_data/folder_" + str(m)):
    if input("File Already Exists, Delete (Y/N) ") == "Y": exit()

    for file in os.listdir('C:/Users/ac913/PycharmProjects/appChallenge/unlabeled_data/folder_' + str(m)):
        os.remove("C:/Users/ac913/PycharmProjects/appChallenge/unlabeled_data/folder_" + str(m) + "/" + file)
else:
    os.mkdir("C:/Users/ac913/PycharmProjects/appChallenge/unlabeled_data/folder_" + str(m))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process, detect posture, and display upper body pose landmarks
    if index % 25 == 0:
        process_and_detect_posture(frame, index)
    index += 1
    # Wait for 1ms and check if 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
