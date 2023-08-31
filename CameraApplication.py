import cv2
import mediapipe as mp
import math

import numpy as np
import torch
from torch import nn
from torch.nn import init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * (640 // 8) * (480 // 8), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.fc_posture = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv_layers(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc_layers(x))

        return self.fc_posture(x)  # out_hand_presence

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                init.constant_(layer.weight, 0.1)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0.1)


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Function to calculate the angle between three points using the cosine rule
def calculate_angle(x1, y1, x2, y2, x3, y3):
    a = euclidean_distance(x2, y2, x3, y3)
    b = euclidean_distance(x1, y1, x3, y3)
    c = euclidean_distance(x1, y1, x2, y2)
    return math.degrees(math.acos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))


# Function to process and display upper body pose landmarks and detect posture
def process_and_detect_posture(image):
    # Convert the BGR image to RGB and process it with MediaPipe Pose.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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

        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow('Upper Body Pose Landmarks', annotated_image)

        black_image = np.zeros((640, 480, 3), dtype = np.uint8)
        mp_drawing.draw_landmarks(
            black_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        return black_image
        # # Save the landmark-only image
        # cv2.imwrite(f"C:/Users/ac913/PycharmProjects/appChallenge/unlabeled_data/folder_{m}/Frame_{idx//25}.jpg",
        #             black_image)
        #
        # cv2.imwrite("C:/Users/ac913/PycharmProjects/appChallenge/actual_frames/folder_8/Color_Frame_" + str(idx//25) + ".jpg",
        #             annotated_image)


# Initialize MediaPipe Pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # Use the correct device index (0 or 1) for your Logitech webcam
model = CNNModel().to(device)  # Replace with your CNNModel initialization code

index = 0

# Initialize the list to store landmark coordinates
landmark_coordinates_list = []

model.load_state_dict(torch.load("C:/Users/ac913/PycharmProjects/appChallenge/models/model_15.pth"))
model.eval()

prev_posture = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    index += 1

    if index % 5 == 0:
        annotated_image = process_and_detect_posture(frame)

        if pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks is not None:
            landmark_coordinates = [
                (landmark.x * frame.shape[1], landmark.y * frame.shape[0])
                for landmark in pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks.landmark
            ]
            landmark_coordinates_list.append(landmark_coordinates)

            annotated_image_tensor = torch.from_numpy(annotated_image).permute(2, 0, 1).unsqueeze(0).float()
            out_hand_presence = model(annotated_image_tensor.to(device))

            predicted_hand_presence = (out_hand_presence >= 0.5).int()

            if prev_posture is None or predicted_hand_presence != prev_posture:
                prev_posture = predicted_hand_presence
                if predicted_hand_presence.item() == 1:
                    print("Good posture")
                else:
                    print("Bad posture")
            # print(predicted_hand_presence)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()