from flask import Flask, render_template, Response, jsonify, request
import subprocess
import cv2
import time
import mediapipe as mp
import math
import atexit
from turbo_flask import Turbo

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'  # Change this to a random secret key
turbo = Turbo(app)

# Create a VideoCapture object to capture camera feed
video_capture = cv2.VideoCapture(1)

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

prev_posture = None

# Create a flag to track if the camera is running
camera_running = True

# Function to stop the camera
def stop_camera():
    global camera_running
    camera_running = False
    video_capture.release()

# Register the function to stop the camera when the Flask app is exited
atexit.register(stop_camera)

# Initialize a global variable for posture
global_posture = "Bad"

@app.route('/')
def index():
    return render_template('home.html', posture=global_posture)

@app.route('/get_started')
def get_started():
    try:
        # Start camera application as a subprocess
        subprocess.Popen(['python', 'cameraapplication.py'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return render_template('home.html')
    except Exception as e:
        return str(e)

# Function to periodically fetch and update the posture
def update_posture():
    while True:
        time.sleep(1)  # Adjust the sleep time as needed

        if prev_posture is not None:
            with app.app_context():
                app.posture = prev_posture

# Start the posture update in a separate thread
import threading
posture_thread = threading.Thread(target=update_posture)
posture_thread.start()

log_file = open("posture_log.txt", "a")

@app.route('/camera_feed')
def camera_feed():
    def generate_frames():
        while camera_running:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Initialize posture as "Good"
            posture = "Bad"

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

            prev_posture = posture

            global_posture = posture  # Update global posture
            print(posture)

            # Append the posture value to the log file
            log_file.write(posture + '\n')

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_posture')
def get_posture():
    return jsonify(posture=global_posture)

# Add a new route to gracefully shutdown the server
@app.route('/shutdown', methods=['POST'])
def shutdown():
    stop_camera()
    log_file.close()  # Close the log file
    return 'Server shutting down...'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)






