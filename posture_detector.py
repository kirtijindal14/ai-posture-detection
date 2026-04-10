import cv2
import mediapipe as mp
import numpy as np
import joblib
import csv
import time

# Load ML model
model = joblib.load("posture_model.pkl")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Create posture log file
log_file = "posture_log.csv"

with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "posture", "angle"])

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera failed to open")
    exit()


def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine))

    return angle


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        # Extract ML features
        features = []
        for lm in landmarks:
            features.append(lm.x)
            features.append(lm.y)

        prediction = model.predict([features])[0]

        # Important landmarks for spine angle
        left_shoulder = landmarks[11]
        left_hip = landmarks[23]
        left_ear = landmarks[7]

        angle = calculate_angle(
            [left_ear.x, left_ear.y],
            [left_shoulder.x, left_shoulder.y],
            [left_hip.x, left_hip.y]
        )

        # Hybrid decision system
        if prediction == 0 or angle > 165:
            status = "Good Posture"
            color = (0, 255, 0)
        else:
            status = "Slouching"
            color = (0, 0, 255)

        # Save posture log
        timestamp = time.time()

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, status, int(angle)])

        # Display text
        cv2.putText(
            frame,
            status,
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

        cv2.putText(
            frame,
            f"Angle: {int(angle)}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # Draw pose skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Posture AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
