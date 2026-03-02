import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Camera failed to open")
    exit()
def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Get body keypoints
        nose = [int(landmarks[mp_pose.PoseLandmark.NOSE].x * w),
                int(landmarks[mp_pose.PoseLandmark.NOSE].y * h)]

        left_shoulder = [int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)]

        right_shoulder = [int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)]

        # Calculate midpoint of shoulders
        mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) // 2,
                        (left_shoulder[1] + right_shoulder[1]) // 2]

        # Create imaginary vertical line
        vertical_point = [mid_shoulder[0], mid_shoulder[1] - 100]

        # Calculate angle between nose and vertical line
        angle = calculate_angle(nose, mid_shoulder, vertical_point)

        if angle > 20:
            status = "Slouching"
            color = (0, 0, 255)
        else:
            status = "Good Posture"
            color = (0, 255, 0)

        cv2.putText(frame, f"Angle: {int(angle)}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, status, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.line(frame, mid_shoulder, nose, color, 2)

    cv2.imshow("Posture AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
