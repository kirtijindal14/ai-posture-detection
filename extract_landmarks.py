import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

data = []

image_folder = "images/train"


def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine))

    return angle


for img_name in os.listdir(image_folder):

    img_path = os.path.join(image_folder, img_name)

    image = cv2.imread(img_path)

    if image is None:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:

        features = []

        for lm in results.pose_landmarks.landmark:
            features.append(lm.x)
            features.append(lm.y)

        # important landmarks
        left_shoulder = results.pose_landmarks.landmark[11]
        left_hip = results.pose_landmarks.landmark[23]
        left_ear = results.pose_landmarks.landmark[7]

        # calculate spine angle
        angle = calculate_angle(
            [left_ear.x, left_ear.y],
            [left_shoulder.x, left_shoulder.y],
            [left_hip.x, left_hip.y]
        )

        # improved labeling rule
        if angle > 170:
            label = 0   # good posture
        elif angle < 150:
            label = 1   # slouch
        else:
            continue    # ignore uncertain posture

        features.append(label)

        data.append(features)


df = pd.DataFrame(data)

df.to_csv("posture_dataset.csv", index=False)

print("Dataset created successfully")
