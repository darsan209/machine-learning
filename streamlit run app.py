# app.py

import streamlit as st
import cv2
import numpy as np
import math
import tempfile
import time
import matplotlib.pyplot as plt

# App Title
st.title("🏀 Ball Speedometer App")

# Upload video
video_file = st.file_uploader("Upload a video of your ball", type=["mp4", "mov", "avi"])

# Parameters
target_speed = st.slider("Target speed (cm/s)", min_value=10, max_value=500, value=100)
pixels_to_cm = st.number_input("Pixels to cm conversion factor", value=0.1)

# Scorecard
score = 0
speed_list = []

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    prev_x, prev_y = None, None
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([30, 150, 50])
        upper_color = np.array([50, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        black_frame = np.zeros_like(frame)

        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if radius > 10:
                cv2.circle(black_frame, (int(x), int(y)), int(radius), (0, 255, 0), -1)

                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time

                if prev_x is not None and prev_y is not None:
                    dx = x - prev_x
                    dy = y - prev_y
                    pixel_distance = math.sqrt(dx**2 + dy**2)
                    cm_distance = pixel_distance * pixels_to_cm
                    speed = cm_distance / dt  # cm/s

                    speed_list.append(speed)

                    if speed > target_speed:
                        score += 1

                prev_x, prev_y = x, y

        # Convert for display in Streamlit
        black_frame = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
        st.image(black_frame, channels="RGB")

    cap.release()

    # Results
    st.subheader("Results")
    st.write(f"Total score: **{score}**")
    if speed_list:
        st.write(f"Max speed: **{max(speed_list):.2f} cm/s**")
        st.write(f"Average speed: **{np.mean(speed_list):.2f} cm/s**")

        # Plot speeds
        fig, ax = plt.subplots()
        ax.plot(speed_list)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Speed (cm/s)")
        ax.set_title("Speed Over Time")
        st.pyplot(fig)
