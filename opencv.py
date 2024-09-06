import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import pytesseract as pt

# Load your pre-trained model
model = tf.keras.models.load_model('object_detection_2.keras')

# Define the object detection function
def object_Detection(image):
    image1 = cv2.resize(image, (224, 224))
    image1 = img_to_array(image1)
    image1 = image1 / 255.0

    h, w, _ = image.shape
    test_arr = image1.reshape(1, 224, 224, 3)

    # Predict coordinates
    cords = model.predict(test_arr)
    denorm = np.array([w, w, h, h])
    coords = cords * denorm
    coords = coords.astype(np.int32)
    return coords

# Function to draw bounding box on the frame
def draw_bbox(image, cords):
    xmin, xmax, ymin, ymax = cords[0]
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    return image

# Streamlit app layout
st.title("Live Object Detection and OCR")

# Start the webcam stream
run = st.checkbox('Start Webcam')

if run:
    # Access the webcam feed
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    stframe = st.empty()  # Placeholder for video stream

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.write("Failed to grab frame")
            break

        # Perform object detection on the frame
        cords = object_Detection(frame)

        # Draw bounding box on the frame
        frame = draw_bbox(frame, cords)

        # Display the frame with bounding box
        stframe.image(frame, channels="BGR")

        # Extract region of interest (ROI) from the detected box
        xmin, xmax, ymin, ymax = cords[0]
        roi = frame[ymin:ymax, xmin:xmax]

        # Extract text from the ROI using pytesseract
        extracted_text = pt.image_to_string(roi)
        st.write("Extracted Text:", extracted_text)

    cap.release()

else:
    st.write("Click 'Start Webcam' to begin live detection.")
