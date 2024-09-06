import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt

warnings.filterwarnings('ignore')

# Load your pre-trained model
model = tf.keras.models.load_model('object_detection_2.keras')

# Define the object detection function
def object_Detection(path):
    # Load the image
    image = load_img(path)
    image = np.array(image, dtype=np.uint8)
    image1 = load_img(path, target_size=(224, 224))
    image1 = img_to_array(image1)
    image1 = image1 / 255.0

    # Get dimensions
    h, w, d = image.shape
    test_arr = image1.reshape(1, 224, 224, 3)

    # Predict coordinates
    cords = model.predict(test_arr)
    denorm = np.array([w, w, h, h])
    coords = cords * denorm
    coords = coords.astype(np.int32)
    return image, coords

# Define the Streamlit app layout
st.title("Object Detection and OCR")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Perform object detection
    path = "temp_image.jpg"
    image, cords = object_Detection(path)

    # Extract coordinates
    xmin, xmax, ymin, ymax = cords[0]
    
    # Show the detected object region
    img = np.array(load_img(path))
    roi = img[ymin:ymax, xmin:xmax]

    # Display the original image with bounding box
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Draw bounding box on the original image
    st.write("Detected Object Region:")
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    st.image(image, caption="Object Detection", use_column_width=True)

