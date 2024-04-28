import streamlit as st
import cv2
import numpy as np
import os
import pytesseract
from pymongo import MongoClient

uri = "mongodb+srv://cse443Prudhvi:pEjVbWv6oJHaHSJA@cluster0.7ournot.mongodb.net/"


cluster = MongoClient(uri)
db = cluster['cseData']  # Replace '<dbname>' with your actual database name
collection = db['numberPlatesData'] 

# Initialize OpenCV Cascade Classifier
PLATE_CASCADE = cv2.CascadeClassifier('indian_license_plate.xml')
MIN_AREA = 300
COLOR = (255, 0, 255)

# Streamlit app
st.title("License Plate Detection")

# Function to detect license plates and extract text
def detect_and_extract_text(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    number_plates = PLATE_CASCADE.detectMultiScale(img_gray, 1.3, 7)

    for (x, y, w, h) in number_plates:
        area = w * h
        if area > MIN_AREA:
            # Draw rectangle around the plate
            cv2.rectangle(image, (x, y), (x + w, y + h), COLOR, 2)
            # Add text label
            cv2.putText(image, "License Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, COLOR, 2)
            # Extract ROI (region of interest)
            img_roi = image[y:y + h, x:x + w]

            # Extract text from ROI using Tesseract OCR
            extracted_text = pytesseract.image_to_string(img_roi)

            new_text = ''
            alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            num = "0123456789"
            for i in extracted_text:
                if i in alpha or i in num:
                    new_text += i

            return new_text, image, (x, y, w, h)

    return None, image, None

# Main Streamlit app
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect License Plate"):
        text, processed_image, roi = detect_and_extract_text(image)

        if text:
            # Draw rectangle around the license plate region
            
            if roi is not None:
                x, y, w, h = roi
                cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            plate_data = collection.find_one({"license_plate_number": text})

            if plate_data:
                st.success(f"Detected License Plate: {text} - Found in Database")
            else:
                st.error(f"Detected License Plate: {text} - Not Found in Database")       

            # Display the processed image
            st.image(processed_image, caption=f"Detected License Plate: {text}", channels="BGR", use_column_width=True)
        else:
            st.error("Number plate no detected")

# Display Streamlit app
st.write("Note: This is a simple demonstration of license plate detection using Streamlit.")
