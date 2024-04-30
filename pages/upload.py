import streamlit as st
import cv2
import numpy as np
import os
import pytesseract
from pymongo import MongoClient
import easyocr

uri = "mongodb+srv://cse443Prudhvi:pEjVbWv6oJHaHSJA@cluster0.7ournot.mongodb.net/"


cluster = MongoClient(uri)
db = cluster['cseData']  # Replace '<dbname>' with your actual database name
collection = db['numberPlatesData'] 

# Initialize OpenCV Cascade Classifier
cascade = cv2.CascadeClassifier('indian_license_plate.xml')
MIN_AREA = 300
COLOR = (255, 0, 255)

# Streamlit app
st.title("License Plate Detection")

# Function to detect license plates and extract text
def detect_and_extract_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect number plate(s)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)

    # Process each detected number plate
    for i, (x, y, w, h) in enumerate(nplate):
        wT, hT, cT = img.shape
        a, b = (int(0.02 * wT), int(0.02 * hT))

        # Crop the number plate region
        plate = img[y + a:y + h - a, x + b:x + w - b, :]

        # Enhance the image to aid in text recognition
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, plate = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
    return plate

# Main Streamlit app
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect License Plate"):
        processed_image = detect_and_extract_text(image)

        reader = easyocr.Reader(['en'])
        result = reader.readtext(processed_image)

        li=[]
        for detection in result:
            license_plate_text = ""
            text = detection[1]
            for i in text:
                if i!=" ":
                    license_plate_text+=i
            li.append(license_plate_text)
        for item in li:
            if(len(item)>6):
                
                plate_data = collection.find_one({"license_plate_number": item})

                if plate_data:
                    st.success(f"Detected License Plate: {item} - Found in Database")
                else:
                    st.error(f"Detected License Plate: {item} - Not Found in Database")

                # Display the processed image
                st.image(processed_image, caption="Processed Image", use_column_width=True)

# Display Streamlit app
st.write("Note: This is a simple demonstration of license plate detection using Streamlit.")
