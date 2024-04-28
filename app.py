import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
import cv2
import av
import pytesseract

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
# Load cascade classifiers
# face_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number (1).xml")
face_cascade = cv2.CascadeClassifier("indian_license_plate.xml")
# /home/vscode/.local/bin/pytesseract
#     /home/vscode/.local/lib/python3.11/site-packages/pytesseract-0.3.10.dist-info/*
#     /home/vscode/.local/lib/python3.11/site-packages/pytesseract/*
# Function to detect license plates and extract text
def detect_and_extract_plate(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = face_cascade.detectMultiScale(img_gray, 1.1, 3)

    for (x, y, w, h) in plates:
        area = w * h
        if area > 300:
            # Draw rectangle around the plate
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Extract ROI (region of interest)
            plate_roi = image[y:y + h, x:x + w]

            # Extract text from ROI using Tesseract OCR
            extracted_text = pytesseract.image_to_string(plate_roi, lang='eng')

            new_text = ''
            alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            num = "0123456789"
            for i in extracted_text:
                if i in alpha or i in num:
                    new_text += i
                    

            return new_text, image, (x, y, w, h)

    return None, image, None

# VideoTransformer class to process video frames
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.latest_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img

        # Detect faces
        faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 3)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return img

# Main Streamlit app
st.title("License Plate Detection")

# Custom component to capture webcam feed
webrtc_ctx = webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    mode=WebRtcMode.SENDRECV, rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
)

# Capture image and detect license plate
if st.button("Capture Image"):
    if webrtc_ctx.video_transformer:
        # Get the last frame from the video transformer
        frame = webrtc_ctx.video_transformer.latest_frame
        

        if frame is not None:
            # Convert frame to BGR format
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Detect and extract license plate
            text, processed_image, roi = detect_and_extract_plate(img)

            if text:
                # Draw rectangle around the license plate region
                if roi is not None:
                    x, y, w, h = roi
                    cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Display the processed image
                st.image(processed_image, caption=f"Detected License Plate: {text}", channels="BGR", use_column_width=True)
            else:
                st.error(text)
        else:
            st.warning("No frames captured. Please enable the webcam and try again.")
    else:
        st.warning("Please enable the webcam first.")
