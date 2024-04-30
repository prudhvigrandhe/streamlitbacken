import streamlit as st
st.set_page_config(page_title="Catch At Toll", page_icon="/workspaces/streamlitbacken/road-barrier.png")
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
import cv2
import av
import pytesseract
from pymongo import MongoClient
import easyocr
# Load cascade classifiers
# face_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number (1).xml")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


uri = "mongodb+srv://cse443Prudhvi:pEjVbWv6oJHaHSJA@cluster0.7ournot.mongodb.net/"


cluster = MongoClient(uri)
db = cluster['cseData']  # Replace '<dbname>' with your actual database name
collection = db['numberPlatesData'] 


cascade = cv2.CascadeClassifier("indian_license_plate.xml")

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

# VideoTransformer class to process video frames
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.latest_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img

        # Detect faces
        faces = cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 3)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return img

# Main Streamlit app
st.title("ANPR")

# Custom component to capture webcam feed
webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoTransformer,
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
            processed_image = detect_and_extract_text(img)
            

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
        else:
            st.warning("No frames captured. Please enable the webcam and try again.")
    else:
        st.warning("Please enable the webcam first.")
