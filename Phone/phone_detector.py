from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# Load the lightweight YOLOv8 Nano model
model = YOLO('yolov8n.pt')

class PhoneDetector(VideoTransformerBase):
    def transform(self, frame):
        # Convert the WebRTC frame to a format OpenCV/YOLO can use
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 detection
        # stream=True is more memory efficient for video
        # Only detect class 39 (bottle) and 67 (phone)
        results = self.model.predict(img, conf=0.4, classes=[39, 67], verbose=False,streaming=True)
        phone_detected = False
        bottle_detected=False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Class ID 67 is 'cell phone' in the COCO dataset
                cls = int(box.cls[0])
                if cls == 67:
                    phone_detected = True
                    # Draw a custom red bounding box for the phone
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(img, "DISTRACTION DETECTED!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                elif cls == 40:
                    bottle_detected = True
                    # Draw a custom red bounding box for the phone
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(img, "Water Bottle", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                


        # Add a global overlay if a phone is in frame
        if phone_detected:
            cv2.putText(img, "PUT YOUR PHONE AWAY", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        if bottle_detected:
            cv2.putText(img, "Drink Some water", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)


        return img

