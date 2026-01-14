import streamlit as st
from phone_detector import PhoneDetector
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
def main():
    st.title("📵 Smart Workspace Monitor")
    st.subheader("Real-Time Distraction Detection")
    st.write("This app uses YOLOv8 to detect if you are using your phone while working.")

    # Setting up the WebRTC streamer
    webrtc_streamer(
        key="phone-detection",
        video_processor_factory=PhoneDetector,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

    st.info("Note: The first time you run this, it will download 'yolov8n.pt' (6MB).")

if __name__ == "__main__":
    main()