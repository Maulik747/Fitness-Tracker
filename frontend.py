import streamlit as st
import requests
import os

# --- Configuration ---
# IMPORTANT: Update this URL if your FastAPI backend is running somewhere else (e.g., a cloud service)
BACKEND_URL = "http://localhost:8000" 
PROCESS_ENDPOINT = f"{BACKEND_URL}/process_video"

st.set_page_config(
    page_title="AI Fitness Coach",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- UI Components ---
st.title("🏋️ AI Fitness Coach")
st.markdown("Upload a video of squats or pushups to get automated rep counting and form feedback.")

uploaded_file = st.file_uploader(
    "Choose a video file (MP4 or MOV)", 
    type=["mp4", "mov"]
)

if uploaded_file is not None:
    st.info("File uploaded successfully. Click 'Analyze Video' to start processing.")
    
    # Button to trigger the analysis
    if st.button("Analyze Video", type="primary"):
        # Display progress indicator
        with st.spinner("Analyzing video... This may take a moment depending on video length."):
            
            # Prepare file for multipart upload
            files = {
                "video_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            
            # Send request to FastAPI backend
            try:
                response = requests.post(PROCESS_ENDPOINT, files=files, timeout=300) # 5 minute timeout
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("Analysis Complete!")
                    
                    # --- Display Results ---
                    st.header("📊 Analysis Summary")
                    
                    # 1. Classification & Reps
                    col1, col2 = st.columns(2)
                    col1.metric("Exercise Detected", data.get("exercise_classified", "N/A"))
                    col2.metric("Total Reps Counted", data.get("reps_counted", 0))

                    # 2. Feedback
                    st.subheader("📝 Coach's Feedback")
                    st.markdown(f"**{data.get('overall_feedback')}**")

                    # 3. Processed Video
                    processed_video_url = BACKEND_URL + data.get("processed_video_url")
                    
                    st.subheader("🎥 Annotated Video")
                    st.markdown("Watch the video below to see the detected keypoints and real-time feedback.")
                    
                    # Streamlit can play the video directly from the backend's static mount
                    st.video(processed_video_url)

                else:
                    st.error(f"Backend Error (Status {response.status_code}): {response.json().get('detail', 'Unknown error.')}")
            
            except requests.exceptions.Timeout:
                st.error("The analysis request timed out (it took longer than 5 minutes). Please try a shorter video.")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to the backend server at {BACKEND_URL}. Ensure the FastAPI server is running.")
            except Exception as e:
                st.exception(f"An unexpected error occurred: {e}")