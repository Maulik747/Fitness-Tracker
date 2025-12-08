import os
import cv2
import tempfile
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any

# --- Configuration ---
PROCESSED_VIDEOS_DIR = "processed_videos"
# Ensure the directory exists
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)

# Initialize FastAPI App
app = FastAPI(title="AI Fitness Coach API", description="Analyzes video for squats and pushups.")

# Mount the directory to serve processed videos statically
app.mount(
    f"/{PROCESSED_VIDEOS_DIR}",
    StaticFiles(directory=PROCESSED_VIDEOS_DIR),
    name="processed_videos"
)

# Initialize MediaPipe Pose Model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ====================================================================
# CORE COMPUTER VISION FUNCTIONS (Adapted from Notebook)
# ====================================================================


FONT = cv2.FONT_HERSHEY_SIMPLEX
BIG = 4
MED = 1.8
THICK_BIG = 4
THICK_MED = 3

def calculate_angle(a: list, b: list, c: list) -> float:
    """Calculates the angle (in degrees) between three 3D keypoints (A, B, C)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 180.0
        
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle_rad)

def classify_exercise(landmarks) -> str:
    """Classifies exercise based on hip vs shoulder vertical alignment."""
    if not landmarks:
        return "Unknown"

    try:
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        
        y_difference = abs(right_shoulder_y - right_hip_y)
        
        # Threshold 0.2 is empirical: Pushup (horizontal) has a small difference. Squat (vertical) has a large difference.
        if y_difference < 0.2:
            return "Pushup"
        else:
            return "Squat"
            
    except Exception:
        return "Unknown"

def process_squat_video(cap: cv2.VideoCapture, out: cv2.VideoWriter, frame_width: int, frame_height: int) -> tuple[int, set]:
    """Processes video frames for Squat analysis (Rep count based on knee angle)."""
    count = 0
    stage = "up"
    form_issues = set()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # CV Processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        frame_feedback = ""

        try:
            landmarks = results.pose_landmarks.landmark
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
            knee_angle = calculate_angle(hip, knee, ankle)
            
            # Rep Counter
            if knee_angle > 165: stage = "up"
            if knee_angle < 100 and stage == "up":
                stage = "down"
                count += 1
                
            # Form Feedback
            if stage == "down" and knee_angle > 105:
                frame_feedback += "Go deeper. "
                form_issues.add("Insufficient Depth")

            # Visualization
            cv2.putText(frame, f"REPS: {count}", (40, 60), FONT, BIG, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"KNEE: {int(knee_angle)}", (40, 120), FONT, BIG, (255, 255, 255), 2, cv2.LINE_AA)
            
            if frame_feedback:
                 cv2.putText(frame, frame_feedback, (frame_width // 2, 100), FONT, BIG, (0, 0, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            
        except Exception:
            cv2.putText(frame, "NO BODY DETECTED", (frame_width // 2 - 150, 30), FONT, BIG, (0, 0, 255), 2, cv2.LINE_AA)

        out.write(frame)
        
    return count, form_issues


def process_pushup_video(cap: cv2.VideoCapture, out: cv2.VideoWriter, frame_width: int, frame_height: int) -> tuple[int, set]:
    """Processes video frames for Pushup analysis (Rep count based on elbow angle)."""
    count = 0
    stage = "up"
    form_issues = set()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # CV Processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        frame_feedback = ""

        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]

            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            back_hip_angle = calculate_angle(shoulder, hip, ankle) 
            
            # Rep Counter
            if elbow_angle > 165: stage = "up"
            if elbow_angle < 90 and stage == "up":
                stage = "down"
                count += 1
                
            # Form Feedback
            if back_hip_angle < 160:
                frame_feedback += "Hips piked. "
                form_issues.add("Hips Piked (Too High)")
            elif back_hip_angle > 200:
                 frame_feedback += "Hips sagging. "
                 form_issues.add("Hips Sagging (Too Low)")

            if stage == "down" and elbow_angle > 100:
                frame_feedback += "Go deeper. "
                form_issues.add("Insufficient Depth")

            # Visualization
            cv2.putText(frame, f"REPS: {count}", (10, 60), FONT, BIG, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"ELBOW: {int(elbow_angle)}", (10, 120), FONT, BIG, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"BACK: {int(back_hip_angle)}", (10, 180), FONT, BIG, (255, 255, 255), 2, cv2.LINE_AA)
            
            if frame_feedback:
                 cv2.putText(frame, frame_feedback, (frame_width // 2, 30), FONT, BIG, (0, 0, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            
        except Exception:
            cv2.putText(frame, "NO BODY DETECTED", (frame_width // 2 - 150, 30), FONT, BIG, (0, 0, 255), 2, cv2.LINE_AA)

        out.write(frame)
        
    return count, form_issues

def create_feedback(exercise_type: str, reps_counted: int, form_issues: set) -> str:
    """Generates the final human-readable feedback string."""
    overall_feedback = f"Your {exercise_type} session analysis is complete. "
    
    if exercise_type == "Unknown":
         return "I could not confidently classify the exercise. Please ensure your full body is visible in the starting frame."
    elif reps_counted == 0:
        overall_feedback += "I couldn't detect any completed reps. Please review the video."
    elif not form_issues:
        overall_feedback += "Great form! Keep up the good work!"
    else:
        overall_feedback += "Here are the main form issues detected: " + ", ".join(list(form_issues)) + ". Focus on correcting these next time."
    
    return overall_feedback

# ====================================================================
# FASTAPI ENDPOINT
# ====================================================================

@app.post("/process_video")
async def process_video_upload(video_file: UploadFile = File(...)) -> Dict[str, Any]:
    """Receives video, processes it, and returns results + video path."""
    if video_file.content_type not in ["video/mp4", "video/mov"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only MP4 and MOV video files are supported.")

    # 1. Save uploaded file to a temporary location
    try:
        temp_file_path = os.path.join(tempfile.gettempdir(), video_file.filename)
        with open(temp_file_path, "wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
            input_path = temp_file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary file: {e}")

    # 2. Define output path
    processed_filename = f"processed_{os.path.basename(video_file.filename)}"
    output_path = os.path.join(PROCESSED_VIDEOS_DIR, processed_filename)
    
    # 3. Process the video (Classification and Analysis)
    
    # Classification requires re-opening the video to read the first frame
    cap_classifier = cv2.VideoCapture(input_path)
    ret, frame = cap_classifier.read()
    if not ret:
        cap_classifier.release()
        os.unlink(input_path) 
        raise HTTPException(status_code=400, detail="Video is empty or unreadable.")
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    exercise_type = classify_exercise(results.pose_landmarks.landmark if results.pose_landmarks else None)
    cap_classifier.release()
    
    # Setup video writer and re-read the input video for full analysis
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    try:
        if exercise_type == "Squat":
            reps_counted, form_issues = process_squat_video(cap, out, frame_width, frame_height)
        elif exercise_type == "Pushup":
            reps_counted, form_issues = process_pushup_video(cap, out, frame_width, frame_height)
        else:
            reps_counted = 0
            form_issues = set()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                cv2.putText(frame, "CLASSIFICATION FAILED", (frame_width // 2 - 150, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                out.write(frame)

        overall_feedback = create_feedback(exercise_type, reps_counted, form_issues)
        
    except Exception as e:
        os.unlink(input_path)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")
    finally:
        cap.release()
        out.release()
        os.unlink(input_path) # Clean up the temporary input file

    # 4. Return results: processed_video_url uses the static mount path
    processed_video_url = f"/{PROCESSED_VIDEOS_DIR}/{processed_filename}"
    
    return {
        "exercise_classified": exercise_type,
        "reps_counted": reps_counted,
        "overall_feedback": overall_feedback,
        "processed_video_url": processed_video_url
    }

# To run the backend: uvicorn backend:app --reload --host 0.0.0.0 --port 8000