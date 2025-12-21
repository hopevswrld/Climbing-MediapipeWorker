"""
Minimal MediaPipe Pose Worker for Railway
Processes videos from Supabase Storage and outputs pose analysis JSON.
Includes background worker that polls analysis_videos table for pending jobs.
"""
import os
import json
import math
import tempfile
import threading
import time
import traceback
from typing import Optional
from contextlib import asynccontextmanager

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

# --- Configuration ---
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
TARGET_FPS = 10  # Sample rate for frame extraction
KEYFRAME_ANGLE_DELTA = 20  # Degrees threshold for keyframe detection
STRAIGHT_ARM_THRESHOLD = 160  # Degrees for "straight arm" classification
POLL_INTERVAL = 5  # Seconds between polling for new jobs

# --- Initialize Supabase client ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose

# --- Landmark name mapping (MediaPipe index -> name) ---
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# --- Background worker control ---
worker_running = True


def calculate_angle(p1: dict, p2: dict, p3: dict) -> Optional[float]:
    """
    Calculate angle at p2 formed by p1-p2-p3.
    Returns angle in degrees, or None if any point has low visibility.
    """
    if not all(p.get("confident", False) for p in [p1, p2, p3]):
        return None
    
    v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"], p1["z"] - p2["z"]])
    v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"], p3["z"] - p2["z"]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))
    return round(angle, 1)


def extract_landmarks(results) -> Optional[dict]:
    """
    Extract landmarks from MediaPipe results into a dictionary.
    Returns None if no pose detected.
    """
    if not results.pose_landmarks:
        return None
    
    landmarks = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        if idx < len(LANDMARK_NAMES):
            landmarks[LANDMARK_NAMES[idx]] = {
                "x": round(lm.x, 4),
                "y": round(lm.y, 4),
                "z": round(lm.z, 4),
                "visibility": round(lm.visibility, 3),
                "confident": lm.visibility >= 0.5
            }
    return landmarks


def compute_metrics(landmarks: dict) -> dict:
    """
    Compute per-frame metrics from landmarks.
    """
    # Center of mass: average of hips
    left_hip = landmarks.get("left_hip", {})
    right_hip = landmarks.get("right_hip", {})
    
    center_of_mass = {
        "x": round((left_hip.get("x", 0) + right_hip.get("x", 0)) / 2, 4),
        "y": round((left_hip.get("y", 0) + right_hip.get("y", 0)) / 2, 4)
    }
    
    # Hip depth: z-axis average of hips
    hip_depth = round((left_hip.get("z", 0) + right_hip.get("z", 0)) / 2, 4)
    
    # Arm angles: shoulder-elbow-wrist
    left_arm_angle = calculate_angle(
        landmarks.get("left_shoulder", {}),
        landmarks.get("left_elbow", {}),
        landmarks.get("left_wrist", {})
    )
    
    right_arm_angle = calculate_angle(
        landmarks.get("right_shoulder", {}),
        landmarks.get("right_elbow", {}),
        landmarks.get("right_wrist", {})
    )
    
    # Leg angles: hip-knee-ankle
    left_leg_angle = calculate_angle(
        landmarks.get("left_hip", {}),
        landmarks.get("left_knee", {}),
        landmarks.get("left_ankle", {})
    )
    
    right_leg_angle = calculate_angle(
        landmarks.get("right_hip", {}),
        landmarks.get("right_knee", {}),
        landmarks.get("right_ankle", {})
    )
    
    # Hip angle: shoulder-hip-knee (using right side as reference)
    hip_angle = calculate_angle(
        landmarks.get("right_shoulder", {}),
        landmarks.get("right_hip", {}),
        landmarks.get("right_knee", {})
    )
    
    # Foot positions relative to hip center
    left_ankle = landmarks.get("left_ankle", {})
    right_ankle = landmarks.get("right_ankle", {})
    
    left_foot_position = {
        "relative_x": round(left_ankle.get("x", 0) - center_of_mass["x"], 4),
        "relative_y": round(left_ankle.get("y", 0) - center_of_mass["y"], 4)
    }
    
    right_foot_position = {
        "relative_x": round(right_ankle.get("x", 0) - center_of_mass["x"], 4),
        "relative_y": round(right_ankle.get("y", 0) - center_of_mass["y"], 4)
    }
    
    return {
        "center_of_mass": center_of_mass,
        "hip_depth": hip_depth,
        "left_arm_angle": left_arm_angle,
        "right_arm_angle": right_arm_angle,
        "left_leg_angle": left_leg_angle,
        "right_leg_angle": right_leg_angle,
        "hip_angle": hip_angle,
        "left_foot_position": left_foot_position,
        "right_foot_position": right_foot_position
    }


def detect_keyframes(pose_sequence: list) -> list:
    """
    Detect keyframes based on right_arm_angle deltas > 20 degrees.
    Returns list of frame indices.
    """
    keyframes = []
    prev_angle = None
    
    for frame in pose_sequence:
        current_angle = frame.get("metrics", {}).get("right_arm_angle")
        if current_angle is not None and prev_angle is not None:
            delta = abs(current_angle - prev_angle)
            if delta > KEYFRAME_ANGLE_DELTA:
                keyframes.append(frame["frame_index"])
        prev_angle = current_angle if current_angle is not None else prev_angle
    
    return keyframes


def compute_analysis_summary(pose_sequence: list) -> dict:
    """
    Compute aggregate analysis metrics from pose sequence.
    """
    total_frames = len(pose_sequence)
    frames_with_pose = sum(1 for f in pose_sequence if f.get("landmarks"))
    
    # Collect right arm angles for extension analysis
    arm_angles = []
    for frame in pose_sequence:
        angle = frame.get("metrics", {}).get("right_arm_angle")
        if angle is not None:
            arm_angles.append(angle)
    
    # Collect leg angles
    leg_angles = []
    for frame in pose_sequence:
        for key in ["left_leg_angle", "right_leg_angle"]:
            angle = frame.get("metrics", {}).get(key)
            if angle is not None:
                leg_angles.append(angle)
    
    # Collect hip depths
    hip_depths = []
    for frame in pose_sequence:
        depth = frame.get("metrics", {}).get("hip_depth")
        if depth is not None:
            hip_depths.append(depth)
    
    # Arm extension stats
    arm_extension = {}
    if arm_angles:
        straight_count = sum(1 for a in arm_angles if a >= STRAIGHT_ARM_THRESHOLD)
        arm_extension = {
            "mean": round(np.mean(arm_angles), 1),
            "min": round(min(arm_angles), 1),
            "max": round(max(arm_angles), 1),
            "straight_arm_percentage": round(100 * straight_count / len(arm_angles), 1)
        }
    
    # Leg extension stats
    leg_extension = {}
    if leg_angles:
        leg_extension = {
            "mean": round(np.mean(leg_angles), 1),
            "min": round(min(leg_angles), 1),
            "max": round(max(leg_angles), 1)
        }
    
    # Hip to wall stats (from hip_depth)
    hip_to_wall = {}
    if hip_depths:
        hip_to_wall = {
            "mean": round(np.mean(hip_depths), 4),
            "min": round(min(hip_depths), 4),
            "max": round(max(hip_depths), 4)
        }
    
    return {
        "total_frames_with_pose": frames_with_pose,
        "detection_rate": round(100 * frames_with_pose / total_frames, 1) if total_frames > 0 else 0,
        "arm_extension": arm_extension,
        "leg_extension": leg_extension,
        "hip_to_wall": hip_to_wall
    }


def process_video(video_path: str) -> dict:
    """
    Process video file and extract pose data.
    Returns dict with video_metadata, pose_sequence, keyframes, analysis_summary.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps > 0 else 0
    
    # Calculate frame sampling interval for target FPS
    frame_interval = max(1, int(fps / TARGET_FPS))
    
    video_metadata = {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": round(duration_seconds, 2),
        "processed_frames": 0,
        "sampled_frames": 0,
        "target_fps": TARGET_FPS
    }
    
    pose_sequence = []
    
    # Process frames with MediaPipe
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=lite, 1=full, 2=heavy (we use 1 for balance)
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_idx = 0
        sampled_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at target FPS
            if frame_idx % frame_interval == 0:
                sampled_count += 1
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                # Extract landmarks and compute metrics
                landmarks = extract_landmarks(results)
                metrics = compute_metrics(landmarks) if landmarks else {}
                
                timestamp_ms = int((frame_idx / fps) * 1000) if fps > 0 else 0
                
                frame_data = {
                    "frame_index": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "landmarks": landmarks,
                    "metrics": metrics,
                    "segmentation_available": False
                }
                pose_sequence.append(frame_data)
            
            frame_idx += 1
    
    cap.release()
    
    # Update metadata
    video_metadata["processed_frames"] = len([f for f in pose_sequence if f["landmarks"]])
    video_metadata["sampled_frames"] = len(pose_sequence)
    
    # Detect keyframes and compute summary
    keyframes = detect_keyframes(pose_sequence)
    analysis_summary = compute_analysis_summary(pose_sequence)
    
    return {
        "video_metadata": video_metadata,
        "pose_sequence": pose_sequence,
        "keyframes": keyframes,
        "analysis_summary": analysis_summary
    }


# =============================================================================
# BACKGROUND WORKER - Polls analysis_videos table for pending jobs
# =============================================================================

def update_job_status(job_id: str, status: str, progress: int, 
                      analysis_result: dict = None, error_message: str = None):
    """
    Update the status of a job in the analysis_videos table.
    """
    update_data = {
        "status": status,
        "progress": progress,
        "updated_at": "now()"
    }
    
    if analysis_result is not None:
        update_data["analysis_result"] = analysis_result
    
    if error_message is not None:
        update_data["error_message"] = error_message
    
    try:
        supabase.table("analysis_videos").update(update_data).eq("id", job_id).execute()
    except Exception as e:
        print(f"[Worker] Failed to update job {job_id}: {e}")


def process_pending_job(job: dict):
    """
    Process a single pending job from the analysis_videos table.
    Downloads video, runs pose analysis, updates database with results.
    """
    job_id = job["id"]
    file_path = job["file_path"]
    
    print(f"[Worker] Processing job {job_id}: {file_path}")
    
    # Step 1: Mark as processing with progress = 5
    update_job_status(job_id, "processing", 5)
    
    try:
        # Step 2: Download video from Supabase Storage
        update_job_status(job_id, "processing", 10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract filename from path
            filename = file_path.split("/")[-1] if "/" in file_path else file_path
            local_video_path = os.path.join(tmpdir, filename)
            
            # Download from analysis-videos bucket
            print(f"[Worker] Downloading video: {file_path}")
            response = supabase.storage.from_("analysis-videos").download(file_path)
            with open(local_video_path, "wb") as f:
                f.write(response)
            
            update_job_status(job_id, "processing", 20)
            
            # Step 3: Process video with MediaPipe
            print(f"[Worker] Running pose analysis...")
            update_job_status(job_id, "processing", 40)
            
            result = process_video(local_video_path)
            
            update_job_status(job_id, "processing", 70)
            
            # Add metadata to result
            result["video_path"] = file_path
            result["job_id"] = job_id
            
            update_job_status(job_id, "processing", 90)
            
            # Step 4: Mark as completed with full result
            print(f"[Worker] Job {job_id} completed successfully")
            update_job_status(job_id, "completed", 100, analysis_result=result)
    
    except Exception as e:
        # Step 5: On error, mark as failed
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[Worker] Job {job_id} failed: {error_msg}")
        print(traceback.format_exc())
        update_job_status(job_id, "failed", 0, error_message=error_msg)


def background_worker_loop():
    """
    Background worker loop that continuously polls for pending jobs.
    Runs in a separate thread, started when the FastAPI app starts.
    """
    global worker_running
    
    print("[Worker] Background worker started")
    
    while worker_running:
        try:
            # Query for ONE pending job, ordered by created_at (oldest first)
            response = (
                supabase.table("analysis_videos")
                .select("*")
                .eq("status", "pending")
                .order("created_at", desc=False)
                .limit(1)
                .execute()
            )
            
            jobs = response.data
            
            if not jobs:
                # No pending jobs, sleep and continue
                time.sleep(POLL_INTERVAL)
                continue
            
            # Process the found job
            job = jobs[0]
            process_pending_job(job)
        
        except Exception as e:
            # Log error but keep the worker running
            print(f"[Worker] Error in worker loop: {e}")
            print(traceback.format_exc())
            time.sleep(POLL_INTERVAL)
    
    print("[Worker] Background worker stopped")


# =============================================================================
# FASTAPI APP WITH LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Starts background worker on startup, stops it on shutdown.
    """
    global worker_running
    
    # Startup: Start the background worker thread
    worker_running = True
    worker_thread = threading.Thread(target=background_worker_loop, daemon=True)
    worker_thread.start()
    print("[App] FastAPI started, background worker running")
    
    yield
    
    # Shutdown: Signal the worker to stop
    worker_running = False
    print("[App] FastAPI shutting down, stopping worker...")


# --- FastAPI app with lifespan ---
app = FastAPI(title="MediaPipe Pose Worker", lifespan=lifespan)


class ProcessRequest(BaseModel):
    bucket: str
    path: str


class ProcessResponse(BaseModel):
    video_path: str
    video_metadata: dict
    pose_sequence: list
    keyframes: list
    analysis_summary: dict


@app.get("/health")
async def health():
    """Health check endpoint for Railway."""
    return {"status": "healthy", "worker_running": worker_running}


@app.post("/process", response_model=ProcessResponse)
async def process_endpoint(request: ProcessRequest):
    """
    Manual endpoint to process a video from Supabase Storage.
    Downloads video, runs MediaPipe Pose, uploads JSON result.
    Note: The background worker handles automatic processing from the database.
    """
    bucket = request.bucket
    path = request.path
    
    # Parse user_id and uuid from path (format: {user_id}/{uuid}.mp4)
    try:
        parts = path.rsplit("/", 1)
        user_id = parts[0] if len(parts) > 1 else ""
        filename = parts[-1]
        uuid = filename.rsplit(".", 1)[0]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path format")
    
    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        local_video_path = os.path.join(tmpdir, filename)
        
        # Download video from Supabase Storage
        try:
            response = supabase.storage.from_(bucket).download(path)
            with open(local_video_path, "wb") as f:
                f.write(response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")
        
        # Process video with MediaPipe
        try:
            result = process_video(local_video_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")
        
        # Add video path to result
        result["video_path"] = path
        result["video_id"] = uuid
        result["video_key"] = f"{bucket}/{path}"
        
        # Prepare output JSON
        output_json = json.dumps(result, indent=2)
        output_path = f"{user_id}/{uuid}.json"
        
        # Upload result to Supabase Storage
        try:
            supabase.storage.from_("analysis-results").upload(
                output_path,
                output_json.encode("utf-8"),
                file_options={"content-type": "application/json", "upsert": "true"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload result: {str(e)}")
    
    return ProcessResponse(
        video_path=path,
        video_metadata=result["video_metadata"],
        pose_sequence=result["pose_sequence"],
        keyframes=result["keyframes"],
        analysis_summary=result["analysis_summary"]
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
