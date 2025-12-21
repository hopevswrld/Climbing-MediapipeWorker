"""
Minimal MediaPipe Pose Worker for Railway
Processes videos from Supabase Storage and outputs pose analysis JSON.

LIFECYCLE: Background worker runs in a non-daemon thread to keep process alive.
"""
import os
import json
import math
import tempfile
import threading
import time
import traceback
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

# --- Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
TARGET_FPS = 10
KEYFRAME_ANGLE_DELTA = 20
STRAIGHT_ARM_THRESHOLD = 160
POLL_INTERVAL = 5

# --- Initialize Supabase client (only if credentials exist) ---
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose

# --- Landmark name mapping ---
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

# --- FastAPI app ---
app = FastAPI(title="MediaPipe Pose Worker")

# --- Worker thread reference ---
worker_thread: Optional[threading.Thread] = None


def calculate_angle(p1: dict, p2: dict, p3: dict) -> Optional[float]:
    if not all(p.get("confident", False) for p in [p1, p2, p3]):
        return None
    v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"], p1["z"] - p2["z"]])
    v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"], p3["z"] - p2["z"]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))
    return round(angle, 1)


def extract_landmarks(results) -> Optional[dict]:
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
    left_hip = landmarks.get("left_hip", {})
    right_hip = landmarks.get("right_hip", {})
    center_of_mass = {
        "x": round((left_hip.get("x", 0) + right_hip.get("x", 0)) / 2, 4),
        "y": round((left_hip.get("y", 0) + right_hip.get("y", 0)) / 2, 4)
    }
    hip_depth = round((left_hip.get("z", 0) + right_hip.get("z", 0)) / 2, 4)
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
    hip_angle = calculate_angle(
        landmarks.get("right_shoulder", {}),
        landmarks.get("right_hip", {}),
        landmarks.get("right_knee", {})
    )
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
    total_frames = len(pose_sequence)
    frames_with_pose = sum(1 for f in pose_sequence if f.get("landmarks"))
    arm_angles = []
    for frame in pose_sequence:
        angle = frame.get("metrics", {}).get("right_arm_angle")
        if angle is not None:
            arm_angles.append(angle)
    leg_angles = []
    for frame in pose_sequence:
        for key in ["left_leg_angle", "right_leg_angle"]:
            angle = frame.get("metrics", {}).get(key)
            if angle is not None:
                leg_angles.append(angle)
    hip_depths = []
    for frame in pose_sequence:
        depth = frame.get("metrics", {}).get("hip_depth")
        if depth is not None:
            hip_depths.append(depth)
    arm_extension = {}
    if arm_angles:
        straight_count = sum(1 for a in arm_angles if a >= STRAIGHT_ARM_THRESHOLD)
        arm_extension = {
            "mean": round(np.mean(arm_angles), 1),
            "min": round(min(arm_angles), 1),
            "max": round(max(arm_angles), 1),
            "straight_arm_percentage": round(100 * straight_count / len(arm_angles), 1)
        }
    leg_extension = {}
    if leg_angles:
        leg_extension = {
            "mean": round(np.mean(leg_angles), 1),
            "min": round(min(leg_angles), 1),
            "max": round(max(leg_angles), 1)
        }
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps > 0 else 0
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
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
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
    video_metadata["processed_frames"] = len([f for f in pose_sequence if f["landmarks"]])
    video_metadata["sampled_frames"] = len(pose_sequence)
    keyframes = detect_keyframes(pose_sequence)
    analysis_summary = compute_analysis_summary(pose_sequence)
    return {
        "video_metadata": video_metadata,
        "pose_sequence": pose_sequence,
        "keyframes": keyframes,
        "analysis_summary": analysis_summary
    }


# =============================================================================
# BACKGROUND WORKER
# =============================================================================

def update_job_status(job_id: str, status: str, progress: int, 
                      analysis_result: dict = None, error_message: str = None):
    if not supabase:
        return
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
        print(f"[Worker] Failed to update job {job_id}: {e}", flush=True)


def process_pending_job(job: dict):
    job_id = job["id"]
    file_path = job["file_path"]
    print(f"[Worker] Processing job {job_id}: {file_path}", flush=True)
    update_job_status(job_id, "processing", 5)
    try:
        update_job_status(job_id, "processing", 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = file_path.split("/")[-1] if "/" in file_path else file_path
            local_video_path = os.path.join(tmpdir, filename)
            print(f"[Worker] Downloading video: {file_path}", flush=True)
            response = supabase.storage.from_("analysis-videos").download(file_path)
            with open(local_video_path, "wb") as f:
                f.write(response)
            update_job_status(job_id, "processing", 20)
            print(f"[Worker] Running pose analysis...", flush=True)
            update_job_status(job_id, "processing", 40)
            result = process_video(local_video_path)
            update_job_status(job_id, "processing", 70)
            result["video_path"] = file_path
            result["job_id"] = job_id
            update_job_status(job_id, "processing", 90)
            print(f"[Worker] Job {job_id} completed successfully", flush=True)
            update_job_status(job_id, "completed", 100, analysis_result=result)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[Worker] Job {job_id} failed: {error_msg}", flush=True)
        print(traceback.format_exc(), flush=True)
        update_job_status(job_id, "failed", 0, error_message=error_msg)


def process_next_job() -> bool:
    if not supabase:
        return False
    try:
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
            return False
        job = jobs[0]
        process_pending_job(job)
        return True
    except Exception as e:
        print(f"[Worker] Error querying for jobs: {e}", flush=True)
        return False


def background_worker_loop():
    """Infinite loop that polls for pending jobs."""
    print("[Worker] Background worker started", flush=True)
    while True:
        try:
            job_found = process_next_job()
            if not job_found:
                time.sleep(POLL_INTERVAL)
        except Exception as e:
            print(f"[Worker] Error in worker loop: {e}", flush=True)
            time.sleep(POLL_INTERVAL)


def start_background_worker():
    """Start the background worker in a non-daemon thread."""
    global worker_thread
    print("[Startup] Starting background worker...", flush=True)
    worker_thread = threading.Thread(
        target=background_worker_loop,
        name="PoseAnalysisWorker",
        daemon=False
    )
    worker_thread.start()
    print("[Startup] Background worker started", flush=True)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
def root():
    """Root health check for Railway."""
    return {"status": "ok"}


@app.get("/health")
def health():
    """Detailed health check."""
    global worker_thread
    worker_alive = worker_thread is not None and worker_thread.is_alive()
    return {
        "status": "healthy",
        "worker_running": worker_alive
    }


class ProcessRequest(BaseModel):
    bucket: str
    path: str


class ProcessResponse(BaseModel):
    video_path: str
    video_metadata: dict
    pose_sequence: list
    keyframes: list
    analysis_summary: dict


@app.post("/process", response_model=ProcessResponse)
def process_endpoint(request: ProcessRequest):
    """Manual endpoint to process a video."""
    bucket = request.bucket
    path = request.path
    try:
        parts = path.rsplit("/", 1)
        user_id = parts[0] if len(parts) > 1 else ""
        filename = parts[-1]
        uuid = filename.rsplit(".", 1)[0]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path format")
    with tempfile.TemporaryDirectory() as tmpdir:
        local_video_path = os.path.join(tmpdir, filename)
        try:
            response = supabase.storage.from_(bucket).download(path)
            with open(local_video_path, "wb") as f:
                f.write(response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")
        try:
            result = process_video(local_video_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")
        result["video_path"] = path
        result["video_id"] = uuid
        result["video_key"] = f"{bucket}/{path}"
        output_json = json.dumps(result, indent=2)
        output_path = f"{user_id}/{uuid}.json"
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


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment
    port = int(os.environ.get("PORT", 8080))
    print(f"[Main] PORT = {port}", flush=True)
    
    # Start background worker BEFORE uvicorn
    # This ensures the worker thread is running
    start_background_worker()
    
    # Run uvicorn - this blocks forever
    print(f"[Main] Starting uvicorn on 0.0.0.0:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
