"""
MediaPipe Pose Analysis Worker for Railway

Background worker with a minimal HTTP health check to satisfy Railway.
"""

import os
import math
import tempfile
import time
import traceback
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from supabase import create_client, Client


# =============================================================================
# CONFIGURATION
# =============================================================================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
PORT = int(os.environ.get("PORT", 8080))

TARGET_FPS = 10
KEYFRAME_ANGLE_DELTA = 20
STRAIGHT_ARM_THRESHOLD = 160
POLL_INTERVAL = 5

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

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


# =============================================================================
# MINIMAL HEALTH CHECK SERVER
# =============================================================================

class HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that responds to health checks."""
    
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass


def start_health_server():
    """Start a minimal HTTP server for Railway health checks."""
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    print(f"[Health] Server listening on port {PORT}", flush=True)
    server.serve_forever()


# =============================================================================
# MEDIAPIPE PROCESSING FUNCTIONS
# =============================================================================

def calculate_angle(p1: dict, p2: dict, p3: dict) -> Optional[float]:
    if not all(p.get("confident", False) for p in [p1, p2, p3]):
        return None
    v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"], p1["z"] - p2["z"]])
    v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"], p3["z"] - p2["z"]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return round(math.degrees(math.acos(cos_angle)), 1)


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
    
    return {
        "center_of_mass": center_of_mass,
        "hip_depth": hip_depth,
        "left_arm_angle": left_arm_angle,
        "right_arm_angle": right_arm_angle,
        "left_leg_angle": left_leg_angle,
        "right_leg_angle": right_leg_angle,
        "hip_angle": hip_angle,
        "left_foot_position": {
            "relative_x": round(left_ankle.get("x", 0) - center_of_mass["x"], 4),
            "relative_y": round(left_ankle.get("y", 0) - center_of_mass["y"], 4)
        },
        "right_foot_position": {
            "relative_x": round(right_ankle.get("x", 0) - center_of_mass["x"], 4),
            "relative_y": round(right_ankle.get("y", 0) - center_of_mass["y"], 4)
        }
    }


def detect_keyframes(pose_sequence: list) -> list:
    keyframes = []
    prev_angle = None
    for frame in pose_sequence:
        current_angle = frame.get("metrics", {}).get("right_arm_angle")
        if current_angle is not None and prev_angle is not None:
            if abs(current_angle - prev_angle) > KEYFRAME_ANGLE_DELTA:
                keyframes.append(frame["frame_index"])
        prev_angle = current_angle if current_angle is not None else prev_angle
    return keyframes


def compute_analysis_summary(pose_sequence: list) -> dict:
    total_frames = len(pose_sequence)
    frames_with_pose = sum(1 for f in pose_sequence if f.get("landmarks"))
    
    arm_angles = [f.get("metrics", {}).get("right_arm_angle") 
                  for f in pose_sequence 
                  if f.get("metrics", {}).get("right_arm_angle") is not None]
    
    leg_angles = []
    for frame in pose_sequence:
        for key in ["left_leg_angle", "right_leg_angle"]:
            angle = frame.get("metrics", {}).get(key)
            if angle is not None:
                leg_angles.append(angle)
    
    hip_depths = [f.get("metrics", {}).get("hip_depth") 
                  for f in pose_sequence 
                  if f.get("metrics", {}).get("hip_depth") is not None]
    
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
                pose_sequence.append({
                    "frame_index": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "landmarks": landmarks,
                    "metrics": metrics,
                    "segmentation_available": False
                })
            frame_idx += 1
    
    cap.release()
    
    video_metadata["processed_frames"] = len([f for f in pose_sequence if f["landmarks"]])
    video_metadata["sampled_frames"] = len(pose_sequence)
    
    return {
        "video_metadata": video_metadata,
        "pose_sequence": pose_sequence,
        "keyframes": detect_keyframes(pose_sequence),
        "analysis_summary": compute_analysis_summary(pose_sequence)
    }


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def update_job_status(job_id: str, status: str, progress: int,
                      analysis_result: dict = None, error_message: str = None):
    update_data = {"status": status, "progress": progress}
    if analysis_result is not None:
        update_data["analysis_result"] = analysis_result
    if error_message is not None:
        update_data["error_message"] = error_message
    
    try:
        supabase.table("analysis_videos").update(update_data).eq("id", job_id).execute()
        print(f"[DB] Job {job_id}: status={status}, progress={progress}", flush=True)
    except Exception as e:
        print(f"[DB] Failed to update job {job_id}: {e}", flush=True)


def fetch_pending_job() -> Optional[dict]:
    try:
        response = (
            supabase.table("analysis_videos")
            .select("*")
            .eq("status", "pending")
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"[DB] Error fetching job: {e}", flush=True)
        return None


# =============================================================================
# JOB PROCESSING
# =============================================================================

def process_job(job: dict):
    job_id = job["id"]
    file_path = job["file_path"]
    
    print(f"[Job] Starting: {job_id}", flush=True)
    update_job_status(job_id, "processing", 5)
    
    try:
        print(f"[Job] Downloading video...", flush=True)
        update_job_status(job_id, "processing", 10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = file_path.split("/")[-1] if "/" in file_path else file_path
            local_path = os.path.join(tmpdir, filename)
            
            response = supabase.storage.from_("analysis-videos").download(file_path)
            with open(local_path, "wb") as f:
                f.write(response)
            
            print(f"[Job] Running analysis...", flush=True)
            update_job_status(job_id, "processing", 30)
            
            result = process_video(local_path)
            result["video_path"] = file_path
            result["job_id"] = job_id
            
            update_job_status(job_id, "completed", 100, analysis_result=result)
            print(f"[Job] Completed: {job_id}", flush=True)
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[Job] Failed: {error_msg}", flush=True)
        print(traceback.format_exc(), flush=True)
        update_job_status(job_id, "failed", 0, error_message=error_msg)


# =============================================================================
# MAIN WORKER LOOP
# =============================================================================

def worker_loop():
    """Main worker loop - runs forever, polls for jobs."""
    print("[Worker] Starting worker loop...", flush=True)
    
    while True:
        try:
            job = fetch_pending_job()
            if job:
                process_job(job)
            else:
                time.sleep(POLL_INTERVAL)
        except Exception as e:
            print(f"[Worker] Error: {e}", flush=True)
            time.sleep(POLL_INTERVAL)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("MEDIAPIPE POSE WORKER", flush=True)
    print("=" * 60, flush=True)
    print(f"Port: {PORT}", flush=True)
    print(f"Poll interval: {POLL_INTERVAL}s", flush=True)
    print("=" * 60, flush=True)
    
    # Start health check server in background thread
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    
    # Run worker loop in main thread (blocks forever)
    worker_loop()
