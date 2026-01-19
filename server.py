"""
MediaPipe Pose Analysis Worker

PURE BACKGROUND WORKER - NO HTTP, NO PORTS, NO SERVERS
Just an infinite loop that polls the database and processes videos.
"""

import os
import math
import signal
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from typing import Optional, Tuple

import cv2
import ffmpeg
import mediapipe as mp
import numpy as np
from supabase import create_client, Client


# =============================================================================
# CONFIGURATION
# =============================================================================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

TARGET_FPS = 15
KEYFRAME_ANGLE_DELTA = 20
STRAIGHT_ARM_THRESHOLD = 160
POLL_INTERVAL = 5

# MediaPipe Confidence Settings
MIN_DETECTION_CONFIDENCE = 0.5   # Minimum confidence for initial pose detection
MIN_TRACKING_CONFIDENCE = 0.7    # Minimum confidence for tracking between frames
MIN_VISIBILITY_THRESHOLD = 0.5   # Minimum visibility to mark a landmark as "confident"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Graceful shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    signal_name = signal.Signals(signum).name
    print(f"\n[Worker] Received {signal_name}, shutting down gracefully...", flush=True)
    shutdown_requested = True


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

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
# HDR TO SDR TRANSCODING
# =============================================================================

def is_hdr_video(video_path: str) -> Tuple[bool, dict]:
    """
    Check if a video is HDR by examining color transfer characteristics.
    Returns (is_hdr, metadata_dict).
    """
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (s for s in probe['streams'] if s['codec_type'] == 'video'),
            None
        )

        if not video_stream:
            return False, {}

        # Extract color metadata
        color_transfer = video_stream.get('color_transfer', '')
        color_primaries = video_stream.get('color_primaries', '')
        color_space = video_stream.get('color_space', '')

        metadata = {
            'color_transfer': color_transfer,
            'color_primaries': color_primaries,
            'color_space': color_space,
            'width': video_stream.get('width'),
            'height': video_stream.get('height'),
            'codec': video_stream.get('codec_name'),
        }

        # HDR indicators: PQ (SMPTE 2084), HLG, or bt2020
        hdr_transfers = ['smpte2084', 'arib-std-b67', 'smpte-st-2084', 'bt2020-10', 'bt2020-12']
        hdr_primaries = ['bt2020']

        is_hdr = (
            color_transfer.lower() in hdr_transfers or
            color_primaries.lower() in hdr_primaries or
            'bt2020' in color_space.lower()
        )

        return is_hdr, metadata

    except Exception as e:
        print(f"[HDR] Error probing video: {e}", flush=True)
        return False, {}


def transcode_hdr_to_sdr(input_path: str, output_path: str) -> bool:
    """
    Transcode HDR video to SDR using tone mapping.
    Uses zscale for color space conversion and Hable tone mapping.
    """
    try:
        print(f"[HDR] Transcoding to SDR...", flush=True)

        # Build the filter chain for HDR to SDR conversion
        # This handles PQ/HLG to SDR conversion with proper tone mapping
        filter_chain = (
            # Convert to linear light
            "zscale=t=linear:npl=100,"
            # Convert to floating point for processing
            "format=gbrpf32le,"
            # Convert to bt709 primaries
            "zscale=p=bt709,"
            # Apply tone mapping (Hable is good for natural look)
            "tonemap=tonemap=hable:desat=0:peak=100,"
            # Convert to bt709 transfer and matrix
            "zscale=t=bt709:m=bt709:r=tv,"
            # Convert to standard pixel format
            "format=yuv420p"
        )

        # Run FFmpeg with the filter chain
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                vf=filter_chain,
                vcodec='libx264',
                crf=23,
                preset='medium',
                acodec='aac',
                audio_bitrate='128k',
                movflags='+faststart'
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        print(f"[HDR] Transcoding complete", flush=True)
        return True

    except ffmpeg.Error as e:
        print(f"[HDR] FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}", flush=True)
        # Fallback: try simpler conversion without zscale (for non-HDR but wide-gamut videos)
        try:
            print(f"[HDR] Trying fallback conversion...", flush=True)
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_path,
                    vf="format=yuv420p",
                    vcodec='libx264',
                    crf=23,
                    preset='medium',
                    acodec='aac',
                    audio_bitrate='128k',
                    movflags='+faststart',
                    colorspace='bt709',
                    color_primaries='bt709',
                    color_trc='bt709'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            print(f"[HDR] Fallback conversion complete", flush=True)
            return True
        except Exception as e2:
            print(f"[HDR] Fallback also failed: {e2}", flush=True)
            return False
    except Exception as e:
        print(f"[HDR] Transcoding error: {e}", flush=True)
        return False


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
                "confident": lm.visibility >= MIN_VISIBILITY_THRESHOLD
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
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
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

def process_next_job():
    """Fetch and process one pending job. Returns True if a job was processed."""
    job = fetch_pending_job()

    if not job:
        return False

    job_id = job["id"]
    file_path = job["file_path"]
    user_id = job.get("user_id")

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

            # Check for HDR and transcode if needed
            print(f"[Job] Checking video format...", flush=True)
            update_job_status(job_id, "processing", 15)

            is_hdr, video_meta = is_hdr_video(local_path)
            processing_path = local_path
            final_file_path = file_path

            if is_hdr:
                print(f"[Job] HDR video detected: {video_meta}", flush=True)
                update_job_status(job_id, "processing", 20)

                # Create SDR version
                sdr_filename = filename.rsplit('.', 1)[0] + '_sdr.mp4'
                sdr_path = os.path.join(tmpdir, sdr_filename)

                if transcode_hdr_to_sdr(local_path, sdr_path):
                    # Upload SDR version to storage
                    print(f"[Job] Uploading SDR version...", flush=True)
                    update_job_status(job_id, "processing", 25)

                    # New path for SDR video
                    sdr_storage_path = file_path.rsplit('.', 1)[0] + '_sdr.mp4'

                    with open(sdr_path, 'rb') as f:
                        sdr_data = f.read()

                    # Upload to storage
                    upload_response = supabase.storage.from_("analysis-videos").upload(
                        sdr_storage_path,
                        sdr_data,
                        {"content-type": "video/mp4", "upsert": "true"}
                    )

                    # Update database record with new file path
                    supabase.table("analysis_videos").update({
                        "file_path": sdr_storage_path,
                        "original_file_path": file_path,  # Keep reference to original
                        "hdr_converted": True
                    }).eq("id", job_id).execute()

                    processing_path = sdr_path
                    final_file_path = sdr_storage_path
                    print(f"[Job] SDR version uploaded: {sdr_storage_path}", flush=True)
                else:
                    print(f"[Job] HDR conversion failed, using original", flush=True)
            else:
                print(f"[Job] Standard SDR video, no conversion needed", flush=True)

            print(f"[Job] Running analysis...", flush=True)
            update_job_status(job_id, "processing", 30)

            result = process_video(processing_path)
            result["video_path"] = final_file_path
            result["job_id"] = job_id
            result["hdr_detected"] = is_hdr
            result["hdr_converted"] = is_hdr and (processing_path != local_path)

            update_job_status(job_id, "completed", 100, analysis_result=result)
            print(f"[Job] Completed: {job_id}", flush=True)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[Job] Failed: {error_msg}", flush=True)
        traceback.print_exc()
        update_job_status(job_id, "failed", 0, error_message=error_msg)

    return True


# =============================================================================
# VIDEO TRANSCODING QUEUE PROCESSING (for post videos)
# =============================================================================

# Track if we've already warned about missing transcoding queue table
_transcode_table_warning_shown = False


def fetch_pending_transcode_job() -> Optional[dict]:
    """
    Fetch a pending video transcoding job from the queue.
    Returns None if no jobs are pending or if the table doesn't exist yet.
    """
    global _transcode_table_warning_shown
    try:
        response = (
            supabase.table("video_transcoding_queue")
            .select("*")
            .eq("status", "pending")
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None
    except Exception as e:
        error_str = str(e).lower()
        # Handle case where table doesn't exist yet (migration not run)
        if "relation" in error_str and "does not exist" in error_str:
            if not _transcode_table_warning_shown:
                print("[Transcode] video_transcoding_queue table not found. "
                      "Run migration to enable post video transcoding.", flush=True)
                _transcode_table_warning_shown = True
            return None
        print(f"[Transcode] Error fetching job: {e}", flush=True)
        return None


def update_transcode_status(job_id: str, status: str, target_path: str = None, error_message: str = None):
    """Update the status of a transcoding job."""
    update_data = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    if target_path:
        update_data["target_path"] = target_path
    if error_message:
        update_data["error_message"] = error_message

    try:
        supabase.table("video_transcoding_queue").update(update_data).eq("id", job_id).execute()
    except Exception as e:
        print(f"[Transcode] Failed to update job {job_id}: {e}", flush=True)


def process_transcode_job():
    """Process one pending transcoding job. Returns True if a job was processed."""
    job = fetch_pending_transcode_job()

    if not job:
        return False

    job_id = job["id"]
    source_bucket = job["source_bucket"]
    source_path = job["source_path"]
    reference_table = job.get("reference_table")
    reference_id = job.get("reference_id")

    print(f"[Transcode] Starting: {job_id} from {source_bucket}/{source_path}", flush=True)
    update_transcode_status(job_id, "processing")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download video
            filename = source_path.split("/")[-1] if "/" in source_path else source_path
            local_path = os.path.join(tmpdir, filename)

            response = supabase.storage.from_(source_bucket).download(source_path)
            with open(local_path, "wb") as f:
                f.write(response)

            # Check if HDR
            is_hdr, video_meta = is_hdr_video(local_path)

            if not is_hdr:
                print(f"[Transcode] Video is not HDR, skipping", flush=True)
                update_transcode_status(job_id, "completed", target_path=source_path)
                return True

            print(f"[Transcode] HDR detected: {video_meta}", flush=True)

            # Transcode to SDR
            sdr_filename = filename.rsplit('.', 1)[0] + '_sdr.mp4'
            sdr_path = os.path.join(tmpdir, sdr_filename)

            if not transcode_hdr_to_sdr(local_path, sdr_path):
                raise Exception("Transcoding failed")

            # Upload SDR version
            sdr_storage_path = source_path.rsplit('.', 1)[0] + '_sdr.mp4'

            with open(sdr_path, 'rb') as f:
                sdr_data = f.read()

            supabase.storage.from_(source_bucket).upload(
                sdr_storage_path,
                sdr_data,
                {"content-type": "video/mp4", "upsert": "true"}
            )

            # Update the reference table with new path
            if reference_table and reference_id:
                if reference_table == "posts":
                    supabase.table("posts").update({
                        "media_url": sdr_storage_path,
                        "original_media_url": source_path
                    }).eq("id", reference_id).execute()
                elif reference_table == "analysis_videos":
                    supabase.table("analysis_videos").update({
                        "file_path": sdr_storage_path,
                        "original_file_path": source_path,
                        "hdr_converted": True
                    }).eq("id", reference_id).execute()

            update_transcode_status(job_id, "completed", target_path=sdr_storage_path)
            print(f"[Transcode] Completed: {job_id}", flush=True)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[Transcode] Failed: {error_msg}", flush=True)
        traceback.print_exc()
        update_transcode_status(job_id, "failed", error_message=error_msg)

    return True


# =============================================================================
# MAIN - INFINITE LOOP, NO HTTP, NO SERVERS
# =============================================================================

def main():
    """
    Main worker loop that polls for pending jobs and processes them.
    Handles both analysis jobs (MediaPipe pose detection) and transcoding jobs (HDR to SDR).
    Supports graceful shutdown via SIGTERM/SIGINT signals.
    """
    print("============================================================", flush=True)
    print("MEDIAPIPE POSE WORKER + VIDEO TRANSCODER", flush=True)
    print("============================================================", flush=True)
    print(f"Poll interval: {POLL_INTERVAL}s", flush=True)
    print("Graceful shutdown: SIGTERM or SIGINT (Ctrl+C)", flush=True)
    print("============================================================", flush=True)

    while not shutdown_requested:
        try:
            # Process analysis jobs (with pose detection)
            analysis_processed = process_next_job()

            # Check for shutdown between job types
            if shutdown_requested:
                break

            # Process transcoding jobs (HDR to SDR only)
            transcode_processed = process_transcode_job()

            if not analysis_processed and not transcode_processed:
                # Sleep in small increments to allow faster shutdown response
                for _ in range(POLL_INTERVAL):
                    if shutdown_requested:
                        break
                    time.sleep(1)
                if not shutdown_requested:
                    print("[Worker] No pending jobs. Sleeping 5s...", flush=True)
        except Exception:
            print("[Worker] ERROR", flush=True)
            traceback.print_exc()
            if not shutdown_requested:
                time.sleep(POLL_INTERVAL)

    print("[Worker] Shutdown complete.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
