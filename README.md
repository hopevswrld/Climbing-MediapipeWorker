# 🧗 Climbing MediaPipe Worker

A background video analysis worker that uses **Google MediaPipe** to extract pose data from climbing videos. Built for the [Ascender](https://github.com/hopevswrld/climbing-social-events-eli) climbing app.

## What It Does

This worker continuously polls a Supabase database for pending video analysis jobs, downloads climbing videos, runs pose estimation using MediaPipe, and stores the results back in the database.

**Extracted Data:**
- 33 body landmarks (x, y, z coordinates + visibility confidence)
- Joint angles (arms, legs, hips)
- Center of mass tracking
- Keyframe detection (significant movement changes)
- Analysis summary (arm extension %, detection rate, etc.)

## Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   Ascender App  │ ───► │    Supabase     │ ◄─── │  MediaPipe      │
│   (iOS/Android) │      │  (DB + Storage) │      │  Worker (AWS)   │
└─────────────────┘      └─────────────────┘      └─────────────────┘
        │                        │                        │
        │  Upload video          │  Poll for pending jobs │
        │  Create job record     │  Download video        │
        └────────────────────────┤  Process with MediaPipe│
                                 │  Store results         │
                                 └────────────────────────┘
```

## Requirements

- Python 3.11+
- Supabase project with:
  - `analysis_videos` table
  - `analysis-videos` storage bucket

## Environment Variables

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
WORKER_ID=worker-1                          # Optional: instance identifier
```

## AWS Secrets Manager

The worker fetches `WORKER_API_KEY` from AWS Secrets Manager:
- **Secret Name:** `WorkerKey`
- **Region:** `us-east-1`
- **Format:** JSON with `WORKER_API_KEY` field, or plain string

This key must match the `WORKER_API_KEY` secret set in Supabase Edge Functions. If the secret cannot be fetched, the worker runs in legacy mode without credit tracking.

**IAM Permissions Required:**
```json
{
  "Effect": "Allow",
  "Action": "secretsmanager:GetSecretValue",
  "Resource": "arn:aws:secretsmanager:us-east-1:*:secret:WorkerKey*"
}
```

## Local Development

```bash
# Clone the repo
git clone https://github.com/hopevswrld/Climbing-MediapipeWorker.git
cd Climbing-MediapipeWorker

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="your-key"

# Run the worker
python server.py
```

## Docker

```bash
# Build
docker build -t mediapipe-worker .

# Run
docker run -e SUPABASE_URL="..." -e SUPABASE_SERVICE_ROLE_KEY="..." mediapipe-worker
```

## AWS Deployment

Deploy as an ECS task or EC2 instance with environment variables configured.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_FPS` | 15 | Frames per second to sample |
| `POLL_INTERVAL` | 5 | Seconds between job polls |
| `MIN_DETECTION_CONFIDENCE` | 0.5 | MediaPipe detection threshold |
| `MIN_TRACKING_CONFIDENCE` | 0.7 | MediaPipe tracking threshold |
| `KEYFRAME_ANGLE_DELTA` | 20° | Angle change to mark keyframe |

## Output Schema

```json
{
  "video_metadata": {
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "duration_seconds": 45.2,
    "processed_frames": 678,
    "detection_rate": 95.3
  },
  "pose_sequence": [
    {
      "frame_index": 0,
      "timestamp_ms": 0,
      "landmarks": {
        "left_shoulder": { "x": 0.45, "y": 0.32, "z": -0.12, "visibility": 0.98 },
        "left_elbow": { "x": 0.38, "y": 0.45, "z": -0.08, "visibility": 0.95 }
      },
      "metrics": {
        "left_arm_angle": 142.5,
        "right_arm_angle": 165.2,
        "hip_depth": -0.15,
        "center_of_mass": { "x": 0.5, "y": 0.6 }
      }
    }
  ],
  "keyframes": [15, 45, 78, 120],
  "analysis_summary": {
    "arm_extension": { "mean": 155.2, "straight_arm_percentage": 42.1 },
    "leg_extension": { "mean": 148.7 },
    "hip_to_wall": { "mean": -0.12 }
  }
}
```

## Database Schema

```sql
CREATE TABLE analysis_videos (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id),
  file_path TEXT NOT NULL,
  status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed
  progress INTEGER DEFAULT 0,
  analysis_result JSONB,
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

## License

MIT

---

Part of the **Ascender** climbing platform 🧗‍♂️


