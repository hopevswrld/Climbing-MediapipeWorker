# Minimal MediaPipe Pose Worker for Railway
# Uses Python 3.11 slim for smaller image size and CPU-only inference

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
# These are minimal deps needed for headless video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .

# Railway injects PORT at runtime - we don't hardcode EXPOSE
# The app will bind to whatever PORT Railway provides

# Use SHELL FORM (not exec form) so $PORT expands at runtime
# Railway injects PORT env var - we default to 8080 if not set
# Running uvicorn directly ensures proper signal handling
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]

