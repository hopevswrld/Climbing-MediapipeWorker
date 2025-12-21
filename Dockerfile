# Minimal MediaPipe Pose Worker for Railway
# Uses Python 3.11 slim for smaller image size and CPU-only inference

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
# These are minimal deps needed for headless video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .

# Expose port (Railway sets PORT env var)
EXPOSE 8080

# Run the FastAPI server
# Railway will set PORT environment variable
CMD ["python", "server.py"]

