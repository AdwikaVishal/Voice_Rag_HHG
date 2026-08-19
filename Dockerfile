FROM python:3.11-slim

# Install system dependencies (ffmpeg is critical for audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy backend and frontend code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose the port (Railway will provide the actual port via $PORT)
EXPOSE 8000

# Start the server, binding to the port provided by Railway
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
