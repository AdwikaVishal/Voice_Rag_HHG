FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy backend and frontend
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Pre-pull the Ollama model (optional but saves time on first request)
# RUN ollama pull qwen2.5:3b

# Expose the port Hugging Face expects
EXPOSE 7860

# Start the server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]