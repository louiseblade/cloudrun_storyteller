# Use NVIDIA's CUDA base image with Ubuntu (CUDA 12.4)
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4 support
RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu124

# Install other Python dependencies
RUN pip3 install flask flask-cors gtts transformers accelerate

# Copy the Flask app
COPY app.py .

# Create directories for audio and offload (model will be mounted)
RUN mkdir -p /app/audio /app/offload /app/goat_70b_local

# Expose the port Cloud Run expects
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV MODEL_PATH=/app/goat_70b_local

# Start the Flask app
CMD ["python3", "app.py"]