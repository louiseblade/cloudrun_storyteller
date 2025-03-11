# Use NVIDIA CUDA 11.8 base image
FROM docker.io/nvidia/cuda:11.8.0-base-ubuntu22.04@sha256:f895871972c1c91eb6a896eee68468f40289395a1e58c492e1be7929d0f8703b

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    locales \
    ninja-build \
    git \
    cuda-toolkit-11-8 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install specific CMake version 3.27.8
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.8/cmake-3.27.8-linux-x86_64.sh -O cmake-install.sh && \
    chmod +x cmake-install.sh && \
    ./cmake-install.sh --skip-license --prefix=/usr/local && \
    rm cmake-install.sh

# Verify CMake version
RUN cmake --version

# Generate locale
RUN locale-gen en_US.UTF-8

# Set working directory
WORKDIR /app

# Install Python dependencies including a pre-built llama-cpp-python wheel
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir "llama-cpp-python[cuda]" --extra-index-url https://abetlen.github.io/llama-cpp-python-cuBLAS-wheels/

# Copy application code (if any)
COPY app.py .

# Create appuser
RUN useradd -m -s /bin/bash appuser

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Set the user to appuser for runtime
USER appuser

# Expose port (if needed by your app)
EXPOSE 8080

# Run the application (adjust command if needed)
CMD ["python3", "app.py"]