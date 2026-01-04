# Medical Image Segmentation Benchmarking Framework
# Multi-stage Docker build for production and development

# Base image with CUDA support
FROM nvidia/cuda:12.9.1-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=all

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Create directories for data and experiments
RUN mkdir -p /app/data /app/experiments /app/logs

# Set permissions
RUN chmod +x main.py run_benchmark.py

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default command for development
CMD ["python", "main.py", "--help"]

# Production stage
FROM base as production

# Copy source code
COPY . .

# Create directories for data and experiments
RUN mkdir -p /app/data /app/experiments /app/logs

# Set permissions
RUN chmod +x main.py run_benchmark.py

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default command for production
CMD ["python", "main.py", "--help"]

# Jupyter stage
FROM development as jupyter

# Install Jupyter
RUN pip install --no-cache-dir jupyter jupyterlab

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Build stage for creating wheels
FROM base as builder

# Install build dependencies
RUN pip install --no-cache-dir build wheel

# Copy source code
COPY . .

# Build the package
RUN python -m build

# Final stage with built package
FROM base as final

# Copy built package from builder stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install the built package
RUN pip install --no-cache-dir /tmp/*.whl

# Create directories
RUN mkdir -p /app/data /app/experiments /app/logs

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default command
CMD ["medical-benchmark", "--help"]
