# Leverage official RunPod CUDA base with Python 3.11
FROM runpod/base:0.4.0-cuda11.8.0

# Set working directory and environment variables
WORKDIR /workspace
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    TRANSFORMERS_CACHE=/workspace/model-cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ocl-icd-opencl-dev \
    opencl-headers && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install torch and torchvision first so they are available for flash-attn's build.
RUN python3.11 -m pip install --upgrade pip && \
    pip install torch==2.2.1 torchvision==0.17.1 && \
    pip install packaging

# Install the remaining dependencies and then install the bleeding-edge transformers and accelerate from GitHub
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install git+https://github.com/huggingface/transformers accelerate && \
    rm requirements.txt

# Force downgrade NumPy to a version below 2 to ensure compatibility
RUN pip install --no-cache-dir "numpy<2"
# Copy application code with proper ownership
COPY --chown=1000:1000 workspace/ .

RUN python3.11 -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='/workspace/model-cache', revision='refs/pr/24')"

# Use RunPod-recommended entrypoint
CMD ["python3.11", "-u", "/workspace/main.py"]
