FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ============================================================
# System dependencies
# ============================================================

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    ninja-build \
    git \
    git-lfs \
    wget \
    curl \
    unzip \
    pkg-config \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenblas-dev \
    libomp-dev \
    htop \
    nano \
    vim \
    tree \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Python virtual environment
# ============================================================

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# ============================================================
# Upgrade pip INSIDE the virtual environment
# ============================================================

RUN pip install --upgrade \
    pip \
    setuptools \
    wheel

# ============================================================
# PyTorch
# ============================================================

RUN pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# ============================================================
# Python research stack
# ============================================================

COPY requirements.txt /tmp/requirements.txt

RUN pip install \
    --no-cache-dir \
    -r /tmp/requirements.txt

# ============================================================
# Workspace
# ============================================================

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]