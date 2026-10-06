# Switch runtime → devel so NVCC is available for the
# Deformable DETR custom CUDA op compilation.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app
# ── headless / stable I/O defaults ───────────────────────────────────────────
ENV HDF5_USE_FILE_LOCKING=FALSE
ENV DISPLAY=
ENV MANI_SKILL_VIEWER=0

RUN mkdir /app
WORKDIR /app

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    python3-opencv \
    python3.11 \
    python3.11-dev \
    python3-pip \
    systemd \
    # build tools for Deformable DETR CUDA ops
    build-essential \
    ninja-build \
    git \
    # headless rendering support for ManiSkill eval
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── pip upgrade ───────────────────────────────────────────────────────────────
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# ── Core numeric / IO ─────────────────────────────────────────────────────────
RUN python3.11 -m pip install \
    numpy==1.26.3 \
    scipy \
    h5py \
    Pillow \
    matplotlib \
    pandas \
    seaborn \
    tqdm \
    einops \
    cython \
    scikit-learn \
    colour \
    wget \
    docker \
    opcua \
    cryptography

# ── PyTorch + xformers ────────────────────────────────────────────────────────
RUN python3.11 -m pip install \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN python3.11 -m pip install \
    xformers==0.0.23.post1 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Detection / segmentation ──────────────────────────────────────────────────
RUN python3.11 -m pip install segmentation-models-pytorch==0.2.1

# ── ViT / backbone utilities (used in model/dino.py, model/vanishing_depth.py)
RUN python3.11 -m pip install \
    transformers \
    timm

# ── Experiment tracking ───────────────────────────────────────────────────────
RUN python3.11 -m pip install wandb

# ── ManiSkill3 (required for eval_policy_maniskill.py) ───────────────────────
RUN python3.11 -m pip install "mani-skill==3.0.1"

# ── Copy project code ─────────────────────────────────────────────────────────
# data/ and demos/ are NOT copied – they are mounted at runtime.
COPY . /app

# ── Compile Deformable DETR custom CUDA ops ───────────────────────────────────
# Requires NVCC (devel image) and the project code present (COPY above).
RUN cd /app/model/deformable_detr/ops && \
    python3.11 setup.py build_ext --inplace

# ── Entrypoint: interactive shell so user can run torchrun ────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]