FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Install necessary packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create an ICD config file to register the NVIDIA EGL implementation with glvnd.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set the working directory
WORKDIR /workspace

# ---------------------------------------------------------------------------
# Select which environment backends to install via --build-arg.
# Examples:
#   docker build --build-arg EXTRAS=dmc            ...   # DMC only
#   docker build --build-arg EXTRAS="dmc,atari"    ...   # DMC + Atari
#   docker build --build-arg EXTRAS=all            ...   # everything
#   docker build --build-arg EXTRAS=isaaclab       ...   # IsaacLab
# Default: dmc
# ---------------------------------------------------------------------------
ARG EXTRAS=dmc

# Install dependencies from pyproject.toml
COPY pyproject.toml .
RUN IFS=',' ; set -- $EXTRAS ; \
    extra_flags="" ; \
    for e in "$@"; do extra_flags="$extra_flags --extra $e"; done ; \
    uv pip install --system --no-project $extra_flags -r pyproject.toml --no-build-isolation-package flatdict

# Set the default MuJoCo rendering backend to EGL, which is compatible with headless
# environments and does not require a display server.
ENV MUJOCO_GL=egl
