FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Install basic utilities and Python
# Ubuntu 24.04 comes with Python 3.12 by default.
# We need to ensure we have a compatible Python version and pip.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    python3 \
    python3-pip \
    python3-dev \
    zsh \
    clang-format \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create user and group to match host
RUN groupadd -g 1001 siml && \
    useradd -u 1005 -g 1001 -d /mnt/home2/giung2/ -s /bin/zsh giung2

# Set working directory
WORKDIR /mnt/home2/giung2/matmuls

# Switch to non-root user
USER giung2

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["/bin/bash"]
