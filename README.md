# matmuls

A library for custom matrix multiplication kernels using Triton, PyTorch, and JAX.

## Installation

This project uses `uv` for dependency management.

```bash
uv sync
```

## Development with Docker (Recommended)

To ensure compatibility with Triton and the latest CUDA features (like PTX 8.7), it is recommended to use the provided Docker environment.

**Prerequisites:**
- Docker with NVIDIA GPU support.

**Quick Start:**
```bash
./scripts/dev_container.sh
```

This script will:
1. Build the Docker image (`matmuls-dev`) based on `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04`.
2. Start an interactive container with your current directory mounted to `/workspace/matmuls`.
3. Mount your home directory to `/mnt/home2/giung2` (as requested).

Once inside the container:
```bash
uv sync
uv run pytest tests/
```

## Usage

You can import the kernels directly from the package:

```python
import torch
from matmuls.kernels.matmul.triton.ops_pt import matmul_fwd

# ... setup tensors ...
output = matmul_fwd(input_a, input_b)
```

## Running Tests

```bash
uv run pytest tests/
```

## Project Structure

- `matmuls/`: Core package containing kernels and models.
- `scripts/`: Benchmark and utility scripts.
- `examples/`: Example scripts and scratchpads.
- `tests/`: Unit tests.
