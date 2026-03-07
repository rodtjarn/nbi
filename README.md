# nbi

A collection of JupyterLab notebooks covering data science and GPU computing, including a hands-on CUDA programming course using CuPy.

## System Prerequisites

These must be installed **before** installing Python packages.

### 1. NVIDIA GPU Driver

Required version: **≥ 520** (for CUDA 12.x support).

```bash
# Arch Linux
sudo pacman -S nvidia nvidia-utils

# Ubuntu / Debian
sudo apt install nvidia-driver-535
```

Verify:
```bash
nvidia-smi
```

### 2. CUDA Toolkit 12.x

Required by `pycuda` (needs `nvcc` on PATH at install time). `cupy-cuda12x` bundles its own CUDA 12 runtime and does not strictly require the toolkit, but it is needed to compile RawKernel code.

```bash
# Arch Linux
sudo pacman -S cuda

# Ubuntu / Debian
sudo apt install nvidia-cuda-toolkit

# Or download directly from NVIDIA:
# https://developer.nvidia.com/cuda-downloads
```

Verify:
```bash
nvcc --version
```

### 3. Python 3.9+

```bash
# Arch Linux
sudo pacman -S python

# Ubuntu / Debian
sudo apt install python3 python3-venv
```

### 4. uv

Fast Python package manager used instead of pip.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:
```bash
uv --version
```

---

## Setup

```bash
git clone https://github.com/rodtjarn/nbi.git
cd nbi

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install all dependencies
uv pip install -r requirements.txt

# Launch JupyterLab
jupyter lab
```

---

## Notebooks

### CUDA Programming Course

A progressive series teaching GPU programming with Python and CuPy.

#### [`01_cuda_basics.ipynb`](01_cuda_basics.ipynb)
- GPU architecture overview (SMs, warps, threads)
- First CUDA kernels with `cp.RawKernel`
- Thread indexing and grid/block configuration
- Global memory read/write patterns
- Benchmarking GPU vs CPU

#### [`02_memory_management.ipynb`](02_memory_management.ipynb)
- Coalesced vs. strided global memory access
- Pinned memory for faster Host↔Device transfers
- Unified Memory (`cp.cuda.ManagedMemory`) and prefetching
- Shared memory tiled matrix transpose
- Roofline model: diagnosing memory-bound vs. compute-bound kernels

#### [`03_warp_level_programming.ipynb`](03_warp_level_programming.ipynb)
- Warp divergence and how to avoid it
- Warp shuffle instructions (`__shfl_down_sync`, `__shfl_up_sync`)
- Occupancy: block size vs. throughput sweep
- Tensor Cores: FP16 vs. FP32 GEMM throughput
- CUDA Graphs: capturing and replaying kernel pipelines

#### [`cuda_self_attention.ipynb`](cuda_self_attention.ipynb)
- Custom self-attention kernel in CUDA C via PyCUDA
- Benchmarks against PyTorch's native implementation

### General Notebooks

#### [`GPU_Matrix_Multiplication.ipynb`](GPU_Matrix_Multiplication.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodtjarn/nbi/blob/main/GPU_Matrix_Multiplication.ipynb)
- Matrix multiplication using PyTorch + CUDA
- Benchmarks GPU vs CPU (NumPy) with correctness verification

#### [`GPU_Matrix_AddSub_Colab.ipynb`](GPU_Matrix_AddSub_Colab.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodtjarn/nbi/blob/main/GPU_Matrix_AddSub_Colab.ipynb)
- Matrix addition and subtraction on GPU using PyTorch
- Designed for Google Colab (Runtime → Change runtime type → T4 GPU)

#### [`MyLab.ipynb`](MyLab.ipynb)
- Softmax implementation with numerical stability
- Bar chart and line chart visualizations

#### [`Hello.ipynb`](Hello.ipynb)
- Hello World
