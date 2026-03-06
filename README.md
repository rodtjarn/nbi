# nbi

A collection of JupyterLab notebooks covering data science and GPU computing.

## Requirements

- Python 3.x
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Setup

### Using uv (recommended)

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
jupyter lab
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Notebooks

### `MyLab.ipynb`
- Softmax function implementation with numerical stability
- Bar chart and line chart visualizations of the softmax output

### `Hello.ipynb`
- Hello World

### `GPU_Matrix_Multiplication.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodtjarn/nbi/blob/main/GPU_Matrix_Multiplication.ipynb)
- Matrix multiplication on an NVIDIA GPU using PyTorch + CUDA
- Benchmarks GPU vs CPU (NumPy) and verifies correctness
- Requires a local NVIDIA GPU with the PyTorch CUDA wheel installed

### `GPU_Matrix_AddSub_Colab.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodtjarn/nbi/blob/main/GPU_Matrix_AddSub_Colab.ipynb)
- Matrix addition and subtraction on GPU using PyTorch
- Designed for **Google Colab** (PyTorch pre-installed, T4 GPU available)
- Benchmarks GPU vs CPU and verifies correctness with a hand-checkable example
- Open in Colab: Runtime → Change runtime type → T4 GPU
