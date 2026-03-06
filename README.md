# MyLab

A JupyterLab notebook demonstrating a softmax function with visualizations.

## Requirements

- Python 3.x
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Setup & Reproduce

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

## Contents

- `MyLab.ipynb` — Main notebook:
  - Hello World
  - Softmax function implementation with numerical stability
  - Bar chart and line chart visualizations of the softmax output
