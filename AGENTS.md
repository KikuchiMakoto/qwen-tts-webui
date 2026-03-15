# Qwen3-TTS WebUI - Agent Guidelines

## Build Commands

```bash
# Install dependencies (CPU PyTorch by default)
uv sync

# Install CPU PyTorch explicitly
uv sync --index-url https://download.pytorch.org/whl/cpu

# Install CUDA 12.4 PyTorch
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install CUDA 12.8 PyTorch
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install ROCm PyTorch (AMD GPU)
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.3

# Install FlashAttention 2 (optional, for GPU)
uv pip install flash-attn --no-build-isolation

# Run the application
uv run streamlit run app.py

# Force CPU execution
FORCE_CPU=1 uv run streamlit run app.py

# Force CUDA execution
FORCE_CUDA=1 uv run streamlit run app.py
```

## Test Commands

```bash
# Run all tests (if test suite exists)
uv run pytest

# Run specific test file
uv run pytest tests/test_engine.py

# Run with coverage
uv run pytest --cov=. --cov-report=html

# Test model loading
uv run python -c "from engine import TTSEngine; engine = TTSEngine(); print('Engine OK')"
```

## Lint Commands

```bash
# Run ruff linter
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Format code with ruff
uv run ruff format .
```

## Code Style Guidelines

### Imports

- Use standard library imports first, then third-party imports, then local imports
- Group imports by blank lines
- Avoid wildcard imports (`from module import *`)
- Use absolute imports for local modules

```python
import os
from pathlib import Path

import numpy as np
import torch
import streamlit as st

from engine import TTSEngine
from voice_store import save_voice
```

### Type Hints

- Use type hints for all function signatures
- Use `|` syntax for union types (Python 3.10+)
- Return types should be explicit

```python
def generate_speech(
    text: str,
    language: str,
    voice_clone_prompt,
    model_size: str = "1.7B",
) -> tuple[np.ndarray, int]:
    """Generate speech from text."""
    pass
```

### Naming Conventions

- **Variables/functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Protected methods**: `_leading_underscore` (same as private)

```python
MAX_Voices = 10

class TTSEngine:
    def __init__(self):
        self._model = None  # Private

    def generate_speech(self, text: str):  # Public
        pass
```

### Docstrings

- Use Google-style docstrings
- Include Args and Returns sections for functions
- Describe purpose briefly

```python
def generate_speech(
    self,
    text: str,
    language: str,
    voice_clone_prompt,
    model_size: str = "1.7B",
) -> tuple[np.ndarray, int]:
    """Generate speech using saved voice clone prompt.

    Args:
        text: Text to synthesize
        language: Output language
        voice_clone_prompt: Prompt created by create_voice_prompt
        model_size: Model size to use ("1.7B" or "0.6B")

    Returns:
        (wav, sample_rate) tuple
    """
```

### Error Handling

- Use specific exceptions
- Provide context in error messages
- Use try-except for recoverable errors
- Let unexpected exceptions propagate

```python
try:
    self._model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
    )
except ImportError as e:
    raise RuntimeError(f"Failed to import model: {e}")
except Exception as e:
    raise RuntimeError(f"Failed to load model {model_id}: {e}")
```

### Device Management

- Priority: CUDA > CPU (ROCm uses CUDA interface)
- Use `get_device()` function for device detection
- Support environment variables for forcing devices: `FORCE_CPU`, `FORCE_CUDA`
- Use `torch.bfloat16` for CUDA, `torch.float32` for CPU

### Attention Implementation

- Use Flash Attention 2 for CUDA when available
- Fall back to SDPA (Scaled Dot-Product Attention) otherwise
- DirectML/other backends use SDPA

### Constants

- Define constants at module level (after imports)
- Use descriptive names
- Group related constants together

```python
# Qwen3-TTS supported languages
SUPPORTED_LANGUAGES = [
    "Japanese", "Chinese", "English", "Korean",
    "German", "French", "Russian", "Portuguese",
    "Spanish", "Italian",
]

# Model IDs
MODEL_IDS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
```

### Streamlit UI Guidelines

- Use containers and columns for layout
- Use session state for caching model and history
- Show completion notifications with toast and audio feedback
- Use spinners for long-running operations
- Provide download buttons for generated audio
- Handle file uploads with appropriate type validation

### File Structure

```
qwen-tts-webui/
├── app.py           # Streamlit UI (main application)
├── engine.py        # Qwen3-TTS engine wrapper
├── voice_store.py   # Voice model persistence
├── pyproject.toml   # uv project configuration
└── README.md        # Documentation
```

### Environment Variables

- `FORCE_CPU=1`: Force CPU execution
- `FORCE_CUDA=1`: Force CUDA execution

### Important Notes

- This project uses PyTorch with qwen-tts package
- Flash Attention 2 is optional but recommended for GPU
- ROCm uses CUDA interface (`torch.cuda.is_available()` returns True)
- Voice models are cached as .pt files in voice_models directory
- Maximum 10 voice models per model size
