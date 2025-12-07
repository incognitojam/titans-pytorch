# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an unofficial PyTorch implementation of [Titans](https://arxiv.org/abs/2501.00663), a neural architecture featuring test-time training with neural memory modules. The repository explores both the paper's original MLP-based memory and experimental architectures including attention-based memory modules.

## Development Commands

### Installation

**Using uv (recommended):**
```bash
# Sync dependencies for examples (required for training)
uv sync --extra examples

# Sync with test dependencies
uv sync --extra test

# Sync all optional dependencies
uv sync --all-extras
```

**Using pip:**
```bash
# Standard installation
pip install titans-pytorch

# Development with examples
pip install .[examples]

# For testing
pip install .[test]
```

### Testing
```bash
# Using uv
uv run pytest
uv run pytest tests/test_titans.py::test_titans

# Using pytest directly
pytest
pytest tests/test_titans.py
pytest tests/test_titans.py::test_titans
pytest tests/test_titans.py::test_mac -v
```

### Training Experiments
```bash
# Using uv
uv run python train_mac.py
uv run python train_implicit_mlp_attn.py

# Direct execution
python train_mac.py
python train_implicit_mlp_attn.py
```

The training scripts download and use the enwik8 dataset from `./data/enwik8.gz`.

## Architecture Overview

### Core Components

**NeuralMemory** (`neural_memory.py`): The central neural memory module implementing test-time training
- Processes sequences in chunks, updating internal weights during inference
- Supports momentum-based updates with configurable orders (1st, 2nd order)
- Uses associative scan for efficient parallel computation
- Configurable memory models (MLP, attention, etc.)
- Key state: `NeuralMemState` containing weights, cache segments, and momentum states
- Supports both parallel processing and sequential inference (with state chaining)

**Memory Models** (`memory_models.py`): Pluggable architectures for the neural memory
- `MemoryMLP`: Standard multi-layer MLP (default, from TTT paper)
- `MemoryAttention`: Attention-based memory module (experimental)
- `MemorySwiGluMLP`: SwiGLU-style MLP with residuals
- `GatedResidualMemoryMLP`: MLP with gated residual connections
- `FactorizedMemoryMLP`: Low-rank factorized weights for smaller chunk sizes

**MemoryAsContextTransformer** (`mac_transformer.py`): Full transformer with Memory-As-Context
- Integrates neural memory at specified layers (`neural_memory_layers` parameter)
- Supports both persistent and long-term memory tokens
- Segmented attention with configurable window sizes
- Optional FlexAttention for GPU optimization
- Can use sliding window or block-diagonal attention patterns

### Key Design Patterns

**State Chaining**: Neural memory supports processing sequences in chunks, maintaining state across calls:
```python
# Parallel processing
retrieved, state = mem(full_sequence)

# Sequential processing (for inference)
retrieved1, state = mem(chunk1)
retrieved2, state = mem(chunk2, state=state)  # state is chained
```

**Weight Residuals**: Neural memory layers can receive weight residuals from previous layers (`neural_mem_weight_residual=True`), allowing gradient flow between memory modules.

**Flexible QKV Views**: Memory modules can derive queries/keys/values from different transformer layers (`neural_memory_qkv_receives_diff_views=True`), enabling the memory to connect to the transformer in flexible ways.

**Chunk-based Processing**: Neural memory processes sequences in chunks (controlled by `chunk_size` and `batch_size` parameters):
- `chunk_size`: Granularity of memory updates
- `batch_size`: How often weights are updated as it traverses the sequence
- Smaller chunks = more granular learning but higher memory usage

### Important Configuration Patterns

When configuring neural memory in transformers:
- `neural_memory_layers`: Tuple of layer indices that use neural memory (e.g., `(2, 4, 6)`)
- `neural_memory_segment_len`: Segment length for neural memory processing
- `neural_memory_batch_size`: Batch size for neural memory weight updates
- `neural_memory_kwargs`: Dict passed to NeuralMemory, including:
  - `momentum`: Enable momentum-based updates
  - `momentum_order`: Order of momentum (1 or 2)
  - `qk_rmsnorm`: Apply RMSNorm to queries/keys
  - `attn_pool_chunks`: Use attention pooling for chunk-derived metadata
  - `use_accelerated_scan`: Use accelerated CUDA scan (requires GPU)
  - `per_parameter_lr_modulation`: Per-layer learned learning rates
  - `spectral_norm_surprises`: Apply spectral normalization to surprise updates (from Muon optimizer)

## Test Structure

Tests in `tests/test_titans.py` are heavily parameterized using pytest fixtures to cover:
- Various sequence lengths
- Different chunk sizes and batch sizes
- Momentum configurations
- Attention pooling variants
- State chaining correctness
- Sequential vs parallel equivalence
- FlexAttention vs standard attention equivalence

## Data and Dependencies

- Training uses enwik8 character-level dataset (95MB of Wikipedia text)
- Requires PyTorch 2.2+ for modern features like FlexAttention
- Uses `assoc-scan` for efficient associative scans
- `tensordict` for managing complex nested states
- `einops` and `einx` for tensor operations
- `hyper-connections` and `x-transformers` for transformer utilities

## Experimental Features

This implementation includes features beyond the paper:
- Weight residuals between neural memory layers (inspired by value residual learning)
- Flexible QKV view selection (addresses the wk @ wv degeneracy concern)
- Spectral normalization of surprises (applying Muon optimizer lessons)
- Multiple memory model architectures (attention, SwiGLU, etc.)
- Per-layer learned learning rate modulation
