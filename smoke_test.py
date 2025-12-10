#!/usr/bin/env python
"""
Smoke test for titans-pytorch.

A minimal script to verify training, forward pass, and sampling work without errors.
Performance is not evaluated - this only checks that the code runs without crashing.

Usage:
    uv run python smoke_test.py
    # or
    python smoke_test.py
"""

import sys
import time
import torch

from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    NeuralMemory,
)


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def test_neural_memory(device):
    """Test NeuralMemory module directly."""
    print("Testing NeuralMemory...", end=" ", flush=True)

    mem = NeuralMemory(
        dim=64,
        chunk_size=8,
    ).to(device)

    # Forward pass
    x = torch.randn(2, 32, 64, device=device)
    retrieved, state = mem(x)

    assert retrieved.shape == x.shape, f"Shape mismatch: {retrieved.shape} vs {x.shape}"
    assert state is not None, "State should not be None"

    # Test state chaining
    x2 = torch.randn(2, 32, 64, device=device)
    retrieved2, state2 = mem(x2, state=state)
    assert retrieved2.shape == x2.shape

    print("OK")


def test_forward_pass(device):
    """Test model forward pass without loss computation."""
    print("Testing forward pass...", end=" ", flush=True)

    model = MemoryAsContextTransformer(
        num_tokens=256,
        dim=64,
        depth=4,
        segment_len=16,
        num_persist_mem_tokens=2,
        num_longterm_mem_tokens=4,
        neural_memory_layers=(1, 2),
        neural_memory_segment_len=4,
        neural_memory_batch_size=32,
        use_flex_attn=False,
        neural_memory_kwargs=dict(
            dim_head=16,
            heads=2,
        )
    ).to(device)

    # Create random token sequence
    seq = torch.randint(0, 256, (2, 64), device=device)

    # Forward pass (logits)
    logits = model(seq)

    assert logits.shape == (2, 64, 256), f"Unexpected logits shape: {logits.shape}"

    print("OK")


def test_training(device):
    """Test training loop with gradient computation."""
    print("Testing training...", end=" ", flush=True)

    model = MemoryAsContextTransformer(
        num_tokens=256,
        dim=64,
        depth=4,
        segment_len=16,
        num_persist_mem_tokens=2,
        num_longterm_mem_tokens=4,
        neural_memory_layers=(1, 2),
        neural_memory_segment_len=4,
        neural_memory_batch_size=32,
        use_flex_attn=False,
        neural_memory_kwargs=dict(
            dim_head=16,
            heads=2,
        )
    ).to(device)

    # Test backward pass works (like existing tests)
    model.train()
    seq = torch.randint(0, 256, (2, 65), device=device)

    # First test: logits backward
    logits = model(seq[:, :-1])
    logits.sum().backward()

    # Clear gradients
    model.zero_grad()

    # Second test: return_loss backward
    loss = model(seq, return_loss=True)
    loss.backward()

    print("OK")


def test_sampling(device):
    """Test autoregressive sampling."""
    print("Testing sampling...", end=" ", flush=True)

    model = MemoryAsContextTransformer(
        num_tokens=256,
        dim=64,
        depth=4,
        segment_len=16,
        num_persist_mem_tokens=2,
        num_longterm_mem_tokens=4,
        neural_memory_layers=(1, 2),
        neural_memory_segment_len=4,
        neural_memory_batch_size=32,
        use_flex_attn=False,
        neural_memory_kwargs=dict(
            dim_head=16,
            heads=2,
        )
    ).to(device)

    model.eval()

    # Prime with some tokens and generate more
    prime = torch.randint(0, 256, (1, 16), device=device)

    with torch.no_grad():
        # seq_len is the total target length; sample returns only generated tokens
        target_total_len = 32
        sample = model.sample(prime, seq_len=target_total_len, use_cache=False)

    expected_generated = target_total_len - 16  # generated tokens only
    assert sample.shape == (1, expected_generated), f"Unexpected sample shape: {sample.shape}"

    print("OK")


def test_with_momentum(device):
    """Test model with momentum-based neural memory updates."""
    print("Testing with momentum...", end=" ", flush=True)

    model = MemoryAsContextTransformer(
        num_tokens=256,
        dim=64,
        depth=4,
        segment_len=16,
        num_persist_mem_tokens=2,
        num_longterm_mem_tokens=4,
        neural_memory_layers=(1, 2),
        neural_memory_segment_len=4,
        neural_memory_batch_size=32,
        use_flex_attn=False,
        neural_memory_kwargs=dict(
            dim_head=16,
            heads=2,
            momentum=True,
            momentum_order=1,
            qk_rmsnorm=True,
        )
    ).to(device)

    seq = torch.randint(0, 256, (2, 65), device=device)

    model.train()
    loss = model(seq, return_loss=True)
    loss.backward()

    print("OK")


def test_advanced_features(device):
    """Test advanced features like weight residuals and different QKV views."""
    print("Testing advanced features...", end=" ", flush=True)

    model = MemoryAsContextTransformer(
        num_tokens=256,
        dim=64,
        depth=4,
        segment_len=16,
        num_persist_mem_tokens=2,
        num_longterm_mem_tokens=4,
        neural_memory_layers=(1, 2),
        neural_memory_segment_len=4,
        neural_memory_batch_size=32,
        neural_mem_weight_residual=True,
        neural_memory_qkv_receives_diff_views=True,
        use_flex_attn=False,
        neural_memory_kwargs=dict(
            dim_head=16,
            heads=2,
            per_parameter_lr_modulation=True,
            attn_pool_chunks=True,
        )
    ).to(device)

    seq = torch.randint(0, 256, (2, 65), device=device)

    model.train()
    loss = model(seq, return_loss=True)
    loss.backward()

    print("OK")


def test_optimizer_step(device):
    """Test full training loop with optimizer steps."""
    print("Testing optimizer step...", end=" ", flush=True)

    model = MemoryAsContextTransformer(
        num_tokens=256,
        dim=64,
        depth=4,
        segment_len=16,
        num_persist_mem_tokens=2,
        num_longterm_mem_tokens=4,
        neural_memory_layers=(1, 2),
        neural_memory_segment_len=4,
        neural_memory_batch_size=32,
        use_flex_attn=False,
        neural_memory_kwargs=dict(
            dim_head=16,
            heads=2,
        )
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    model.train()
    for step in range(2):
        seq = torch.randint(0, 256, (2, 65), device=device)
        loss = model(seq, return_loss=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # Clone all parameter data to avoid memory aliasing during optimizer step
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.data = param.data.clone()

        optimizer.step()
        optimizer.zero_grad()

    print("OK")


def main():
    start_time = time.time()

    print("=" * 60)
    print("Titans PyTorch Smoke Test")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device}")
    print()

    tests = [
        test_neural_memory,
        test_forward_pass,
        test_training,
        test_sampling,
        test_with_momentum,
        test_advanced_features,
        test_optimizer_step,
    ]

    failed = []
    for test_fn in tests:
        try:
            test_fn(device)
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((test_fn.__name__, str(e)))

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests failed in {elapsed:.1f}s")
        for name, error in failed:
            print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print(f"SUCCESS: All {len(tests)} tests passed in {elapsed:.1f}s")
        sys.exit(0)


if __name__ == "__main__":
    main()
