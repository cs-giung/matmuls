import jax
import jax.numpy as jnp
from matmuls.kernels.batch_invariant import matmul


def test_correctness():
    """
    Verify that batch-invariant matmul matches jnp.matmul results within tolerance.
    """
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    M, K, N = 1024, 1024, 1024
    a = jax.random.normal(k1, (M, K), dtype=jnp.float32)
    b = jax.random.normal(k2, (K, N), dtype=jnp.float32)

    ref = jnp.matmul(a, b)
    out = matmul(a, b)  # Expects JIT compilation inside or can be JITed

    # Block until ready to ensure computation is done
    out.block_until_ready()

    print(f"Ref Mean: {jnp.mean(ref)}, Max: {jnp.max(ref)}")
    print(f"Out Mean: {jnp.mean(out)}, Max: {jnp.max(out)}")

    # Check non-zero
    assert jnp.max(jnp.abs(out)) > 0, "Output should not be all zeros"

    # Check close
    # TF32/BF16 noise can be significant for 1024x1024 accumulation
    diff = jnp.max(jnp.abs(out - ref))
    print(f"Correctness Max Diff: {diff}")

    assert diff < 2e-1, f"Mismatch with reference. Diff: {diff}"


def test_invariance():
    """
    Verify that batch-invariant matmul is strictly bitwise identical
    regardless of batch size (batch invariance).
    """
    # Use deterministic inputs similar to user verification
    B, D = 256, 1024

    # Use Linspace/Ranges
    a = jnp.linspace(-100, 100, B * D).reshape(B, D)
    b = jnp.linspace(-100, 100, D * D).reshape(D, D)

    # 1. Baseline (jnp.matmul) check
    # Note: JAX might be invariant depending on backend/XLA, but we check anyway
    out1_ref = jnp.matmul(a[:1], b)
    out2_ref = jnp.matmul(a, b)[:1]
    diff_ref = jnp.max(jnp.abs(out1_ref - out2_ref))
    print(f"Baseline (jnp.matmul) Difference: {diff_ref}")

    # 2. Invariant Kernel check
    out1 = matmul(a[:1], b)
    out2 = matmul(a, b)[:1]
    diff = jnp.max(jnp.abs(out1 - out2))
    print(f"Invariant Kernel Difference: {diff}")

    # Strict 0.0 check implies strict bitwise identity
    assert diff == 0.0, f"Kernel failed invariance check. Diff: {diff}"
