import numpy as np


def quantize(
    w: np.ndarray,  # [n, k]
    q_bits: int,
    q_groupsize: int = None,
):
    assert q_bits in [2, 4, 8]

    n, k = w.shape
    if q_groupsize is None:
        q_groupsize = k

    w_reshaped = w.reshape(n, -1, q_groupsize)
    w_max = np.maximum(0, np.max(w_reshaped, axis=-1))
    w_min = np.minimum(0, np.min(w_reshaped, axis=-1))
    q_max = (1 << (q_bits - 1)) - 1
    q_min = 0 - (1 << (q_bits - 1))
    s = (w_max - w_min) / (q_max - q_min)  # [n, k // q_groupsize]
    z = w_min  # [n, k // q_groupsize]

    g_idx = np.arange(k) // q_groupsize
    q = np.round((w - z[:, g_idx]) / s[:, g_idx]).astype(np.int32)  # [n, k]

    if q_bits in [2, 4, 8]:
        features_per_int = 32 // q_bits
        shift = np.arange(features_per_int) * q_bits
        q = q.reshape(n, k // features_per_int, features_per_int)
        q = np.bitwise_or.reduce(q << shift, axis=-1).astype(np.int32)

    return q, s, z


def dequantize(
    q: np.ndarray,  # [n, k // features_per_int]
    s: np.ndarray,  # [n, k // q_groupsize]
    z: np.ndarray,  # [n, k // q_groupsize]
    q_bits: int,
    q_groupsize: int = None,
):
    assert q_bits in [2, 4, 8]

    features_per_int = 32 // q_bits
    n = q.shape[0]
    k = q.shape[1] * features_per_int
    if q_groupsize is None:
        q_groupsize = k // s.shape[1]

    g_idx = np.arange(k) // q_groupsize
    shift = np.arange(features_per_int) * q_bits

    q = np.broadcast_to(q[:, :, None], (n, k // features_per_int, features_per_int))
    q = np.right_shift(q, shift[None, None, :])
    q = np.bitwise_and(q, (1 << q_bits) - 1)
    q = q.reshape(n, k)

    return q.astype(s.dtype) * s[:, g_idx] + z[:, g_idx]
