import torch
import triton
from matmuls.kernels.batch_invariant.core.matmul_triton import matmul_kernel_persistent


def get_compute_units():
    """
    Returns the number of streaming multiprocessors (SMs) or equivalent compute units
    for the available accelerator. Assigns the value to NUM_SMS.
    """
    NUM_SMS = None
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
    else:
        # Fallback for older torch or no accelerator logic
        if torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"

    # Use match/case for device-specific logic (Python 3.10+)
    match device_type:
        case "cuda":
            device_properties = torch.cuda.get_device_properties(0)
            NUM_SMS = device_properties.multi_processor_count
        case "xpu":
            device_properties = torch.xpu.get_device_properties(0)
            NUM_SMS = device_properties.max_compute_units
        case _:
            print("No CUDA or XPU device available. Using CPU.")
            # For CPU, you might want to use the number of CPU cores
            NUM_SMS = torch.get_num_threads()

    return NUM_SMS


def matmul_persistent(a: torch.Tensor, b: torch.Tensor):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    NUM_SMS = get_compute_units()
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    # Configs based on dtype (from reference)
    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
    }

    config = configs.get(
        dtype, configs[torch.float32]
    )  # Default to float32 config if unknown

    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        **config,
        num_stages=3,
        num_warps=8,
    )
    return c


# context manager removed as per user request
# Alias for user convenience
matmul = matmul_persistent
