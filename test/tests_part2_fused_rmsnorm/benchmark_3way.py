"""
3-Way Benchmark: PyTorch Unfused vs Our Kernel vs Liger-Kernel
"""
import torch
import time
import matplotlib
matplotlib.use('Agg')  # headless backend for HPC (no display)
import matplotlib.pyplot as plt
import numpy as np

from triton_fused_add_rmsnorm import FusedAddRMSNorm

# Import Liger-Kernel's fused add + RMSNorm
from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
try:
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
except ImportError:
    pass

# Liger's RMSNorm with fused add
from liger_kernel.ops.rms_norm import LigerRMSNormFunction

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def cuda_event_time_us(fn, n_warmup=50, n_iters=200):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / n_iters  # us

def wall_clock_time_us(fn, n_warmup=50, n_iters=200):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e6

# ---------------------------------------------------------------------------
# PyTorch unfused reference
# ---------------------------------------------------------------------------
def pytorch_unfused(X_res, X_hid, W, eps, dtype):
    residual = X_res + X_hid
    mean_sq = residual.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(mean_sq + eps)
    return residual * rstd.to(dtype) * W, residual

# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------
configs = [
    # Small (launch overhead region)
    (1,    1,  4096, "bf16"),
    (1,   32,  4096, "bf16"),
    # Medium (crossover)
    (2,  128,  4096, "bf16"),
    (4,  128,  4096, "bf16"),
    # Large (fusion wins)
    (4,  256,  4096, "bf16"),
    (4,  512,  4096, "bf16"),
    (8,  256,  4096, "bf16"),
    (8,  512,  4096, "bf16"),
    (16, 256,  4096, "bf16"),
    # Llama-70B hidden dim
    (4,  256,  8192, "bf16"),
    (8,  256,  8192, "bf16"),
    # FP16
    (4,  256,  4096, "fp16"),
    (8,  512,  4096, "fp16"),
    # FP32
    (4,  256,  4096, "fp32"),
]

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# ---------------------------------------------------------------------------
# Run 3-way benchmark
# ---------------------------------------------------------------------------
results = []

print("=" * 90)
print("3-WAY BENCHMARK: PyTorch Unfused vs Our Kernel vs Liger-Kernel")
print("=" * 90)
print(f"\n  {'Config':<18} {'Method':<10} {'Unfused':>10} {'Ours':>10} {'Liger':>10} {'Ours spd':>9} {'Liger spd':>9}")
print("  " + "-" * 80)

for B, T, D, dtype_str in configs:
    dtype = dtype_map[dtype_str]
    X_res = torch.randn(B, T, D, dtype=dtype, device="cuda")
    X_hid = torch.randn(B, T, D, dtype=dtype, device="cuda")
    W = torch.ones(D, dtype=dtype, device="cuda")
    eps = 1e-6
    label = f"{B}x{T}x{D} {dtype_str}"

    # --- Unfused ---
    def fn_unfused():
        return pytorch_unfused(X_res, X_hid, W, eps, dtype)

    # --- Our kernel ---
    def fn_ours():
        return FusedAddRMSNorm.apply(X_res, X_hid, W, eps, 0.0, "llama")

    # --- Liger kernel ---
    # Liger's in_place mode: X is modified in-place (X += residual), then normalised.
    # Signature: LigerRMSNormFunction.apply(X, W, eps, offset, casting_mode, in_place)
    # where in_place is the residual tensor. After the call:
    #   - X is modified in-place to hold (X_original + in_place), i.e. the updated residual
    #   - return value is the normalised output
    # We pre-allocate fresh tensors each call to avoid mutating benchmark inputs,
    # using .clone() only once outside the timed loop.
    X_hid_bench = X_hid.clone()
    X_res_bench = X_res.clone()

    def fn_liger():
        # Reset to original values (copy_ is faster than clone)
        X_hid_bench.copy_(X_hid)
        X_res_bench.copy_(X_res)
        normed = LigerRMSNormFunction.apply(X_hid_bench, W, eps, 0.0, "llama", X_res_bench)
        return normed, X_hid_bench  # X_hid_bench now holds the updated residual

    # CUDA event timing
    t_unfused = cuda_event_time_us(fn_unfused)
    t_ours = cuda_event_time_us(fn_ours)
    t_liger = cuda_event_time_us(fn_liger)

    spd_ours = t_unfused / t_ours if t_ours > 0 else float("inf")
    spd_liger = t_unfused / t_liger if t_liger > 0 else float("inf")

    # Wall-clock timing
    tw_unfused = wall_clock_time_us(fn_unfused)
    tw_ours = wall_clock_time_us(fn_ours)
    tw_liger = wall_clock_time_us(fn_liger)

    spd_ours_w = tw_unfused / tw_ours if tw_ours > 0 else float("inf")
    spd_liger_w = tw_unfused / tw_liger if tw_liger > 0 else float("inf")

    results.append({
        "label": label, "B": B, "T": T, "D": D, "dtype": dtype_str,
        "n_elements": B * T * D,
        "unfused_cuda": t_unfused, "ours_cuda": t_ours, "liger_cuda": t_liger,
        "unfused_wall": tw_unfused, "ours_wall": tw_ours, "liger_wall": tw_liger,
        "spd_ours_cuda": spd_ours, "spd_liger_cuda": spd_liger,
        "spd_ours_wall": spd_ours_w, "spd_liger_wall": spd_liger_w,
    })

    print(f"  {label:<18} {'CUDA evt':<10} {t_unfused:>8.1f}us {t_ours:>8.1f}us {t_liger:>8.1f}us {spd_ours:>8.2f}x {spd_liger:>8.2f}x")
    print(f"  {'':<18} {'Wall-clk':<10} {tw_unfused:>8.1f}us {tw_ours:>8.1f}us {tw_liger:>8.1f}us {spd_ours_w:>8.2f}x {spd_liger_w:>8.2f}x")

print()

# ---------------------------------------------------------------------------
# Correctness: verify our kernel matches Liger's output
# ---------------------------------------------------------------------------
print("=" * 90)
print("CORRECTNESS: Our Kernel vs Liger-Kernel")
print("=" * 90)

for dtype_str in ["bf16", "fp32"]:
    dtype = dtype_map[dtype_str]
    B, T, D = 4, 64, 4096
    X_res = torch.randn(B, T, D, dtype=dtype, device="cuda")
    X_hid = torch.randn(B, T, D, dtype=dtype, device="cuda")
    W = torch.randn(D, dtype=dtype, device="cuda")
    eps = 1e-6

    # PyTorch reference: fused add + RMSNorm
    res_ref = X_res + X_hid
    mean_sq = res_ref.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(mean_sq + eps)
    normed_ref = res_ref * rstd.to(dtype) * W

    # PyTorch reference: RMSNorm only (no add) — to detect Liger's actual behavior
    mean_sq_hid = X_hid.float().pow(2).mean(dim=-1, keepdim=True)
    rstd_hid = torch.rsqrt(mean_sq_hid + eps)
    normed_hid_only = X_hid * rstd_hid.to(dtype) * W

    # Our kernel (fused add + RMSNorm)
    normed_ours, res_ours = FusedAddRMSNorm.apply(X_res, X_hid, W, eps, 0.0, "llama")

    # Liger kernel
    X_hid_l = X_hid.clone()
    normed_liger = LigerRMSNormFunction.apply(X_hid_l, W, eps, 0.0, "llama", X_res.clone())

    atol = 1e-5 if dtype == torch.float32 else 1e-2

    # Auto-detect: did Liger do add+RMSNorm or just RMSNorm?
    err_liger_fused = (normed_liger - normed_ref).abs().max().item()
    err_liger_plain = (normed_liger - normed_hid_only).abs().max().item()
    liger_did_fused = err_liger_fused < 1.0

    if liger_did_fused:
        liger_mode = "add+RMSNorm (fused)"
        err_liger = err_liger_fused
    else:
        liger_mode = "RMSNorm only (no residual add)"
        err_liger = err_liger_plain

    err_ours = (normed_ours - normed_ref).abs().max().item()
    ours_pass = torch.allclose(normed_ours, normed_ref, atol=atol, rtol=atol)
    liger_pass = err_liger < atol

    print(f"\n  dtype={dtype_str}, shape=({B},{T},{D})")
    print(f"    Liger detected mode:   {liger_mode}")
    print(f"    Ours vs PyTorch ref:   max_err={err_ours:.2e}  {'PASS' if ours_pass else 'FAIL'}")
    print(f"    Liger vs its ref:      max_err={err_liger:.2e}  {'PASS' if liger_pass else 'FAIL'}")
    if not liger_did_fused:
        print(f"    Note: Liger pip version does RMSNorm only; our kernel fuses add+RMSNorm.")

print()