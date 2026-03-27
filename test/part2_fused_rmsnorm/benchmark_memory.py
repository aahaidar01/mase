"""
Memory Benchmark: Fused vs Unfused Add+RMSNorm
===============================================
Measures peak GPU memory during forward+backward pass.
Kernel fusion eliminates storing the intermediate residual tensor separately,
reducing peak memory by ~1 tensor of size (B, T, D).
"""
import torch
import gc
from triton_fused_add_rmsnorm import FusedAddRMSNorm, FusedAddRMSNormModule


def pytorch_unfused_fwd_bwd(X_res, X_hid, W, eps=1e-6):
    """Unfused: add then RMSNorm, with backward."""
    X_res = X_res.clone().requires_grad_(True)
    X_hid = X_hid.clone().requires_grad_(True)
    W = W.clone().requires_grad_(True)

    residual = X_res + X_hid
    mean_sq = residual.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(mean_sq + eps)
    normed = residual * rstd.to(residual.dtype) * W

    loss = normed.sum()
    loss.backward()
    return normed


def fused_fwd_bwd(X_res, X_hid, W, eps=1e-6):
    """Fused: single Triton kernel, with backward."""
    X_res = X_res.clone().requires_grad_(True)
    X_hid = X_hid.clone().requires_grad_(True)
    W = W.clone().requires_grad_(True)

    normed, residual = FusedAddRMSNorm.apply(X_res, X_hid, W, eps, 0.0, "llama")

    loss = normed.sum() + residual.sum() * 0  # use both outputs
    loss.backward()
    return normed


def measure_peak_memory(fn, *args, n_repeats=5):
    """Run fn and return peak GPU memory in MB."""
    peaks = []
    for _ in range(n_repeats):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        fn(*args)
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
        peaks.append(peak)
    return min(peaks)  # most stable = minimum


# =========================================================================
# Run memory benchmark
# =========================================================================
print("=" * 70)
print("MEMORY BENCHMARK: Fused vs Unfused (forward + backward)")
print("=" * 70)

configs = [
    (1, 64, 4096,    "Llama-7B single token batch"),
    (4, 128, 4096,   "Llama-7B small batch"),
    (8, 256, 4096,   "Llama-7B medium batch"),
    (16, 512, 4096,  "Llama-7B large batch"),
    (1, 64, 8192,    "Llama-70B hidden dim"),
    (4, 256, 8192,   "Llama-70B batch"),
]

print(f"\n{'Config':<35} {'Unfused':>10} {'Fused':>10} {'Saved':>10} {'Reduction':>10}")
print("-" * 75)

all_results = []
for B, T, D, label in configs:
    try:
        X_res = torch.randn(B, T, D, dtype=torch.bfloat16, device="cuda")
        X_hid = torch.randn(B, T, D, dtype=torch.bfloat16, device="cuda")
        W = torch.randn(D, dtype=torch.bfloat16, device="cuda")

        mem_unfused = measure_peak_memory(pytorch_unfused_fwd_bwd, X_res, X_hid, W)
        mem_fused = measure_peak_memory(fused_fwd_bwd, X_res, X_hid, W)

        saved = mem_unfused - mem_fused
        pct = (saved / mem_unfused) * 100 if mem_unfused > 0 else 0

        print(f"({B},{T},{D}) {label:<25} {mem_unfused:>8.1f}MB {mem_fused:>8.1f}MB {saved:>+8.1f}MB {pct:>8.1f}%")
        all_results.append({
            "config": f"({B},{T},{D})",
            "label": label,
            "unfused_mb": mem_unfused,
            "fused_mb": mem_fused,
            "saved_mb": saved,
            "pct": pct,
        })

        del X_res, X_hid, W
        gc.collect()
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print(f"({B},{T},{D}) {label:<25}  OOM — skipped")
        gc.collect()
        torch.cuda.empty_cache()

if all_results:
    avg_pct = sum(r["pct"] for r in all_results) / len(all_results)
    print(f"\nAverage memory reduction: {avg_pct:.1f}%")
    print("\nKey insight: Fusion eliminates storing the intermediate residual")
    print("tensor separately in memory — one fewer (B,T,D) allocation per site.")