"""
7B Model End-to-End: Fused Add+RMSNorm on Llama-2-7B

Measures BOTH unfused and fused latency on the same model load:
  1. Load model -> time unfused forward
  2. Patch layers -> time fused forward
  3. Compare correctness + real measured speedup
"""
import torch
import torch.nn as nn
import types
import time
import gc
import os
from triton_fused_add_rmsnorm import FusedAddRMSNorm, FusedAddRMSNormModule

def benchmark_forward(model, input_ids, n_warmup=10, n_iters=50, label=""):
    """Time forward pass using CUDA events."""
    for _ in range(n_warmup):
        with torch.no_grad():
            model(input_ids)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        with torch.no_grad():
            model(input_ids)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / n_iters
    print(f"    {label}: {ms:.2f} ms")
    return ms


def patch_decoder_layer(layer):
    """
    Patch a single decoder layer to fuse post-attn add+RMSNorm.

    IMPORTANT: This is a separate function (not inlined in a loop) so that
    norm_weight and eps are captured correctly per-layer via closure scope.
    If this were inside a for-loop, all layers would share the last layer's
    variables (Python closure-in-a-loop bug).
    """
    post_norm = layer.post_attention_layernorm
    eps = getattr(post_norm, 'variance_epsilon', getattr(post_norm, 'eps', 1e-6))
    norm_weight = post_norm.weight

    def fused_forward(self, hidden_states, **kwargs):
        residual = hidden_states

        # 1. Pre-attention norm (unchanged)
        hidden_states = self.input_layernorm(hidden_states)

        # 2. Self attention
        attn_out = self.self_attn(hidden_states=hidden_states, **kwargs)
        if isinstance(attn_out, tuple):
            hidden_states = attn_out[0]
        else:
            hidden_states = attn_out

        # 3. FUSED: residual + attn_output -> post_attention_layernorm
        hidden_states, residual = FusedAddRMSNorm.apply(
            residual, hidden_states,
            norm_weight, eps, 0.0, "llama"
        )

        # 4. MLP
        hidden_states = self.mlp(hidden_states)

        # 5. Second residual add (unfused)
        hidden_states = residual + hidden_states

        return hidden_states

    layer.forward = types.MethodType(fused_forward, layer)


def test_7b_model(model_name, dtype=torch.float16):
    print(f"\n{'='*70}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*70}")

    # ---- 1. Load model ----
    print(f"  Loading {model_name} in {dtype}...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params/1e9:.1f}B params")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # ---- 2. Find decoder layers ----
    decoder_layers = []
    for name, mod in model.named_modules():
        if "DecoderLayer" in type(mod).__name__:
            decoder_layers.append((name, mod))

    n_layers = len(decoder_layers)
    print(f"  Decoder layers: {n_layers}")

    # ---- 3. Benchmark UNFUSED ----
    print(f"\n  LATENCY (measured, CUDA events, 50 iters):")

    test_configs = [
        (1, 64,   "batch=1, seq=64"),
        (1, 256,  "batch=1, seq=256"),
        (4, 64,   "batch=4, seq=64"),
    ]

    results_by_config = {}
    for batch, seq_len, label in test_configs:
        try:
            input_ids = torch.randint(100, 5000, (batch, seq_len), device="cuda")
            with torch.no_grad():
                _ = model(input_ids)
        except torch.cuda.OutOfMemoryError:
            print(f"    {label}: OOM - skipped")
            continue
        unfused_ms = benchmark_forward(model, input_ids, label=f"Unfused  ({label})")
        results_by_config[label] = {"input_ids": input_ids, "unfused_ms": unfused_ms}

    # ---- 4. Reference output for correctness ----
    input_ids_corr = torch.randint(100, 5000, (1, 64), device="cuda")
    with torch.no_grad():
        ref_out = model(input_ids_corr).logits.clone()
    ref_topk = ref_out[0, -1].topk(10).indices.tolist()

    # ---- 5. Patch layers (each call creates its own closure scope) ----
    print(f"\n  Patching {n_layers} layers...")
    for name, layer in decoder_layers:
        patch_decoder_layer(layer)
    print(f"  Patched {n_layers} fusion sites")

    # ---- 6. Benchmark FUSED ----
    for label, data in results_by_config.items():
        fused_ms = benchmark_forward(model, data["input_ids"], label=f"Fused    ({label})")
        data["fused_ms"] = fused_ms

    # ---- 7. Correctness ----
    with torch.no_grad():
        fused_out = model(input_ids_corr).logits

    abs_diff = (fused_out - ref_out).abs()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()
    logit_range = ref_out.max().item() - ref_out.min().item()
    rel_err = max_abs_err / logit_range if logit_range > 0 else float('inf')

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_out[0, -1].unsqueeze(0).float(),
        fused_out[0, -1].unsqueeze(0).float()
    ).item()

    fused_topk5 = fused_out[0, -1].topk(5).indices.tolist()
    topk5_match = ref_topk[:5] == fused_topk5

    correctness_pass = (rel_err < 0.01) and (cos_sim > 0.999) and topk5_match

    print(f"\n  CORRECTNESS:")
    print(f"    Max abs err:       {max_abs_err:.4e}")
    print(f"    Mean abs err:      {mean_abs_err:.4e}")
    print(f"    Logit range:       {logit_range:.2f}")
    print(f"    Relative error:    {rel_err:.6f}  {'PASS' if rel_err < 0.01 else 'FAIL'}")
    print(f"    Cosine similarity: {cos_sim:.6f}  {'PASS' if cos_sim > 0.999 else 'FAIL'}")
    print(f"    Top-5 match:       {'PASS' if topk5_match else 'FAIL'}")
    print(f"      ref:   {ref_topk[:5]}")
    print(f"      fused: {fused_topk5}")
    print(f"    Overall:           {'PASS' if correctness_pass else 'FAIL'}")

    # ---- 8. Speedup summary ----
    print(f"\n  SPEEDUP SUMMARY (measured):")
    print(f"  {'Config':<25} {'Unfused':>10} {'Fused':>10} {'Saving':>10} {'Speedup':>10}")
    print(f"  {'-'*65}")
    for label, data in results_by_config.items():
        if "fused_ms" in data:
            saving = data["unfused_ms"] - data["fused_ms"]
            speedup = data["unfused_ms"] / data["fused_ms"] if data["fused_ms"] > 0 else 0
            print(f"  {label:<25} {data['unfused_ms']:>8.2f}ms {data['fused_ms']:>8.2f}ms {saving:>+8.2f}ms {speedup:>9.3f}x")

    # ---- 9. Cleanup ----
    del model, ref_out, fused_out
    gc.collect()
    torch.cuda.empty_cache()

    return correctness_pass


# =========================================================================
# Run
# =========================================================================
results = {}

try:
    ok = test_7b_model("meta-llama/Llama-2-7b-hf")
    results["Llama-2-7B"] = "PASSED" if ok else "FAILED"
except torch.cuda.OutOfMemoryError:
    print("\n  OOM: Model too large for this GPU")
    results["Llama-2-7B"] = "OOM"
except Exception as e:
    print(f"\n  ERROR: {e}")
    import traceback; traceback.print_exc()
    results["Llama-2-7B"] = f"ERROR: {str(e)[:100]}"

gc.collect()
torch.cuda.empty_cache()

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for name, status in results.items():
    print(f"  {name:<20} {status}")
print(f"{'='*70}")