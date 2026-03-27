"""
Test: Module-Level Fused RMSNorm + Residual Pass
=================================================
Tests the module-level MASE pass that patches decoder layers to use
the fused Triton kernel. No FX tracing — works on any HF model.

Usage:
    python -u test_module_pass.py
"""

import sys
import os
import torch
import torch.nn as nn
import types
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_fused_add_rmsnorm import FusedAddRMSNorm, FusedAddRMSNormModule


# ---------------------------------------------------------------------------
# Inline the module pass (so we don't need MASE installed on HPC)
# ---------------------------------------------------------------------------
_RMSNORM_CLASS_NAMES = frozenset({
    "RMSNorm", "LlamaRMSNorm", "MistralRMSNorm", "GemmaRMSNorm",
    "Qwen2RMSNorm", "InternLMRMSNorm", "SimpleRMSNorm",
})
_DECODER_LAYER_CLASS_NAMES = frozenset({
    "LlamaDecoderLayer", "MistralDecoderLayer", "GemmaDecoderLayer",
    "Qwen2DecoderLayer", "InternLMDecoderLayer",
})


def _is_rmsnorm(m):
    return any(name in type(m).__name__ for name in _RMSNORM_CLASS_NAMES)

def _is_decoder_layer(m):
    return any(name in type(m).__name__ for name in _DECODER_LAYER_CLASS_NAMES)

def _get_eps(m):
    return getattr(m, "variance_epsilon", getattr(m, "eps", 1e-6))

def _get_offset(m):
    return 1.0 if "Gemma" in type(m).__name__ else 0.0


def _patch_decoder_layer(layer, casting_mode):
    post_norm = getattr(layer, "post_attention_layernorm", None)
    if post_norm is None or not _is_rmsnorm(post_norm):
        return False

    eps = _get_eps(post_norm)
    offset = _get_offset(post_norm)
    norm_weight = post_norm.weight

    def fused_forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states=hidden_states, **kwargs)
        if isinstance(attn_out, tuple):
            hidden_states = attn_out[0]
        else:
            hidden_states = attn_out
        hidden_states, residual = FusedAddRMSNorm.apply(
            residual, hidden_states,
            norm_weight, eps, offset, casting_mode,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    layer.forward = types.MethodType(fused_forward, layer)
    return True


def fused_rmsnorm_residual_transform_pass(network, pass_args=None):
    pass_args = pass_args or {}
    casting_mode = pass_args.get("casting_mode", "llama")
    fused_layers = []
    for name, module in network.named_modules():
        if _is_decoder_layer(module):
            if _patch_decoder_layer(module, casting_mode):
                fused_layers.append(name)
    return network, {"num_fused": len(fused_layers), "fused_layers": fused_layers}


# ===========================================================================
# Tests
# ===========================================================================

def test_tiny_llama():
    """Test on tiny Llama — no download, random weights."""
    print("\n" + "=" * 70)
    print("TEST: Module Pass on Tiny Llama (2 layers, random weights)")
    print("=" * 70)

    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=128,
        vocab_size=1000, use_cache=False,
    )
    model = LlamaForCausalLM(config).to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params, {config.num_hidden_layers} layers")

    # Reference
    input_ids = torch.randint(0, 1000, (1, 32), device="cuda")
    with torch.no_grad():
        ref = model(input_ids).logits.clone()
    ref_topk = ref[0, -1].topk(5).indices.tolist()
    print(f"  Ref top-5: {ref_topk}")

    # Apply module pass
    model, info = fused_rmsnorm_residual_transform_pass(model, {"casting_mode": "llama"})
    print(f"  Fused: {info['num_fused']} layers — {info['fused_layers']}")

    # Correctness
    with torch.no_grad():
        fused = model(input_ids).logits
    err = (fused - ref).abs().max().item()
    fused_topk = fused[0, -1].topk(5).indices.tolist()
    topk_match = ref_topk == fused_topk

    print(f"  Max abs err: {err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")
    print(f"  Top-5 match: {'PASS' if topk_match else 'FAIL'}")
    print(f"    ref:   {ref_topk}")
    print(f"    fused: {fused_topk}")

    ok = err < 1e-4 and topk_match and info["num_fused"] == 2
    print(f"  Overall: {'PASS' if ok else 'FAIL'}")
    return ok


def test_casting_modes():
    """Test all 3 casting modes."""
    print("\n" + "=" * 70)
    print("TEST: Module Pass — Casting Modes")
    print("=" * 70)

    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=64,
        vocab_size=500, use_cache=False,
    )

    all_pass = True
    for mode in ["llama", "gemma", "none"]:
        model = LlamaForCausalLM(config).to("cuda").eval()
        input_ids = torch.randint(0, 500, (1, 8), device="cuda")

        with torch.no_grad():
            ref = model(input_ids).logits.clone()

        model, info = fused_rmsnorm_residual_transform_pass(model, {"casting_mode": mode})

        with torch.no_grad():
            fused = model(input_ids).logits

        err = (fused - ref).abs().max().item()
        ok = err < 1e-3 and info["num_fused"] == 1
        all_pass &= ok
        print(f"  mode={mode:<6}  fused={info['num_fused']}  max_err={err:.2e}  {'PASS' if ok else 'FAIL'}")

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_llama_7b():
    """Test on real Llama-2-7B (if available)."""
    print("\n" + "=" * 70)
    print("TEST: Module Pass on Llama-2-7B")
    print("=" * 70)

    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"  SKIPPED: Could not load model — {e}")
        return True  # Don't fail suite

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params/1e9:.1f}B params")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    input_ids = torch.randint(100, 5000, (1, 64), device="cuda")

    # Reference
    with torch.no_grad():
        ref = model(input_ids).logits.clone()
    ref_topk = ref[0, -1].topk(5).indices.tolist()

    # Apply pass
    model, info = fused_rmsnorm_residual_transform_pass(model, {"casting_mode": "llama"})
    print(f"  Fused: {info['num_fused']} layers")

    # Correctness
    with torch.no_grad():
        fused = model(input_ids).logits

    abs_diff = (fused - ref).abs()
    max_err = abs_diff.max().item()
    mean_err = abs_diff.mean().item()
    logit_range = ref.max().item() - ref.min().item()
    rel_err = max_err / logit_range if logit_range > 0 else float('inf')

    cos_sim = torch.nn.functional.cosine_similarity(
        ref[0, -1].unsqueeze(0).float(),
        fused[0, -1].unsqueeze(0).float(),
    ).item()

    fused_topk = fused[0, -1].topk(5).indices.tolist()
    topk_match = ref_topk == fused_topk

    ok = (rel_err < 0.01) and (cos_sim > 0.999) and topk_match

    print(f"\n  CORRECTNESS:")
    print(f"    Max abs err:       {max_err:.4e}")
    print(f"    Mean abs err:      {mean_err:.4e}")
    print(f"    Logit range:       {logit_range:.2f}")
    print(f"    Relative error:    {rel_err:.6f}  {'PASS' if rel_err < 0.01 else 'FAIL'}")
    print(f"    Cosine similarity: {cos_sim:.6f}  {'PASS' if cos_sim > 0.999 else 'FAIL'}")
    print(f"    Top-5 match:       {'PASS' if topk_match else 'FAIL'}")
    print(f"      ref:   {ref_topk}")
    print(f"      fused: {fused_topk}")
    print(f"    Overall:           {'PASS' if ok else 'FAIL'}")

    # Latency comparison
    print(f"\n  LATENCY:")

    # Unfused — need to reload
    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()

    n_warmup, n_iters = 10, 50
    for _ in range(n_warmup):
        with torch.no_grad():
            model(input_ids)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iters):
        with torch.no_grad():
            model(input_ids)
    e.record()
    torch.cuda.synchronize()
    unfused_ms = s.elapsed_time(e) / n_iters
    print(f"    Unfused: {unfused_ms:.2f} ms")

    # Fused
    model, _ = fused_rmsnorm_residual_transform_pass(model, {"casting_mode": "llama"})

    for _ in range(n_warmup):
        with torch.no_grad():
            model(input_ids)
    torch.cuda.synchronize()
    s2 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    s2.record()
    for _ in range(n_iters):
        with torch.no_grad():
            model(input_ids)
    e2.record()
    torch.cuda.synchronize()
    fused_ms = s2.elapsed_time(e2) / n_iters

    saving = unfused_ms - fused_ms
    speedup = unfused_ms / fused_ms if fused_ms > 0 else 0
    print(f"    Fused:   {fused_ms:.2f} ms")
    print(f"    Saving:  {saving:+.2f} ms ({speedup:.3f}x)")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return ok


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    all_pass = True
    all_pass &= test_tiny_llama()
    all_pass &= test_casting_modes()
    all_pass &= test_llama_7b()

    print("\n" + "=" * 70)
    if all_pass:
        print("MODULE PASS: ALL TESTS PASSED")
    else:
        print("MODULE PASS: SOME TESTS FAILED")
    print("=" * 70)

    sys.exit(0 if all_pass else 1)
