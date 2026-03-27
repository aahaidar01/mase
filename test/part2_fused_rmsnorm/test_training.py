"""
Test: Training with Fused Add+RMSNorm Kernel
=============================================

Verifies that the fused kernel works correctly in a full training loop:
1. Gradients flow through the fused autograd Function
2. Optimizer steps update all weights (including fused RMSNorm weight)
3. Loss decreases over epochs
4. Training trajectory matches unfused baseline

Usage:
    python -u test_training.py

Runs on any CUDA GPU. Uses a tiny Llama model (~1.7M params) so no
HuggingFace download needed.
"""

import sys
import os
import torch
import torch.nn as nn
import types
import copy
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_fused_add_rmsnorm import FusedAddRMSNorm


# ---------------------------------------------------------------------------
# Module-level pass (inlined for standalone HPC use)
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


def apply_fused_pass(model, casting_mode="llama"):
    count = 0
    for name, module in model.named_modules():
        if _is_decoder_layer(module):
            if _patch_decoder_layer(module, casting_mode):
                count += 1
    return count


# ===========================================================================
# Test 1: Training loop on tiny Llama
# ===========================================================================
def test_training_tiny_llama():
    """
    Train a tiny Llama for several steps with and without fusion.
    Verify:
    - Loss decreases (model is learning)
    - Gradients are non-zero on all parameters
    - RMSNorm weights receive gradients and update
    - Fused and unfused training produce similar loss trajectories
    """
    print("\n" + "=" * 70)
    print("TEST: Training Loop — Tiny Llama (fused vs unfused)")
    print("=" * 70)

    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        vocab_size=1000,
        use_cache=False,
    )

    # Create two identical models from the same weights
    torch.manual_seed(42)
    model_unfused = LlamaForCausalLM(config).to("cuda").train()

    model_fused = copy.deepcopy(model_unfused)
    n_fused = apply_fused_pass(model_fused, "llama")
    print(f"  Model: {sum(p.numel() for p in model_unfused.parameters()):,} params, "
          f"{config.num_hidden_layers} layers")
    print(f"  Fused layers: {n_fused}")

    # Same optimizer config for both
    lr = 1e-3
    opt_unfused = torch.optim.AdamW(model_unfused.parameters(), lr=lr)
    opt_fused = torch.optim.AdamW(model_fused.parameters(), lr=lr)

    # Training data (fixed random batch, same for both)
    torch.manual_seed(123)
    n_steps = 20
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device="cuda")
    labels = input_ids.clone()

    losses_unfused = []
    losses_fused = []

    print(f"\n  Training for {n_steps} steps (batch={batch_size}, seq={seq_len}, lr={lr})")
    print(f"\n  {'Step':<6} {'Unfused Loss':<16} {'Fused Loss':<16} {'Diff':<14} {'Match'}")
    print(f"  {'-' * 68}")

    all_match = True

    for step in range(n_steps):
        # --- Unfused step ---
        opt_unfused.zero_grad()
        out_u = model_unfused(input_ids=input_ids, labels=labels)
        loss_u = out_u.loss
        loss_u.backward()
        opt_unfused.step()
        losses_unfused.append(loss_u.item())

        # --- Fused step ---
        opt_fused.zero_grad()
        out_f = model_fused(input_ids=input_ids, labels=labels)
        loss_f = out_f.loss
        loss_f.backward()
        opt_fused.step()
        losses_fused.append(loss_f.item())

        diff = abs(loss_u.item() - loss_f.item())
        # Allow increasing divergence over steps due to compounding rounding
        tol = 1e-3 * (1 + step * 0.5)
        match = diff < tol
        all_match &= match

        if step < 5 or step == n_steps - 1 or not match:
            print(f"  {step:<6} {loss_u.item():<16.6f} {loss_f.item():<16.6f} "
                  f"{diff:<14.2e} {'OK' if match else 'DIVERGED'}")

    # --- Check 1: Loss decreased ---
    unfused_decreased = losses_unfused[-1] < losses_unfused[0]
    fused_decreased = losses_fused[-1] < losses_fused[0]
    print(f"\n  Loss decreased (unfused): {losses_unfused[0]:.4f} -> {losses_unfused[-1]:.4f} "
          f"{'PASS' if unfused_decreased else 'FAIL'}")
    print(f"  Loss decreased (fused):   {losses_fused[0]:.4f} -> {losses_fused[-1]:.4f} "
          f"{'PASS' if fused_decreased else 'FAIL'}")

    # --- Check 2: Final losses are close ---
    final_diff = abs(losses_unfused[-1] - losses_fused[-1])
    final_rel = final_diff / abs(losses_unfused[-1]) if losses_unfused[-1] != 0 else float('inf')
    final_close = final_rel < 0.05  # within 5% after 20 steps
    print(f"  Final loss difference:    {final_diff:.6f} (rel: {final_rel:.4f}) "
          f"{'PASS' if final_close else 'FAIL'}")

    return unfused_decreased and fused_decreased and final_close


# ===========================================================================
# Test 2: Gradient flow check
# ===========================================================================
def test_gradient_flow():
    """
    Verify that after one backward pass with the fused kernel:
    - All parameters have non-None gradients
    - RMSNorm weights specifically receive gradients
    - Gradient magnitudes are reasonable (not zero, not exploding)
    """
    print("\n" + "=" * 70)
    print("TEST: Gradient Flow — All Parameters Receive Gradients")
    print("=" * 70)

    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
        vocab_size=500, use_cache=False,
    )

    torch.manual_seed(42)
    model = LlamaForCausalLM(config).to("cuda").train()
    n_fused = apply_fused_pass(model, "llama")

    input_ids = torch.randint(0, 500, (2, 16), device="cuda")
    labels = input_ids.clone()

    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()

    total_params = 0
    params_with_grad = 0
    zero_grad_params = []
    none_grad_params = []
    rmsnorm_grads = {}

    for name, p in model.named_parameters():
        total_params += 1
        if p.grad is None:
            none_grad_params.append(name)
        elif p.grad.abs().max().item() == 0:
            zero_grad_params.append(name)
        else:
            params_with_grad += 1

        # Track RMSNorm weight gradients specifically
        if "layernorm" in name.lower() or "rmsnorm" in name.lower():
            if p.grad is not None:
                rmsnorm_grads[name] = p.grad.norm().item()

    print(f"  Total parameters: {total_params}")
    print(f"  Parameters with nonzero grad: {params_with_grad}")

    if none_grad_params:
        print(f"  Parameters with None grad: {len(none_grad_params)}")
        for n in none_grad_params[:5]:
            print(f"    - {n}")
    else:
        print(f"  Parameters with None grad: 0  PASS")

    if zero_grad_params:
        print(f"  Parameters with zero grad: {len(zero_grad_params)}")
        for n in zero_grad_params[:5]:
            print(f"    - {n}")
    else:
        print(f"  Parameters with zero grad: 0  PASS")

    print(f"\n  RMSNorm weight gradients:")
    for name, gnorm in rmsnorm_grads.items():
        print(f"    {name}: grad_norm={gnorm:.6f}")

    all_have_grad = len(none_grad_params) == 0
    none_zero = len(zero_grad_params) == 0
    rmsnorm_ok = len(rmsnorm_grads) > 0 and all(v > 0 for v in rmsnorm_grads.values())

    ok = all_have_grad and none_zero and rmsnorm_ok
    print(f"\n  Overall: {'PASS' if ok else 'FAIL'}")
    return ok


# ===========================================================================
# Test 3: Weight update verification
# ===========================================================================
def test_weight_updates():
    """
    Verify that optimizer.step() actually changes the model weights,
    including the RMSNorm weights that are used by the fused kernel.
    """
    print("\n" + "=" * 70)
    print("TEST: Weight Updates — Optimizer Modifies All Weights")
    print("=" * 70)

    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=64,
        vocab_size=500, use_cache=False,
    )

    torch.manual_seed(42)
    model = LlamaForCausalLM(config).to("cuda").train()
    apply_fused_pass(model, "llama")

    # Snapshot weights before training
    weights_before = {}
    for name, p in model.named_parameters():
        weights_before[name] = p.data.clone()

    # One training step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    input_ids = torch.randint(0, 500, (2, 8), device="cuda")

    optimizer.zero_grad()
    out = model(input_ids=input_ids, labels=input_ids.clone())
    out.loss.backward()
    optimizer.step()

    # Check which weights changed
    changed = 0
    unchanged = []
    rmsnorm_changed = {}

    for name, p in model.named_parameters():
        diff = (p.data - weights_before[name]).abs().max().item()
        if diff > 0:
            changed += 1
        else:
            unchanged.append(name)

        if "layernorm" in name.lower() or "rmsnorm" in name.lower():
            rmsnorm_changed[name] = diff

    total = len(weights_before)
    print(f"  Total parameters: {total}")
    print(f"  Weights updated: {changed}/{total}")

    if unchanged:
        print(f"  Unchanged weights: {len(unchanged)}")
        for n in unchanged[:5]:
            print(f"    - {n}")
    else:
        print(f"  Unchanged weights: 0  PASS")

    print(f"\n  RMSNorm weight updates:")
    for name, diff in rmsnorm_changed.items():
        status = "updated" if diff > 0 else "UNCHANGED"
        print(f"    {name}: max_diff={diff:.2e} ({status})")

    rmsnorm_ok = all(v > 0 for v in rmsnorm_changed.values()) if rmsnorm_changed else False
    ok = changed == total and rmsnorm_ok
    print(f"\n  Overall: {'PASS' if ok else 'FAIL'}")
    return ok


# ===========================================================================
# Test 4: Multi-epoch overfitting
# ===========================================================================
def test_overfit_single_batch():
    """
    Train on the same batch for 50 steps. The fused model should overfit
    (loss → ~0) just like the unfused model. This proves the fused kernel
    doesn't break the optimization landscape.
    """
    print("\n" + "=" * 70)
    print("TEST: Overfitting — Fused Model Memorises Single Batch")
    print("=" * 70)

    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=64,
        vocab_size=100, use_cache=False,
    )

    torch.manual_seed(42)
    model = LlamaForCausalLM(config).to("cuda").train()
    apply_fused_pass(model, "llama")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    input_ids = torch.randint(0, 100, (2, 16), device="cuda")
    labels = input_ids.clone()

    n_steps = 50
    initial_loss = None
    final_loss = None

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == n_steps - 1:
            final_loss = loss.item()

        if step < 3 or step == n_steps - 1:
            print(f"  Step {step:>3}: loss = {loss.item():.6f}")

    reduction = (initial_loss - final_loss) / initial_loss * 100
    overfit_ok = final_loss < initial_loss * 0.1  # Loss dropped by 90%+

    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Reduction:    {reduction:.1f}%")
    print(f"  Overall: {'PASS' if overfit_ok else 'FAIL'} "
          f"(threshold: >90% reduction)")
    return overfit_ok


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
    all_pass &= test_gradient_flow()
    all_pass &= test_weight_updates()
    all_pass &= test_overfit_single_batch()
    all_pass &= test_training_tiny_llama()

    print("\n" + "=" * 70)
    if all_pass:
        print("TRAINING TESTS: ALL PASSED")
    else:
        print("TRAINING TESTS: SOME FAILED")
    print("=" * 70)

    sys.exit(0 if all_pass else 1)
