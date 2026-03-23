# ADLS Project Context — Kernel-Fusion-Aware Optimisation Pipeline in MASE

> **Purpose of this file:** Paste this into the system prompt or first message of any new LLM chat to give it full context about the project. It contains everything needed to continue development without re-explaining anything.

---

## Who I Am

- **Name:** Dorian
- **Role:** MSc Applied Machine Learning student, Imperial College London (EEE department)
- **Supervisor:** Dr Sonali Parbhoo
- **Course:** Advanced Deep Learning Systems (ADLS) 2024/25
- **Other work:** Also working on a thesis project called CBRA (Causal Bayesian Reasoning Agent) — separate from this

---

## The Group Project

We are building a **kernel-fusion-aware optimisation pipeline inside MASE** for the ADLS module. MASE is an open-source PyTorch FX-based ML compiler maintained by the DeepWok Lab at Imperial College London.

- **Repo:** https://github.com/DeepWok/mase/tree/main
- **Core software stack:** `src/chop/`
- **Transform passes:** `chop.passes.transform` (quantise, prune, etc.)
- **Search/NAS:** Uses Optuna for multi-objective search
- **Related repos:** `DeepWok/mase-triton`, `DeepWok/mase-triton-examples`, `DeepWok/mase-cuda`

### The Three Parts

| Part | Owner | Description |
|------|-------|-------------|
| **Part 1** | Other group member(s) | **FlexAttention Integration** — Replace `F.scaled_dot_product_attention` with PyTorch's FlexAttention API. Programmable `score_mod` functions (causal, sliding window, ALiBi) compiled into fused Triton kernels via `torch.compile`. Targets Llama, BERT, Mistral. |
| **Part 2** | **Dorian (me)** | **Custom Triton Kernel: Fused RMSNorm + Residual** — Hand-written Triton GPU kernel fusing residual addition and RMSNorm into a single pass. Eliminates 2 redundant global memory round-trips per transformer layer. |
| **Part 3** | Other group member(s) | **Automated Search Pipeline** — Optuna-based Pareto search over (bit-width × fusion strategy) producing accuracy/latency/memory Pareto frontiers on 3+ models. Key contribution: adding `LatencyRunner` to MASE's search loop (previously only optimised accuracy/memory). |

### Pipeline Architecture (data flows top-to-bottom)

```
ML Model (BERT / Llama / Mistral)
    ↓
Quantise Transform Pass
    ↓
FlexAttention Fusion (Part 1)
    ↓
RMSNorm + Residual Fusion (Part 2)
    ↓
LatencyRunner + Eval
    ↓
Optuna Search → Pareto Frontier
```

### Key Gap We Fill

MASE's search infrastructure had a `LatencyRunner` that was considered but never implemented. Search currently optimises for accuracy/memory but ignores actual runtime. We close this loop, enabling joint optimisation over accuracy, latency, and memory.

### Fallback Design

- Part 3 (Pipeline + Search) is load-bearing and works independently
- Parts 1 & 2 are additive — partial success still yields a complete automated system with Pareto results on 3 models
- If Triton kernel is blocked → wrap Liger-Kernel as fallback
- If FlexAttention hits `torch.compile` graph breaks → document as standalone benchmarks

### Risks

| Severity | Risk | Mitigation |
|----------|------|------------|
| HIGH | `torch.compile` graph breaks | Compile `flex_attention` function only, not full model |
| HIGH | Triton kernel debugging | Start from Liger-Kernel reference |
| MED | HPC queue times / GPU access | Start search jobs early, run overnight, parallelise configs |
| MED | FlexAttention API | Pin torch 2.6; causal-only first; sliding window/ALiBi are stretch goals |
| LOW | Integration / merge conflicts | Each part in its own feature branch; pipeline owner integrates |

### Experiments & Deliverables

1. **Pareto Search:** Accuracy vs. Latency vs. Memory — Optuna search (bit-width × fusion strategy) on 3+ models
2. **Incremental Optimisation Gains:** Baseline → +Quantisation → +RMSNorm fusion → +FlexAttention (shows individual and cumulative benefit)
3. **Sequence Length Scaling:** Latency vs. seq-length per fusion strategy (reveals asymptotic complexity)
4. **Batch Size Scaling:** Latency/throughput across batch sizes (1, 4, 16, 32, 128) per fusion strategy
5. **Peak GPU Memory Profiling:** `torch.cuda.max_memory_allocated()` per optimisation stage

---

## Part 2 — Current Status (COMPLETE: Standalone Kernel)

### What Was Built

Two Python files that work together:

#### `triton_fused_add_rmsnorm.py`

Contains:

1. **`_fused_add_rmsnorm_fwd_kernel`** — Triton forward kernel. Each program instance processes one row (one token position) of the (B×T, D) tensor. Loads X_residual and X_hidden once, computes `residual = X_res + X_hid`, then `normed = (residual / RMS(residual)) * weight`, stores both outputs and caches rstd for backward. BLOCK_SIZE = next_power_of_2(hidden_dim).

2. **`_fused_add_rmsnorm_bwd_kernel`** — Triton backward kernel. Receives dL/d(normed) and dL/d(residual), computes dL/dX_residual, dL/dX_hidden (both equal since they're summed), and per-row dL/dWeight partials. Weight gradient is reduced across rows in PyTorch (two-stage reduction pattern from Liger-Kernel).

3. **`FusedAddRMSNorm`** — `torch.autograd.Function` connecting kernels to PyTorch autograd.

4. **`FusedAddRMSNormModule`** — `torch.nn.Module` wrapper with learnable weight. Drop-in replacement for MASE transform pass integration.

#### `test_fused_add_rmsnorm.py`

- Forward correctness: 144 tests (8 shapes × 3 dtypes × 3 casting modes × 2 offsets)
- Backward correctness: 36 tests (gradient agreement with PyTorch reference)
- nn.Module wrapper test
- Latency benchmark (fused vs unfused, 6 configurations)
- Memory benchmark (peak GPU allocation)

### Casting Modes Supported

| Mode | Behaviour | Used by |
|------|-----------|---------|
| `"llama"` | Only rstd computed in fp32 | Llama 2, Llama 3 |
| `"gemma"` | Everything cast to fp32; weight offset = 1.0 | Gemma, Gemma 2 |
| `"none"` | No casting, but reductions still accumulated in fp32 | General |

### Bug Found and Fixed

**Problem:** 8/144 forward tests failed — all `bf16 + casting_mode="none"`. The sum-of-squares reduction was done entirely in bf16, causing accumulation errors across large hidden dimensions.

**Fix:** In all three `"none"` branches (forward kernel, backward kernel, PyTorch reference), cast to fp32 before reduction operations even though the mode says "none". This matches Liger-Kernel's approach.

**Result after fix:** 144/144 forward, 36/36 backward — all passing.

### Benchmark Results (Google Colab, Tesla T4)

| Configuration | Unfused (µs) | Fused (µs) | Speedup |
|--------------|-------------|-----------|---------|
| Single token, Llama-7B (bf16) | 94.4 | 78.7 | 1.20× |
| **Batch inference, Llama-7B (bf16)** | **740.7** | **150.7** | **4.92×** |
| **Long seq, Llama-7B (bf16)** | **1235.1** | **289.0** | **4.27×** |
| **Batch inference, Llama-70B (bf16)** | **323.3** | **83.7** | **3.86×** |
| Single token, Llama-7B (fp32) | 72.7 | 81.4 | 0.89× |
| Batch, Llama-7B (fp32) | 360.3 | 147.2 | 2.45× |

**Memory:** 60% peak reduction (80 MB → 32 MB) at shape (4, 512, 4096) bf16.

**Note:** T4 numbers are for validation. Final report benchmarks should come from Imperial HPC A100s.

### Reference Implementations Used

- **Liger-Kernel** (`linkedin/Liger-Kernel`): `src/liger_kernel/ops/rms_norm.py` for RMSNorm kernel design, casting modes, backward pass derivation. Their `FusedAddRMSNorm` (PR #812) is the closest existing implementation to what we built.
- **Unsloth** (`unslothai/unsloth`): `kernels/rms_layernorm.py` — secondary reference for Triton RMSNorm patterns.

---

## Part 2 — Next Steps (TODO)

### Immediate: MASE Integration

1. **Fork MASE** — single group fork at `your-group/mase`, each member on a feature branch:
   - `feature/flex-attention` (Part 1)
   - `feature/fused-rmsnorm-residual` (Part 2 — Dorian)
   - `feature/latency-search-pipeline` (Part 3)

2. **Place kernel in MASE source tree** — suggested location:
   ```
   src/chop/passes/transform/fused_rmsnorm/
       __init__.py
       triton_fused_add_rmsnorm.py    (the kernel)
       fused_rmsnorm_pass.py          (the FX graph transform pass)
   ```

3. **Write the MASE transform pass** — a function that:
   - Walks the PyTorch FX graph
   - Pattern-matches on: `add` node followed by `RMSNorm` call
   - Replaces both nodes with a single `FusedAddRMSNormModule` call
   - Follows existing conventions in `chop.passes.transform.quantize`

4. **Keep upstream synced:**
   ```bash
   git remote add upstream https://github.com/DeepWok/mase.git
   git fetch upstream
   git merge upstream/main
   ```

### Then: Additional Benchmarks

- Sequence length scaling: sweep T ∈ [128, 256, 512, 1024, 2048, 4096], fixed batch
- Batch size scaling: sweep B ∈ [1, 4, 16, 32, 128], fixed seq length
- Full LlamaDecoderLayer integration: time a single layer forward pass with/without fusion
- Run everything on Imperial HPC A100s for final numbers

### Then: Connect to Part 3

- Make fused kernel a toggleable option in the Optuna search config
- Part 3's search pipeline explores (bit-width × fusion_strategy) where fusion_strategy includes fused_rmsnorm as one axis

---

## Technical Details for Reference

### RMSNorm Formula

```
residual_out  = X_residual + X_hidden
rms           = sqrt( (1/D) * sum(residual_out²) + eps )
normed_out    = (residual_out / rms) * (weight + offset)
```

Where offset = 0.0 for Llama, 1.0 for Gemma.

### RMSNorm Backward

```
Let r = residual, rstd = 1/RMS(r), w = weight + offset

dL/dr = rstd * (dNormed * w) - (rstd³ / D) * sum(dNormed * w * r) * r
dL/dWeight = sum_over_rows(dNormed * r * rstd)

Since residual = X_res + X_hid:
    dL/dX_res = dL/dr + dL/d(residual_out)
    dL/dX_hid = dL/dr + dL/d(residual_out)
```

### Kernel Design Choices

- **One row per program instance** — each Triton program handles one token position
- **Single BLOCK_SIZE tile** — `triton.next_power_of_2(hidden_dim)`, max 65536
- **fp32 accumulation for all reductions** — even in "none" casting mode, sum-of-squares and dot products are accumulated in fp32 to prevent bf16 rounding errors
- **Two-stage weight gradient reduction** — per-row partials computed in Triton, summed across rows in PyTorch (matches Liger-Kernel pattern)
- **num_warps** — `min(max(BLOCK_SIZE // 256, 1), 16)`

### Environment Tested

- PyTorch 2.10.0+cu128
- Triton 3.6.0
- GPU: Tesla T4 (Google Colab)
- Target for final benchmarks: NVIDIA A100 (Imperial HPC)

---

## File Inventory

| File | Description | Status |
|------|-------------|--------|
| `triton_fused_add_rmsnorm.py` | Triton kernel + autograd + nn.Module | ✅ Complete |
| `test_fused_add_rmsnorm.py` | Tests + benchmarks | ✅ Complete |
| `Part2_FusedAddRMSNorm_Triton_Kernel.ipynb` | Colab notebook running everything | ✅ Complete |
| `Part2_FusedAddRMSNorm_DevLog.docx` | Formatted development log document | ✅ Complete |
| MASE transform pass | FX graph pattern matching | ❌ TODO |
| HPC A100 benchmarks | Final report numbers | ❌ TODO |
| Sequence/batch scaling experiments | Deliverables 3 & 4 | ❌ TODO |
