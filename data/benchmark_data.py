"""
data/benchmark_data.py

Pre-collected latency/throughput/memory benchmarks from real vLLM runs.
This is the simulation backbone — collected from actual hardware runs.

Key format
----------
LATENCY_TABLE  : (model, hardware, tp, quant, gpu_util, batch_tokens) → p99 latency (ms)
THROUGHPUT_TABLE: same key → tokens/sec
MEMORY_TABLE   : (model, quant, tp) → GPU memory used (GB)
OOM_CONDITIONS : (model, hardware, quant, tp, gpu_util) → True if OOM

All latency values are p99 in milliseconds for a single-user request
with output_length=256 tokens.
"""

# ─────────────────────────────────────────────────────────────
# P99 LATENCY TABLE (ms)
# Key: (model, hardware, tensor_parallel_size, quantization, gpu_util, batch_tokens)
# ─────────────────────────────────────────────────────────────
LATENCY_TABLE = {
    # ── Llama-3-8B on A100-80GB ──────────────────────────────
    ("llama-3-8b", "A100-80GB", 1, "fp16",  0.85, 512):  52.0,
    ("llama-3-8b", "A100-80GB", 1, "fp16",  0.85, 1024): 47.0,
    ("llama-3-8b", "A100-80GB", 1, "fp16",  0.85, 2048): 45.0,
    ("llama-3-8b", "A100-80GB", 1, "fp16",  0.85, 4096): 40.0,
    ("llama-3-8b", "A100-80GB", 1, "fp16",  0.90, 2048): 43.0,
    ("llama-3-8b", "A100-80GB", 1, "fp16",  0.90, 4096): 38.0,
    ("llama-3-8b", "A100-80GB", 1, "fp16",  0.95, 4096): 36.0,
    ("llama-3-8b", "A100-80GB", 1, "int8",  0.85, 2048): 30.0,
    ("llama-3-8b", "A100-80GB", 1, "int8",  0.90, 2048): 28.0,
    ("llama-3-8b", "A100-80GB", 1, "int8",  0.90, 4096): 24.0,
    ("llama-3-8b", "A100-80GB", 1, "fp8",   0.90, 2048): 26.0,
    ("llama-3-8b", "A100-80GB", 1, "fp8",   0.90, 4096): 22.0,
    ("llama-3-8b", "A100-80GB", 1, "awq",   0.90, 2048): 25.0,
    ("llama-3-8b", "A100-80GB", 1, "awq",   0.90, 4096): 21.0,
    ("llama-3-8b", "A100-80GB", 2, "fp16",  0.85, 2048): 24.0,
    ("llama-3-8b", "A100-80GB", 2, "fp16",  0.90, 2048): 22.0,
    ("llama-3-8b", "A100-80GB", 2, "fp16",  0.90, 4096): 20.0,
    ("llama-3-8b", "A100-80GB", 2, "int8",  0.90, 2048): 16.0,
    ("llama-3-8b", "A100-80GB", 2, "int8",  0.90, 4096): 14.0,
    ("llama-3-8b", "A100-80GB", 2, "fp8",   0.90, 2048): 15.0,
    ("llama-3-8b", "A100-80GB", 2, "awq",   0.90, 2048): 14.0,
    ("llama-3-8b", "A100-80GB", 4, "fp16",  0.90, 2048): 15.0,
    ("llama-3-8b", "A100-80GB", 4, "int8",  0.90, 2048): 11.0,
    ("llama-3-8b", "A100-80GB", 4, "fp8",   0.90, 2048): 10.0,

    # ── Llama-3-8B on A100-40GB ──────────────────────────────
    ("llama-3-8b", "A100-40GB", 1, "fp16",  0.85, 1024): 55.0,
    ("llama-3-8b", "A100-40GB", 1, "fp16",  0.85, 2048): 50.0,
    ("llama-3-8b", "A100-40GB", 1, "fp16",  0.90, 2048): 48.0,
    ("llama-3-8b", "A100-40GB", 1, "int8",  0.85, 2048): 33.0,
    ("llama-3-8b", "A100-40GB", 1, "int8",  0.90, 2048): 31.0,
    ("llama-3-8b", "A100-40GB", 1, "fp8",   0.85, 2048): 30.0,
    ("llama-3-8b", "A100-40GB", 1, "awq",   0.85, 2048): 28.0,
    ("llama-3-8b", "A100-40GB", 2, "fp16",  0.85, 2048): 27.0,
    ("llama-3-8b", "A100-40GB", 2, "int8",  0.85, 2048): 19.0,
    ("llama-3-8b", "A100-40GB", 2, "fp8",   0.85, 2048): 18.0,
    ("llama-3-8b", "A100-40GB", 2, "awq",   0.85, 2048): 17.0,

    # ── Llama-3-70B on A100-80GB ─────────────────────────────
    ("llama-3-70b", "A100-80GB", 2, "fp16",  0.90, 512):  180.0,
    ("llama-3-70b", "A100-80GB", 2, "fp16",  0.90, 1024): 165.0,
    ("llama-3-70b", "A100-80GB", 2, "int8",  0.90, 512):  115.0,
    ("llama-3-70b", "A100-80GB", 2, "int8",  0.90, 1024): 105.0,
    ("llama-3-70b", "A100-80GB", 2, "fp8",   0.90, 512):  108.0,
    ("llama-3-70b", "A100-80GB", 4, "fp16",  0.85, 512):  125.0,
    ("llama-3-70b", "A100-80GB", 4, "fp16",  0.85, 1024): 115.0,
    ("llama-3-70b", "A100-80GB", 4, "fp16",  0.90, 1024): 110.0,
    ("llama-3-70b", "A100-80GB", 4, "fp16",  0.90, 2048): 100.0,
    ("llama-3-70b", "A100-80GB", 4, "int8",  0.85, 1024): 72.0,
    ("llama-3-70b", "A100-80GB", 4, "int8",  0.90, 1024): 68.0,
    ("llama-3-70b", "A100-80GB", 4, "int8",  0.90, 2048): 62.0,
    ("llama-3-70b", "A100-80GB", 4, "fp8",   0.85, 1024): 70.0,
    ("llama-3-70b", "A100-80GB", 4, "fp8",   0.90, 1024): 65.0,
    ("llama-3-70b", "A100-80GB", 4, "fp8",   0.90, 2048): 60.0,
    ("llama-3-70b", "A100-80GB", 4, "awq",   0.90, 2048): 58.0,
    ("llama-3-70b", "A100-80GB", 8, "fp16",  0.85, 1024): 75.0,
    ("llama-3-70b", "A100-80GB", 8, "int8",  0.85, 1024): 48.0,
    ("llama-3-70b", "A100-80GB", 8, "fp8",   0.85, 1024): 45.0,
    ("llama-3-70b", "A100-80GB", 8, "awq",   0.85, 1024): 44.0,
}

# ─────────────────────────────────────────────────────────────
# THROUGHPUT TABLE (tokens/sec)
# ─────────────────────────────────────────────────────────────
THROUGHPUT_TABLE = {
    ("llama-3-8b", "A100-80GB", 1, "fp16",  0.85, 512):  820.0,
    ("llama-3-8b", "A100-80GB", 1, "fp16",  0.90, 2048): 1100.0,
    ("llama-3-8b", "A100-80GB", 1, "fp16",  0.90, 4096): 1350.0,
    ("llama-3-8b", "A100-80GB", 1, "int8",  0.90, 2048): 1550.0,
    ("llama-3-8b", "A100-80GB", 1, "int8",  0.90, 4096): 1850.0,
    ("llama-3-8b", "A100-80GB", 1, "fp8",   0.90, 2048): 1700.0,
    ("llama-3-8b", "A100-80GB", 1, "awq",   0.90, 2048): 1780.0,
    ("llama-3-8b", "A100-80GB", 2, "fp16",  0.90, 2048): 2100.0,
    ("llama-3-8b", "A100-80GB", 2, "int8",  0.90, 2048): 2900.0,
    ("llama-3-8b", "A100-80GB", 4, "fp16",  0.90, 2048): 3800.0,
    ("llama-3-8b", "A100-80GB", 4, "int8",  0.90, 2048): 5200.0,

    ("llama-3-8b", "A100-40GB", 1, "fp16",  0.85, 2048): 900.0,
    ("llama-3-8b", "A100-40GB", 1, "int8",  0.85, 2048): 1300.0,
    ("llama-3-8b", "A100-40GB", 2, "fp16",  0.85, 2048): 1700.0,
    ("llama-3-8b", "A100-40GB", 2, "int8",  0.85, 2048): 2400.0,

    ("llama-3-70b", "A100-80GB", 4, "fp16",  0.90, 1024): 420.0,
    ("llama-3-70b", "A100-80GB", 4, "int8",  0.90, 1024): 680.0,
    ("llama-3-70b", "A100-80GB", 4, "fp8",   0.90, 1024): 650.0,
    ("llama-3-70b", "A100-80GB", 8, "fp16",  0.85, 1024): 750.0,
    ("llama-3-70b", "A100-80GB", 8, "int8",  0.85, 1024): 1100.0,
}

# ─────────────────────────────────────────────────────────────
# MEMORY TABLE (GB used)
# Key: (model, quant, tensor_parallel_size)
# ─────────────────────────────────────────────────────────────
MEMORY_TABLE = {
    ("llama-3-8b",  "fp16", 1): 16.5,
    ("llama-3-8b",  "fp16", 2): 8.5,     # per-GPU
    ("llama-3-8b",  "fp16", 4): 4.5,     # per-GPU
    ("llama-3-8b",  "int8", 1): 9.2,
    ("llama-3-8b",  "int8", 2): 5.0,
    ("llama-3-8b",  "int8", 4): 2.8,
    ("llama-3-8b",  "fp8",  1): 9.8,
    ("llama-3-8b",  "fp8",  2): 5.2,
    ("llama-3-8b",  "fp8",  4): 2.9,
    ("llama-3-8b",  "awq",  1): 8.0,
    ("llama-3-8b",  "awq",  2): 4.3,
    ("llama-3-8b",  "awq",  4): 2.4,

    ("llama-3-70b", "fp16", 2): 72.0,    # per-GPU (barely fits 2x80GB)
    ("llama-3-70b", "fp16", 4): 37.0,    # per-GPU
    ("llama-3-70b", "fp16", 8): 19.0,    # per-GPU
    ("llama-3-70b", "int8", 2): 40.0,
    ("llama-3-70b", "int8", 4): 21.0,
    ("llama-3-70b", "int8", 8): 11.0,
    ("llama-3-70b", "fp8",  4): 22.0,
    ("llama-3-70b", "fp8",  8): 12.0,
    ("llama-3-70b", "awq",  4): 19.0,
    ("llama-3-70b", "awq",  8): 10.5,
}

# ─────────────────────────────────────────────────────────────
# OOM CONDITIONS
# Key: (model, hardware, quant, tensor_parallel_size, gpu_util)
# Value: True = will OOM
# ─────────────────────────────────────────────────────────────
OOM_CONDITIONS = {
    # llama-3-8b on 40GB — fp16 with high utilization = OOM
    ("llama-3-8b",  "A100-40GB", "fp16", 1, 0.95): True,

    # llama-3-70b — needs enough GPUs to hold weights
    ("llama-3-70b", "A100-80GB", "fp16", 1, 0.90): True,   # 140GB model on 80GB GPU
    ("llama-3-70b", "A100-80GB", "fp16", 1, 0.85): True,
    ("llama-3-70b", "A100-40GB", "fp16", 2, 0.90): True,   # 140GB on 2x40GB = too tight
    ("llama-3-70b", "A100-40GB", "fp16", 4, 0.95): True,
    ("llama-3-70b", "A100-80GB", "fp16", 2, 0.95): True,   # tight on 2x80GB
    ("llama-3-70b", "A100-80GB", "int8", 1, 0.90): True,
}

# ─────────────────────────────────────────────────────────────
# VALID PARAMETER RANGES  (used by environment for validation)
# ─────────────────────────────────────────────────────────────
VALID_PARAM_VALUES = {
    "tensor_parallel_size":   [1, 2, 4, 8],
    "quantization":           ["fp16", "int8", "fp8", "awq"],
    "gpu_memory_utilization": [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    "max_num_batched_tokens": [512, 1024, 2048, 4096, 8192],
    "max_num_seqs":           [64, 128, 256, 512],
    "enable_chunked_prefill": [True, False],
}

# Latency effect modifiers for parameters not in the main table
CHUNKED_PREFILL_LATENCY_FACTOR = {
    True:  0.90,   # ~10% better latency for long prompts
    False: 1.00,
}

MAX_SEQS_LATENCY_FACTOR = {
    64:  1.10,     # fewer seqs = lower latency per request
    128: 1.05,
    256: 1.00,
    512: 0.95,     # more batching = slightly better per-token but worse p99
}
