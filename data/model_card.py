# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "gpt2": {
        "hf_id": "openai-community/gpt2",
        "params_b": 0.124,
        "hf_token": False,
        "description": "GPT-2 Small (124M) — no HF token required",
        "ram_weights_gb": 0.25,
        "startup_s": 10,
    },
    "smollm2-135m": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "params_b": 0.135,
        "hf_token": False,
        "description": "SmolLM2-135M-Instruct — modern tiny model",
        "ram_weights_gb": 0.27,
        "startup_s": 20,
    },
    "gemma-3-270m": {
        "hf_id": "google/gemma-3-270m-it",
        "params_b": 0.270,
        "hf_token": True,
        "description": "Gemma-3-270M-IT — hardest task, KV cache is key",
        "ram_weights_gb": 0.54,
        "startup_s": 40,
    },
}

# ─────────────────────────────────────────────────────────────
# VALID PARAMETER VALUES
# ─────────────────────────────────────────────────────────────
VALID_PARAM_VALUES = {
    "dtype": ["float32", "float16", "bfloat16"],
    "max_model_len": [128, 192, 256],
    "max_num_batched_tokens": [64, 128, 256, 512],
    "max_num_seqs": [1, 2, 4, 8],
}

# ─────────────────────────────────────────────────────────────
# PARAMETERS THAT REQUIRE A vLLM RESTART vs REQUEST-ONLY CHANGES
# ─────────────────────────────────────────────────────────────
REQUIRES_RESTART = {"dtype", "max_model_len"}
REQUEST_ONLY = {"max_num_batched_tokens", "max_num_seqs"}

# ─────────────────────────────────────────────────────────────
# RAM LIMITS
# Base system after stopping Docker: ~3.36GB
# HF Spaces total: 8GB
# Safe headroom for vLLM: 8 - 3.36 = ~4.6GB available
# ─────────────────────────────────────────────────────────────
RAM_SAFETY_LIMIT_GB = 4.0
 
# ─────────────────────────────────────────────────────────────
# vLLM SERVER SETTINGS
# ─────────────────────────────────────────────────────────────
VLLM_PORT = 8100
VLLM_STARTUP_TIMEOUT_S = 180
 
# ─────────────────────────────────────────────────────────────
# BENCHMARK SETTINGS
# Minimal to stay within 20-min inference script runtime limit.
# ─────────────────────────────────────────────────────────────
BENCH_WARMUP_REQUESTS = 2
BENCH_TIMED_REQUESTS = 5
BENCH_MAX_TOKENS = 32
BENCH_PROMPT = "Summarize in one sentence: The quick brown fox jumps over the lazy dog."