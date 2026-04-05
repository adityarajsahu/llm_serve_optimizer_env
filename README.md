---
title: LLM Serve Optimizer Env
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# LLM Serve Optimizer Environment

An **Reinforcement Learning environment** for optimizing LLM deployment configurations to meet target latency and throughput requirements. Built using [OpenEnv](https://github.com/meta-pytorch/OpenEnv), the environment runs [vLLM](https://github.com/vllm-project/vllm) inference server and benchmarks it after each configuration change — providing an RL agent with real-world performance feedback.

The environment is **platform and hardware agnostic** — it can run on CPU, GPU, and AI accelerators such as Google TPU, with vLLM transparently handling the backend. The reference deployment targets CPU-only hardware (HF Spaces free tier), but the same environment and agent script work without modification on any hardware vLLM supports.

---

## What This Is

Deploying an LLM in production requires tuning several engine parameters (data types, context lengths, batching settings) to balance speed, throughput, and memory usage. Getting this right is non-trivial and typically requires manual experimentation.

This project frames that problem as a **reinforcement learning task**:

- An **agent** observes the current deployment configuration and its measured performance (p99 latency, throughput, RAM usage).
- The agent sends a single `(parameter, value)` action to the environment.
- The environment **restarts or reconfigures** the vLLM server with the new setting, runs a real benchmark, and returns a **reward** based on how much the performance improved and whether the target was met.
- The agent's goal is to find the optimal configuration within a limited number of steps.

The environment server is a FastAPI/WebSocket application built with OpenEnv. Clients connect over WebSocket and interact with the environment using a standard `reset → step → step → ...` loop.

---

## Supported Models

The environment supports any model that vLLM can serve. The three pre-configured models are deliberately small to stay within the reference deployment's resource limits (HF Spaces: 2 vCPU, ~8 GB RAM). All three are publicly available on Hugging Face and require no HF token. On GPU or TPU hardware, larger models can be used by extending the model registry in `data/model_card.py`.

> **Note:** Pythia-70M-Deduped is used by **two** tasks (`easy_pythia_p99` and `extreme_pythia_p99_tput_ram_optimize`) with different performance targets and reward structures.

| Model                     | HF ID                                 | Parameters | Tasks                                                                       |
| ------------------------- | ------------------------------------- | ---------- | --------------------------------------------------------------------------- |
| **Pythia-70M-Deduped**    | `EleutherAI/pythia-70m-deduped`       | 70M        | `easy_pythia_p99`, `extreme_pythia_p99_tput_ram_optimize`                   |
| **GPT-2 Small**           | `openai-community/gpt2`               | 124M       | `medium_gpt2_p99_tput`                                                      |
| **SmolLM2-135M-Instruct** | `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M       | `hard_smollm2_stricter_p99_tput`                                            |

---

## Tasks

The environment defines **four tasks** of increasing difficulty. Each task specifies which model to deploy, performance targets, and a step budget. Task IDs are descriptive strings that encode the model and the objectives being optimized.

> **Targets are calibrated to real CPU measurements** on this host. They reflect realistic improvements achievable within the step budget — not aspirational "best case" figures.

### `easy_pythia_p99` — Pythia-70M-Deduped

- **Baseline:** p99 = 1789 ms · throughput = 24.3 tok/s · RAM ≈ 93 GB (system total)
- **Goal:** p99 latency ≤ **920 ms** (latency-only target)
- **Steps:** 5
- **Hint:** Switching `dtype` from `float32` → `bfloat16` is typically sufficient.

### `medium_gpt2_p99_tput` — GPT-2 (124M)

- **Baseline:** p99 = 2017 ms · throughput = 17.2 tok/s · RAM ≈ 92 GB
- **Goal:** p99 latency ≤ **1750 ms** AND throughput ≥ **19 tok/s**
- **Steps:** 5
- **Hint:** `bfloat16` helps latency; tuning `max_num_batched_tokens` improves throughput.

### `hard_smollm2_stricter_p99_tput` — SmolLM2-135M-Instruct

- **Baseline:** p99 = 2610 ms · throughput = 12.6 tok/s · RAM ≈ 93 GB
- **Goal:** p99 latency ≤ **2650 ms** AND throughput ≥ **13 tok/s**
- **Steps:** 5
- **Note:** RAM pressure is a real concern here — the agent must also keep memory usage in check.

### `extreme_pythia_p99_tput_ram_optimize` — Pythia-70M-Deduped _(new)_

- **Baseline:** p99 = 1789 ms · throughput = 24.3 tok/s · RAM ≈ 93 GB
- **Goal:** **Three simultaneous objectives:**
  1. p99 latency ≤ **780 ms** (below the float32 baseline of ~789 ms)
  2. throughput ≥ **42 tok/s**
  3. **Minimize total system RAM** — every GB saved below 108 GB earns additional reward
- **Steps:** 5
- **Note:** This is the only task where RAM savings directly contribute to both the per-step reward (up to `+0.30`) and the final score (up to 30% of total). Combines `bfloat16` + low `max_num_batched_tokens` + `max_num_seqs=1` for the best RAM profile.

---

## Tunable Parameters

At each step the agent picks **one** parameter and sets it to a new value from the allowed list:

| Parameter                | Allowed Values                   | Effect                                                         |
| ------------------------ | -------------------------------- | -------------------------------------------------------------- |
| `dtype`                  | `float32`, `float16`, `bfloat16` | Weight precision. `bfloat16` is fastest on modern CPU.         |
| `max_model_len`          | `128`, `192`, `256`              | Max KV-cache context. Lower = less RAM, faster startup.        |
| `max_num_batched_tokens` | `64`, `128`, `256`, `512`        | Batch budget. Higher = better throughput, slight latency cost. |
| `max_num_seqs`           | `1`, `2`, `4`, `8`               | Parallel sequences. Higher = throughput boost, higher p99.     |

Parameters that affect engine architecture (`dtype`, `max_model_len`, `max_num_batched_tokens`, `max_num_seqs`) trigger a **full vLLM restart**. The environment handles process lifecycle automatically.

---

## How Rewards Are Decided

Each action returns a **per-step reward** used during training, and a **final score** (0–1) used for evaluation.

### Per-Step Reward

The reward is computed by `TaskGrader` in `server/graders.py` after the benchmark completes:

**If vLLM fails to start** (OOM or invalid config): `reward = -0.3`

**`easy_pythia_p99` (latency only):**

| Signal | Max |
|---|---|
| Proportional latency improvement vs. previous step | `+0.50` |
| Hitting latency target (≤ 920 ms) | `+0.30` |
| Beating 70% of target (≤ 644 ms) | `+0.20` |

**`medium_gpt2_p99_tput` (latency + throughput):**

| Signal | Max |
|---|---|
| Proportional latency improvement | `+0.35` |
| Throughput progress toward 19 tok/s target | `+0.35` |
| Both targets met simultaneously | `+0.20` |
| RAM below 108 GB total | `+0.10` |

**`hard_smollm2_stricter_p99_tput` (latency + throughput + RAM):**

| Signal | Max |
|---|---|
| Proportional latency improvement | `+0.35` |
| Throughput progress toward 13 tok/s target | `+0.30` |
| Both targets met simultaneously | `+0.25` |
| RAM below 109 GB total | `+0.10` |

**`extreme_pythia_p99_tput_ram_optimize` (latency + throughput + RAM savings — _new_):**

| Signal | Max |
|---|---|
| Latency improvement vs. previous step | `+0.25` |
| Throughput progress toward 42 tok/s target | `+0.25` |
| Both latency AND throughput targets met simultaneously | `+0.20` |
| RAM savings below 108 GB ceiling (continuous, scales with GB saved) | `+0.30` |

> The RAM savings window spans `108 GB − 93.18 GB ≈ 14.8 GB`. Keeping RAM at the system baseline earns the full `+0.30`; being at or above 108 GB earns `+0.00`.

All per-step rewards are capped at `1.0`.

### Final Score (Evaluation)

Computed from the **best configuration found across all steps**.

**Tasks `easy_pythia_p99`, `medium_gpt2_p99_tput`, `hard_smollm2_stricter_p99_tput`:**

| Outcome                                 | Score                                                   |
| --------------------------------------- | ------------------------------------------------------- |
| vLLM failed                             | `0.0`                                                   |
| Both latency and throughput targets met | `0.7 + 0.3 × (target / best_latency)`, capped at `1.0` |
| Latency target met only                 | `0.55`                                                  |
| Throughput target met only              | `0.35`                                                  |
| Neither target met                      | `(target_lat / best_p99) × 0.3` (partial credit)        |

**Task `extreme_pythia_p99_tput_ram_optimize` (three-objective scoring):**

| Component | Weight | Notes |
|---|---|---|
| Latency (p99 vs 780 ms target) | **40%** | Partial credit even below target; small bonus for beating it |
| Throughput (vs 42 tok/s target) | **30%** | Proportional: `tput / 42`, capped at 1.0 |
| RAM savings (below 108 GB ceiling) | **30%** | `(108 - ram_used) / 14.82 GB`, capped at 1.0 |

An **invalid action** (unknown parameter or out-of-range value) incurs a small penalty of `-0.05` and wastes a step.

---

## Project Structure

```
llm_serve_optimizer_env/
├── server/
│   ├── app.py                    # FastAPI app entry point (OpenEnv server)
│   ├── environment.py            # Core RL environment logic (reset, step, state)
│   ├── simulator.py              # vLLM process lifecycle + real benchmarking
│   ├── graders.py                # Task definitions, reward functions, final scoring
├── data/
│   ├── model_card.py             # Model registry & valid parameter values
│   └── baseline_cache.json       # Pre-computed baseline metrics for each task
├── tests/
│   ├── test_environment.py       # Unit tests for the RL environment logic
│   └── test_simulator.py         # Unit tests for the vLLM simulator
├── client.py                     # OpenEnv WebSocket client for the environment
├── models.py                     # Pydantic models: Action, Observation, State
├── inference.py                  # LLM agent script (mandatory submission entry point)
├── openenv.yaml                  # OpenEnv deployment config
├── pyproject.toml                # Python project metadata
├── Dockerfile                    # Docker image (HF Spaces compatible)
└── requirements.txt              # Python dependencies
```

---

## How the Benchmark Works

### Episode Start (`reset`)

Since baseline metrics for each task's default configuration are deterministic (same model, same initial params, same hardware), they are **pre-computed and saved** to `data/baseline_cache.json` to avoid redundant compute on every episode start. The `reset()` endpoint returns the cached baseline instantly, allowing the agent to begin tuning immediately.

The cache covers all four task IDs with their measured baselines (float32, default params):

| Task ID | Model | Baseline p99 | Baseline tput | Baseline RAM |
|---|---|---|---|---|
| `easy_pythia_p99` | pythia-70m-deduped | 1789 ms | 24.3 tok/s | 93.18 GB |
| `medium_gpt2_p99_tput` | gpt2 | 2017 ms | 17.2 tok/s | 92.06 GB |
| `hard_smollm2_stricter_p99_tput` | smollm2-135m | 2610 ms | 12.6 tok/s | 92.67 GB |
| `extreme_pythia_p99_tput_ram_optimize` | pythia-70m-deduped | 1789 ms | 24.3 tok/s | 93.18 GB |

> **Note:** RAM values reflect total system RAM in use (measured via `psutil.virtual_memory().used`) at benchmark time on the reference host, not model weights size.

### Each Step (`step`)

When the agent sends an action:

1. **Parameter validation** — illegal parameter or value gives `-0.05` reward.
2. **vLLM restart** — practically all parameters require a restart. The environment manages the subprocess, waits for the `/health` endpoint, and times out after 180 s.
3. **Warmup** — 2 warmup requests are sent to prime the server.
4. **Timed benchmark** — 5 requests are sent; p50/p99 latency and tokens/s throughput are measured from real HTTP round-trips.
5. **Reward computation** — scores are computed against task targets and returned to the agent.

---

## Running `inference.py` (Mandatory Submission Script)

`inference.py` is the agent script. It connects to the running environment server, queries an external LLM (acting as the RL policy), and steps through the environment.

### Prerequisites

The environment server must already be running (see Docker section below).

The script reads credentials from a `.env` file in the project root. Create one with the following variables:

```dotenv
# LLM API endpoint (used for the agent's reasoning)
API_BASE_URL= ... # https://api.groq.com/openai/v1

# Model to use for agent decisions
MODEL_NAME= ... # llama-3.3-70b-versatile

# API Key for the LLM provider (Groq API key in this case)
HF_TOKEN= ... # <groq_api_key_in_my_case>
```

> **Note:** The project uses the [Groq API](https://groq.com/) with the `llama-3.3-70b-versatile` model for agent inference. The `HF_TOKEN` must the the API Key for the LLM provider being used.

### Running Inference

```bash
# Install OpenEnv environment dependencies
pip install -r requirements.txt

# Run the agent against all four tasks
python inference.py
```

The script runs all four tasks in sequence (`easy_pythia_p99`, `medium_gpt2_p99_tput`, `hard_smollm2_stricter_p99_tput`, `extreme_pythia_p99_tput_ram_optimize`) and prints structured `[START]`, `[STEP]`, and `[END]` log lines for each — the mandatory format required by the OpenEnv submission spec — followed by a summary table.

---

## Docker — Building and Running Locally

The Dockerfile builds a CPU-only image based on `vllm/vllm-openai-cpu`. During the build, all three unique model checkpoints are cloned locally to reduce runtime latency (Pythia-70M, GPT-2, SmolLM2-135M — note that `extreme_pythia_p99_tput_ram_optimize` reuses the same Pythia-70M weights as `easy_pythia_p99`).

> ⚠️ The build step downloads ~700 MB of model weights. It may take several minutes.

### Build the image

```bash
docker build -f Dockerfile -t llm-serve-optimizer-env .
```

### Run the container

```bash
docker run --rm -d --name llm-serve-optimizer-env-container -p 7860:7860 llm-serve-optimizer-env
```

### Verify it's healthy

```bash
curl http://localhost:7860/health
```

### Run tests inside the container

```bash
docker exec llm-serve-optimizer-env-container pytest
```

### Stop the container

```bash
docker stop llm-serve-optimizer-env-container
```

---

## Environment Details

- **Framework:** [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — WebSocket-based RL environment server built on FastAPI
- **Inference backend:** [vLLM](https://github.com/vllm-project/vllm) — hardware agnostic; runs on CPU, GPU, and AI accelerators (e.g. Google TPU) without environment code changes
- **Reference deployment:** HF Spaces free tier (2 vCPU, 8 GB RAM, CPU-only)
- **Agent LLM:** Groq API — `llama-3.3-70b-versatile`
- **Port:** `7860` (default OpenEnv / HF Spaces port)
- **CI/CD:** GitHub Actions — runs pytest inside Docker on every push to `main`; deploys to HF Spaces only if all tests pass
