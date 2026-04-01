---
title: LLM Serve Optimizer Env
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# LLM Serve Optimizer Environment

An **Reinforcement Learning environment** for optimizing LLM deployment configurations to meet target latency and throughput requirements. Built using [OpenEnv](https://huggingface.co/openenv), the environment runs a real [vLLM](https://github.com/vllm-project/vllm) inference server on CPU and benchmarks it after each configuration change — providing an RL agent with real-world performance feedback.

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

Due to CPU-only hardware constraints (HF Spaces: 2 vCPU, ~8 GB RAM), only very small models are supported. All three models are publicly available on Hugging Face and require no HF token:

| Model                     | HF ID                                 | Parameters | Task     |
| ------------------------- | ------------------------------------- | ---------- | -------- |
| **Pythia-70M-Deduped**    | `EleutherAI/pythia-70m-deduped`       | 70M        | `easy`   |
| **GPT-2 Small**           | `openai-community/gpt2`               | 124M       | `medium` |
| **SmolLM2-135M-Instruct** | `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M       | `hard`   |

---

## Tasks

The environment defines three tasks of increasing difficulty. Each task specifies which model to deploy, performance targets, and a step budget.

### Easy — Pythia-70M-Deduped

- **Goal:** Get p99 latency ≤ **150 ms**
- **Throughput target:** None
- **Steps:** 5

### Medium — GPT-2 (124M)

- **Goal:** Get p99 latency ≤ **300 ms** AND throughput ≥ **40 tok/s**
- **Steps:** 5

### Hard — SmolLM2-135M-Instruct

- **Goal:** Get p99 latency ≤ **450 ms** AND throughput ≥ **70 tok/s**
- **Steps:** 5
- **Note:** RAM pressure is a real concern here — the agent must also keep memory usage in check.

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

**Easy task (latency only):**

- Up to `+0.5` for proportional latency improvement over the previous step
- `+0.3` bonus for hitting the latency target (≤ 150 ms)
- `+0.2` additional bonus for beating 70% of the target (≤ 105 ms)

**Medium task (latency + throughput):**

- Up to `+0.35` for latency improvement
- Up to `+0.35` proportional to how close throughput is to the 40 tok/s target
- `+0.2` bonus when **both** targets are simultaneously met
- `+0.1` bonus if RAM usage stays below 3 GB

**Hard task (latency + throughput + RAM):**

- Up to `+0.35` for latency improvement
- Up to `+0.30` proportional to throughput progress toward 70 tok/s
- `+0.25` bonus when **both** targets are simultaneously met
- `+0.10` bonus if RAM stays below 4 GB

All per-step rewards are capped at `1.0`.

### Final Score (Evaluation)

Computed from the **best configuration found across all steps**:

| Outcome                                 | Score                                                  |
| --------------------------------------- | ------------------------------------------------------ |
| vLLM failed                             | `0.0`                                                  |
| Both latency and throughput targets met | `0.7 + 0.3 × (target / best_latency)`, capped at `1.0` |
| Latency target met only                 | `0.55`                                                 |
| Throughput target met only              | `0.35`                                                 |
| Neither target met                      | `progress × 0.3` (partial credit)                      |

An **invalid action** (unknown parameter or out-of-range value) incurs a small penalty of `-0.05` and wastes a step.

---

## Project Structure

```
llm_serve_optimizer_env/
├── server/
│   ├── app.py            # FastAPI app entry point (OpenEnv server)
│   ├── environment.py    # Core RL environment logic (reset, step, state)
│   ├── simulator.py      # vLLM process lifecycle + real benchmarking
│   └── graders.py        # Task definitions, reward functions, final scoring
├── data/
│   └── model_card.py     # Model registry & valid parameter values
├── client.py             # OpenEnv WebSocket client for the environment
├── models.py             # Pydantic models: Action, Observation, State
├── inference.py          # LLM agent script (mandatory submission entry point)
├── demo.py               # Demo/debug script
├── openenv.yaml          # OpenEnv deployment config
├── Dockerfile            # CPU-only Docker image (HF Spaces compatible)
└── requirements.txt      # Python dependencies
```

---

## How the Benchmark Works

When the agent sends an action:

1. **Parameter validation** — illegal parameter or value gives `-0.05` reward.
2. **vLLM restart** (if needed) — practically all parameters require a restart. The environment manages the subprocess, waits for the `/health` endpoint, and times out after 180 s.
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
API_BASE_URL=https://api.groq.com/openai/v1

# Model to use for agent decisions
MODEL_NAME=llama-3.3-70b-versatile

# API Key for the LLM provider (Groq API key in this case)
HF_TOKEN=<your_groq_api_key_here>
```

> **Note:** The project uses the [Groq API](https://console.groq.com/) with the `llama-3.3-70b-versatile` model for agent inference. The `HF_TOKEN` variable name is kept for compatibility with the submission spec, but its value should be your Groq API key.

### Running

```bash
# Install dependencies (if not in Docker)
pip install -r requirements.txt

# Run the agent against the 'easy' task
python inference.py
```

The script will print a step-by-step table of actions, metrics, and rewards, and a summary at the end.

By default, only the `easy` task is enabled. To run `medium` or `hard`, uncomment the relevant lines in `inference.py`:

```python
TASKS = [
    "easy",
    "medium",
    "hard"
]
```

---

## Docker — Building and Running Locally

The Dockerfile builds a CPU-only image based on `vllm/vllm-openai-cpu`. During the build, all three supported models are cloned locally so no internet access is needed at runtime.

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

- **Framework:** [OpenEnv](https://huggingface.co/openenv) — WebSocket-based RL environment server built on FastAPI
- **Inference backend:** [vLLM](https://github.com/vllm-project/vllm) running in CPU-only mode
- **Hardware target:** HF Spaces free tier (2 vCPU, 8 GB RAM)
- **Agent LLM:** Groq API — `llama-3.3-70b-versatile`
- **Port:** `7860` (default OpenEnv / HF Spaces port)
