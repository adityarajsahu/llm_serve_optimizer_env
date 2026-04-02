"""
inference.py — Mandatory submission inference script.

MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory.
- Participants must use OpenAI Client for all LLM calls using above variables.

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=easy env=llm-serve-optimizer model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action=dtype=bfloat16 reward=0.10 done=false error=null
    [STEP] step=2 action=max_model_len=128 reward=0.30 done=false error=null
    [END] success=true steps=2 rewards=0.10,0.30
"""

import json
import os
import sys
import textwrap
import time
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "http://localhost:7860",
)

BENCHMARK = "llm-serve-optimizer"
TASKS: List[str] = ["easy", "medium", "hard"]
MAX_STEPS = 5
TEMPERATURE = 0.1
MAX_TOKENS = 100
SUCCESS_THRESHOLD = 0.1

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert MLOps engineer optimizing LLM deployments using vLLM on CPU.
    Your goal is to minimize inference latency while meeting throughput targets.

    Each step choose ONE parameter to change.
    Respond ONLY with a JSON object — no explanation, no markdown:
    {"parameter": "<param_name>", "value": <value>}

    Valid parameters and their allowed values:
    dtype                  : "float32", "float16", "bfloat16"
    max_model_len          : 128, 192, 256
    max_num_batched_tokens : 64, 128, 256, 512
    max_num_seqs           : 1, 2, 4, 8

    Key knowledge for CPU inference:
    - bfloat16 is the BEST dtype for CPU — faster than float32, works on ALL models
    - float16 works well on Pythia and GPT-2
    - max_model_len controls KV cache size — THE biggest RAM consumer
      * max_model_len=256 → high RAM usage (risky for larger models)
      * max_model_len=192 → balanced
      * max_model_len=128 → lowest RAM, fastest startup, least context
    - For SmolLM2-135M: bfloat16 is often best and reduce max_model_len to save RAM
    - Increasing max_num_batched_tokens helps throughput with small latency cost
    - max_num_seqs > 1 increases throughput but may raise per-request p99
    - A vLLM startup failure costs -0.3 reward — avoid OOM configs
    - Respond with ONLY the JSON object, nothing else.
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )

def build_user_prompt(obs, step: int, history: List[str]) -> str:
    lines = [
        f"Step {step} — Task: {obs.task_id}",
        f"Model   : {obs.model} ({obs.model_hf_id})",
        f"Goal    : {obs.task_description}",
        "",
        "Current deployment config:",
        json.dumps(obs.current_params, indent=4),
        "",
        "Real vLLM CPU measurements:",
        f"  p99 latency  : {obs.latency_p99_ms:.0f} ms   ← target ≤ {obs.target_latency_ms:.0f} ms",
        f"  p50 latency  : {obs.latency_p50_ms:.0f} ms",
        f"  throughput   : {obs.throughput_tok_per_sec:.1f} tok/s",
        f"  RAM used     : {obs.ram_used_gb:.2f} / {obs.ram_total_gb:.1f} GB",
    ]

    if obs.target_throughput > 0:
        lines.append(f"  tput target  : ≥ {obs.target_throughput:.0f} tok/s")

    lines += [
        "",
        f"Steps remaining : {obs.steps_remaining}",
        f"Last feedback   : {obs.last_action_feedback}",
    ]

    if obs.constraint_violated:
        lines.append(
            "⚠️  vLLM FAILED TO START — likely OOM or bad config. "
            "Reduce max_model_len to 128 or change dtype."
        )

    if history:
        lines += ["", "Action history:"]
        lines += [f"  {h}" for h in history[-4:]]

    lines += ["", 'Respond with exactly one JSON: {"parameter": "...", "value": ...}']
    return "\n".join(lines)

def run_task(task_id: str, llm_client: OpenAI) -> dict:
    from client import LLMServeEnv
    from models import ServeAction

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        with LLMServeEnv(
            base_url=ENV_BASE_URL, message_timeout_s=600.0
        ).sync() as env:

            result = env.reset(task_id=task_id)
            obs    = result.observation
            step   = 0

            while not result.done and step < MAX_STEPS:
                step += 1
                error_msg: Optional[str] = None

                try:
                    completion = llm_client.chat.completions.create(
                        model = MODEL_NAME,
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": build_user_prompt(obs, step, history)},
                        ],
                        temperature = TEMPERATURE,
                        max_tokens = MAX_TOKENS,
                        response_format = {"type": "json_object"},
                    )
                    raw_content = completion.choices[0].message.content
                    json_content = json.loads(raw_content)
                    param = json_content["parameter"]
                    value = json_content["value"]
                    action = ServeAction(parameter=param, value=value)
                except Exception as exc:
                    error_msg = str(exc)
                    print(f"[DEBUG] LLM error at step {step}: {exc}", flush=True, file=sys.stderr)
                    log_step(step=step, action="null", reward=0.0, done=False, error=error_msg)
                    rewards.append(0.0)
                    steps_taken = step
                    continue

                result = env.step(action)
                obs = result.observation

                reward = result.reward or 0.0
                done = result.done
                if obs.constraint_violated:
                    error_msg = obs.last_action_feedback or "constraint_violated"

                rewards.append(reward)
                steps_taken = step

                action_str = f"{action.parameter}:{action.value}"
                history.append(
                    f"Step {step}: {action_str} → "
                    f"p99={obs.latency_p99_ms:.0f}ms "
                    f"tput={obs.throughput_tok_per_sec:.1f}tok/s "
                    f"RAM={obs.ram_used_gb:.2f}GB "
                    f"reward={reward:+.3f}"
                )

                log_step(step=step, action=action_str, reward=reward,
                         done=done, error=error_msg)

                if done:
                    break

            state = env.state()
            from server.graders import ALL_TASKS, TaskGrader
            final_score = TaskGrader().final_score(
                ALL_TASKS[task_id],
                type("M", (), {
                    "failed": obs.constraint_violated,
                    "latency_p99_ms": state.best_latency_ms,
                    "throughput_tok_per_sec": state.best_throughput,
                    "ram_used_gb": obs.ram_used_gb,
                })(),
            )
            success = final_score >= SUCCESS_THRESHOLD

    except Exception as exc:
        # print(f"[DEBUG] Task '{task_id}' aborted: {exc}", flush=True, file=sys.stderr)
        final_score = 0.0

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return {
        "task_id": task_id,
        "final_score": final_score,
        "steps_used": steps_taken,
        "rewards": rewards,
    }

def main() -> None:
    missing = [v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN") if not os.getenv(v)]
    # if missing:
    #     print(f"[inference.py] WARNING: Missing env vars: {', '.join(missing)}", flush=True)

    # print(f"[inference.py] API_BASE_URL : {API_BASE_URL}",                       flush=True)
    # print(f"[inference.py] MODEL_NAME   : {MODEL_NAME}",                         flush=True)
    # print(f"[inference.py] ENV_BASE_URL : {ENV_BASE_URL}",                       flush=True)
    # print(f"[inference.py] HF_TOKEN     : {'set ✓' if API_KEY else 'NOT SET ✗'}", flush=True)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "EMPTY")
    results    = []
    t_total    = time.time()

    for task_id in TASKS:
        results.append(run_task(task_id, llm_client))

    total_elapsed = time.time() - t_total

    # # ── Summary table (informational, not part of the mandatory format) ────────
    # print(f"\n{'═'*70}", flush=True)
    # print(f"{'Task':<20} | {'Score':>6} | {'Steps':>5} | {'Rewards'}", flush=True)
    # print(f"{'─'*70}", flush=True)
    # for r in results:
    #     rewards_str = ",".join(f"{x:.2f}" for x in r["rewards"])
    #     print(
    #         f"{r['task_id']:<20} | "
    #         f"{r['final_score']:>6.3f} | "
    #         f"{r['steps_used']:>5} | "
    #         f"{rewards_str}",
    #         flush=True,
    #     )
    # print(f"{'─'*70}", flush=True)
    # avg = sum(r["final_score"] for r in results) / len(results) if results else 0.0
    # print(f"{'Average score':<20} | {avg:>6.3f}", flush=True)
    # print(f"Total elapsed: {total_elapsed:.1f}s", flush=True)
    # print(f"{'═'*70}\n",  flush=True)

    # if avg == 0.0:
    #     print("[inference.py] ERROR: All scores 0.0 — check server connection.", flush=True)
    #     sys.exit(1)


if __name__ == "__main__":
    main()