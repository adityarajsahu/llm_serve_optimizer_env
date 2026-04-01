"""
inference.py — Mandatory submission inference script.

Environment variables required:
    API_BASE_URL   LLM API endpoint  (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier  (e.g. Qwen/Qwen2.5-1.5B-Instruct)
    HF_TOKEN       Hugging Face API key (also used by vLLM for Gemma-3)

    ENV_BASE_URL   OpenEnv server URL (default: http://localhost:7860)

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py
"""

import os
import re
import json
import sys
import textwrap
import time
from pydantic import BaseModel

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS_PER_TASK = 5
TEMPERATURE = 0.1
MAX_TOKENS = 100
TASKS = [
    "easy", 
    "medium", 
    "hard"
]

class ActionResponse(BaseModel):
    parameter: str
    value: str | int

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
    - float16 works on GPT-2 and SmolLM2 but CRASHES on Gemma-3 (do NOT use float16 with Gemma)
    - max_model_len controls KV cache size — THE biggest RAM consumer
      * max_model_len=256 → high RAM usage (risky for larger models)
      * max_model_len=192 → balanced
      * max_model_len=128 → lowest RAM, fastest startup, least context
    - For Gemma-3-270M: ALWAYS use bfloat16 and reduce max_model_len first to avoid OOM
    - Increasing max_num_batched_tokens helps throughput with small latency cost
    - max_num_seqs > 1 increases throughput but may raise per-request p99
    - A vLLM startup failure costs -0.3 reward — avoid OOM configs
    - Respond with ONLY the JSON object, nothing else.
""").strip()


def build_user_prompt(obs, step: int, history: list[str]) -> str:
    lines = [
        f"Step {step} — Task: {obs.task_id}",
        f"Model   : {obs.model} ({obs.model_hf_id})",
        f"Goal    : {obs.task_description}",
        "",
        "Current deployment config:",
        json.dumps(obs.current_params, indent = 4),
        "",
        "Real vLLM CPU measurements:",
        f"  p99 latency  : {obs.latency_p99_ms:.0f} ms   ← target ≤ {obs.target_latency_ms:.0f} ms",
        f"  p50 latency  : {obs.latency_p50_ms:.0f} ms",
        f"  throughput   : {obs.throughput_tok_per_sec:.1f} tok/s",
        f"  RAM used     : {obs.ram_used_gb:.2f} / {obs.ram_total_gb:.1f} GB",
    ]

    if obs.target_throughput > 0:
        lines.append(
            f"  tput target  : ≥ {obs.target_throughput:.0f} tok/s"
        )

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

    lines += [
        "",
        'Respond with exactly one JSON: {"parameter": "...", "value": ...}',
    ]
    return "\n".join(lines)


def run_task(task_id: str, llm_client: OpenAI) -> dict:
    from client import LLMServeEnv
    from models import ServeAction

    history: list[str] = []
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"TASK : {task_id}")
    print(f"{'='*60}")

    with LLMServeEnv(base_url = ENV_BASE_URL, message_timeout_s = 600.0).sync() as env:
        result = env.reset(task_id = task_id)
        obs = result.observation
        step = 0

        print(f"Model   : {obs.model_hf_id}")
        print(f"Baseline: p99={obs.latency_p99_ms:.0f}ms | "
              f"tput={obs.throughput_tok_per_sec:.1f}tok/s | "
              f"RAM={obs.ram_used_gb:.2f}GB")
        print(f"Target  : p99 ≤ {obs.target_latency_ms:.0f}ms", end="")
        if obs.target_throughput > 0:
            print(f" AND tput ≥ {obs.target_throughput:.0f}tok/s", end="")
        print(f"\n{'─'*60}")

        while not result.done and step < MAX_STEPS_PER_TASK:
            step += 1

            try:
                completion = llm_client.chat.completions.create(
                    model = MODEL_NAME,
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_prompt(obs, step, history)},
                    ],
                    temperature = TEMPERATURE,
                    max_tokens = MAX_TOKENS,
                    response_format={"type": "json_object"}
                )
                raw_content = completion.choices[0].message.content
                json_content = json.loads(raw_content)
                # match = re.search(r'\{.*\}', raw_content.replace('\n', ''), re.DOTALL)
                # if not match:
                #     print(f"  [Step {step}] Failed to find JSON in LLM output — skipping")
                #     continue
                # action_parsed = ActionResponse.model_validate_json(match.group(0))
            except Exception as exc:
                print(f"  [Step {step}] LLM error: {exc} — skipping")
                import traceback
                traceback.print_exc()
                continue

            # if not action_parsed or not action_parsed.parameter:
            #     print(f"  [Step {step}] Bad JSON output — skipping")
            #     continue

            action = ServeAction(
                parameter = json_content["parameter"],
                value = json_content["value"],
            )

            result = env.step(action)
            obs = result.observation

            icon = "💥" if obs.constraint_violated else ("✅" if obs.reward > 0.3 else "  ")
            print(
                f"  Step {step:2d} {icon}  "
                f"{action.parameter:<25} = {str(action.value):<10}  "
                f"p99={obs.latency_p99_ms:7.0f}ms  "
                f"tput={obs.throughput_tok_per_sec:7.1f}tok/s  "
                f"RAM={obs.ram_used_gb:.2f}GB  "
                f"reward={result.reward:+.3f}"
            )
            history.append(
                f"Step {step}: {action.parameter}={action.value} → "
                f"p99={obs.latency_p99_ms:.0f}ms "
                f"tput={obs.throughput_tok_per_sec:.1f}tok/s "
                f"RAM={obs.ram_used_gb:.2f}GB "
                f"reward={result.reward:+.3f}"
            )

        state = env.state()
        from server.graders import ALL_TASKS, TaskGrader
        final_score = TaskGrader().final_score(
            ALL_TASKS[task_id],
            type("M", (), {
                "failed":                 obs.constraint_violated,
                "latency_p99_ms":         state.best_latency_ms,
                "throughput_tok_per_sec": state.best_throughput,
                "ram_used_gb":            obs.ram_used_gb,
            })(),
        )

        elapsed = time.time() - t_start
        print(f"{'─'*60}")
        print(f"  Best p99     : {state.best_latency_ms:.0f} ms")
        print(f"  Best tput    : {state.best_throughput:.1f} tok/s")
        print(f"  Failed starts: {state.failed_starts}")
        print(f"  Final score  : {final_score:.3f} / 1.000")
        print(f"  Elapsed      : {elapsed:.1f}s")

        return {
            "task_id": task_id,
            "final_score": final_score,
            "best_latency_ms": state.best_latency_ms,
            "best_throughput": state.best_throughput,
            "total_reward": state.total_reward,
            "target_hit": state.target_hit,
            "failed_starts": state.failed_starts,
            "steps_used": state.step_count,
            "elapsed_s": round(elapsed, 1),
        }


def main() -> None:
    missing = [v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN") if not os.getenv(v)]
    if missing:
        print(f"[inference.py] WARNING: Missing env vars: {', '.join(missing)}")

    print(f"[inference.py] API_BASE_URL : {API_BASE_URL}")
    print(f"[inference.py] MODEL_NAME   : {MODEL_NAME}")
    print(f"[inference.py] ENV_BASE_URL : {ENV_BASE_URL}")
    print(f"[inference.py] HF_TOKEN     : {'set ✓' if API_KEY else 'NOT SET ✗'}")

    llm_client  = OpenAI(base_url = API_BASE_URL, api_key = API_KEY or "EMPTY")
    results = []
    t_total = time.time()

    for task_id in TASKS:
        results.append(run_task(task_id, llm_client))

    total_elapsed = time.time() - t_total

    print(f"\n{'═'*65}")
    print(f"{'Task':<30} | {'Score':>6} | {'Best p99':>10} | {'Steps':>5} | {'Time':>6}")
    print(f"{'─'*65}")
    for r in results:
        print(
            f"{r['task_id']:<30} | "
            f"{r['final_score']:>6.3f} | "
            f"{r['best_latency_ms']:>7.0f} ms | "
            f"{r['steps_used']:>5} | "
            f"{r['elapsed_s']:>5.1f}s"
        )
    print(f"{'─'*65}")
    avg = sum(r["final_score"] for r in results) / len(results)
    print(f"{'Average score':<30} | {avg:>6.3f}")
    print(f"{'Total elapsed':<30}   {total_elapsed:.1f}s")
    print(f"{'═'*65}\n")

    if avg == 0.0:
        print("[inference.py] ERROR: All scores 0.0 — check server connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()