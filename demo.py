"""
demo.py — Run the environment locally without any server.

Drives the environment object directly — no WebSocket needed.
Use this to verify the simulator and graders work before starting the server.

Usage
-----
    python demo.py                           # expert agent on all 3 tasks
    python demo.py --task easy               # single task
    python demo.py --random                  # random agent (shows contrast)

NOTE: This will actually start real vLLM processes and run benchmarks.
      Make sure vllm is installed: pip install vllm
"""

import sys
import os
import argparse
import random

sys.path.insert(0, os.path.dirname(__file__))

from server.environment import LLMServeEnvironment
from server.graders import ALL_TASKS
from models import ServeAction

EXPERT_STRATEGIES = {
    "easy": [
        ServeAction(parameter = "dtype", value = "float16"),
        ServeAction(parameter = "max_model_len", value = 128),
        ServeAction(parameter = "max_num_batched_tokens", value = 256),
        ServeAction(parameter = "dtype", value = "bfloat16"),
        ServeAction(parameter = "max_num_seqs", value = 2),
    ],
    "medium": [
        ServeAction(parameter = "dtype", value = "bfloat16"),
        ServeAction(parameter = "max_model_len", value = 192),
        ServeAction(parameter = "max_num_batched_tokens", value = 256),
        ServeAction(parameter = "max_num_seqs", value = 4),
        ServeAction(parameter = "max_model_len", value = 128),
    ],
    "hard": [
        ServeAction(parameter = "max_model_len", value = 128),
        ServeAction(parameter = "dtype", value = "bfloat16"),
        ServeAction(parameter = "max_num_batched_tokens", value = 128),
        ServeAction(parameter = "dtype", value = "float16"),
        ServeAction(parameter = "max_num_seqs", value = 2),
    ],
}

ALL_VALID_ACTIONS = [
    ServeAction(parameter = "dtype", value = "float32"),
    ServeAction(parameter = "dtype", value = "float16"),
    ServeAction(parameter = "dtype", value = "bfloat16"),
    ServeAction(parameter = "max_model_len", value = 128),
    ServeAction(parameter = "max_model_len", value = 192),
    ServeAction(parameter = "max_model_len", value = 256),
    ServeAction(parameter = "max_num_batched_tokens", value = 64),
    ServeAction(parameter = "max_num_batched_tokens", value = 128),
    ServeAction(parameter = "max_num_batched_tokens", value = 256),
    ServeAction(parameter = "max_num_seqs", value = 1),
    ServeAction(parameter = "max_num_seqs", value = 2),
    ServeAction(parameter = "max_num_seqs", value = 4),
]


def run_task(task_id: str, use_random: bool = False):
    env  = LLMServeEnvironment()
    obs  = env.reset(task_id=task_id)
    task = ALL_TASKS[task_id]

    print(f"\n{'═'*65}")
    print(f"  TASK   : {task_id.upper()}  ({'RANDOM' if use_random else 'EXPERT'} AGENT)")
    print(f"{'═'*65}")
    print(f"  Model  : {obs.model} ({obs.model_hf_id})")
    print(f"  Target : p99 ≤ {obs.target_latency_ms:.0f}ms", end="")
    if obs.target_throughput > 0:
        print(f"  AND ≥ {obs.target_throughput:.0f} tok/s", end="")
    print(f"\n  Base   : p99={obs.latency_p99_ms:.0f}ms  "
          f"tput={obs.throughput_tok_per_sec:.1f}tok/s  "
          f"RAM={obs.ram_used_gb:.2f}GB")
    print(f"{'─'*65}")

    actions = EXPERT_STRATEGIES.get(task_id, []) if not use_random else []
    step    = 0

    while not obs.done:
        if use_random:
            action = random.choice(ALL_VALID_ACTIONS)
        elif step < len(actions):
            action = actions[step]
        else:
            action = random.choice(ALL_VALID_ACTIONS)

        obs  = env.step(action)
        step += 1

        if obs.constraint_violated:
            icon = "💥 FAIL"
        elif obs.reward > 0.3:
            icon = "🟢"
        elif obs.reward > 0:
            icon = "🟡"
        elif obs.reward < 0:
            icon = "🔴"
        else:
            icon = "⬜"

        print(
            f"  Step {step:2d} {icon}  "
            f"{action.parameter:<25} = {str(action.value):<10}  "
            f"p99={obs.latency_p99_ms:7.0f}ms  "
            f"RAM={obs.ram_used_gb:.2f}GB  "
            f"reward={obs.reward:+.3f}"
        )

    # Final summary
    state = env.state
    print(f"{'─'*65}")
    print(f"  Best p99      : {state.best_latency_ms:.0f}ms  "
          f"(target ≤{task.target_latency_ms:.0f}ms)  "
          f"{'✅ HIT' if state.target_hit else '❌ MISSED'}")
    if task.target_throughput > 0:
        tput_ok = state.best_throughput >= task.target_throughput
        print(f"  Best tput     : {state.best_throughput:.1f} tok/s  "
              f"(target ≥{task.target_throughput:.0f})  "
              f"{'✅' if tput_ok else '❌'}")
    print(f"  Failed starts : {state.failed_starts}")
    print(f"  Total reward  : {state.total_reward:.3f}")

    from server.graders import TaskGrader
    final = TaskGrader().final_score(
        task,
        type("M", (), {
            "failed":                 obs.constraint_violated,
            "latency_p99_ms":         state.best_latency_ms,
            "throughput_tok_per_sec": state.best_throughput,
            "ram_used_gb":            obs.ram_used_gb,
        })(),
    )
    print(f"  Final score   : {final:.3f} / 1.000")

    # Stop vLLM process cleanly
    env._simulator.stop()
    return final


def main():
    parser = argparse.ArgumentParser(description="LLM Deploy Env — Local Demo")
    parser.add_argument(
        "--task",
        choices=list(ALL_TASKS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    tasks  = list(ALL_TASKS.keys()) if args.task == "all" else [args.task]
    scores = [run_task(t, use_random=args.random) for t in tasks]

    if len(scores) > 1:
        print(f"\n{'═'*65}")
        print(f"  AVERAGE SCORE: {sum(scores)/len(scores):.3f} / 1.000")
        print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()