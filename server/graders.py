from dataclasses import dataclass

@dataclass
class TaskConfig:
    task_id: str
    model_key: str
    description: str
    initial_params: dict
    target_latency_ms: float
    target_throughput: float
    max_steps: int
    difficulty: str
    baseline_ram_gb: float = 0.0
    target_ram_gb: float = 0.0

TASK_EASY = TaskConfig(
    task_id = "easy_pythia_p99",
    model_key = "pythia-70m-deduped",
    description = "Deploy Pythia-70M on CPU. Default float32 is slower than bfloat16 on modern CPUs. Tune dtype (try bfloat16) to reduce p99 latency below 920ms. Pythia is tiny so RAM is not a concern — focus on speed. You have 5 steps.",
    initial_params = {
        "dtype": "float32",
        "max_model_len": 128,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 1,
    },
    target_latency_ms = 920.0,
    target_throughput = 0.0,
    max_steps = 5,
    difficulty = "easy_pythia_p99"
)

TASK_MEDIUM = TaskConfig(
    task_id = "medium_gpt2_p99_tput",
    model_key = "gpt2",
    description = "Deploy GPT-2 (124M) on CPU for a real-time API. Hit BOTH: p99 latency ≤ 1750ms AND throughput ≥ 19 tok/s. bfloat16 is faster than float32 on modern CPUs. Tuning max_num_batched_tokens helps throughput. You have 5 steps.",
    initial_params = {
        "dtype": "float32",
        "max_model_len": 128,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 1,
    },
    target_latency_ms = 1750.0,
    target_throughput = 19.0,
    max_steps = 5,
    difficulty = "medium_gpt2_p99_tput"
)

TASK_HARD = TaskConfig(
    task_id = "hard_smollm2_stricter_p99_tput",
    model_key = "smollm2-135m",
    description = "Deploy SmolLM2-135M on CPU. WARNING: RAM usage might be something to optimize as well, try keeping memory low. Target: p99 ≤ 2650ms AND throughput ≥ 13 tok/s. Try bfloat16, reduce max_model_len, and tune max_num_batched_tokens. You have 5 steps.",
    initial_params = {
        "dtype": "float32",
        "max_model_len": 128,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 1,
    },
    target_latency_ms = 2650.0,
    target_throughput = 13.0,
    max_steps = 5,
    difficulty = "hard_smollm2_stricter_p99_tput"
)

TASK_EXTREME = TaskConfig(
    task_id = "extreme_pythia_p99_tput_ram_optimize",
    model_key = "pythia-70m-deduped",
    description = "Deploy Pythia-70M on CPU with THREE simultaneous goals: (1) p99 latency ≤ 780 ms — below the float32 baseline; (2) throughput ≥ 42 tok/s — keep the model fast; (3) minimize RAM — every GB saved below 108 GB (total system) earns extra reward. bfloat16 helps latency and throughput. Lower max_num_batched_tokens and max_num_seqs reduce RAM and KV-cache pressure. max_model_len=128 is already the minimum. You have 5 steps.",
    initial_params = {
        "dtype": "float32",
        "max_model_len": 128,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 1,
    },
    target_latency_ms = 780.0,
    target_throughput = 42.0,
    max_steps = 5,
    difficulty = "extreme_pythia_p99_tput_ram_optimize",
    baseline_ram_gb = 93.18,
    target_ram_gb = 108.0,
)

ALL_TASKS = {
    "easy_pythia_p99": TASK_EASY,
    "medium_gpt2_p99_tput": TASK_MEDIUM,
    "hard_smollm2_stricter_p99_tput": TASK_HARD,
    "extreme_pythia_p99_tput_ram_optimize": TASK_EXTREME,
}

class TaskGrader:
    def grade(self, task: TaskConfig, metrics, previous_latency: float, step_number: int) -> float:
        if metrics.failed:
            return -0.3

        if task.task_id == "easy_pythia_p99":
            return self._grade_easy(task, metrics, previous_latency)
        elif task.task_id == "medium_gpt2_p99_tput":
            return self._grade_medium(task, metrics, previous_latency)
        elif task.task_id == "hard_smollm2_stricter_p99_tput":
            return self._grade_hard(task, metrics, previous_latency)
        elif task.task_id == "extreme_pythia_p99_tput_ram_optimize":
            return self._grade_extreme(task, metrics, previous_latency)
        return 0.0

    def final_score(self, task: TaskConfig, metrics) -> float:
        if metrics.failed:
            return 0.0

        lat_ok = metrics.latency_p99_ms <= task.target_latency_ms
        tput_ok = (
            task.target_throughput == 0.0
            or metrics.throughput_tok_per_sec >= task.target_throughput
        )

        if task.task_id == "extreme_pythia_p99_tput_ram_optimize":
            return self._final_score_extreme(task, metrics, lat_ok, tput_ok)

        if lat_ok and tput_ok:
            ratio = task.target_latency_ms / max(metrics.latency_p99_ms, 1.0)
            return min(round(0.7 + 0.3 * (ratio - 1.0), 3), 1.0)
        elif lat_ok:
            return 0.55
        elif tput_ok:
            return 0.35
        else:
            progress = min(
                task.target_latency_ms / max(metrics.latency_p99_ms, 1.0), 1.0
            )
            return round(progress * 0.3, 3)

    def _final_score_extreme(self, task: TaskConfig, metrics, lat_ok: bool, tput_ok: bool) -> float:
        if lat_ok:
            lat_ratio = task.target_latency_ms / max(metrics.latency_p99_ms, 1.0)
            lat_score = min(0.40 + 0.10 * (lat_ratio - 1.0), 0.50)
        else:
            lat_progress = min(task.target_latency_ms / max(metrics.latency_p99_ms, 1.0), 1.0)
            lat_score = round(lat_progress * 0.40, 3)

        tput_ratio = min(metrics.throughput_tok_per_sec / max(task.target_throughput, 1.0), 1.0)
        tput_score = round(tput_ratio * 0.30, 3)

        savings_window = max(task.target_ram_gb - task.baseline_ram_gb, 1.0)
        ram_savings = max(task.target_ram_gb - metrics.ram_used_gb, 0.0)
        ram_ratio = min(ram_savings / savings_window, 1.0)
        ram_score = round(ram_ratio * 0.30, 3)

        total = round(lat_score + tput_score + ram_score, 3)
        return min(total, 1.0)

    def _grade_easy(self, task: TaskConfig, metrics, prev_latency: float) -> float:
        reward = 0.0
        latency = metrics.latency_p99_ms

        if latency < prev_latency and prev_latency > 0:
            improvement = (prev_latency - latency) / prev_latency
            reward += min(0.5, improvement * 2.0)

        if latency <= task.target_latency_ms:
            reward += 0.3

        if latency <= task.target_latency_ms * 0.7:
            reward += 0.2

        return min(round(reward, 3), 1.0)

    def _grade_medium(self, task: TaskConfig, metrics, prev_latency) -> float:
        reward = 0.0
        latency = metrics.latency_p99_ms
        tput = metrics.throughput_tok_per_sec

        if latency < prev_latency and prev_latency > 0:
            reward += min(0.35, (prev_latency - latency) / prev_latency * 1.5)
        
        tput_ratio = min(tput / task.target_throughput, 1.0)
        reward += 0.35 * tput_ratio

        if latency <= task.target_latency_ms and tput >= task.target_throughput:
            reward += 0.2

        if metrics.ram_used_gb < 108.0:
            reward += 0.1

        return min(round(reward, 3), 1.0)

    def _grade_hard(self, task: TaskConfig, metrics, prev_latency: float) -> float:
        reward = 0.0
        latency = metrics.latency_p99_ms
        tput = metrics.throughput_tok_per_sec

        if latency < prev_latency and prev_latency > 0:
            reward += min(0.35, (prev_latency - latency) / prev_latency * 2.0)

        tput_ratio = min(tput / task.target_throughput, 1.0)
        reward += 0.30 * tput_ratio

        if latency <= task.target_latency_ms and tput >= task.target_throughput:
            reward += 0.25

        if metrics.ram_used_gb < 109.0:
            reward += 0.10

        return min(round(reward, 3), 1.0)

    def _grade_extreme(self, task: TaskConfig, metrics, prev_latency: float) -> float:
        reward = 0.0
        latency = metrics.latency_p99_ms
        tput    = metrics.throughput_tok_per_sec
        if latency < prev_latency and prev_latency > 0:
            reward += min(0.25, (prev_latency - latency) / prev_latency * 2.5)

        tput_ratio = min(tput / max(task.target_throughput, 1.0), 1.0)
        reward += 0.25 * tput_ratio

        if latency <= task.target_latency_ms and tput >= task.target_throughput:
            reward += 0.20

        savings_window = max(task.target_ram_gb - task.baseline_ram_gb, 1.0)
        ram_savings = max(task.target_ram_gb - metrics.ram_used_gb, 0.0)
        ram_ratio = min(ram_savings / savings_window, 1.0)
        reward += 0.30 * ram_ratio

        return min(round(reward, 3), 1.0)