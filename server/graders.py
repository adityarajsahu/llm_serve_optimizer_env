from dataclasses import dataclass

@dataclass
class TaskConfig:
    task_id: str
    model: str
    hardware: str
    num_gpus: int
    description: str
    initial_params: dict
    target_latency_ms: float
    target_throughput: float
    max_steps: int
    difficulty: str

TASK_EASY = TaskConfig(
    task_id = "easy",
    model = "llama-3-8b",
    hardware = "A100-80GB",
    num_gpus = 1,
    description = "You are deploying Llama-3-8B on a single A100-80GB GPU. The default config gives ~45ms mean latency. Your goal: get mean latency below 20ms. You have 5 steps. Try changing quantization and tensor parallelism.",
    initial_params = {
        "tensor_parallel_size": 1,
        "quantization": "fp16",
        "gpu_memory_utilization": 0.85,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 256,
        "enable_chunked_prefill": False,
    },
    target_latency_ms = 20.0,
    target_throughput = 0.0,
    max_steps = 5,
    difficulty = "easy"
)

TASK_MEDIUM = TaskConfig(
    task_id = "medium",
    model = "llama-3-8b",
    hardware = "A100-40GB",
    num_gpus = 2,
    description = "You are deploying Llama-3-8B on 2x A100-40GB GPUs for a production API. You need BOTH: mean latency < 25ms AND throughput > 1500 tok/s. Memory is tight on 40GB — an OOM will cost you. You have 8 steps.",
    initial_params = {
        "tensor_parallel_size": 1,
        "quantization": "fp16",
        "gpu_memory_utilization": 0.85,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 256,
        "enable_chunked_prefill": False,
    },
    target_latency_ms = 25.0,
    target_throughput = 1500.0,
    max_steps = 8,
    difficulty = "medium"
)

TASK_HARD = TaskConfig(
    task_id = "hard",
    model = "llama-3-70b",
    hardware = "A100-80GB",
    num_gpus = 4,
    description = "You are deploying Llama-3-70B on 4x A100-80GB GPUs. Target: mean latency < 80ms AND throughput > 600 tok/s. fp16 alone won't fit — quantization and tensor parallelism choices are critical. OOM will cost you heavily. You have 10 steps.",
    initial_params = {
        "tensor_parallel_size": 2,
        "quantization": "fp16",
        "gpu_memory_utilization": 0.90,
        "max_num_batched_tokens": 512,
        "max_num_seqs": 128,
        "enable_chunked_prefill": False,
    },
    target_latency_ms = 80.0,
    target_throughput = 600.0,
    max_steps = 10,
    difficulty = "hard"
)

ALL_TASKS = {
    "easy": TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard": TASK_HARD
}

class TaskGrader:
    def grade(self, task: TaskConfig, metrics, previous_latency: float, step_number: int) -> float:
        if metrics.oom:
            return -0.5

        if task.task_id == "easy":
            return self._grade_easy(task, metrics, previous_latency)
        elif task.task_id == "medium":
            return self._grade_medium(task, metrics, previous_latency)
        elif task.task_id == "hard":
            return self._grade_hard(task, metrics, previous_latency)
        return 0.0

    def final_score(self, task: TaskConfig, metrics) -> float:
        if metrics.oom:
            return 0.0

        if task.task_id == "easy":
            if metrics.mean_e2el_ms <= task.target_latency_ms:
                return 1.0
            elif metrics.mean_e2el_ms <= task.target_latency_ms * 1.5:
                return 0.5
            else:
                return max(0.0, 1.0 - (metrics.mean_e2el_ms / 45.0))
        elif task.task_id == "medium":
            latency_ok = metrics.mean_e2el_ms <= task.target_latency_ms
            tput_ok = metrics.throughput_token_per_sec >= task.target_throughput
            
            if latency_ok and tput_ok:
                return 1.0
            elif latency_ok or tput_ok:
                return 0.5
            else:
                return 0.2
        elif task.task_id == "hard":
            latency_ok = metrics.mean_e2el_ms <= task.target_latency_ms
            tput_ok = metrics.throughput_token_per_sec >= task.target_throughput

            if latency_ok and tput_ok:
                return 1.0
            elif latency_ok:
                return 0.6
            elif metrics.mean_e2el_ms <= task.target_latency_ms * 1.25:
                return 0.3
            else:
                return 0.1
            
        return 0.0

    def _grade_easy(self, task: TaskConfig, metrics, prev_latency: float) -> float:
        reward = 0.0
        latency = metrics.mean_e2el_ms
        initial = 45.0

        if latency < prev_latency:
            step_improvement = (prev_latency - latency) / initial
            reward += min(0.40, step_improvement * 2.0)

        if latency <= task.target_latency_ms:
            reward += 0.30
        elif latency <= task.target_latency_ms * 1.5:
            reward += 0.10

        return min(round(reward, 3), 1.0)

    def _grade_medium(self, task: TaskConfig, metrics, prev_latency) -> float:
        reward = 0.0
        latency = metrics.mean_e2el_ms
        tput = metrics.throughput_token_per_sec
        initial_latency = 50.0

        if latency < prev_latency:
            reward += min(0.35, (prev_latency - latency) / initial_latency * 2.0)
        if latency <= task.target_latency_ms:
            reward += 0.10

        tput_ratio = min(tput / task.target_throughput, 1.0)
        reward += 0.35 * tput_ratio

        if latency <= task.target_latency_ms and tput >= task.target_throughput:
            reward += 0.20

        if metrics.gpu_memory_used_gb < 38.0:
            reward += 0.10

        return min(round(reward, 3), 1.0)

    def _grade_hard(self, task: TaskConfig, metrics, prev_latency: float) -> float:
        reward = 0.0
        latency = metrics.mean_e2el_ms
        tput = metrics.throughput_token_per_sec
        initial_latency = 165.0

        if latency < prev_latency:
            reward += min(0.40, (prev_latency - latency) / initial_latency * 2.5)
        if latency <= task.target_latency_ms:
            reward += 0.10

        tput_ratio = min(tput / task.target_throughput, 1.0)
        reward += 0.30 * tput_ratio

        if latency <= task.target_latency_ms and tput >= task.target_throughput:
            reward += 0.20

        if metrics.gpu_memory_used_gb < 60.0:
            reward += 0.10

        return min(round(reward, 3), 1.0)