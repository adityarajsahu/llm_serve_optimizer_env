from typing import Any
from openenv.core.env_server import Action, Observation, State

class ServeAction(Action):
    parameter: str
    value: Any

class ServeObservation(Observation):
    model: str
    model_hf_id: str
    hardware: str
    current_params: dict
    latency_p50_ms: float
    latency_p99_ms: float
    throughput_tok_per_sec: float
    ram_used_gb: float
    ram_total_gb: float
    task_id: str
    task_description: str
    target_latency_ms: float
    target_throughput: float
    steps_remaining: int
    legal_parameters: list
    reward: float = 0.0
    done: bool = False
    last_action_feedback: str = ""
    constraint_violated: bool = False

class ServeState(State):
    task_id: str = ""
    best_latency_ms: float = float('inf')
    initial_latency_ms: float = float('inf')
    best_throughput: float = 0.0
    total_reward: float = 0.0
    target_hit: bool = False
    failed_starts: int = 0