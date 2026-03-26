from typing import Any, Optional
from openenv.core.env_server import Action, Observation, State

class ServeAction(Action):
    parameter: str
    value: Any

class ServeObservation(Observation):
    model: str
    hardware: str
    num_gpus_available: int
    current_params: dict
    mean_e2el_ms: float
    throughput_token_per_sec: float
    gpu_memory_used_gb: float # VRAM actually utilized
    gpu_memory_total_gb: float # total GPU VRAM
    task_id: str
    task_description: str
    target_e2el_ms: float
    target_throughput: float
    steps_remaining: int
    legal_parameters: list
    reward: float = 0.0
    done: bool = False
    last_action_feedback: str = ""
    constraint_violated: bool = False

class ServeState(State):
    task_id: str = ""
    best_e2el_ms: float = float('inf')
    initial_e2el_ms: float = float('inf')
    best_throughput: float = 0.0
    total_reward: float = 0.0
    target_hit: bool = False
    oom_count: int = 0