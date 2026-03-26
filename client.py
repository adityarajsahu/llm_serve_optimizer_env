from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import ServeAction, ServeObservation, ServeState

class LLMServeEnv(EnvClient[ServeAction, ServeObservation, ServeState]):
    def _step_payload(self, action: ServeAction) -> dict:
        return {
            "parameter": action.parameter,
            "value": action.value
        }

    def _reset_payload(self, **kwargs) -> dict:
        return kwargs

    def _parse_result(self, payload: dict) -> StepResult[ServeObservation]:
        obs_data = payload.get("observation", {})
        obs = ServeObservation(
            model = obs_data.get("model", ""),
            hardware = obs_data.get("hardware", ""),
            num_gpus_available = obs_data.get("num_gpus_available", 1),
            current_params = obs_data.get("current_params", {}),
            mean_e2el_ms = obs_data.get("mean_e2el_ms", 0.0),
            throughput_token_per_sec = obs_data.get("throughput_token_per_sec", 0.0),
            gpu_memory_used_gb = obs_data.get("gpu_memory_used_gb", 0.0),
            gpu_memory_total_gb = obs_data.get("gpu_memory_total_gb", 80.0),
            task_id = obs_data.get("task_id", ""),
            task_description = obs_data.get("task_description", ""),
            target_e2el_ms = obs_data.get("target_e2el_ms", 0.0),
            target_throughput = obs_data.get("target_throughput", 0.0),
            steps_remaining = obs_data.get("steps_remaining", 0),
            legal_parameters = obs_data.get("legal_parameters", []),
            reward = obs_data.get("reward", 0.0),
            done = obs_data.get("done", False),
            last_action_feedback = obs_data.get("last_action_feedback", ""),
            constraint_violated = obs_data.get("constraint_violated", False),
        )
        return StepResult(
            observation = obs,
            reward = payload.get("reward", 0.0),
            done = payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> ServeState:
        return ServeState(
            episode_id = payload.get("episode_id", ""),
            step_count = payload.get("step_count", 0),
            task_id = payload.get("task_id", ""),
            best_e2el_ms = payload.get("best_e2el_ms", float('inf')),
            initial_e2el_ms = payload.get("initial_e2el_ms", float('inf')),
            best_throughput = payload.get("best_throughput", 0.0),
            total_reward = payload.get("total_reward", 0.0),
            target_hit = payload.get("target_hit", False),
            oom_count = payload.get("oom_count", 0),
        )