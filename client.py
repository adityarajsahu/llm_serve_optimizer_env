from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import ServeAction, ServeObservation, ServeState

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
            model = obs_data.get("model") or "",
            model_hf_id = obs_data.get("model_hf_id") or "",
            hardware = obs_data.get("hardware") or "",
            current_params = obs_data.get("current_params") or {},
            latency_p50_ms = obs_data.get("latency_p50_ms") or 0.0,
            latency_p99_ms = obs_data.get("latency_p99_ms") or 0.0,
            throughput_tok_per_sec = obs_data.get("throughput_tok_per_sec") or 0.0,
            ram_used_gb = obs_data.get("ram_used_gb") or 0.0,
            ram_total_gb = obs_data.get("ram_total_gb") or 8.0,
            task_id = obs_data.get("task_id") or "",
            task_description = obs_data.get("task_description") or "",
            target_latency_ms = obs_data.get("target_latency_ms") or 0.0,
            target_throughput = obs_data.get("target_throughput") or 0.0,
            steps_remaining = obs_data.get("steps_remaining") or 0,
            legal_parameters = obs_data.get("legal_parameters") or [],
            reward = obs_data.get("reward") or 0.0,
            done = obs_data.get("done") or False,
            last_action_feedback = obs_data.get("last_action_feedback") or "",
            constraint_violated = obs_data.get("constraint_violated") or False
        )

        return StepResult(
            observation = obs,
            reward = payload.get("reward", 0.0),
            done = payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> ServeState:
        return ServeState(
            episode_id = payload.get("episode_id") or "",
            step_count = payload.get("step_count") or 0,
            task_id = payload.get("task_id") or "",
            best_latency_ms = payload.get("best_latency_ms") if payload.get("best_latency_ms") is not None else float('inf'),
            initial_latency_ms = payload.get("initial_latency_ms") if payload.get("initial_latency_ms") is not None else float('inf'),
            best_throughput = payload.get("best_throughput") or 0.0,
            total_reward = payload.get("total_reward") or 0.0,
            target_hit = payload.get("target_hit") or False,
            failed_starts = payload.get("failed_starts") or 0
        )