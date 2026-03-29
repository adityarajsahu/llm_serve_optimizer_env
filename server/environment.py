import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from uuid import uuid4
from openenv.core.env_server.interfaces import Environment

from models import ServeAction, ServeObservation, ServeState
from server.simulator import LatencySimulator
from server.graders import TaskGrader, ALL_TASKS
from data.model_card import VALID_PARAM_VALUES, MODEL_REGISTRY

class LLMServeEnvironment(Environment):
    def __init__(self):
        self._simulator = LatencySimulator()
        self._grader = TaskGrader()
        self._task = None
        self._current_params = {}
        self._current_metrics = None
        self._serve_state = ServeState(episode_id = str(uuid4()), step_count = 0)

    def reset(self, task_id: str = "easy") -> ServeObservation:
        if task_id not in ALL_TASKS:
            task_id = "easy"

        self._task = ALL_TASKS[task_id]
        self._current_params = self._task.initial_params.copy()
        self._current_metrics = self._simulator.simulate(
            model_key = self._task.model_key,
            params = self._current_params,
            changed_param = None
        )

        self._serve_state = ServeState(
            episode_id = str(uuid4()),
            step_count = 0,
            task_id = task_id,
            best_latency_ms = self._current_metrics.latency_p99_ms,
            initial_latency_ms = self._current_metrics.latency_p99_ms,
            best_throughput = self._current_metrics.throughput_tok_per_sec,
            total_reward = 0.0,
            target_hit = False,
            failed_starts = 0
        )

        return self._build_observation(
            reward = 0.0,
            done = False,
            feedback = f"Episode started. Baseline p99={self._current_metrics.latency_p99_ms:.0f}ms. RAM used={self._current_metrics.ram_used_gb:.2f}GB. Target: p99 ≤ {self._task.target_latency_ms:.0f}ms. Begin tuning."
        )

    def step(self, action: ServeAction) -> ServeObservation:
        previous_latency = self._current_metrics.latency_p99_ms

        feedback, valid = self._apply_action(action)
        if not valid:
            self._serve_state.step_count += 1
            return self._build_observation(
                reward = -0.05,
                done = self._is_done(),
                feedback = feedback
            )

        self._current_metrics = self._simulator.simulate(
            model_key = self._task.model_key,
            params = self._current_params,
            changed_param = action.parameter
        )

        reward = self._grader.grade(
            task = self._task,
            metrics = self._current_metrics,
            previous_latency = previous_latency,
            step_number = self._serve_state.step_count
        )

        self._serve_state.step_count += 1
        self._serve_state.total_reward += reward

        if self._current_metrics.failed:
            self._serve_state.failed_starts += 1
            feedback += " ⚠️  vLLM failed to start. Try reducing max_model_len or changing dtype."
        else:
            if self._current_metrics.latency_p99_ms < self._serve_state.best_latency_ms:
                self._serve_state.best_latency_ms = self._current_metrics.latency_p99_ms
                feedback += f" ✓ New best: {self._serve_state.best_latency_ms:.0f}ms"

            if self._current_metrics.throughput_tok_per_sec > self._serve_state.best_throughput:
                self._serve_state.best_throughput = self._current_metrics.throughput_tok_per_sec

        done = self._is_done()
        if done:
            if self._serve_state.target_hit:
                feedback += " 🎉 TARGET HIT! Episode complete."
            else:
                feedback += f" Episode ended. Best p99: {self._serve_state.best_latency_ms:.0f}ms."

        return self._build_observation(
            reward = reward,
            done = done,
            feedback = feedback
        )

    @property
    def state(self) -> ServeState:
        return self._serve_state

    def _apply_action(self, action: ServeAction) -> tuple[str, bool]:
        param = action.parameter
        value = action.value

        legal = VALID_PARAM_VALUES.get(param)
        if legal is None:
            return f"Unknown parameter: '{param}'. Legal params: {list(VALID_PARAM_VALUES.keys())}", False

        if value not in legal:
            return f"Invalid value '{value}' for '{param}'. Legal values: {legal}", False

        old = self._current_params.get(param)
        self._current_params[param] = value
        return f"Set {param} = {value} (was {old})", True

    def _is_done(self) -> bool:
        if not self._current_metrics.failed:
            lat_ok = (
                self._current_metrics.latency_p99_ms <= self._task.target_latency_ms
            )
            tput_ok = (
                self._task.target_throughput == 0.0 
                or self._current_metrics.throughput_tok_per_sec >= self._task.target_throughput
            )
            if lat_ok and tput_ok:
                self._serve_state.target_hit = True
                return True

        if self._serve_state.step_count >= self._task.max_steps:
            return True
        
        return False

    def _build_observation(self, reward: float, done: bool, feedback: str) -> ServeObservation:
        m = self._current_metrics
        model_info = MODEL_REGISTRY[self._task.model_key]

        return ServeObservation(
            model = self._task.model_key,
            model_hf_id=model_info["hf_id"],
            hardware = "CPU",
            current_params = self._current_params.copy(),
            latency_p50_ms = m.latency_p50_ms,
            latency_p99_ms = m.latency_p99_ms,
            throughput_tok_per_sec = m.throughput_tok_per_sec,
            ram_used_gb = m.ram_used_gb,
            ram_total_gb = self._simulator.ram_total_gb,
            task_id = self._task.task_id,
            task_description = self._task.description,
            target_latency_ms = self._task.target_latency_ms,
            target_throughput = self._task.target_throughput,
            steps_remaining = self._task.max_steps - self._serve_state.step_count,
            legal_parameters = list(VALID_PARAM_VALUES.keys()),
            reward = reward,
            done = done,
            last_action_feedback = feedback,
            constraint_violated = m.failed
        )