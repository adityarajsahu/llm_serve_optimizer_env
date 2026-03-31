import sys
import os
import pytest
import unittest.mock as mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.environment import LLMServeEnvironment
from server.simulator import SimulationResult
from models import ServeAction, ServeObservation, ServeState

def make_mock_result(latency = 500.0, tput = 20.0, ram = 1.5, failed = False):
    return SimulationResult(
        latency_p99_ms = latency,
        latency_p50_ms = latency * 0.65,
        throughput_tok_per_sec = tput,
        ram_used_gb = ram,
        failed = failed,
        config_note = "mocked"
    )

def make_env(task_id = "easy", initial_latency = 500.0):
    env = LLMServeEnvironment()
    env._simulator.simulate = mock.MagicMock(
        return_value = make_mock_result(latency = initial_latency)
    )
    env._simulator.ram_total_gb = 8.0
    obs = env.reset(task_id = task_id)
    return env, obs

class TestReset:
    def test_reset_returns_observation(self):
        env, obs = make_env()
        assert isinstance(obs, ServeObservation)

    def test_reset_state_is_clean(self):
        env, obs = make_env()
        assert env.state.step_count == 0
        assert env.state.total_reward == 0.0
        assert env.state.target_hit is False
        assert env.state.failed_starts == 0

    def test_reset_task_id_set(self):
        env, obs = make_env("easy")
        assert env.state.task_id == "easy"

    def test_reset_unknown_task_defaults_to_easy(self):
        env = LLMServeEnvironment()
        env._simulator.simulate = mock.MagicMock(
            return_value = make_mock_result()
        )
        env._simulator.ram_total_gb = 8.0
        obs = env.reset(task_id = "nonexistent")
        assert obs.task_id == "easy"

    def test_reset_hardware_is_cpu(self):
        env, obs = make_env()
        assert obs.hardware == "CPU"

    def test_reset_has_ram_fields(self):
        env, obs = make_env()
        assert hasattr(obs, "ram_used_gb")
        assert hasattr(obs, "ram_total_gb")

    def test_reset_all_tasks(self):
        for task_id in ["easy", "medium", "hard"]:
            env = LLMServeEnvironment()
            env._simulator.simulate = mock.MagicMock(
                return_value = make_mock_result()
            )
            env._simulator.ram_total_gb = 8.0
            obs = env.reset(task_id = task_id)
            assert obs.task_id == task_id
        
    def test_reset_initial_params_have_dtype(self):
        env, obs = make_env()
        assert "dtype" in obs.current_params
        assert "max_model_len" in obs.current_params

class TestStep:
    def test_step_increments_count(self):
        env, _ = make_env()
        env.step(ServeAction(parameter = "dtype", value = "float16"))
        assert env.state.step_count == 1

    def test_valid_dtype_change(self):
        env, _ = make_env()
        env._simulator.simulate.return_value = make_mock_result(latency = 300.0)
        obs = env.step(ServeAction(parameter = "dtype", value = "float16"))
        assert isinstance(obs, ServeObservation)
        assert obs.latency_p99_ms == 300.0
    
    def test_invalid_parameter_negative_reward(self):
        env, _ = make_env()
        obs = env.step(ServeAction(parameter = "quantization", value = "int8"))
        assert obs.reward == -0.05

    def test_invalid_parameter_tensor_parallel(self):
        env, _ = make_env()
        obs = env.step(ServeAction(parameter = "tensor_parallel_size", value = 2))
        assert obs.reward == -0.05

    def test_invalid_parameter_gpu_memory(self):
        env, _ = make_env()
        obs = env.step(ServeAction(parameter = "gpu_memory_utilization", value = 0.9))
        assert obs.reward == -0.05

    def test_invalid_dtype_value(self):
        env, _ = make_env()
        obs = env.step(ServeAction(parameter = "dtype", value = "fp8"))
        assert obs.reward == -0.05

    def test_invalid_max_model_len_value(self):
        env, _ = make_env()
        obs = env.step(ServeAction(parameter = "max_model_len", value = 1024))
        assert obs.reward == -0.05

    def test_improvement_gives_positive_reward(self):
        env, _ = make_env(initial_latency = 1000.0)
        env._simulator.simulate.return_value = make_mock_result(latency = 500.0)
        obs = env.step(ServeAction(parameter = "dtype", value = "float16"))
        assert obs.reward > 0

    def test_failed_start_gives_negative_reward(self):
        env, _ = make_env()
        env._simulator.simulate.return_value = make_mock_result(failed = True)
        obs = env.step(ServeAction(parameter = "dtype", value = "bfloat16"))
        assert obs.reward == -0.3
        assert obs.constraint_violated is True
        assert env.state.failed_starts == 1

    def test_steps_remaining_decrements(self):
        env, initial_obs = make_env()
        initial_remaining = initial_obs.steps_remaining
        obs = env.step(ServeAction(parameter = "dtype", value = "float16"))
        assert obs.steps_remaining == initial_remaining - 1

    def test_episode_ends_at_max_steps(self):
        env, _ = make_env("easy")
        obs = None
        for _ in range(5):
            obs = env.step(ServeAction(parameter = "max_num_seqs", value = 2))
        assert obs.done is True

    def test_episode_ends_on_target_hit(self):
        env, _ = make_env("easy")
        env._simulator.simulate.return_value = make_mock_result(latency = 300.0)
        obs = env.step(ServeAction(parameter = "dtype", value = "float16"))
        assert obs.done is True
        assert env.state.target_hit is True

    def test_best_latency_tracked(self):
        env, _ = make_env(initial_latency = 1000.0)
        env._simulator.simulate.return_value = make_mock_result(latency = 400.0)
        env.step(ServeAction(parameter = "dtype", value = "float16"))
        assert env.state.best_latency_ms == 400.0

class TestState:
    def test_state_type(self):
        env, _ = make_env()
        assert isinstance(env.state, ServeState)

    def test_state_episode_id_not_empty(self):
        env, _ = make_env()
        assert len(env.state.episode_id) > 0

    def test_state_resets_between_episodes(self):
        env, _ = make_env()
        env.step(ServeAction(parameter = "dtype", value = "float16"))
        assert env.state.step_count == 1
 
        env._simulator.simulate.return_value = make_mock_result()
        env.reset("easy")
        assert env.state.step_count == 0
        assert env.state.total_reward == 0.0