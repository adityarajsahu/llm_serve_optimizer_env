import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

vllm_available = True
try:
    import vllm
except ImportError:
    vllm_available = False

pytestmark = pytest.mark.skipif(
    not vllm_available,
    reason = "vLLM not installed - skipping real benchmark tests"
)

class TestSimulatorBasic:
    def test_model_registry_has_all_models(self):
        from data.model_card import MODEL_REGISTRY
        assert "gpt2" in MODEL_REGISTRY
        assert "smollm2-135m" in MODEL_REGISTRY
        assert "gemma-3-270m" in MODEL_REGISTRY

    def test_valid_param_values_no_quantization(self):
        from data.model_card import VALID_PARAM_VALUES
        assert "quantization" not in VALID_PARAM_VALUES
        assert "dtype" in VALID_PARAM_VALUES
        assert "max_model_len" in VALID_PARAM_VALUES
        assert "max_num_batched_tokens" in VALID_PARAM_VALUES
        assert "max_num_seqs" in VALID_PARAM_VALUES

    def test_max_model_len_values(self):
        from data.model_card import VALID_PARAM_VALUES
        assert VALID_PARAM_VALUES["max_model_len"] == [128, 192, 256]

    def test_restart_policy(self):
        from data.model_card import REQUIRES_RESTART, REQUEST_ONLY
        assert "dtype" in REQUIRES_RESTART
        assert "max_model_len" in REQUIRES_RESTART
        assert "quantization" not in REQUIRES_RESTART
        assert "max_num_batched_tokens" in REQUEST_ONLY
        assert "max_num_seqs" in REQUEST_ONLY

    def test_ram_safety_limit_set(self):
        from data.model_card import RAM_SAFETY_LIMIT_GB
        assert RAM_SAFETY_LIMIT_GB > 0
        assert RAM_SAFETY_LIMIT_GB <= 8.0

class TestSimulatorRAM:
    def test_insufficient_ram_returns_failure(self):
        from server.simulator import LatencySimulator
        import unittest.mock as mock
 
        sim = LatencySimulator()
        with mock.patch(
            "server.simulator._ram_available_gb", return_value = 0.5
        ):
            result = sim.simulate(
                model_key = "gpt2",
                params = {
                    "dtype": "float16", 
                    "max_model_len": 128,
                    "max_num_batched_tokens": 64, 
                    "max_num_seqs": 1
                },
                changed_param = None,
            )
        assert result.failed
        sim.stop()

    def test_simulation_result_failure_factory(self):
        from server.simulator import SimulationResult
        r = SimulationResult.failure("test failure")
        assert r.failed
        assert r.latency_p99_ms == float('inf')
        assert r.throughput_tok_per_sec == 0.0

class TestSimulatorReal:
    def test_gpt2_float16_benchmark(self):
        from server.simulator import LatencySimulator
        sim = LatencySimulator()
 
        result = sim.simulate(
            model_key = "gpt2",
            params = {
                "dtype": "float16",
                "max_model_len": 128,
                "max_num_batched_tokens": 64,
                "max_num_seqs": 1,
            },
            changed_param = None,
        )
 
        assert not result.failed
        assert result.latency_p99_ms > 0
        assert result.latency_p50_ms > 0
        assert result.latency_p50_ms <= result.latency_p99_ms
        assert result.throughput_tok_per_sec > 0
        assert result.ram_used_gb > 0
        sim.stop()

    def test_float16_faster_than_float32(self):
        from server.simulator import LatencySimulator
        sim = LatencySimulator()
 
        params_base = {
            "max_model_len": 128, 
            "max_num_batched_tokens": 64, 
            "max_num_seqs": 1
        }
 
        r_fp32 = sim.simulate(
            "gpt2", {
                **params_base, 
                "dtype": "float32"
            }, 
            changed_param = None
        )

        r_fp16 = sim.simulate(
            "gpt2", 
            {
                **params_base, 
                "dtype": "float16"
            }, 
            changed_param = "dtype"
        )
 
        assert not r_fp32.failed
        assert not r_fp16.failed
        assert r_fp16.latency_p99_ms < r_fp32.latency_p99_ms
        sim.stop()

    def test_smaller_max_model_len_uses_less_ram(self):
        from server.simulator import LatencySimulator
        sim = LatencySimulator()
 
        params_base = {
            "dtype": "float16", 
            "max_num_batched_tokens": 64, 
            "max_num_seqs": 1
        }
 
        r_256 = sim.simulate(
            "gpt2", 
            {
                **params_base, 
                "max_model_len": 256
            }, 
            changed_param = None
        )

        r_128 = sim.simulate(
            "gpt2", 
            {
                **params_base, 
                "max_model_len": 128
            }, 
            changed_param = "max_model_len"
        )
 
        assert not r_256.failed
        assert not r_128.failed
        assert r_128.ram_used_gb <= r_256.ram_used_gb + 0.1 # Added 0.1 for floating point inaccuracies due to OS and Docker overhead
        sim.stop()

    def test_no_restart_for_request_only_param(self):
        from server.simulator import LatencySimulator
        sim = LatencySimulator()
 
        params = {
            "dtype": "float16", 
            "max_model_len": 128,
            "max_num_batched_tokens": 64, 
            "max_num_seqs": 1
        }
        sim.simulate("gpt2", params, changed_param = None)
        proc_before = sim._vllm._proc
 
        params["max_num_seqs"] = 2
        sim.simulate("gpt2", params, changed_param = "max_num_seqs")
        proc_after = sim._vllm._proc
 
        assert proc_before is proc_after
        sim.stop()