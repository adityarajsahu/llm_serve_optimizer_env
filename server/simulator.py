import math
from data.benchmark_data import (
    LATENCY_TABLE,
    THROUGHPUT_TABLE,
    MEMORY_TABLE,
    OOM_CONDITIONS,
    VALID_PARAM_VALUES,
    CHUNKED_PREFILL_LATENCY_FACTOR,
    MAX_SEQS_LATENCY_FACTOR
)

class SimulationResult:
    def __init__(self, mean_e2el_ms: float, throughput_token_per_sec: float, gpu_memory_used_gb: float, oom: bool, config_note: str = ""):
        self.mean_e2el_ms = mean_e2el_ms
        self.throughput_token_per_sec = throughput_token_per_sec
        self.gpu_memory_used_gb = gpu_memory_used_gb
        self.oom = oom
        self.config_note = config_note

class LatencySimulator:
    QUANT_LATENCY_RATIO = {
        "fp16": 1.00,
        "fp8":  0.58,
        "int8": 0.62,
        "awq":  0.56,
    }

    QUANT_MEMORY_RATIO = {
        "fp16": 1.00,
        "fp8":  0.59,
        "int8": 0.56,
        "awq":  0.49,
    }

    TP_SPEEDUP = {
        1: 1.00,
        2: 1.80,
        4: 3.20,
        8: 5.60,
    }

    def simulate(self, model: str, hardware: str, params: dict) -> SimulationResult:
        tp = int(params.get("tensor_parallel_size", 1))
        quant = str(params.get("quantization", "fp16"))
        util = float(params.get("gpu_memory_utilization", 0.90))
        batch = int(params.get("max_num_batched_tokens", 2048))
        seqs = int(params.get("max_num_seqs", 256))
        chunked = bool(params.get("enable_chunked_prefill", False))

        if self._is_oom(model, hardware, quant, tp, util):
            return SimulationResult(
                mean_e2el_ms = float('inf'),
                throughput_token_per_sec = 0.0,
                gpu_memory_used_gb = self.gpu_total(hardware) * util + 5,
                oom = True,
                config_note = "OOM: not enough GPU memory for this configuration",
            )

        exact_key = (model, hardware, tp, quant, util, batch)
        latency_mean = LATENCY_TABLE.get(exact_key)
        throughput = THROUGHPUT_TABLE.get(exact_key)
        memory = MEMORY_TABLE.get(exact_key)
        note = "exact match"

        if latency_mean is None:
            latency_mean, note = self._interpolate_latency(model, hardware, tp, quant, util, batch)
        if throughput is None:
            throughput = self._interpolate_throughput(model, hardware, tp, quant, util, batch, latency_mean)
        if memory is None:
            memory = self._interpolate_memory(model, quant, tp)

        chunked_factor = CHUNKED_PREFILL_LATENCY_FACTOR.get(chunked, 1.0)
        seqs_factor = MAX_SEQS_LATENCY_FACTOR.get(seqs, 1.0)
        latency_mean = latency_mean * chunked_factor * seqs_factor

        gpu_cap = self.gpu_total(hardware)
        kv_overhead = gpu_cap * util * 0.15   # KV cache scales with allocated GPU pool
        total_memory = min(memory + kv_overhead, gpu_cap)  # hard ceiling = physical GPU memory

        return SimulationResult(
            mean_e2el_ms = latency_mean,
            throughput_token_per_sec = round(throughput, 0),
            gpu_memory_used_gb = round(total_memory, 1),
            oom = False,
            config_note = note
        )

    def _interpolate_latency(self, model: str, hardware: str, tp: int, quant: str, util: float, batch: int) -> tuple[float, str]:
        base_key = self._find_base_key(model, hardware, batch)
        if base_key is None:
            base_latency = 120.0 if "70b" in model else 50.0
        else:
            base_latency = LATENCY_TABLE[base_key]
            base_tp = base_key[2]
            base_quant = base_key[3]
            base_latency /= self.QUANT_LATENCY_RATIO.get(base_quant, 1.0)
            base_latency *= self.TP_SPEEDUP.get(base_tp, 1.0) / self.TP_SPEEDUP.get(1, 1.0)

        latency = base_latency
        latency *= self.QUANT_LATENCY_RATIO.get(quant, 1.0)
        latency /= self.TP_SPEEDUP.get(tp, 1.0)

        ref_batch = 2048
        if batch != ref_batch:
            batch_factor = 1.0 - 0.08 * math.log2(max(batch, 512) / ref_batch)
            batch_factor = max(0.70, min(batch_factor, 1.15))
            latency *= batch_factor

        latency *= max(0.95, 1.05 - util * 0.10)

        return round(latency, 1), "interpolated"

    def _interpolate_throughput(self, model: str, hardware: str, tp: int, quant: str, util: float, batch: int, latency_mean: float) -> float:
        for key, tput in THROUGHPUT_TABLE.items():
            if key[0] == model and key[1] == hardware and key[2] == tp and key[3] == quant:
                ref_batch = key[5]
                batch_scale = math.sqrt(batch / max(ref_batch, 1))
                return round(tput * batch_scale, 0)

        tokens_per_request = 256
        concurrency = min(batch // 64, 16)
        return round((1000 / latency_mean) * tokens_per_request * max(concurrency, 1), 0)

    def _interpolate_memory(self, model: str, quant: str, tp: int) -> float:
        fp16_mem = MEMORY_TABLE.get((model, "fp16", 1))
        if fp16_mem is None:
            fp16_mem = 140.0 if "70b" in model else 17.0
        
        per_gpu = fp16_mem / tp
        per_gpu *= self.QUANT_MEMORY_RATIO.get(quant, 1.0)
        return round(per_gpu, 1)

    def _find_base_key(self, model: str, hardware: str, batch: int):
        candidates = [
            k for k in LATENCY_TABLE if k[0] == model and k[1] == hardware
        ]
        if not candidates:
            return None

        return min(candidates, key = lambda k: abs(k[5] - batch))

    def _is_oom(self, model: str, hardware: str, quant: str, tp: int, util: float) -> bool:
        if OOM_CONDITIONS.get((model, hardware, quant, tp, util), False):
            return True
        mem_per_gpu = self._interpolate_memory(model, quant, tp)
        gpu_total = self.gpu_total(hardware)
        if mem_per_gpu > gpu_total * util:
            return True
        return False

    def gpu_total(self, hardware: str) -> float:
        return {
            "A100-80GB": 80.0,
            "A100-40GB": 40.0,
            "H100-80GB": 80.0,
            "V100-32GB": 32.0,
        }.get(hardware, 80.0)