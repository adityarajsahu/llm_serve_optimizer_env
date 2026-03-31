import os
import time
import subprocess
import statistics
import psutil
import requests
from typing import Optional

from data.model_card import (
    MODEL_REGISTRY, 
    VALID_PARAM_VALUES, 
    REQUIRES_RESTART, 
    REQUEST_ONLY, 
    RAM_SAFETY_LIMIT_GB, 
    VLLM_PORT, 
    VLLM_STARTUP_TIMEOUT_S, 
    BENCH_WARMUP_REQUESTS, 
    BENCH_TIMED_REQUESTS, 
    BENCH_MAX_TOKENS, 
    BENCH_PROMPT
)

def _ram_used_gb() -> float:
    try:
        return round(psutil.virtual_memory().used / (1024 ** 3), 2)
    except Exception:
        return 0.0

def _ram_total_gb() -> float:
    try:
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        return 8.0

def _ram_available_gb() -> float:
    try:
        return round(psutil.virtual_memory().available / (1024 ** 3), 2)
    except Exception:
        return 4.0

class SimulationResult:
    def __init__(self, latency_p99_ms: float, latency_p50_ms: float, throughput_tok_per_sec: float, ram_used_gb: float, failed: bool, config_note: str = ""):
        self.latency_p99_ms = latency_p99_ms
        self.latency_p50_ms = latency_p50_ms
        self.throughput_tok_per_sec = throughput_tok_per_sec
        self.ram_used_gb = ram_used_gb
        self.failed = failed
        self.config_note = config_note

    @classmethod
    def failure(cls, note: str) -> "SimulationResult":
        return cls(
            latency_p99_ms = float('inf'),
            latency_p50_ms = float('inf'),
            throughput_tok_per_sec = 0.0,
            ram_used_gb = _ram_used_gb(),
            failed = True,
            config_note = note
        )

class vLLMProcess:
    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None

    def start(self, model_key: str, params: dict) -> bool:
        self.stop()

        model_info = MODEL_REGISTRY[model_key]
        hf_id = model_info["hf_id"]
        
        # If a local clone exists (e.g., from Docker build), use it instead to avoid cache volumes
        local_model_name = hf_id.split("/")[-1]
        local_dir = os.path.join(os.environ.get("HOME", "/home/user"), "app", "models", local_model_name)
        if os.path.exists(local_dir):
            serve_id = local_dir
        else:
            serve_id = hf_id
            
        dtype = params.get("dtype", "float32")
        max_len = int(params.get("max_model_len", 192))

        cmd = [
            "vllm", "serve",
            "--model", serve_id,
            "--port", str(VLLM_PORT),
            "--dtype", dtype,
            "--max-model-len", str(max_len)
        ]

        hf_token = os.environ.get("HF_TOKEN", "")
        if model_info["hf_token"] and hf_token:
            cmd += ["--huggingface-token", hf_token]

        print(f"[vLLM] Starting: {' '.join(cmd)}")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("[vLLM] vllm not installed")
            return False

        url = f"http://localhost:{VLLM_PORT}/health"
        deadline = time.time() + VLLM_STARTUP_TIMEOUT_S

        while time.time() < deadline:
            if self._proc.poll() is not None:
                print("[vLLM] Process died during startup.")
                return False
            
            try:
                r = requests.get(url, timeout = 2)
                if r.status_code == 200:
                    print(f"[vLLM] Healthy. RAM used: {_ram_used_gb():.2f}GB / {_ram_total_gb():.1f}GB")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(3)

        print("[vLLM] Startup timed out.")
        self.stop()
        return False

    def benchmark(self, model_key: str, params: dict) -> SimulationResult:
        hf_id = MODEL_REGISTRY[model_key]["hf_id"]
        local_model_name = hf_id.split("/")[-1]
        local_dir = os.path.join(os.environ.get("HOME", "/home/user"), "app", "models", local_model_name)
        if os.path.exists(local_dir):
            serve_id = local_dir
        else:
            serve_id = hf_id

        url = f"http://localhost:{VLLM_PORT}/v1/completions"
        payload = {
            "model": serve_id,
            "prompt": BENCH_PROMPT,
            "max_tokens": BENCH_MAX_TOKENS,
            "temperature": 0.0,
            "n": 1
        }

        for _ in range(BENCH_WARMUP_REQUESTS):
            try:
                requests.post(url, json = payload, timeout = 120)
            except Exception:
                pass

        latencies = []
        tokens_generated = 0
        t_wall = time.time()

        for _ in range(BENCH_TIMED_REQUESTS):
            t0 = time.time()
            try:
                resp = requests.post(url, json = payload, timeout = 30)
                elapsed_ms = (time.time() - t0) * 1000
                latencies.append(elapsed_ms)
                if resp.ok:
                    tokens_generated += resp.json().get("usage", {}).get("completion_tokens", BENCH_MAX_TOKENS)
            except Exception as e:
                print(f"[vLLM] Request failed: {e}")

        total_s = max(time.time() - t_wall, 0.001)

        if not latencies:
            return SimulationResult.failure("All benchmark requests failed")

        latencies.sort()
        p50 = statistics.median(latencies)
        p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
        tput = tokens_generated / total_s

        print(f"[vLLM] Benchmark: p50={p50:.0f}ms p99={p99:.0f}ms tput={tput:.1f}tok/s RAM={_ram_used_gb():.2f}GB")

        return SimulationResult(
            latency_p99_ms = round(p99, 1),
            latency_p50_ms = round(p50, 1),
            throughput_tok_per_sec = round(tput, 1),
            ram_used_gb = _ram_used_gb(),
            failed = False,
            config_note = "real vLLM CPU"
        )

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout = 10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            print("[vLLM] Stopped.")
        self._proc = None
    
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

class LatencySimulator:
    def __init__(self):
        self._vllm = vLLMProcess()
        self._active_model: Optional[str] = None
        self._active_params: dict = {}
        self.ram_total_gb = _ram_total_gb()

        print(f"[Simulator] Real vLLM CPU mode. System RAM: {self.ram_total_gb:.1f}GB. Available: {_ram_available_gb():.2f}GB. Safety limit: {RAM_SAFETY_LIMIT_GB}GB.")

    def simulate(self, model_key: str, params: dict, changed_param: Optional[str] = None) -> SimulationResult:
        available = _ram_available_gb()
        if available < RAM_SAFETY_LIMIT_GB:
            return SimulationResult.failure(f"Insufficient RAM: only {available:.2f}GB available, need at least {RAM_SAFETY_LIMIT_GB}GB. Try reducing max_model_len.")

        needs_restart = (
            changed_param is None
            or changed_param in REQUIRES_RESTART
            or model_key != self._active_model
            or not self._vllm.is_running()
        )

        if needs_restart:
            print(f"[Simulator] Restarting vLLM (param={changed_param}, model={model_key})")
            ok = self._vllm.start(model_key, params)
            if not ok:
                return SimulationResult.failure(f"vLLM failed to start (model={model_key}, dtype={params.get('dtype')}, max_model_len={params.get('max_model_len')}). Try reducing max_model_len or switching dtype.")
            self._active_model = model_key
            self._active_params = params.copy()
        else:
            print(f"[Simulator] Reusing vLLM (request-only change: {changed_param})")
            self._active_params = params.copy()

        return self._vllm.benchmark(model_key, params)

    def stop(self):
        self._vllm.stop()
    
    # def gpu_total(self, *_) -> float:
    #     return self.ram_total_gb