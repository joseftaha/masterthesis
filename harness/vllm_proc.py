"""Subprocess wrapper around a single ``vllm serve`` instance.

The controller owns a node-wide allocation (4x A40). It starts/stops a
single vLLM OpenAI-compatible server at a time on that allocation. The
*warm* mode means the server is already running with the requested model
loaded; the *cold* mode means we start the server now and wait for
readiness; *switch* means stop the current server and start a new one
with a different model.

Local NVMe is used as the model cache via the ``HF_HOME`` env var so no
remote download path is ever exercised at measurement time. Use
``download_models.py`` once during setup to populate the cache.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass

import requests


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@dataclass
class ServeSpec:
    name: str               # e.g. Meta-Llama-3-8B-Instruct
    model_path: str         # local snapshot dir or hub id (cache must be warm)
    tensor_parallel: int
    quantization: str | None
    gpu_memory_utilization: float
    max_model_len: int
    seed: int
    extra_args: list[str]


class VLLMServer:
    def __init__(self, spec: ServeSpec, log_path: str, port: int | None = None):
        self.spec = spec
        self.port = port or _free_port()
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.t_started: float | None = None
        self.t_ready: float | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _argv(self) -> list[str]:
        s = self.spec
        argv = [
            "vllm", "serve", s.model_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--served-model-name", s.name,
            "--tensor-parallel-size", str(s.tensor_parallel),
            "--gpu-memory-utilization", str(s.gpu_memory_utilization),
            "--max-model-len", str(s.max_model_len),
            "--dtype", "bfloat16",
            "--seed", str(s.seed),
            "--disable-log-requests",
        ]
        if s.quantization:
            argv += ["--quantization", s.quantization]
        argv += s.extra_args
        return argv

    def start(self) -> None:
        env = os.environ.copy()
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        # Pin to all GPUs of the allocation; tensor-parallel-size selects fan-out.
        self.t_started = time.monotonic()
        self.proc = subprocess.Popen(
            self._argv(),
            stdout=open(self.log_path, "ab"),
            stderr=subprocess.STDOUT,
            env=env, preexec_fn=os.setsid,
        )

    def wait_ready(self, timeout_s: float = 1800.0) -> float:
        assert self.proc is not None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"vllm exited rc={self.proc.returncode}; see {self.log_path}")
            try:
                r = requests.get(f"{self.url}/v1/models", timeout=2.0)
                if r.status_code == 200 and r.json().get("data"):
                    self.t_ready = time.monotonic()
                    return self.t_ready - (self.t_started or self.t_ready)
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise TimeoutError(f"vllm did not become ready within {timeout_s}s")

    def stop(self, grace: float = 30.0) -> float:
        if self.proc is None:
            return 0.0
        t0 = time.monotonic()
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            self.proc.wait()
        return time.monotonic() - t0
