import json
import math
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

from .records import Completion


class InferenceError(RuntimeError):
    pass


class InferenceClient:
    """One synchronous inference subprocess with a private inherited socket."""

    def __init__(self, *, role, gpu, model, tokenizer, budget, context, concurrency,
                 version, timeout, transfer, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{role}.log"
        import pynvml as nvml

        nvml.nvmlInit()
        try:
            ordered = sorted(
                (nvml.nvmlDeviceGetPciInfo(handle).busId, nvml.nvmlDeviceGetUUID(handle))
                for handle in (nvml.nvmlDeviceGetHandleByIndex(index)
                               for index in range(nvml.nvmlDeviceGetCount()))
            )
        finally:
            nvml.nvmlShutdown()
        ordinal = next((index for index, (_, uuid) in enumerate(ordered) if uuid == gpu), None)
        if ordinal is None:
            raise InferenceError(f"Inference GPU {gpu} was not found by NVML")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(ordinal), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_NO_USAGE_STATS="1",
                   PYTHONUNBUFFERED="1")
        options = dict(role=role, model=str(model), tokenizer=str(tokenizer), budget=budget,
                       context=context, concurrency=concurrency, version=version,
                       transfer=transfer, expected_uuid=gpu)
        parent, child = socket.socketpair()
        parent.settimeout(timeout)
        self.socket = parent
        self.reader = parent.makefile("rb")
        self.process = None
        try:
            with self.log_path.open("ab") as log:
                log_start = log.seek(0, os.SEEK_END)
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "alpha_grpo.inference_worker",
                     str(child.fileno()), json.dumps(options)],
                    env=env, pass_fds=(child.fileno(),), stdout=log, stderr=log,
                    start_new_session=True,
                )
        except BaseException:
            self.close()
            raise
        finally:
            child.close()
        try:
            self.startup = self._receive()
            if (self.startup["version"] != version or self.startup["role"] != role
                    or self.startup["physical_device"] != gpu):
                raise InferenceError("Inference worker started with the wrong device, role, or version")
        except BaseException as error:
            self.close()
            if isinstance(error, InferenceError):
                error.add_note(f"{role} worker log: {self.log_path}")
                try:
                    with self.log_path.open("rb") as log:
                        end = log.seek(0, os.SEEK_END)
                        log.seek(max(log_start, end - 16 * 1024))
                        tail = log.read().decode("utf-8", errors="replace").strip()
                except OSError:
                    tail = ""
                if tail:
                    error.add_note(tail)
            raise

    def _receive(self):
        try:
            line = self.reader.readline()
        except OSError as error:
            raise InferenceError(f"Inference response timed out or connection failed. See {self.log_path}") from error
        if not line:
            raise InferenceError(f"Inference worker exited. See {self.log_path}")
        response = json.loads(line)
        if "error" in response:
            raise InferenceError(response["error"])
        return response["result"]

    def call(self, operation, **arguments):
        try:
            self.socket.sendall(
                json.dumps(dict(operation=operation, **arguments), allow_nan=False).encode() + b"\n"
            )
        except OSError as error:
            raise InferenceError(f"Could not send {operation} to inference worker") from error
        return self._receive()

    def generate(self, token_ids, *, version, count, max_tokens, temperature, top_p, seed):
        result = self.call("generate", token_ids=token_ids, version=version, count=count,
                           max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed)
        completions = [Completion(**item) for item in result]
        if len(completions) != count:
            raise InferenceError("Wrong number of completions")
        return completions

    def score(self, sequences, *, version):
        results = self.call("score", sequences=sequences, version=version)
        lengths = [len(item[1]) for item in sequences]
        if len(results) != len(lengths) or any(
            len(scores) != length or not all(map(math.isfinite, scores))
            for scores, length in zip(results, lengths)
        ):
            raise InferenceError("Invalid completion log probabilities")
        return results

    def close(self):
        self.reader.close()
        self.socket.close()
        if self.process is None:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                continue
            # Kill any engine descendants that survived the worker exit.
            if sig == signal.SIGKILL:
                break
        self.process.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
