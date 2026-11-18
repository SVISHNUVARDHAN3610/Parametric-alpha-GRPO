import logging
import os
import shutil
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from .checkpoint import export_model, require_space
from .inference import InferenceClient, InferenceError
from .weights import model_fingerprint

LOGGER = logging.getLogger(__name__)


class WeightSynchronizer(Protocol):
    def synchronize(self, version: int) -> str: ...


class Policy:
    def __init__(self, *, start, source, tokenizer, trainer, mode, timeout, root: Path):
        self.start = start
        self.source = source
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.timeout = timeout
        self.root = root
        self.mode = "checkpoint" if mode == "auto" and (root / "nccl-timeout").exists() else mode
        self.client: InferenceClient | None = None
        self.group = None
        self.acknowledged_version = None
        self.snapshot: Path | None = None

    def _timeout(self):
        # The native communicator may block inside CUDA. A poisoned process
        # cannot safely keep training or broadcast another set of weights.
        (self.root / "nccl-timeout").write_text(
            "Resume the latest committed checkpoint with checkpoint synchronization.\n",
            encoding="utf-8",
        )
        os._exit(75)

    def _collective(self, request, send):
        timer = threading.Timer(self.timeout + 10, self._timeout)
        timer.daemon = True
        timer.start()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(request)
                result = send()
                pending.result()
                return result
        finally:
            timer.cancel()

    def _nccl(self, version):
        import torch
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

        if self.group is None:
            with socket.socket() as address:
                address.bind(("127.0.0.1", 0))
                port = address.getsockname()[1]
            init = dict(master_address="127.0.0.1", master_port=port, world_size=2)
            self.group = self._collective(
                lambda: self.client.call("init", rank_offset=1, **init),
                lambda: NCCLWeightTransferEngine.trainer_init(init),
            )
        self.client.call("start", version=version)
        parameters = list(self.trainer.model.named_parameters())
        metadata = dict(names=[name for name, _ in parameters],
                        shapes=[list(value.shape) for _, value in parameters],
                        dtype_names=["float16"] * len(parameters), packed=False)

        def send():
            with torch.no_grad():
                NCCLWeightTransferEngine.trainer_send_weights(
                    ((name, value.detach().to(dtype=torch.float16)) for name, value in parameters),
                    dict(group=self.group, packed=False),
                )
                torch.cuda.current_stream().synchronize()

        self._collective(lambda: self.client.call("receive", **metadata), send)
        self.client.call("finish", version=version)

    def _checkpoint(self, version):
        self.close()
        weights = self.root / "sync-weights"
        weights.mkdir(exist_ok=True)
        require_space(weights, sum(p.numel() for p in self.trainer.parameters) * 2 + 64 * 2**20)
        destination = weights / f"policy-{version:06d}"
        if destination.exists():
            shutil.rmtree(destination)
        export_model(self.trainer.model, self.tokenizer, destination)
        self.snapshot = destination
        self.client = self.start(destination, version, self.mode != "checkpoint")

    def _verify(self, version):
        health = self.client.call("health")
        if health["version"] != version or health["role"] != "policy":
            raise InferenceError("Policy did not acknowledge synchronized weights")
        if self.client.call("fingerprint", version=version) != model_fingerprint(self.trainer.model):
            raise InferenceError("Policy weight samples differ from the trainer's FP16 weights")
        prompt = self.tokenizer.encode("Two plus two is", add_special_tokens=False)
        response = self.tokenizer.encode(" four.", add_special_tokens=False)
        sequence = [(prompt, response)]
        expected = self.trainer.score(sequence)[0]
        actual = self.client.score(sequence, version=version)[0]
        if any(abs(left - right) > 0.25 for left, right in zip(expected, actual, strict=True)):
            raise InferenceError(f"Policy canary disagrees with the trainer: {expected} vs {actual}")

    def synchronize(self, version):
        if self.acknowledged_version == version:
            return "unchanged"
        self.acknowledged_version = None
        if self.client is None and version == 0:
            self.client = self.start(self.source, 0, self.mode != "checkpoint")
            backend = "initial"
        elif self.client is None or self.mode == "checkpoint":
            self._checkpoint(version)
            backend = "checkpoint"
        else:
            try:
                self._nccl(version)
                self._verify(version)
            except Exception as error:
                if self.mode == "nccl":
                    raise
                LOGGER.warning("NCCL synchronization failed. Switching to checkpoint: %s", error)
                self.mode = "checkpoint"
                self._checkpoint(version)
                backend = "checkpoint"
            else:
                backend = "nccl"
        if backend != "nccl":
            self._verify(version)
        self.acknowledged_version = version
        return backend

    def close(self):
        self.acknowledged_version = None
        if self.client is not None:
            self.client.close()
            self.client = None
        self.group = None
        if self.snapshot is not None:
            shutil.rmtree(self.snapshot)
            self.snapshot = None
