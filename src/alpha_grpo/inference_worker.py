import json
import math
import os
import signal
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from uuid import UUID


def selected_scores(ids, rows):
    if rows is None or len(ids) != len(rows):
        return None
    values = []
    for token, row in zip(ids, rows, strict=True):
        value = row.get(token) if row is not None else None
        if value is None or not math.isfinite(value.logprob):
            return None
        values.append(value.logprob)
    return values


def checked_device_uuid(cuda_uuid_bytes, expected):
    actual = UUID(bytes=bytes(cuda_uuid_bytes))
    requested = UUID(expected.lower().removeprefix("gpu-"))
    if actual != requested:
        raise RuntimeError(f"CUDA device UUID mismatch on device 0: expected GPU-{requested}, got GPU-{actual}")
    return expected


def configure_cuda_linker(directory: Path, maps: Path = Path("/proc/self/maps")):
    """Expose the loaded CUDA driver under the name required by `-lcuda`."""
    try:
        mappings = maps.open(encoding="utf-8", errors="surrogateescape")
    except OSError:
        return
    with mappings:
        for mapping in mappings:
            fields = mapping.rstrip("\n").split(maxsplit=5)
            if len(fields) < 6:
                continue
            driver = Path(fields[5])
            if not driver.is_absolute() or not (
                driver.name == "libcuda.so" or driver.name.startswith("libcuda.so.")
            ) or not driver.is_file():
                continue
            (directory / "libcuda.so").symlink_to(driver)
            search_path = os.environ.get("LIBRARY_PATH")
            os.environ["LIBRARY_PATH"] = (
                f"{search_path}{os.pathsep}{directory}" if search_path else str(directory)
            )
            return


class Engine:
    def __init__(self, options):
        import torch

        self.physical_device = checked_device_uuid(
            torch.cuda.get_device_properties(0).uuid.bytes, options["expected_uuid"],
        )
        # PyTorch has loaded the versioned driver. JIT builds in the vLLM child
        # inherit the private link directory through the compiler search path.
        configure_cuda_linker(Path(options["cuda_link_dir"]))
        from vllm import LLM
        from vllm.config import WeightTransferConfig

        self.version = options["version"]
        self.role = options["role"]
        self.updating = False
        self.concurrency = options["concurrency"]
        self.transfer = options["transfer"]
        self.llm = LLM(
            model=options["model"], tokenizer=options["tokenizer"],
            dtype="float16", tensor_parallel_size=1, enforce_eager=True,
            max_model_len=options["context"], max_num_seqs=self.concurrency,
            max_num_batched_tokens=max(options["context"], 1024),
            gpu_memory_utilization=options["budget"], enable_prefix_caching=False,
            generation_config="vllm", logprobs_mode="raw_logprobs",
            worker_extension_cls="alpha_grpo.weights.FingerprintWorker",
            weight_transfer_config=WeightTransferConfig(backend="nccl") if self.transfer else None,
        )

    def execute(self, operation, **args):
        if operation == "health":
            return dict(version=self.version, role=self.role,
                        physical_device=self.physical_device)
        if operation == "fingerprint":
            if self.updating or args["version"] != self.version:
                raise RuntimeError("Cannot inspect a stale or partially updated policy")
            return self.llm.collective_rpc("alpha_grpo_fingerprint")[0]
        if operation in {"generate", "score"}:
            if self.updating or args.pop("version") != self.version:
                raise RuntimeError("Inference version is stale or being updated")
            if operation == "generate":
                if self.role != "policy":
                    raise RuntimeError("The reference engine only scores responses")
                return self.generate(**args)
            return self.score(**args)
        if self.role != "policy" or not self.transfer:
            raise RuntimeError(f"Weight transfer is unavailable for {self.role}")
        if operation == "init":
            self.llm.init_weight_transfer_engine({"init_info": args})
        elif operation == "start":
            if self.updating or args["version"] <= self.version:
                raise RuntimeError("Invalid weight update version")
            self.updating = True
            self.llm.start_weight_update(is_checkpoint_format=True)
        elif operation == "receive" and self.updating:
            self.llm.update_weights({"update_info": args})
        elif operation == "finish" and self.updating:
            self.llm.finish_weight_update()
            self.version = args["version"]
            self.updating = False
        else:
            raise ValueError(f"Invalid operation {operation}")
        return self.version

    def generate(self, token_ids, count, max_tokens, temperature, top_p, seed):
        from vllm import SamplingParams

        completions = []
        for offset in range(0, count, self.concurrency):
            size = min(self.concurrency, count - offset)
            params = [
                SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                               top_k=0, seed=(seed + offset + index) % (2**31 - 1),
                               logprobs=0, detokenize=False)
                for index in range(size)
            ]
            outputs = self.llm.generate([{"prompt_token_ids": token_ids}] * size, params,
                                        use_tqdm=False)
            for request in outputs:
                item = request.outputs[0]
                # With detokenization disabled, vLLM returns the sampled stop token.
                ids = list(item.token_ids)
                if not ids or len(ids) > max_tokens:
                    raise ValueError("Invalid completion length")
                completions.append(dict(token_ids=ids, logprobs=selected_scores(ids, item.logprobs),
                                        finish_reason=item.finish_reason))
        return completions

    def score(self, sequences):
        from vllm import SamplingParams

        # The engine context reserves one position for this unused output.
        results = []
        params = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0,
                                detokenize=False, ignore_eos=True)
        for offset in range(0, len(sequences), self.concurrency):
            chunk = sequences[offset:offset + self.concurrency]
            inputs = [{"prompt_token_ids": prompt + response} for prompt, response in chunk]
            outputs = self.llm.generate(inputs, params, use_tqdm=False)
            for (prompt, response), output in zip(chunk, outputs, strict=True):
                if output.prompt_token_ids != prompt + response:
                    raise ValueError("vLLM returned a different scoring sequence")
                rows = output.prompt_logprobs
                scores = selected_scores(response, rows[len(prompt):] if rows is not None else None)
                if scores is None:
                    raise ValueError("Missing teacher-forced log probabilities")
                results.append(scores)
        return results


def watch_parent(parent):
    while os.getppid() == parent:
        time.sleep(1)
    os.killpg(os.getpgrp(), signal.SIGKILL)


def main():
    fd, raw_options = sys.argv[1:]
    threading.Thread(target=watch_parent, args=(os.getppid(),), daemon=True).start()
    with socket.socket(fileno=int(fd)) as channel, channel.makefile("rb") as reader:
        def send(**message):
            channel.sendall(json.dumps(message, allow_nan=False).encode() + b"\n")

        try:
            engine = Engine(json.loads(raw_options))
            send(result=engine.execute("health"))
        except Exception:
            send(error=traceback.format_exc())
            return
        for raw in reader:
            try:
                send(result=engine.execute(**json.loads(raw)))
            except Exception:
                send(error=traceback.format_exc())


if __name__ == "__main__":
    main()
