import hashlib
import importlib.metadata
import json
import logging
import os
import random
import shutil
import signal
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

from .checkpoint import atomic_directory, export_model, load_resume, require_space, save_resume
from .data import TASKS, digest, load_curated, model_snapshot, shuffled, write_prompts
from .inference import InferenceClient, InferenceError
from .records import Group, RolloutBatch, RunState, Trajectory
from .sync import Policy
from .verifiers import load_registry, reward

LOGGER = logging.getLogger(__name__)
GIB = 2**30


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_json(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def hardware():
    import pynvml as nvml

    nvml.nvmlInit()
    devices = []
    for index in range(2):
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        uuid = nvml.nvmlDeviceGetUUID(handle)
        name = nvml.nvmlDeviceGetName(handle)
        devices.append(dict(index=index, uuid=uuid, name=name,
                            bytes=nvml.nvmlDeviceGetMemoryInfo(handle).total))
    return sorted(devices, key=lambda item: nvml.nvmlDeviceGetPciInfo(
        nvml.nvmlDeviceGetHandleByIndex(item["index"])).busId)


class ResourceMonitor:
    def __init__(self, devices, root, checkpoint_root):
        self.devices = devices
        self.root = root
        self.checkpoint_root = checkpoint_root
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.error = None
        self.minimum_free = [device["bytes"] for device in devices]

    def sample(self):
        import psutil
        import pynvml as nvml

        try:
            handles = [nvml.nvmlDeviceGetHandleByIndex(item["index"]) for item in self.devices]
            while not self.stop.is_set():
                free_gpu = [nvml.nvmlDeviceGetMemoryInfo(handle).free for handle in handles]
                self.minimum_free = [min(old, new) for old, new in
                                     zip(self.minimum_free, free_gpu, strict=True)]
                append_json(self.root / "resources.jsonl", dict(
                    time=time.time(), gpu_free_bytes=free_gpu,
                    host_available_bytes=psutil.virtual_memory().available,
                    working_free_bytes=shutil.disk_usage(self.root).free,
                    checkpoint_free_bytes=shutil.disk_usage(self.checkpoint_root).free,
                ))
                self.stop.wait(0.25)
        except Exception as error:
            self.error = error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def inference_launcher(study, gpu, source, root):
    engine = study.engine

    def start(role, weights, version, transfer):
        budget = engine.reference_budget
        if role == "policy":
            budget = engine.policy_budget
            if transfer:
                # vLLM allocates receive and layer-reload buffers outside its KV cache budget.
                budget = min(budget, engine.nccl_policy_budget)
        return InferenceClient(
            role=role, gpu=gpu, model=weights, tokenizer=source,
            budget=budget,
            context=study.training.max_prompt + study.training.max_completion + 1,
            concurrency=engine.max_sequences, version=version, timeout=engine.timeout_seconds,
            startup_timeout=engine.startup_timeout_seconds,
            transfer=transfer, log_dir=root,
        )

    return start


def _model_and_verifier(study, data, stage):
    from transformers import AutoTokenizer

    manifest, rows = load_curated(data, stage)
    if manifest["model"] != study.model or manifest["dataset"] != study.dataset:
        raise ValueError("Dataset manifest belongs to a different study")
    if study.model_revision != "main" and not manifest["model_revision"].startswith(study.model_revision):
        raise ValueError("Model revision differs from the prepared dataset")
    revision, source = model_snapshot(study.model, manifest["model_revision"])
    if revision != manifest["model_revision"]:
        raise ValueError("Model snapshot changed")
    tokenizer = AutoTokenizer.from_pretrained(source)
    registry = load_registry(data / "if_functions.py", manifest["verifier_digest"])
    return manifest, rows, source, tokenizer, registry


def _completion_text(tokenizer, completion):
    return tokenizer.decode(completion.token_ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)


def prescreen(study, data, output, seed):
    manifest, rows, source, tokenizer, registry = _model_and_verifier(study, data, "prepared")
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = output.with_suffix(".prescreen.jsonl")
    if output.exists() or audit.exists():
        raise FileExistsError("Use a new prescreen output path")
    start = inference_launcher(study, hardware()[1]["uuid"], source, output.parent / "prescreen-logs")
    accepted = []
    with start("policy", source, 0, False) as client:
        for task in TASKS:
            count = 0
            pool = shuffled((row for row in rows if row.task == task), seed)
            for item in pool:
                completions = client.generate(
                    item.token_ids, version=0, count=8, max_tokens=study.training.max_completion,
                    temperature=1, top_p=1, seed=request_seed(seed, item.source_id),
                )
                passed = sum(reward(item, _completion_text(tokenizer, record), registry)
                             for record in completions)
                append_json(audit, dict(task=task, source_id=item.source_id, pass_count=passed,
                                        accepted=passed < 8))
                if passed < 8:
                    accepted.append(item)
                    count += 1
                if count == 500:
                    break
            if count != 500:
                raise RuntimeError(f"Only {count} eligible {task} prompts. See {audit}.")
            LOGGER.info("Prescreen selected 500 %s prompts", task)
    with atomic_directory(output) as temporary:
        write_prompts(temporary, shuffled(accepted, seed))
        shutil.copyfile(data / "if_functions.py", temporary / "if_functions.py")
        shutil.copyfile(audit, temporary / "prescreen.jsonl")
        manifest.update(stage="curated", seed=seed, prepared_sha256=manifest["prompts_sha256"],
                        prompts_sha256=digest(temporary / "prompts.jsonl"))
        write_json(temporary / "manifest.json", manifest)


def request_seed(seed, *parts):
    material = json.dumps([seed, *parts]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") % (2**31 - 1)


def collect(study, experiment, seed, prompts, state, policy, reference, tokenizer, registry, root):
    from .objective import gradient_weight_shares, group_advantages

    config = study.training
    groups = []
    limit = config.filter_cap if experiment.filter_constant else config.prompts_per_step
    if policy.acknowledged_version != state.version:
        raise RuntimeError("Collecting against an unsynchronized policy is forbidden")
    state.collection_id += 1
    for _ in range(limit):
        prompt = prompts[state.cursor % len(prompts)]
        rollout_seed = request_seed(seed, state.cursor, state.version)
        state.cursor += 1
        completions = policy.client.generate(
            prompt.token_ids, version=state.version, count=config.group_size,
            max_tokens=config.max_completion, temperature=experiment.temperature,
            top_p=experiment.top_p, seed=rollout_seed,
        )
        state.attempted_groups += 1
        lengths = [len(item.token_ids) for item in completions]
        state.generated_tokens += sum(lengths)
        try:
            rewards = [reward(prompt, _completion_text(tokenizer, item), registry) for item in completions]
        except Exception as error:
            append_json(root / "verification-errors.jsonl", dict(
                source_id=prompt.source_id, version=state.version, error=repr(error),
            ))
            raise
        advantages = group_advantages(rewards, experiment.normalize_std)
        constant = min(rewards) == max(rewards)
        reject = experiment.filter_constant and constant
        append_json(root / "rollouts.jsonl", dict(
            version=state.version, collection=state.collection_id, cursor=state.cursor,
            source_id=prompt.source_id, task=prompt.task, rewards=rewards, lengths=lengths,
            finish_reasons=[item.finish_reason for item in completions],
            constant=constant, rejected=reject,
            gradient_weight_share=gradient_weight_shares(lengths, advantages, experiment.alpha),
        ))
        if reject:
            continue
        sequences = [(prompt.token_ids, item.token_ids) for item in completions]
        missing = [index for index, item in enumerate(completions) if item.logprobs is None]
        if missing:
            recovered = policy.client.score([sequences[index] for index in missing], version=state.version)
            for index, scores in zip(missing, recovered, strict=True):
                completions[index] = replace(completions[index], logprobs=scores)
        ref_scores = reference.score(sequences, version=0)
        groups.append(Group(prompt, [
            Trajectory(completion, value, advantage, scores)
            for completion, value, advantage, scores in zip(
                completions, rewards, advantages, ref_scores, strict=True
            )
        ]))
        if len(groups) == config.prompts_per_step:
            break
    if len(groups) != config.prompts_per_step:
        return RolloutBatch(state.version, [])
    return RolloutBatch(state.version, groups)


def latest_checkpoint(root):
    return next((path for path in sorted(root.glob("step-*"), reverse=True)
                 if (path / "complete").exists()), None)


def save_evaluation_snapshot(root, trainer, tokenizer, version):
    if version in {125, 250, 375}:
        destination = root / "snapshots" / f"step-{version:06d}"
        if not (destination / "complete").is_file():
            export_model(trainer.model, tokenizer, destination)


def store_checkpoint(root, trainer, state, identity):
    destination = root / f"step-{state.version:06d}"
    if destination.exists():
        if (destination / "complete").is_file():
            return
        raise RuntimeError(f"An incomplete checkpoint occupies {destination}")
    previous = latest_checkpoint(root)
    for path in root.glob("step-*"):
        if path != previous and (path / "complete").is_file():
            shutil.rmtree(path)
    bytes_per_parameter = 12 if trainer.offload else 7
    require_space(root, sum(p.numel() for p in trainer.parameters) * bytes_per_parameter + 256 * 2**20)
    save_resume(destination, trainer, state, identity)


@contextmanager
def stop_on_signal():
    stop = threading.Event()
    signals = (signal.SIGINT, signal.SIGTERM)
    old = {item: signal.getsignal(item) for item in signals}
    for item in signals:
        signal.signal(item, lambda *_: stop.set())
    try:
        yield stop
    finally:
        for item, handler in old.items():
            signal.signal(item, handler)


def _truncate_uncommitted(path, key, boundary):
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = []
        for index, line in enumerate(lines):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break
                raise
            if value[key] <= boundary:
                kept.append(line)
        temporary = path.with_suffix(".repair")
        temporary.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        temporary.replace(path)


def train(study, experiment_name, seed, data, root, checkpoint_root, resume, gate=False,
          steps_override=None, max_new_steps=None):
    devices = hardware()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    from .trainer import Trainer

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("Expected one visible T4 in the trainer process")
    if "T4" not in devices[1]["name"]:
        raise RuntimeError("Expected a second physical T4 for both inference workers")
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if not resume and ((root / "run.json").exists() or latest_checkpoint(checkpoint_root)):
        raise FileExistsError("Run already exists. Use --resume or a fresh directory.")
    experiment = study.experiments[experiment_name]
    if seed not in experiment.seeds:
        raise ValueError("Seed is not registered for this experiment")
    if steps_override is not None:
        study.training = replace(study.training, steps=steps_override)
    manifest, rows, source, tokenizer, registry = _model_and_verifier(study, data, "curated")
    if len(rows) != 1500 or max(len(row.token_ids) for row in rows) > study.training.max_prompt:
        raise ValueError("Curated inputs do not match the study prompt limits")
    prompts = shuffled(rows, seed)
    random.seed(seed)
    torch.manual_seed(seed)
    identity = json.loads(json.dumps(dict(
        experiment=experiment_name, seed=seed, objective=asdict(experiment),
        training=asdict(study.training), data=manifest["prompts_sha256"],
        model=manifest["model_revision"], dataset=manifest["dataset_revision"],
        verifier=manifest["verifier_digest"],
    )))
    run_metadata = dict(
        identity=identity, engine=asdict(study.engine), devices=devices,
        versions={item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
        cuda=torch.version.cuda, source_version=os.environ.get("ALPHA_GRPO_GIT_REVISION"),
    )
    with ExitStack() as stack:
        monitor = stack.enter_context(ResourceMonitor(devices, root, checkpoint_root))
        stop = stack.enter_context(stop_on_signal())
        trainer = Trainer(source, tokenizer, study.training, experiment)
        state = RunState()
        if resume:
            previous_run = json.loads((root / "run.json").read_text(encoding="utf-8"))
            if previous_run["identity"] != identity:
                raise ValueError("Output directory belongs to a different experiment")
            previous = latest_checkpoint(checkpoint_root)
            if previous is None:
                raise FileNotFoundError("No complete checkpoint to resume")
            state = load_resume(previous, trainer, identity)
            _truncate_uncommitted(root / "rollouts.jsonl", "cursor", state.cursor)
            _truncate_uncommitted(root / "steps.jsonl", "version", state.version)
            for snapshot in (root / "snapshots").glob("step-*"):
                if snapshot.name.removeprefix("step-").isdigit() and int(snapshot.name[5:]) > state.version:
                    shutil.rmtree(snapshot)
        else:
            store_checkpoint(checkpoint_root, trainer, state, identity)
        write_json(root / "run.json", run_metadata)
        LOGGER.info("Starting %s, seed %d, from policy version %d", experiment_name, seed, state.version)
        save_evaluation_snapshot(root, trainer, tokenizer, state.version)
        target = study.training.steps
        if max_new_steps is not None:
            target = min(target, state.version + max_new_steps)
        start = inference_launcher(study, devices[1]["uuid"], source, root)
        policy = Policy(start=lambda path, version, transfer: start("policy", path, version, transfer),
                        source=source, tokenizer=tokenizer, trainer=trainer,
                        mode=study.engine.sync, timeout=study.engine.timeout_seconds, root=root)
        stack.callback(policy.close)
        policy.synchronize(state.version)
        reference = stack.enter_context(start("reference", source, 0, False))
        probe = [(prompts[0].token_ids, tokenizer.encode(" The answer is 4.", add_special_tokens=False))]
        baseline_ref = reference.score(probe, version=0)
        baseline_fingerprint = reference.call("fingerprint", version=0)
        if gate:
            policy_probe = policy.client.score(probe, version=state.version)[0]
            if state.version == 0 and any(abs(a - b) > 0.05 for a, b in
                                          zip(baseline_ref[0], policy_probe, strict=True)):
                raise RuntimeError("Initial policy and reference scores disagree")
            samples = policy.client.generate(
                prompts[0].token_ids, version=state.version, count=2, max_tokens=16,
                temperature=1, top_p=1, seed=seed,
            )
            sample_inputs = [(prompts[0].token_ids, item.token_ids) for item in samples]
            rescored = policy.client.score(sample_inputs, version=state.version)
            for item, scores in zip(samples, rescored, strict=True):
                if item.logprobs and any(abs(a - b) > 0.25 for a, b in
                                         zip(item.logprobs, scores, strict=True)):
                    raise RuntimeError("Sampled and teacher-forced policy scores disagree")
            try:
                policy.client.score(probe, version=state.version + 1)
            except InferenceError:
                pass
            else:
                raise RuntimeError("Policy accepted a stale version")
        while state.version < target and not stop.is_set():
            started = time.monotonic()
            batches = collect(study, experiment, seed, prompts, state, policy, reference,
                              tokenizer, registry, root)
            collected = time.monotonic()
            if not batches.groups:
                state.empty_attempts += 1
                state.pending_collection_seconds += collected - started
                if state.empty_attempts >= 3:
                    raise RuntimeError("Three consecutive collection attempts found no mixed groups")
                continue
            state.empty_attempts = 0
            metrics = trainer.update(batches)
            state.version += 1
            updated = time.monotonic()
            backend = policy.synchronize(state.version)
            synchronized = time.monotonic()
            if gate or state.version % 75 == 0:
                if (reference.score(probe, version=0) != baseline_ref
                        or reference.call("fingerprint", version=0) != baseline_fingerprint):
                    raise RuntimeError("Frozen reference probe changed")
            collection_seconds = collected - started + state.pending_collection_seconds
            state.pending_collection_seconds = 0
            append_json(root / "steps.jsonl", dict(
                **asdict(state), **metrics, accepted_groups=len(batches.groups),
                trajectories=batches.size, sync=backend, collect_seconds=collection_seconds,
                update_seconds=updated - collected, sync_seconds=synchronized - updated,
            ))
            if state.version % 75 == 0:
                store_checkpoint(checkpoint_root, trainer, state, identity)
            save_evaluation_snapshot(root, trainer, tokenizer, state.version)
            if state.version == 1 or state.version % 10 == 0:
                LOGGER.info("Step %d/%d, %d groups, sync %s, %.1fs",
                            state.version, target, len(batches.groups), backend,
                            synchronized - started)
        store_checkpoint(checkpoint_root, trainer, state, identity)
        if gate:
            if monitor.error:
                raise RuntimeError("Resource measurement failed") from monitor.error
            if min(monitor.minimum_free) < GIB:
                raise RuntimeError(f"Below 1 GiB sampled GPU headroom: {monitor.minimum_free}")
            steps_file = root / "steps.jsonl"
            steps = [json.loads(line) for line in steps_file.read_text(encoding="utf-8").splitlines()
                     ] if steps_file.is_file() else []
            step_seconds = [entry["collect_seconds"] + entry["update_seconds"] + entry["sync_seconds"]
                            for entry in steps]
            average = mean(step_seconds) if step_seconds else None
            write_json(root / "gate.json", dict(
                completed=state.version == study.training.steps, steps=state.version,
                minimum_gpu_free_bytes=monitor.minimum_free, backend=policy.mode,
                trainer_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                trainer_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                mean_step_seconds=average,
                projected_375_step_hours=375 * average / 3600 if average is not None else None,
            ))
