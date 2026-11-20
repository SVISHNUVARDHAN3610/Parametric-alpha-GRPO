import json
import random
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path


@contextmanager
def atomic_directory(destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        yield temporary
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def export_model(model, tokenizer, destination):
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    require_space(destination.parent, sum(p.numel() for p in model.parameters()) * 2 + 64 * 2**20)
    with atomic_directory(destination) as temporary:
        state = {}
        converted = {}
        for name, value in model.state_dict(keep_vars=True).items():
            # Preserve tied parameters so save_pretrained can serialize one copy.
            if id(value) not in converted:
                dtype = torch.float16 if value.is_floating_point() else value.dtype
                converted[id(value)] = value.detach().to(device="cpu", dtype=dtype)
            state[name] = converted[id(value)]
        model.save_pretrained(temporary, state_dict=state, max_shard_size="2GB",
                              safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        (temporary / "complete").touch()


def require_space(path: Path, requested: int):
    if shutil.disk_usage(path).free < requested:
        raise RuntimeError(f"Insufficient free disk at {path} for {requested / 2**30:.2f} GiB")


def save_resume(path, trainer, sampler_state, identity):
    import torch

    with atomic_directory(path) as temporary:
        torch.save(dict(
            model=None if trainer.offload else trainer.model.state_dict(),
            optimizer=trainer.optimizer.state_dict(),
            masters=None if trainer.master_parameters is None else
            [parameter.detach() for parameter in trainer.master_parameters],
            scheduler=trainer.scheduler.state_dict(),
            scale=trainer.scale, scale_growth=trainer.scale_growth,
            python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),
        ), temporary / "training.pt")
        (temporary / "metadata.json").write_text(
            json.dumps(dict(identity=identity, state=asdict(sampler_state)), indent=2),
            encoding="utf-8",
        )
        (temporary / "complete").touch()


def load_resume(path, trainer, identity):
    import torch

    from .records import RunState

    if not (path / "complete").is_file():
        raise ValueError(f"Incomplete checkpoint: {path}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["identity"] != identity:
        raise ValueError("Checkpoint configuration or dataset does not match this run")
    # Restore only checkpoints created by this repository and trusted by the user.
    saved = torch.load(path / "training.pt", map_location="cpu", weights_only=True)
    model_state = saved.pop("model")
    if model_state is not None:
        trainer.model.load_state_dict(model_state)
    if trainer.master_parameters is not None:
        with torch.no_grad():
            for parameter, gpu, value in zip(
                trainer.master_parameters, trainer.parameters, saved.pop("masters"), strict=True,
            ):
                parameter.copy_(value)
                gpu.copy_(parameter)
    trainer.optimizer.load_state_dict(saved.pop("optimizer"))
    trainer.scheduler.load_state_dict(saved["scheduler"])
    trainer.scale = saved["scale"]
    trainer.scale_growth = saved["scale_growth"]
    random.setstate(saved["python_rng"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    return RunState(**metadata["state"])
