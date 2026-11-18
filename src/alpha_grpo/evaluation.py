import json
import hashlib
import os
from pathlib import Path

from .checkpoint import atomic_directory
from .data import model_snapshot, read_metadata
from .verifiers import math_reward

TASKS = {
    "gsm8k_cot": (8, "openai/gsm8k", "exact_match"),
    "math500": (0, "HuggingFaceH4/MATH-500", "exact_match"),
    "ifeval": (0, "google/IFEval", "prompt_level_strict_acc"),
    "mmlu": (5, "cais/mmlu", "acc"),
    "truthfulqa_mc2": (0, "truthfulqa/truthful_qa", "acc"),
}
SAMPLED_SEEDS = (42, 1337, 2026)


def math_results(doc, results):
    return {"exact_match": math_reward(doc["answer"], results[0], allow_text=True)}


def evaluate(model, revision, data, output, tasks, device, greedy_only, limit,
             label=None, training_seed=None):
    from .campaign import hardware

    if device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = hardware()[0]["uuid"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from huggingface_hub import HfApi
    from lm_eval import simple_evaluate
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable

    manifest = read_metadata(data)
    if Path(model).is_dir():
        source = str(Path(model).resolve())
        fingerprint = hashlib.sha256()
        for path in sorted(Path(source).glob("*.safetensors")):
            fingerprint.update(path.name.encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    fingerprint.update(chunk)
        model_revision = fingerprint.hexdigest()
    else:
        if model == manifest["model"] and revision == "main":
            revision = manifest["model_revision"]
        model_revision, source = model_snapshot(model, revision)
    benchmarks = dict(manifest["benchmarks"])
    for name in tasks:
        repo = TASKS[name][1]
        if repo not in benchmarks:
            benchmarks[repo] = HfApi().dataset_info(repo).sha

    class PinnedTasks(TaskManager):
        def _get_config(self, name):
            config = super()._get_config(name)
            repo = config.get("dataset_path")
            if repo in benchmarks:
                config["dataset_kwargs"] = dict(config.get("dataset_kwargs") or {},
                                                revision=benchmarks[repo])
            return config

    class StrictHFLM(HFLM):
        def tok_batch_encode(self, strings, padding_side="left", left_truncate_len=None,
                             truncation=False):
            ids, masks = super().tok_batch_encode(strings, padding_side, None, False)
            if left_truncate_len and ids.shape[1] > left_truncate_len:
                raise ValueError(f"Evaluation prompt exceeds available context ({left_truncate_len})")
            return ids, masks

        def _loglikelihood_tokens(self, requests, *args, **kwargs):
            if any(len(context) + len(completion) - 1 > self.max_length
                   for _, context, completion in requests):
                raise ValueError("Evaluation choice exceeds model context")
            return super()._loglikelihood_tokens(requests, *args, **kwargs)

    lm = StrictHFLM(pretrained=source, dtype=torch.float16 if device == "gpu" else torch.float32,
                    device="cuda:0" if device == "gpu" else "cpu", batch_size=1, max_length=4096,
                    truncation=False)
    manager = PinnedTasks()
    with atomic_directory(output) as temporary:
        (temporary / "protocol.json").write_text(json.dumps(dict(
            model=model, model_revision=model_revision, datasets=benchmarks, limit=limit,
            tasks=tasks, device=device, fewshot_seed=42, max_new_tokens=512,
            label=label or output.name, training_seed=training_seed,
            checkpoint_step=(int(Path(model).name[5:]) if Path(model).name.startswith("step-")
                             and Path(model).name[5:].isdigit() else None),
        ), indent=2), encoding="utf-8")
        with (temporary / "items.jsonl").open("w", encoding="utf-8") as stream:
            for task in tasks:
                fewshot, _, metric = TASKS[task]
                generation = task in {"gsm8k_cot", "math500", "ifeval"}
                for sample_seed in ((None, *SAMPLED_SEEDS) if generation and not greedy_only else (None,)):
                    options = None
                    if generation:
                        options = dict(do_sample=sample_seed is not None, max_gen_toks=512)
                        if sample_seed is not None:
                            options.update(temperature=0.2, top_p=1.0, top_k=0)
                    selected = task
                    if task == "math500":
                        selected = ConfigurableTask(config=dict(
                            task="math500", dataset_path=TASKS[task][1], test_split="test",
                            dataset_kwargs=dict(revision=benchmarks[TASKS[task][1]]),
                            output_type="generate_until", doc_to_text="problem",
                            doc_to_target="answer", num_fewshot=0,
                            generation_kwargs=dict(until=[], do_sample=False, temperature=0.0),
                            process_results=math_results,
                            metric_list=[dict(metric="exact_match", aggregation="mean",
                                              higher_is_better=True)],
                        ))
                    result = simple_evaluate(
                        model=lm, tasks=[selected], task_manager=manager, num_fewshot=fewshot,
                        apply_chat_template=True, fewshot_as_multiturn=True, limit=limit,
                        log_samples=True, bootstrap_iters=1000, gen_kwargs=options,
                        random_seed=42, numpy_random_seed=42, fewshot_random_seed=42,
                        torch_random_seed=sample_seed or 42,
                    )
                    samples = result.pop("samples")
                    mode = "greedy" if sample_seed is None else f"sample-{sample_seed}"
                    for subtask, items in samples.items():
                        for item in items:
                            if generation and item["filter"] not in {"strict-match", "none"}:
                                continue
                            response = item["resps"][0][0] if generation else None
                            token_count = (len(lm.tokenizer.encode(response, add_special_tokens=False))
                                           if isinstance(response, str) else None)
                            row = dict(task=task, subtask=subtask, mode=mode,
                                       item=item["doc_id"], score=float(item[metric]),
                                       output_tokens=token_count)
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                    (temporary / f"{task}-{mode}.json").write_text(json.dumps(
                        dict(results=result, samples=samples), default=handle_non_serializable,
                    ), encoding="utf-8")
