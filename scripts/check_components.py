"""Probe training, inference, and policy synchronization on Kaggle."""

import argparse
import json
import logging
import os
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from alpha_grpo.campaign import hardware, inference_launcher
from alpha_grpo.config import load_study
from alpha_grpo.data import model_snapshot
from alpha_grpo.records import Completion, Group, PromptRecord, RolloutBatch, Trajectory
from alpha_grpo.sync import Policy


def load_probe_trainer(study, source, tokenizer, optimizer, *, sync_probe):
    import torch

    from alpha_grpo.trainer import Trainer

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("Expected one visible T4 for the trainer check")
    config = replace(study.training, optimizer=optimizer or study.training.optimizer)
    if sync_probe:
        # Make the diagnostic update visible in the FP16 weight fingerprint.
        config = replace(config, steps=1, lr=1e-4)
    return Trainer(source, tokenizer, config, study.experiments["control-a05"])


def update_probe(trainer, prompt, answers, old_scores, reference_scores):
    records = [
        Trajectory(Completion(ids, old, "probe"), reward, reward - 0.5, reference)
        for ids, old, reference, reward in zip(
            answers, old_scores, reference_scores, (1.0, 0.0), strict=True,
        )
    ]
    group = Group(PromptRecord("probe", "gsm8k", [], prompt, {}), records)
    metrics = trainer.update(RolloutBatch(0, [group]))
    if metrics["gradient_norm"] <= 0:
        raise RuntimeError("Expected a nonzero gradient for the probe batch")
    return {"optimizer": trainer.training.optimizer, "trajectories": len(records), **metrics}


def check_sync(study, source, tokenizer, trainer, prompt, answers, mode, out, gpu):
    start = inference_launcher(study, gpu["uuid"], source, out)
    policy = Policy(
        start=lambda path, version, transfer: start("policy", path, version, transfer),
        source=source, tokenizer=tokenizer, trainer=trainer, mode=mode,
        timeout=study.engine.timeout_seconds, root=out,
    )
    pairs = [(prompt, answer) for answer in answers]
    with ExitStack() as stack:
        stack.callback(policy.close)
        policy.synchronize(0)
        reference = stack.enter_context(start("reference", source, 0, False))
        initial_fingerprint = policy.client.call("fingerprint", version=0)
        reference_fingerprint = reference.call("fingerprint", version=0)
        old_scores = policy.client.score(pairs, version=0)
        reference_scores = reference.score(pairs, version=0)
        metrics = update_probe(trainer, prompt, answers, old_scores, reference_scores)
        backend = policy.synchronize(1)
        if policy.client.call("fingerprint", version=1) == initial_fingerprint:
            raise RuntimeError("The probe update did not change the sampled FP16 policy weights")
        if (reference.call("fingerprint", version=0) != reference_fingerprint
                or reference.score(pairs, version=0) != reference_scores):
            raise RuntimeError("The frozen reference changed during the probe")
        sample, = policy.client.generate(
            prompt, version=1, count=1, max_tokens=min(16, study.training.max_completion),
            temperature=1.0, top_p=1.0, seed=42,
        )
        return {
            "gpu": gpu, "logs": str(out), "sync": backend,
            "policy_version": policy.acknowledged_version, "reference_unchanged": True,
            "completion": tokenizer.decode(sample.token_ids, skip_special_tokens=True),
            "completion_tokens": len(sample.token_ids), **metrics,
        }


def check_inference(study, source, tokenizer, prompt, known_answer, component, out, gpu):
    start = inference_launcher(study, gpu["uuid"], source, out)
    report = {"gpu": gpu, "logs": str(out)}
    with ExitStack() as stack:
        policy = (stack.enter_context(start("policy", source, 0, False))
                  if component in {"policy", "both"} else None)
        reference = (stack.enter_context(start("reference", source, 0, False))
                     if component in {"reference", "both"} else None)
        answer = known_answer
        if policy is not None:
            sample, = policy.generate(prompt, version=0, count=1,
                                      max_tokens=min(16, study.training.max_completion),
                                      temperature=1.0, top_p=1.0, seed=42)
            answer = sample.token_ids
            if not answer:
                raise RuntimeError("Policy returned an empty token sequence")
            report["completion"] = tokenizer.decode(answer, skip_special_tokens=True)
            report["sampled_logprobs_available"] = sample.logprobs is not None
            report["policy_logprobs"], = policy.score([(prompt, answer)], version=0)
        if reference is not None:
            report["reference_logprobs"], = reference.score([(prompt, answer)], version=0)
        if component == "both":
            report["max_logprob_difference"] = max(
                abs(policy_score - reference_score)
                for policy_score, reference_score in zip(
                    report["policy_logprobs"], report["reference_logprobs"], strict=True,
                )
            )
        report["completion_tokens"] = len(answer)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("trainer", "policy", "reference", "both", "sync"))
    parser.add_argument("--config", type=Path, default=Path("configs/study.yaml"))
    parser.add_argument("--out", type=Path, help="Directory for worker logs")
    parser.add_argument("--optimizer", choices=("gpu", "cpu"), help="Trainer and sync checks")
    parser.add_argument("--sync", choices=("checkpoint", "nccl"),
                        help="Transfer backend for the sync check (default: checkpoint)")
    args = parser.parse_args()
    uses_trainer = args.component in {"trainer", "sync"}
    if args.optimizer and not uses_trainer:
        parser.error("--optimizer requires the trainer or sync check")
    if args.sync and args.component != "sync":
        parser.error("--sync requires the sync check")
    if args.component == "trainer" and args.out is not None:
        parser.error("--out is for inference worker logs")
    if uses_trainer:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logging.basicConfig(level=logging.INFO)
    study = load_study(args.config)
    gpu = hardware()[1] if args.component != "trainer" else None
    if gpu is not None and "T4" not in gpu["name"]:
        raise RuntimeError("Expected a T4 as the second physical GPU")
    revision, source = model_snapshot(study.model, study.model_revision)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 2?"}], tokenize=True,
        add_generation_prompt=True, return_dict=False,
    )
    answers = [tokenizer.encode(text, add_special_tokens=False)
               for text in ("The answer is 4.", "The answer is 7.")]
    if uses_trainer:
        trainer = load_probe_trainer(study, source, tokenizer, args.optimizer,
                                     sync_probe=args.component == "sync")
    if args.component == "trainer":
        scores = trainer.score([(prompt, answer) for answer in answers])
        report = update_probe(trainer, prompt, answers, scores, scores)
    else:
        mode = args.sync or "checkpoint"
        name = f"sync-{mode}" if args.component == "sync" else args.component
        out = args.out or Path("/kaggle/working/alpha-grpo/component-checks") / name
        print(f"Worker logs: {out}", flush=True)
        if args.component == "sync":
            report = check_sync(study, source, tokenizer, trainer, prompt, answers, mode, out, gpu)
        else:
            report = check_inference(study, source, tokenizer, prompt, answers[0], args.component, out, gpu)
    print(json.dumps({"component": args.component, "model_revision": revision, **report},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
