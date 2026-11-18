import argparse
import logging
import shutil
from dataclasses import replace
from pathlib import Path

from .config import load_study


def _parser():
    parser = argparse.ArgumentParser(prog="alpha-grpo")
    parser.add_argument("--config", type=Path, default=Path("configs/study.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare-data", help="Normalize data and audit benchmark overlap")
    preparation.add_argument("--out", type=Path, required=True)
    preparation.add_argument("--verifier-revision", default="main")
    screening = commands.add_parser("prescreen", help="Build the shared 1,500-prompt manifest")
    screening.add_argument("--data", type=Path, required=True)
    screening.add_argument("--out", type=Path, required=True)
    screening.add_argument("--seed", type=int, default=42)
    for command in ("preflight", "smoke", "train"):
        task = commands.add_parser(command)
        task.add_argument("--data", type=Path, required=True)
        task.add_argument("--out", type=Path, required=True)
        task.add_argument("--checkpoint-root", type=Path)
        task.add_argument("--experiment", default="control-a05")
        task.add_argument("--seed", type=int, default=42)
        task.add_argument("--sync", choices=("auto", "nccl", "checkpoint"))
        task.add_argument("--optimizer", choices=("gpu", "cpu"))
        task.add_argument("--microbatch", type=int, choices=(1, 2))
        task.add_argument("--resume", action="store_true")
        task.add_argument("--max-new-steps", type=int)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--revision", default="main")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument("--tasks", nargs="+",
                            choices=("gsm8k_cot", "math500", "ifeval", "mmlu", "truthfulqa_mc2"))
    evaluation.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    evaluation.add_argument("--greedy-only", action="store_true")
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--label")
    evaluation.add_argument("--training-seed", type=int)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--runs", type=Path, nargs="*", default=[])
    analysis.add_argument("--evaluations", type=Path, nargs="*", default=[])
    analysis.add_argument("--baseline", type=Path)
    analysis.add_argument("--out", type=Path, required=True)
    export = commands.add_parser("export", help="Copy a complete model snapshot for distribution")
    export.add_argument("--snapshot", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--hub-repo")
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "evaluate":
        from .evaluation import TASKS, evaluate

        evaluate(args.model, args.revision, args.data, args.out, args.tasks or list(TASKS),
                 args.device, args.greedy_only, args.limit, args.label, args.training_seed)
        return
    if args.command == "analyze":
        from .analysis import analyze

        if not args.runs and not args.evaluations:
            parser.error("Provide training runs or evaluations")
        analyze(args.runs, args.evaluations, args.baseline, args.out)
        return
    if args.command == "export":
        from .checkpoint import atomic_directory

        if not (args.snapshot / "complete").is_file():
            parser.error("Provide a complete evaluation snapshot")
        with atomic_directory(args.out) as temporary:
            shutil.copytree(args.snapshot, temporary, dirs_exist_ok=True)
        if args.hub_repo:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(args.hub_repo, repo_type="model", exist_ok=True, private=True)
            api.upload_folder(repo_id=args.hub_repo, repo_type="model", folder_path=args.out)
        return
    study = load_study(args.config)
    if args.command == "prepare-data":
        from .data import prepare

        prepare(study, args.out, args.verifier_revision)
    elif args.command == "prescreen":
        from .campaign import prescreen

        prescreen(study, args.data, args.out, args.seed)
    else:
        from .campaign import train

        if args.experiment not in study.experiments:
            parser.error(f"Unknown experiment: {args.experiment}")
        if args.max_new_steps is not None and args.max_new_steps < 1:
            parser.error("--max-new-steps must be positive")
        for option in ("optimizer", "microbatch"):
            value = getattr(args, option)
            if value is not None:
                study.training = replace(study.training, **{option: value})
        if args.sync is not None:
            study.engine = replace(study.engine, sync=args.sync)
        train(study, args.experiment, args.seed, args.data, args.out,
              args.checkpoint_root or args.out / "checkpoints", args.resume,
              gate=args.command != "train",
              steps_override={"preflight": 1, "smoke": 30}.get(args.command),
              max_new_steps=args.max_new_steps)
