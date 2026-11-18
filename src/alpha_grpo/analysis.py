import json
import random
from collections import defaultdict
from statistics import mean, stdev

from .checkpoint import atomic_directory


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def paired_interval(candidate, baseline, *, draws=1000):
    if candidate.keys() != baseline.keys() or not candidate:
        raise ValueError("Paired evaluation requires identical, nonempty benchmark item sets")
    differences = [candidate[key] - baseline[key] for key in sorted(candidate)]
    rng = random.Random(42)
    estimates = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(draws))
    return [estimates[int(0.025 * draws)], estimates[min(draws - 1, int(0.975 * draws))]]


def absolute_interval(scores):
    return paired_interval(scores, dict.fromkeys(scores, 0.0))


def _plot(plt, output, name, x_label, y_label, populate):
    figure, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    populate(axis)
    axis.set(xlabel=x_label, ylabel=y_label)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=8)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"{name}.{extension}", dpi=180)
    plt.close(figure)


def analyze(runs, evaluations, baseline, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyarrow as pa
    import pyarrow.parquet as pq

    eval_summary = []
    seed_scores = defaultdict(dict)
    baseline_groups = defaultdict(dict)
    baseline_protocol = None
    if baseline is not None:
        baseline_protocol = json.loads((baseline / "protocol.json").read_text(encoding="utf-8"))
        for row in jsonl(baseline / "items.jsonl"):
            key = (row["subtask"], row["item"])
            if key in baseline_groups[row["task"], row["mode"]]:
                raise ValueError("Duplicate baseline evaluation item")
            baseline_groups[row["task"], row["mode"]][row["subtask"], row["item"]] = row["score"]
    for directory in evaluations:
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        if baseline_protocol is not None:
            for key in ("datasets", "limit", "fewshot_seed", "max_new_tokens"):
                if baseline_protocol[key] != protocol[key]:
                    raise ValueError(f"Paired comparison has mismatched {key} in {directory}")
        grouped = defaultdict(list)
        for row in jsonl(directory / "items.jsonl"):
            grouped[row["task"], row["mode"]].append(row)
        for (task, mode), items in grouped.items():
            scores = {(row["subtask"], row["item"]): row["score"] for row in items}
            if len(scores) != len(items):
                raise ValueError(f"Duplicate evaluation items in {directory}")
            lengths = [row["output_tokens"] for row in items if row["output_tokens"] is not None]
            score = mean(scores.values())
            comparison = baseline_groups.get((task, mode))
            if baseline is not None and comparison is None:
                raise ValueError(f"Baseline lacks {task} / {mode}")
            row = dict(label=protocol["label"], evaluation=str(directory), task=task, mode=mode,
                       score=score, absolute_95=absolute_interval(scores), items=len(items), mean_output_tokens=mean(lengths) if lengths else None,
                       delta=score - mean(comparison.values()) if comparison else None,
                       paired_95=paired_interval(scores, comparison) if comparison else None)
            eval_summary.append(row)
            if protocol["training_seed"] is not None:
                identity = (protocol["label"], protocol.get("checkpoint_step"), task, mode)
                if protocol["training_seed"] in seed_scores[identity]:
                    raise ValueError(f"Duplicate training seed for {identity}")
                seed_scores[identity][protocol["training_seed"]] = score
    for row in eval_summary:
        row["pareto"] = (
            not any(other["task"] == row["task"] and other["mode"] == row["mode"]
                    and other["mean_output_tokens"] is not None
                    and other["mean_output_tokens"] <= row["mean_output_tokens"]
                    and other["score"] >= row["score"]
                    and (other["score"] > row["score"]
                         or other["mean_output_tokens"] < row["mean_output_tokens"])
                    for other in eval_summary)
            if row["mean_output_tokens"] is not None else None
        )

    run_summary = []
    trajectories = []
    drift = defaultdict(list)
    for directory in runs:
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        label = f"{run['identity']['experiment']} / {run['identity']['seed']}"
        steps = jsonl(directory / "steps.jsonl")
        groups = jsonl(directory / "rollouts.jsonl")
        accepted = [group for group in groups if not group["rejected"]]
        mixed = [group for group in accepted if not group["constant"]]
        total_seconds = sum(
            step["collect_seconds"] + step["update_seconds"] + step["sync_seconds"] for step in steps
        )
        run_summary.append(dict(
            label=label, updates=len(steps), candidate_groups=len(groups), accepted_groups=len(accepted),
            unique_prompts=len({group["source_id"] for group in groups}),
            generated_responses=sum(len(group["lengths"]) for group in groups),
            generated_tokens=sum(sum(group["lengths"]) for group in groups),
            constant_group_rate=mean(group["constant"] for group in groups) if groups else None,
            sync_seconds=sum(step["sync_seconds"] for step in steps), step_seconds=total_seconds,
            accepted_groups_per_hour=3600 * len(accepted) / total_seconds if total_seconds else None,
            mixed_groups_per_hour=3600 * len(mixed) / total_seconds if total_seconds else None,
        ))
        for group in groups:
            for length, value, share in zip(
                group["lengths"], group["rewards"], group["gradient_weight_share"], strict=True,
            ):
                trajectories.append(dict(
                    run=label, version=group["version"], task=group["task"], reward=value,
                    length=length, share=share, rejected=group["rejected"], source_id=group["source_id"],
                ))
                drift[label, group["task"], value, (group["version"] // 25) * 25].append(length)

    variability = [
        dict(label=label, checkpoint_step=step, task=task, mode=mode, seeds=scores,
             standard_deviation=stdev(scores.values()) if len(scores) > 1 else None)
        for (label, step, task, mode), scores in seed_scores.items()
    ]
    with atomic_directory(output) as temporary:
        (temporary / "summary.json").write_text(json.dumps(
            dict(evaluations=eval_summary, training_seed_variability=variability, runs=run_summary),
            indent=2, allow_nan=False,
        ), encoding="utf-8")
        if trajectories:
            pq.write_table(pa.Table.from_pylist(trajectories), temporary / "trajectories.parquet")

        def frontier(axis):
            for task in sorted({row["task"] for row in eval_summary}):
                rows = [row for row in eval_summary if row["task"] == task and row["mode"] == "greedy"
                        and row["mean_output_tokens"] is not None]
                if rows:
                    axis.scatter([row["mean_output_tokens"] for row in rows],
                                 [row["score"] for row in rows], label=task)
                for row in rows:
                    if row["pareto"]:
                        axis.annotate(row["label"], (row["mean_output_tokens"], row["score"]), fontsize=7)

        def length_drift(axis):
            for label, task, reward in sorted({key[:3] for key in drift}):
                points = sorted((key[3], mean(values)) for key, values in drift.items()
                                if key[:3] == (label, task, reward))
                axis.plot([point[0] for point in points], [point[1] for point in points],
                          label=f"{label} / {task} / reward {reward:g}")

        def weight_proxy(axis):
            buckets = defaultdict(list)
            for row in trajectories:
                if row["share"] is not None and not row["rejected"]:
                    buckets[row["run"], ((row["length"] - 1) // 32) * 32 + 16].append(row["share"])
            for label in sorted({key[0] for key in buckets}):
                points = sorted((length, mean(values)) for (run, length), values in buckets.items()
                                if run == label)
                axis.plot([point[0] for point in points], [point[1] for point in points],
                          marker=".", label=label)

        def theoretical_share(axis):
            ratios = list(range(1, 101))
            for alpha in (0.0, 0.5, 1.0):
                axis.plot(ratios, [ratio ** alpha / (1 + ratio ** alpha) for ratio in ratios],
                          label=f"alpha={alpha:g}")

        _plot(plt, temporary, "two-response-weight-share", "Long / short response length",
              "Long-response weight share at equal absolute advantage", theoretical_share)

        def efficiency(axis):
            for run in run_summary:
                if run["mixed_groups_per_hour"] is not None:
                    axis.scatter(run["generated_tokens"], run["mixed_groups_per_hour"], label=run["label"])

        _plot(plt, temporary, "accuracy-length", "Mean output tokens", "Accuracy", frontier)
        _plot(plt, temporary, "length-drift", "Policy updates (25-step bins)", "Mean response tokens", length_drift)
        _plot(plt, temporary, "length-weight-proxy", "Response tokens (bin center)", "Mean weight share", weight_proxy)
        _plot(plt, temporary, "filter-efficiency", "All generated tokens",
              "Mixed-reward groups / hour including synchronization", efficiency)
