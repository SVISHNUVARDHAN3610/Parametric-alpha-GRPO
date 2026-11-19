import hashlib
import inspect
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from urllib.request import Request, urlopen

from .checkpoint import atomic_directory
from .records import PromptRecord
from .verifiers import constraint_specs, load_registry, parse_gold

BENCHMARKS = (
    ("openai/gsm8k", "main", "test", "question"),
    ("HuggingFaceH4/MATH-500", None, "test", "problem"),
    ("google/IFEval", None, "train", "prompt"),
)
MATH_SOURCE = "SevenAIGmbH/MATH-lighteval"
TASKS = ("gsm8k", "math", "ifeval")
OOD_BENCHMARKS = ("cais/mmlu", "truthfulqa/truthful_qa")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def final_question(text):
    # The Ai2 math and GSM8K rows can include worked examples in one message.
    fields = re.split(r"(?m)^\s*Question:\s*", text)
    return re.split(r"(?m)^\s*Answer:\s*", fields[-1], maxsplit=1)[0].strip()


def bit_profile(ids):
    masks = defaultdict(int)
    for index, token in enumerate(ids):
        masks[token] |= 1 << index
    return masks


def lcs_length(ids, profile):
    row = 0
    for token in ids:
        present = row | profile.get(token, 0)
        row = present & ~(present - ((row << 1) | 1))
    return row.bit_count()


def is_contaminated(ids, benchmarks):
    counts = Counter(ids)
    for size, profile in benchmarks:
        minimum = min(size, len(ids))
        threshold = max(13, (3 * minimum + 4) // 5)
        if minimum < threshold:
            continue
        # Multiset overlap is an upper bound on LCS.
        overlap = sum(min(count, profile.get(token, 0).bit_count())
                      for token, count in counts.items())
        if overlap >= threshold and lcs_length(ids, profile) >= threshold:
            return True
    return False


def model_snapshot(model, revision):
    from huggingface_hub import HfApi, snapshot_download

    sha = HfApi().model_info(model, revision=revision).sha
    path = snapshot_download(
        model, revision=sha,
        allow_patterns=[
            "*.json", "*.safetensors", "pytorch_model.bin", "pytorch_model-*.bin",
            "*.txt", "*.model", "*.jinja",
        ],
    )
    snapshot = Path(path)
    if not any(file.is_file() for pattern in ("*.safetensors", "pytorch_model*.bin")
               for file in snapshot.glob(pattern)):
        raise FileNotFoundError(f"No model weights found in snapshot for {model}@{sha}: {path}")
    return sha, path


def read_metadata(root):
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def read_prompts(root):
    return [PromptRecord(**json.loads(line)) for line in
            (root / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]


def write_prompts(root, rows):
    with (root / "prompts.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _dataset(repo, name, split, *, revision="main"):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(repo, revision=revision).sha
    return sha, load_dataset(repo, name, split=split, revision=sha)


def _verifier_source(path, revision):
    if revision == "main":
        url = "https://api.github.com/repos/allenai/open-instruct/commits/main"
        with urlopen(Request(url, headers={"User-Agent": "alpha-grpo"}), timeout=30) as response:
            revision = json.load(response)["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full Ai2 open-instruct commit SHA")
    url = f"https://raw.githubusercontent.com/allenai/open-instruct/{revision}/open_instruct/if_functions.py"
    with urlopen(url, timeout=30) as response:
        path.write_bytes(response.read())
    return revision


def prepare(study, output, verifier_revision):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    model_sha, source = model_snapshot(study.model, study.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(source)
    dataset_sha, rows = _dataset(study.dataset, None, "train", revision=study.dataset_revision)
    math_sha, originals = _dataset(MATH_SOURCE, "default", "train")
    levels = defaultdict(list)
    for row in originals:
        levels[normalize(row["problem"])].append(row["level"])

    def tokenize(text):
        return tokenizer.encode(normalize(text), add_special_tokens=False)

    benchmark_revisions = {}
    benchmarks = []
    for repo, name, split, field in BENCHMARKS:
        sha, items = _dataset(repo, name, split)
        benchmark_revisions[repo] = sha
        for item in items:
            ids = tokenize(item[field])
            benchmarks.append((len(ids), bit_profile(ids)))
    for repo in OOD_BENCHMARKS:
        benchmark_revisions[repo] = HfApi().dataset_info(repo).sha

    with atomic_directory(output) as temporary:
        verifier_file = temporary / "if_functions.py"
        verifier_sha = _verifier_source(verifier_file, verifier_revision)
        verifier_digest = digest(verifier_file)
        registry = load_registry(verifier_file, verifier_digest)
        audit = defaultdict(Counter)
        prepared = []
        for index, row in enumerate(rows):
            task = row["dataset"].strip().lower()
            if task not in TASKS:
                raise ValueError(f"Unknown dataset tag {row['dataset']!r} on row {index}")
            messages = [dict(message) for message in row["messages"]]
            content = messages[-1]["content"]
            if task != "ifeval":
                content = final_question(content)
                messages = [dict(role="user", content=content)]
            if task == "math" and levels[normalize(content)] not in (
                ["Level 1"], ["Level 2"], ["Level 3"],
            ):
                audit[task]["level_or_join"] += 1
                continue
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False,
            )
            if not encoded or len(encoded) > study.training.max_prompt:
                audit[task]["prompt_length"] += 1
                continue
            if is_contaminated(tokenize(content), benchmarks):
                audit[task]["lcs"] += 1
                continue
            if task == "ifeval":
                specs = constraint_specs(row)
                for item in specs:
                    if item["func_name"] not in registry:
                        raise ValueError(f"Unsupported constraint {item['func_name']!r} on row {index}")
                    inspect.signature(registry[item["func_name"]]).bind("", **{
                        key: value for key, value in item.items()
                        if key != "func_name" and value is not None
                    })
                verifier = dict(constraints=specs)
            else:
                answer = row["ground_truth"].strip()
                if not parse_gold(answer):
                    audit[task]["unparseable_gold"] += 1
                    continue
                verifier = dict(answer=answer)
            prepared.append(PromptRecord(f"{dataset_sha}:{index}", task, messages, encoded, verifier))
            audit[task]["eligible"] += 1
        write_prompts(temporary, prepared)
        manifest = dict(
            dataset=study.dataset, dataset_revision=dataset_sha, model=study.model, model_revision=model_sha,
            math_source=MATH_SOURCE, math_revision=math_sha, benchmarks=benchmark_revisions,
            verifier_revision=verifier_sha, verifier_digest=verifier_digest, audit=audit,
            stage="prepared", prompts_sha256=digest(temporary / "prompts.jsonl"),
        )
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def shuffled(rows, seed):
    result = list(rows)
    random.Random(seed).shuffle(result)
    return result


def load_curated(root, stage):
    manifest = read_metadata(root)
    if manifest["stage"] != stage or manifest["prompts_sha256"] != digest(root / "prompts.jsonl"):
        raise ValueError(f"Invalid or modified {stage} dataset manifest")
    return manifest, read_prompts(root)
