import hashlib
import importlib.util
import json
import re
from functools import lru_cache
from pathlib import Path


def constraint_specs(row):
    for field in ("ground_truth", "constraint"):
        value = row.get(field)
        try:
            decoded = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            decoded = [decoded]
        if isinstance(decoded, list) and decoded and all(isinstance(item, dict) for item in decoded):
            return [
                {"func_name": row.get("constraint_type"), **item} for item in decoded
            ]
    raise ValueError("Instruction constraint arguments are missing or malformed")


def load_registry(path: Path, digest: str):
    import langdetect

    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Instruction verifier source fingerprint mismatch")
    spec = importlib.util.spec_from_file_location("alpha_grpo_upstream_if", path)
    module = importlib.util.module_from_spec(spec)
    langdetect.DetectorFactory.seed = 0
    spec.loader.exec_module(module)
    return module.IF_FUNCTIONS_MAP


@lru_cache(maxsize=4096)
def parse_gold(answer):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse

    options = dict(fallback_mode="no_fallback", extraction_mode="first_match", parsing_timeout=3)
    if "$" not in answer and ("\\" in answer or any(char in answer for char in "^_={}")):
        parsed = parse(f"${answer}$", extraction_config=(LatexExtractionConfig(),), **options)
        if parsed:
            return parsed
    return parse(answer, extraction_config=(LatexExtractionConfig(), ExprExtractionConfig()), **options)


def math_reward(answer, response, *, allow_text=False):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    gold = parse_gold(answer)
    if not gold:
        if allow_text:
            def clean(text):
                text = re.sub(r"^\s*(?:final answer|answer):\s*", "", text, flags=re.I)
                text = text.strip().strip("$").strip().removesuffix(".")
                match = re.fullmatch(r"\\(?:boxed|text)\{(.*)\}", text)
                if match:
                    text = match[1]
                return " ".join(text.strip().casefold().split())

            boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", response)
            lines = response.strip().splitlines()
            candidate = boxed[-1] if boxed else lines[-1] if lines else ""
            return float(bool(clean(answer)) and clean(answer) == clean(candidate))
        raise ValueError(f"Unparseable math label: {answer!r}")
    if "\\boxed" in response:
        candidate = response[response.rfind("\\boxed"):]
        extraction = (LatexExtractionConfig(boxed_match_priority=0),)
    else:
        lines = response.strip().splitlines()
        candidate = lines[-1] if lines else ""
        extraction = (LatexExtractionConfig(), ExprExtractionConfig())
    predicted = parse(candidate, extraction_config=extraction, fallback_mode="no_fallback",
                      extraction_mode="first_match", parsing_timeout=3)
    return float(bool(predicted) and verify(gold, predicted, timeout_seconds=3))


def reward(prompt, response, registry):
    if prompt.task != "ifeval":
        return math_reward(prompt.verifier["answer"], response)
    return float(bool(response.strip()) and all(
        registry[item["func_name"]](response, **{
            key: value for key, value in item.items() if key != "func_name" and value is not None
        }) for item in prompt.verifier["constraints"]
    ))
