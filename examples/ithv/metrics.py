"""Paper convention: S_RER = RER_OH - RER_EH (fractions; lower is more EH-oriented)."""

import hashlib
import json
import math
from pathlib import Path


def relative_error_reduction(accuracy, base_accuracy):
    if accuracy is None or base_accuracy is None:
        return None
    if any(isinstance(x, bool) or not math.isfinite(x) or not 0 <= x <= 1 for x in (accuracy, base_accuracy)):
        raise ValueError("Accuracies must be finite fractions in [0, 1]")
    return (accuracy - base_accuracy) / (1 - base_accuracy) if base_accuracy < 1 else None


def relative_metrics(summary, base):
    oh = relative_error_reduction(summary["obvious"]["accuracy"], base["obvious"]["accuracy"])
    eh = relative_error_reduction(summary["elusive"]["accuracy"], base["elusive"]["accuracy"])
    return {"RER_OH": oh, "RER_EH": eh, "S_RER": oh - eh if oh is not None and eh is not None else None}


def add_relative_metrics(summary, base_file, current_manifest):
    """Require an explicit, protocol-matched base evaluation; never invent a reference."""
    base_file = Path(base_file)
    payload = json.loads(base_file.read_text(encoding="utf-8"))
    if set(payload) != {"base"}:
        raise ValueError("--base-summary must refer to a released base evaluation")
    previous = json.loads((base_file.parent / "evaluation_manifest.json").read_text(encoding="utf-8"))
    current = json.loads(Path(current_manifest).read_text(encoding="utf-8"))
    for key in ("test_sha256", "judge_id", "judge_prompt_sha256", "base_weights", "runtime"):
        if current[key] != previous[key]:
            raise ValueError(f"Base evaluation protocol differs: {key}")
    for key in ("base_model", "image_root", "split", "num_samples", "temperature", "top_p",
                "max_new_tokens", "min_pixels", "max_pixels", "batch_questions", "seed"):
        if current["config"][key] != previous["config"][key]:
            raise ValueError(f"Base evaluation settings differ: {key}")
    for key in current["protocol_sha256"]:
        if current["protocol_sha256"][key] != previous["protocol_sha256"].get(key):
            raise ValueError("Base evaluation code/rubric differs")
    summary.update(relative_metrics(summary, payload["base"]))
    summary["base_summary_sha256"] = hashlib.sha256(base_file.read_bytes()).hexdigest()
