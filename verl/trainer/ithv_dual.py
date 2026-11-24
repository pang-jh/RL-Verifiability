
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


DUAL_STATE_FILE = "ithv_dual_state.json"


@dataclass
class ITHVDualConfig:
    enabled: bool = False
    risk_budget: float = 0.14
    lambda_init: Optional[float] = None  # None means 0.5 * the training known-accuracy fraction.
    learning_rate: float = 0.1
    lambda_max: float = 1.0
    ema_beta: float = 0.8
    monitor_prompts: int = 66
    advantage_scale: float = 2.0  # Match the old symmetric reward's centered advantage scale.

    def post_init(self):
        for name in ("risk_budget", "learning_rate", "lambda_max", "ema_beta", "advantage_scale"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"ITHV dual {name} must be finite")
        if not 0 <= self.risk_budget <= 1 or not 0 <= self.ema_beta < 1:
            raise ValueError("ITHV risk_budget must be in [0, 1] and ema_beta in [0, 1)")
        if self.learning_rate <= 0 or self.lambda_max <= 0 or self.advantage_scale <= 0:
            raise ValueError("ITHV dual learning_rate, lambda_max and advantage_scale must be positive")
        if self.lambda_init is not None and (
            not math.isfinite(self.lambda_init) or not 0 <= self.lambda_init <= self.lambda_max
        ):
            raise ValueError("ITHV dual lambda_init must be in [0, lambda_max]")
        if isinstance(self.monitor_prompts, bool) or not isinstance(self.monitor_prompts, int) or self.monitor_prompts < 1:
            raise ValueError("ITHV monitor_prompts must be positive")


def human_weight(acc, category):
    """NA is outside the measured constraint, not assumed safe."""
    if category not in {"obvious", "elusive", "not_applicable"}:
        raise ValueError(f"Unknown ITHV category: {category!r}")
    if acc is None:
        if category != "not_applicable":
            raise ValueError("OH/EH rows require human accuracy")
        return None
    if category == "not_applicable" or isinstance(acc, bool):
        raise ValueError("NA must have acc=null; human accuracy cannot be boolean")
    if not isinstance(acc, (int, float)) or not math.isfinite(acc) or not 0 <= acc <= 1:
        raise ValueError(f"Invalid human accuracy: {acc!r}")
    return 1.0 - acc


def training_population(dataset):
    """Inspect retained dataset rows, without opening images/tokenizing them again."""
    rows = getattr(dataset, "dataset", dataset)
    count = known = 0
    digest = hashlib.sha256()
    for row in rows:
        weight = human_weight(row.get("acc"), row.get("hallucination_type"))
        known += weight is not None
        count += 1
        # Include question/reference/image paths so restoring a different population fails.
        payload = {key: row.get(key) for key in (
            "sample_id", "acc", "hallucination_type", "task_type", "question", "prompt", "answer", "images"
        )}
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    if not count or not known:
        raise ValueError("ITHV dual requires a nonempty dataset with human-accuracy annotations")
    return {"prompts": count, "known_prompts": known, "known_fraction": known / count,
            "sha256": digest.hexdigest()}


class ITHVDualController:
    def __init__(self, config: ITHVDualConfig, population: dict):
        config.post_init()
        self.config = config
        self.population = dict(population)
        self.known_fraction = float(population["known_fraction"])
        if not 0 < self.known_fraction <= 1:
            raise ValueError("ITHV known_fraction must be in (0, 1]")
        self.lambda_value = config.lambda_init if config.lambda_init is not None else 0.5 * self.known_fraction
        if self.lambda_value > config.lambda_max:
            raise ValueError("ITHV lambda_max is below the automatic initial multiplier")
        self.risk_ema = None
        self.updates = 0
        self.last_step = 0
        self.active_step = None

    def begin_step(self, step: int):
        if self.active_step is not None or step != self.last_step + 1:
            raise RuntimeError("ITHV dual steps must be consecutive, with one update per outer step")
        self.active_step = step
        self.step_lambda = self.lambda_value
        self.seen_uids = set()
        self.monitor_count = self.known_count = self.unknown_answers = self.known_answers = 0
        self.cost_lower = self.cost_upper = 0.0

    @property
    def monitor_complete(self):
        return self.active_step is not None and self.monitor_count == self.config.monitor_prompts

    def observe(self, uids, metrics: dict, generations: int):
        """Take a fixed prefix of raw prompt groups, before any DAPO filtering.

        Unknown judge outcomes have cost in [0, w]. Use the upper endpoint for
        control, while keeping these answers masked from policy gradients.
        """
        if self.active_step is None:
            raise RuntimeError("Call begin_step before collecting ITHV risk")
        keys = ("difficulty_known", "difficulty_weight", "difficulty_cost", "difficulty_cost_upper",
                "reward_valid", "accuracy")
        if any(key not in metrics or len(metrics[key]) != len(uids) for key in keys):
            raise ValueError("ITHV dual requires per-answer correctness, validity and difficulty metrics")
        validate_difficulty_metrics(metrics, len(uids))
        groups = {}
        for index, uid in enumerate(uids):
            groups.setdefault(str(uid), []).append(index)
        selected = list(groups.items())[: self.config.monitor_prompts - self.monitor_count]
        for uid, indices in selected:
            if uid in self.seen_uids or len(indices) != generations:
                raise ValueError("ITHV raw groups must have unique UIDs and the configured rollout count")
            known = [metrics["difficulty_known"][i] for i in indices]
            weights = [metrics["difficulty_weight"][i] for i in indices]
            if len(set(known)) != 1 or len(set(weights)) != 1 or known[0] not in (0, 1):
                raise ValueError("ITHV human difficulty must be constant within each prompt group")
        for uid, indices in selected:
            self.seen_uids.add(uid)
            self.monitor_count += 1
            if metrics["difficulty_known"][indices[0]]:
                self.known_count += 1
                self.known_answers += len(indices)
                self.unknown_answers += sum(metrics["reward_valid"][i] == 0 for i in indices)
                self.cost_lower += sum(metrics["difficulty_cost"][i] for i in indices) / generations
                self.cost_upper += sum(metrics["difficulty_cost_upper"][i] for i in indices) / generations

    def finish_step(self):
        if not self.monitor_complete:
            raise RuntimeError("ITHV risk monitor is incomplete; refusing an adaptive-stopping estimate")
        metrics = {"dual/lambda_used": self.step_lambda, "dual/risk_budget": self.config.risk_budget,
                   "dual/monitor_prompts": self.monitor_count, "dual/known_prompts": self.known_count,
                   "dual/known_fraction": self.known_fraction}
        if self.known_count:
            lower = self.cost_lower / self.known_count
            upper = self.cost_upper / self.known_count
            self.risk_ema = upper if self.risk_ema is None else (
                self.config.ema_beta * self.risk_ema + (1 - self.config.ema_beta) * upper
            )
            self.lambda_value = min(self.config.lambda_max, max(
                0.0, self.step_lambda + self.config.learning_rate * (self.risk_ema - self.config.risk_budget)
            ))
            self.updates += 1
            metrics.update({"dual/risk_lower": lower, "dual/risk_upper": upper,
                            "dual/risk_ema": self.risk_ema,
                            "dual/violation": upper - self.config.risk_budget,
                            "dual/unknown_fraction": self.unknown_answers / self.known_answers})
        metrics.update({"dual/lambda_next": self.lambda_value, "dual/updates": self.updates,
                        "dual/update_skipped": float(self.known_count == 0),
                        "dual/at_upper_bound": float(self.lambda_value >= self.config.lambda_max)})
        self.last_step = self.active_step
        self.active_step = None
        return metrics

    def state_dict(self, global_step):
        if self.active_step is not None or self.last_step != global_step:
            raise RuntimeError("ITHV dual can only be checkpointed at its completed outer-step boundary")
        return {"version": 1, "global_step": global_step, "config": asdict(self.config),
                "population": self.population, "lambda_value": self.lambda_value,
                "risk_ema": self.risk_ema, "updates": self.updates}

    def load_state_dict(self, state, global_step):
        if self.active_step is not None or state.get("version") != 1 or state.get("global_step") != global_step:
            raise ValueError("ITHV dual checkpoint version/step mismatch")
        if state.get("config") != asdict(self.config) or state.get("population") != self.population:
            raise ValueError("ITHV dual configuration or training population differs from checkpoint")
        lam, ema, updates = state["lambda_value"], state["risk_ema"], state["updates"]
        if not math.isfinite(lam) or not 0 <= lam <= self.config.lambda_max:
            raise ValueError("Invalid saved ITHV multiplier")
        if ema is not None and (not math.isfinite(ema) or not 0 <= ema <= 1):
            raise ValueError("Invalid saved ITHV EMA")
        if not isinstance(updates, int) or not 0 <= updates <= global_step:
            raise ValueError("Invalid saved ITHV update count")
        self.lambda_value, self.risk_ema, self.updates, self.last_step = lam, ema, updates, global_step

    def save(self, folder, global_step):
        path = Path(folder) / DUAL_STATE_FILE
        state = self.state_dict(global_step)
        fd, temporary = tempfile.mkstemp(prefix=".ithv_dual_", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def validate_difficulty_metrics(metrics, size):
    keys = ("difficulty_known", "difficulty_weight", "difficulty_cost", "difficulty_cost_upper",
            "reward_valid", "accuracy")
    if any(key not in metrics or len(metrics[key]) != size for key in keys):
        raise ValueError("Missing per-answer ITHV difficulty metrics")
    for i in range(size):
        known, weight, lower, upper, valid, correct = (metrics[key][i] for key in keys)
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in (known, weight, lower, upper, valid, correct)):
            raise ValueError("ITHV difficulty metrics must be finite numbers")
        if known not in (0, 1) or valid not in (0, 1) or correct not in (0, 1) or not 0 <= weight <= 1:
            raise ValueError("ITHV correctness/validity/known flags must be binary and weight in [0, 1]")
        if not known and weight != 0:
            raise ValueError("Unmeasured ITHV rows must have zero measured difficulty weight")
        expected_lower = weight * (1 - correct) if valid else 0.0
        expected_upper = expected_lower if valid else weight
        if not math.isclose(lower, expected_lower, abs_tol=1e-7) or not math.isclose(upper, expected_upper, abs_tol=1e-7):
            raise ValueError("ITHV difficulty cost is inconsistent with correctness/validity/weight")


def add_dual_eval_metadata(summary, state_path):
    """Read checkpoint budget for reporting only; never update controller state."""
    path = Path(state_path)
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        upper = summary["difficulty"]["risk_upper"]
        summary["dual"] = {
            "risk_budget": state["config"]["risk_budget"], "lambda_next": state["lambda_value"],
            "risk_ema_train": state["risk_ema"],
            "feasible_at_eval_decoding": upper is not None and upper <= state["config"]["risk_budget"],
        }
    return summary


def summarize_risk_records(records, rows_by_id=None):
    """Prompt-mean proxy risk L_H with explicit bounds for unknown judgments.

    Missing judgments retain their rollout denominators in the risk bounds
    rather than silently conditioning the reported risk on judge success.
    """
    costs_lower = costs_upper = valid_cost = 0.0
    known_prompts = known_answers = valid_answers = 0
    seen_ids = set()
    for record in records:
        sample_id = record["sample_id"]
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate evaluation sample_id: {sample_id}")
        seen_ids.add(sample_id)
        source = rows_by_id[record["sample_id"]] if rows_by_id is not None else record
        for key in ("question", "answer", "task_type", "hallucination_type"):
            if source.get(key) != record.get(key):
                raise ValueError(f"Evaluation input differs from saved record for {record['sample_id']}: {key}")
        if "acc" not in source:
            raise ValueError("Risk reporting requires source acc, including explicit null for NA")
        if "acc" in record and record["acc"] != source["acc"]:
            raise ValueError(f"Human accuracy differs from saved record for {sample_id}")
        weight = human_weight(source["acc"], record["hallucination_type"])
        if weight is None:
            continue
        samples = record["samples"]
        if not samples:
            raise ValueError("Risk reporting requires at least one answer per known prompt")
        known_prompts += 1
        known_answers += len(samples)
        lower = upper = 0.0
        for sample in samples:
            if type(sample["valid"]) is not bool:
                raise ValueError("Evaluation validity must be boolean")
            if sample["valid"]:
                if sample["correct"] not in (0, 1):
                    raise ValueError("Valid answers must have binary correctness")
                cost = weight * (1 - sample["correct"])
                valid_cost += cost
                lower += cost
                upper += cost
                valid_answers += 1
            else:
                if sample["correct"] is not None:
                    raise ValueError("Unknown judgments must have correct=null")
                upper += weight
        costs_lower += lower / len(samples)
        costs_upper += upper / len(samples)
    return {"risk_lower": costs_lower / known_prompts if known_prompts else None,
            "risk_upper": costs_upper / known_prompts if known_prompts else None,
            "risk_valid_only": valid_cost / valid_answers if valid_answers else None,
            "known_prompts": known_prompts, "known_answers": known_answers,
            "unknown_answers": known_answers - valid_answers}
