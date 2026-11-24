
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import read_rows


CATEGORIES = ("obvious", "elusive", "not_applicable")


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Required by dataclasses in the risk helper.
    spec.loader.exec_module(module)
    return module


@dataclass
class EvalConfig:
    label: str
    step: int
    base_model: Path
    run_dir: Path
    test_file: Path
    output_dir: Path
    protocol_root: Path
    entry_file: Path
    image_root: Path
    judge_model: Path
    split: str = "test"
    base_summary: Path = None
    judge_url: str = "http://127.0.0.1:8003"
    num_samples: int = 4
    temperature: float = 0.8
    top_p: float = 1.0
    max_new_tokens: int = 256
    min_pixels: int = 262144
    max_pixels: int = 1003520
    batch_questions: int = 2
    judge_batch_size: int = 8
    seed: int = 42

    @property
    def checkpoint(self):
        return self.run_dir / "checkpoints" / f"global_step_{self.step}"

    def validate(self):
        if type(self.step) is not int or (self.step != 0 if self.label == "base" else self.step < 1):
            raise ValueError("Use step 0 for base; otherwise select a positive checkpoint on validation")
        if self.label not in {"base", "dapo", "dapo_v", "grpo", "grpo_v"}:
            raise ValueError(f"Unknown evaluation variant: {self.label}")
        if self.split not in {"validation", "test"}:
            raise ValueError("Evaluation split must be validation or test")
        for value in (
            self.num_samples,
            self.max_new_tokens,
            self.min_pixels,
            self.max_pixels,
            self.batch_questions,
            self.judge_batch_size,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("Sampling, batch and pixel counts must be positive integers")
        if (
            not math.isfinite(self.temperature)
            or self.temperature <= 0
            or not 0 < self.top_p <= 1
            or self.min_pixels > self.max_pixels
        ):
            raise ValueError("Invalid sampling or image-resolution settings")
        if self.judge_batch_size > 32:
            raise ValueError("JUDGE_BATCH_SIZE must not exceed the existing judge protocol's limit of 32")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path, records):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_protocol(config):
    root = config.protocol_root
    reward = load_source("ithv_shared_eval_reward", root / "examples/reward_function/ithv.py")
    risk = load_source("ithv_shared_eval_risk", root / "verl/trainer/ithv_dual.py")
    return reward, risk


def load_rows(config, risk):
    """Read the published split and resolve images without modifying the files."""
    rows = read_rows(config.test_file, config.image_root, check_images=True)
    for row in rows:
        risk.human_weight(row["acc"], row["hallucination_type"])
    return rows


def validate_adapter(adapter, base_model):
    from safetensors import safe_open

    adapter = Path(adapter)
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    saved_base = config.get("base_model_name_or_path")
    if not saved_base or Path(saved_base).resolve() != Path(base_model).resolve():
        raise ValueError(f"Adapter base mismatch at {adapter}: {saved_base} != {base_model}")
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt", device="cpu") as saved:
        keys = set(saved.keys())
        if not keys or any(".lora_A." not in key and ".lora_B." not in key for key in keys):
            raise ValueError("Expected a nonempty LoRA-only adapter")
        for key in keys:
            counterpart = (
                key.replace(".lora_A.", ".lora_B.") if ".lora_A." in key else key.replace(".lora_B.", ".lora_A.")
            )
            if counterpart not in keys:
                raise ValueError(f"Missing paired LoRA tensor: {counterpart}")
    return adapter


def attach_adapter(model, adapter):
    """PEFT may warn on missing keys; require the complete expected tensor set."""
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    saved = load_file(str(Path(adapter) / "adapter_model.safetensors"))
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, local_files_only=True)
    # The supported recipes freeze embeddings and never resize the vocabulary.
    # PEFT's automatic embedding-export heuristic is not part of LoRA restore.
    actual = get_peft_model_state_dict(model, save_embedding_layers=False)
    if actual.keys() != saved.keys():
        raise ValueError(
            f"LoRA tensor set mismatch: expected {len(actual)}, saved {len(saved)}; "
            f"missing={sorted(actual.keys() - saved.keys())[:4]}, "
            f"unexpected={sorted(saved.keys() - actual.keys())[:4]}"
        )
    for key, tensor in actual.items():
        expected = saved[key].to(dtype=tensor.dtype)
        if tensor.shape != expected.shape or not torch.equal(tensor.detach().cpu(), expected):
            raise ValueError(f"LoRA weight was not restored exactly: {key}")
    print(f"verified {len(saved)} policy LoRA tensors", flush=True)
    return model.eval()


def build_messages(prompt):
    content = []
    for index, part in enumerate(prompt.split("<image>")):
        if index:
            content.append({"type": "image"})
        if part:
            content.append({"type": "text", "text": part})
    return [{"role": "user", "content": content}]


def open_rgb(path):
    from PIL import Image

    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.load()
    return image


def process_image(image, minimum, maximum):
    # Same two-stage integer resize as EasyR1 verl.utils.dataset.process_image.
    if image.width * image.height > maximum:
        factor = math.sqrt(maximum / (image.width * image.height))
        image = image.resize((int(image.width * factor), int(image.height * factor)))
    if image.width * image.height < minimum:
        factor = math.sqrt(minimum / (image.width * image.height))
        image = image.resize((int(image.width * factor), int(image.height * factor)))
    return image


def configure_generation(model, processor):
    if model.config.model_type != "llavaonevision1_5":
        return
    vocab = processor.tokenizer.get_vocab()
    for field, token in (
        ("vision_start_token_id", "<|vision_start|>"),
        ("image_token_id", "<|image_pad|>"),
        ("video_token_id", "<|video_pad|>"),
    ):
        value = vocab.get(token)
        if value is None or getattr(model.config, field, value) not in (None, value):
            raise ValueError(f"LLaVA special-token mismatch: {field}")
        setattr(model.config, field, value)


def generate_answers(config, model, processor, rows):
    import torch
    from tqdm.auto import tqdm

    answers = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), config.batch_questions), desc=f"{config.label} generation"):
            chunk = rows[start : start + config.batch_questions]
            texts = [
                processor.apply_chat_template(
                    build_messages(row["prompt"]), add_generation_prompt=True, tokenize=False
                )
                for row in chunk
            ]
            images = [process_image(open_rgb(row["images"][0]), config.min_pixels, config.max_pixels) for row in chunk]
            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                max_new_tokens=config.max_new_tokens,
                num_return_sequences=config.num_samples,
            )
            decoded = processor.batch_decode(
                output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if len(decoded) != len(chunk) * config.num_samples:
                raise ValueError("Generated response count differs from the requested sampling protocol")
            answers.extend([decoded[i * config.num_samples : (i + 1) * config.num_samples] for i in range(len(chunk))])
    return answers


def score_rows(config, reward, rows, answers):
    if len(answers) != len(rows) or any(len(group) != config.num_samples for group in answers):
        raise ValueError("Generated responses do not match the complete ordered test set")
    inputs = []
    for row, group in zip(rows, answers, strict=True):
        image = open_rgb(row["images"][0])
        for answer in group:
            inputs.append(
                {
                    "response": answer,
                    "response_length": len(answer.split()),
                    "ground_truth": row["answer"],
                    **{key: row[key] for key in ("question", "acc", "task_type", "hallucination_type", "sample_id")},
                    "multi_modal_data": {"images": [image]},
                }
            )
    scores = reward.compute_score(
        inputs,
        expected_judge_model=str(config.judge_model),
        reward_mode="correctness",
        judge_url=config.judge_url,
        judge_timeout=600,
        judge_retries=2,
        judge_batch_size=config.judge_batch_size,
        cache_size=8192,
        audit_path=str(config.output_dir / f"judge_audit_{config.label}.jsonl"),
    )
    if len(scores) != len(inputs):
        raise ValueError("Judge returned an incorrect number of scores")
    records = []
    for index, (row, group) in enumerate(zip(rows, answers, strict=True)):
        samples = []
        for answer, score in zip(
            group, scores[index * config.num_samples : (index + 1) * config.num_samples], strict=True
        ):
            valid = score["reward_valid"] == 1.0
            samples.append(
                {
                    "generation": answer,
                    "correct": int(score["accuracy"]) if valid else None,
                    "valid": valid,
                    "route": "exact"
                    if score["exact_match"] == 1.0
                    else ("skipped" if score["judge_skipped"] == 1.0 else "judge"),
                }
            )
        records.append(
            {
                **{
                    key: row[key]
                    for key in ("sample_id", "hallucination_type", "acc", "task_type", "question", "answer")
                },
                "samples": samples,
            }
        )
    return records


def validate_records(config, records, rows):
    if [record["sample_id"] for record in records] != [row["sample_id"] for row in rows]:
        raise ValueError("Saved results do not match the complete ordered test set")
    for record in records:
        if len(record["samples"]) != config.num_samples:
            raise ValueError("Saved results have a different number of samples per question")
        for sample in record["samples"]:
            if (
                type(sample["valid"]) is not bool
                or not isinstance(sample["generation"], str)
                or (sample["valid"] and type(sample["correct"]) is not int)
                or (sample["valid"] and sample["correct"] not in (0, 1))
                or (not sample["valid"] and sample["correct"] is not None)
            ):
                raise ValueError("Invalid cached correctness/validity record")


def summarize(records, rows, risk):
    summary = {}
    for category in ("overall", *CATEGORIES):
        samples = [
            s
            for r in records
            if category == "overall" or r["hallucination_type"] == category
            for s in r["samples"]
            if s["valid"]
        ]
        correct = sum(s["correct"] for s in samples)
        summary[category] = {
            "accuracy": correct / len(samples) if samples else None,
            "correct": correct,
            "valid": len(samples),
        }
    summary["judge_skipped"] = sum(not s["valid"] for r in records for s in r["samples"])
    summary["difficulty"] = risk.summarize_risk_records(records, {row["sample_id"]: row for row in rows})
    return summary


def prepare_manifest(config, source_files, health):
    import platform

    import peft
    import torch
    import transformers

    settings = {
        key: str(value.resolve()) if isinstance(value, Path) else value for key, value in asdict(config).items()
    }
    root = config.protocol_root
    files = [
        Path(__file__),
        config.entry_file,
        root / "examples/reward_function/ithv.py",
        root / "verl/trainer/ithv_dual.py",
        config.base_model / "config.json",
    ]
    files.extend([root / "verl/models/llava_onevision15.py", root / "verl/utils/tokenizer.py",
                  root / "examples/ithv/data.py", root / "examples/ithv/metrics.py"])
    local_model_code = config.base_model / "modeling_llavaonevision1_5.py"
    if local_model_code.is_file():
        files.append(local_model_code)
    manifest = {
        "config": settings,
        "test_sha256": sha256(config.test_file),
        "protocol_sha256": {str(path.resolve()): sha256(path) for path in files},
        "checkpoint": [file_signature(path) for path in source_files],
        "base_weights": [file_signature(path) for path in sorted(config.base_model.glob("*.safetensors"))],
        "judge_id": health["judge_id"],
        "judge_prompt_sha256": health["prompt_sha256"],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    path = config.output_dir / "evaluation_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"评测配置、权重或裁判与缓存不一致，请使用新的 OUTPUT_DIR: {config.output_dir}")
    elif any(config.output_dir.iterdir()):
        raise ValueError(f"输出目录包含无 manifest 的旧文件，请使用新的 OUTPUT_DIR: {config.output_dir}")
    else:
        atomic_json(path, manifest)


def print_table(config, summary, file=None):
    def cell(value):
        return f"{value:.4f}" if value is not None else "n/a"

    print(
        f"ITHV {config.split} sampled eval (STEP={config.step}, {config.num_samples} samples/question, temp={config.temperature})",
        file=file,
    )
    print(
        f"{'model':<14} {'overall':>8} {'OH':>8} {'EH':>8} {'NA':>8} {'L_H':>8} {'skipped':>8}",
        file=file,
    )
    values = [summary[key]["accuracy"] for key in ("overall", *CATEGORIES)]
    values.append(summary["difficulty"]["risk_upper"])
    print(
        f"{config.label:<14} " + " ".join(f"{cell(v):>8}" for v in values) + f" {summary['judge_skipped']:>8}",
        file=file,
    )
    if summary["difficulty"]["unknown_answers"]:
        print("L_H reports the conservative upper bound; see risk_lower/risk_upper in summary.json.", file=file)
    if "S_RER" in summary:
        print(f"S_RER (OH minus EH, fraction): {cell(summary['S_RER'])}", file=file)


def run_evaluation(config, checkpoint_files, load_model, dual_state_file):
    """Evaluate only the explicitly named actor checkpoint; never select on test."""
    import requests
    import torch

    config.validate()
    sources = checkpoint_files(config)  # Fail before contacting the judge/loading a model.
    reward, risk = load_protocol(config)
    rows = load_rows(config, risk)
    session = requests.Session()
    session.trust_env = False
    response = session.get(config.judge_url.rstrip("/") + "/health", timeout=30)
    response.raise_for_status()
    health = response.json()
    if (
        health.get("frozen") is not True
        or health.get("prompt_sha256") != reward.PROMPT_SHA256
        or health.get("verdict_schema") != {"correct": "boolean"}
        or not health.get("judge_id")
    ):
        raise RuntimeError("Frozen judge protocol mismatch")
    if Path(health.get("model_path", "")).resolve() != config.judge_model.resolve():
        raise RuntimeError("Unexpected judge model")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_manifest(config, sources, health)
    print(
        f"{config.label}: step={config.step}, {config.split} questions={len(rows)}, judge={health['judge_id'][:16]}", flush=True
    )
    results_file = config.output_dir / f"results_{config.label}.jsonl"
    generations_file = config.output_dir / f"generations_{config.label}.jsonl"
    if results_file.exists():
        records = read_jsonl(results_file)
        print("[cache] using completed, protocol-matched results", flush=True)
    else:
        if generations_file.exists():
            saved = read_jsonl(generations_file)
            if [item["sample_id"] for item in saved] != [row["sample_id"] for row in rows]:
                raise ValueError("Cached generations do not match the complete ordered test set")
            answers = [item["generations"] for item in saved]
            print("[cache] reusing completed generations; skipping model inference", flush=True)
        else:
            if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() or not torch.cuda.is_available():
                raise RuntimeError("请显式设置 CUDA_VISIBLE_DEVICES 为一张可用 GPU；评测不会自动占卡")
            model, processor = load_model(config)
            try:
                torch.manual_seed(config.seed)  # Reset AFTER model/adapter loading.
                answers = generate_answers(config, model, processor, rows)
            finally:
                del model, processor
                torch.cuda.empty_cache()
            atomic_jsonl(
                generations_file,
                [
                    {"sample_id": row["sample_id"], "generations": group}
                    for row, group in zip(rows, answers, strict=True)
                ],
            )
        print(f"[{config.label}] 生成完成，正在评分...", flush=True)
        client = reward._get_client(config.judge_url, str(config.judge_model), 600, 2)
        if client.metadata.get("judge_id") != health["judge_id"]:
            raise RuntimeError("Judge identity changed after evaluation preflight; no results were published")
        records = score_rows(config, reward, rows, answers)
        validate_records(config, records, rows)
        atomic_jsonl(results_file, records)
    validate_records(config, records, rows)
    summary = summarize(records, rows, risk)
    summary.update(checkpoint_step=config.step, run_dir=str(config.run_dir.resolve()))
    risk.add_dual_eval_metadata(summary, config.checkpoint / dual_state_file)
    if config.base_summary is not None:
        from .metrics import add_relative_metrics

        add_relative_metrics(summary, config.base_summary, config.output_dir / "evaluation_manifest.json")
    atomic_json(config.output_dir / "summary.json", {config.label: summary})
    with (config.output_dir / "result.txt").open("w", encoding="utf-8") as stream:
        print_table(config, summary, file=stream)
    print_table(config, summary)
    print(f"results saved to {config.output_dir}", flush=True)
    return summary
