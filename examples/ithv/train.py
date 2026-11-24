
import argparse
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path

from .data import DEFAULT_DATA_DIR, validate_splits


ROOT = Path(__file__).resolve().parents[2]
RECIPES = ("dapo", "dapo_v", "grpo", "grpo_v")
BACKBONES = ("qwen25_3b", "qwen25_7b", "llava_ov15_8b")
PAPER_BUDGETS = {"qwen25_3b": 0.14, "qwen25_7b": 0.05, "llava_ov15_8b": 0.14}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", choices=RECIPES, required=True)
    p.add_argument("--backbone", choices=BACKBONES, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--judge-model", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpus", required=True, help="Explicit physical GPU IDs, e.g. 0 or 0,1")
    p.add_argument("--judge-url", default="http://127.0.0.1:8003")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--risk-budget", type=float, help="DAPO-V: backbone-specific paper budget; GRPO-V: 0.14")
    p.add_argument("--resume", action="store_true", help="Restore the latest complete checkpoint in this output")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print all effective overrides; no GPU, judge or writes")
    return p


def build_overrides(args):
    if args.recipe.startswith("grpo") and args.backbone != "llava_ov15_8b":
        raise ValueError("This release contains GRPO/GRPO-V main experiments only for LLaVA")
    if not re.fullmatch(r"[0-9]+(,[0-9]+)*", args.gpus) or len(set(args.gpus.split(","))) != len(args.gpus.split(",")):
        raise ValueError("--gpus must contain unique comma-separated physical GPU IDs")
    world_size = len(args.gpus.split(","))
    if (21 * 8) % world_size:
        raise ValueError("GPU count must divide the 168-response optimizer mini-batch")
    if args.epochs < 1 or not 0 <= args.seed < 2**32 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Invalid epochs, seed or learning rate")
    grpo, dual = args.recipe.startswith("grpo"), args.recipe.endswith("_v")
    budget = args.risk_budget if args.risk_budget is not None else (0.14 if grpo else PAPER_BUDGETS[args.backbone])
    if not math.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("Risk budget must be in [0, 1]")
    llava = args.backbone == "llava_ov15_8b"
    # These are saved training settings, not an ablation of packing.
    padding_free = not llava or args.recipe == "dapo_v"
    output = args.output_dir.expanduser().resolve()
    reward_kwargs = {
        "reward_lambda": 0.0,
        "reward_mode": "correctness" if dual else "fixed",
        "judge_url": args.judge_url,
        "expected_judge_model": str(args.judge_model.expanduser().resolve()),
        "judge_timeout": 600.0,
        "judge_retries": 2 if args.recipe == "dapo_v" and args.backbone == "qwen25_3b" else 5,
        "judge_retry_wait": 30.0,
        "judge_batch_size": 8, "cache_size": 4096,
        "audit_path": str(output / "reward_audit.jsonl"),
    }
    return {
        "data.train_files": str(args.data_dir.resolve() / "train.jsonl"),
        "data.val_files": str(args.data_dir.resolve() / "validation.jsonl"),
        "data.image_dir": str(args.image_root.expanduser().resolve()),
        "data.mini_rollout_batch_size": 21 if grpo or (22 * 8) % world_size else 22,
        "data.seed": args.seed,
        "worker.actor.model.model_path": str(args.model_path.expanduser().resolve()),
        "worker.actor.model.trust_remote_code": llava,
        "worker.actor.padding_free": padding_free,
        "worker.ref.padding_free": padding_free,
        "worker.actor.optim.lr": args.learning_rate,
        "worker.actor.clip_ratio_high": 0.2 if grpo else 0.28,
        "worker.actor.enable_dual_clip": not grpo,
        "worker.actor.mask_kl_with_reward_valid": grpo,
        "worker.actor.loss_avg_mode": "seq" if grpo else "token",
        "worker.actor.loss_type": "default",
        "worker.rollout.enforce_eager": llava,
        "worker.rollout.seed": args.seed,
        "worker.reward.reward_function": str(ROOT / "examples/reward_function/ithv.py") + ":compute_score",
        "worker.reward.reward_function_kwargs": reward_kwargs,
        "algorithm.ithv_grpo_variant": args.recipe if grpo else None,
        "algorithm.norm_adv_by_std": not dual,
        "algorithm.ithv_dual.enabled": dual,
        "algorithm.ithv_dual.risk_budget": budget,
        "algorithm.ithv_dual.monitor_prompts": 63 if grpo else 66,
        "algorithm.disable_kl": not grpo,
        "algorithm.use_kl_loss": grpo,
        "algorithm.kl_type": "fixed",
        "algorithm.kl_penalty": "low_var_kl",
        "algorithm.kl_coef": 0.04 if grpo else 0.0,
        "algorithm.online_filtering": not grpo,
        "trainer.n_gpus_per_node": world_size,
        "trainer.total_epochs": args.epochs,
        "trainer.experiment_name": output.name,
        "trainer.logger": ["console", "file", "wandb"] if args.wandb else ["console", "file"],
        "trainer.save_checkpoint_path": str(output / "checkpoints"),
        "trainer.find_last_checkpoint": args.resume,
        # A GRPO/V checkpoint has an additional recipe guard in the trainer.
    }


def preflight_judge(args):
    import importlib.util
    import requests

    path = ROOT / "examples/reward_function/ithv.py"
    spec = importlib.util.spec_from_file_location("release_reward_preflight", path)
    reward = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward)
    session = requests.Session()
    session.trust_env = False
    response = session.get(args.judge_url.rstrip("/") + "/health", timeout=30)
    response.raise_for_status()
    health = response.json()
    if (health.get("frozen") is not True or health.get("prompt_sha256") != reward.PROMPT_SHA256
            or health.get("verdict_schema") != {"correct": "boolean"} or not health.get("judge_id")):
        raise RuntimeError("Frozen judge protocol mismatch")
    if Path(health.get("model_path", "")).resolve() != args.judge_model.expanduser().resolve():
        raise RuntimeError("Frozen judge model does not match --judge-model")


def main(argv=None):
    args = parser().parse_args(argv)
    overrides = build_overrides(args)
    validate_splits(args.data_dir)
    command = [sys.executable, "-m", "verl.trainer.main", f"config={ROOT / 'configs/main.yaml'}"]
    command += [f"{key}={json.dumps(value)}" for key, value in overrides.items()]
    if args.dry_run:
        print(json.dumps({"config": "configs/main.yaml", "overrides": overrides}, indent=2))
        print(shlex.join(command))
        return
    for model in (args.model_path, args.judge_model):
        if not (model.expanduser() / "config.json").is_file():
            raise FileNotFoundError(f"Local model config not found: {model}")
    validate_splits(args.data_dir, args.image_root, check_images=True)
    checkpoints = args.output_dir.expanduser().resolve() / "checkpoints"
    existing = checkpoints.is_dir() and any(checkpoints.glob("global_step_*"))
    if existing and not args.resume:
        raise FileExistsError("Output already contains checkpoints; use --resume or a new output directory")
    if args.resume and not (checkpoints / "checkpoint_tracker.json").is_file():
        raise FileNotFoundError("--resume requires this output's checkpoint tracker; refusing to restart from base")
    preflight_judge(args)
    from .runtime import prepare_run

    run_lock = prepare_run(args, overrides)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # The server is separate; local HTTP requests must not use a proxy.
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = ",".join(filter(None, [os.environ.get(key, ""), "localhost", "127.0.0.1", "::1"]))
    os.chdir(ROOT)
    # Keep the output lease alive across exec; Ray is not attached to another job.
    assert run_lock >= 0
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
