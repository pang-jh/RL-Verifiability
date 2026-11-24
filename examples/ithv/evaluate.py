"""Evaluate base or one explicitly selected main-experiment checkpoint on ITHV."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from . import eval_ithv_common as common
from .data import DEFAULT_DATA_DIR, validate_splits


ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", choices=("base", "dapo", "dapo_v", "grpo", "grpo_v"), required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--judge-model", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--step", type=int, help="Select on validation before evaluating test; no automatic fallback")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--judge-url", default="http://127.0.0.1:8003")
    p.add_argument("--gpu", required=True, help="One explicitly selected GPU")
    p.add_argument("--base-summary", type=Path, help="Protocol-matched base summary.json, for S_RER")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--batch-questions", type=int, default=2)
    p.add_argument("--dry-run", action="store_true", help="Print configuration only; no GPU, network or writes")
    p.add_argument("--check-only", action="store_true", help="Check actual images/adapter headers without inference")
    return p


def configuration(args):
    if not args.gpu.isdigit():
        raise ValueError("--gpu must identify exactly one physical GPU")
    if args.recipe == "base":
        if args.run_dir is not None or args.step not in (None, 0) or args.base_summary is not None:
            raise ValueError("Base evaluation does not accept a training run, nonzero step or base-summary")
        step, run_dir = 0, ROOT
    else:
        if args.run_dir is None or args.step is None or args.step < 1:
            raise ValueError("A trained recipe requires --run-dir and an explicit positive --step")
        step, run_dir = args.step, args.run_dir.expanduser().resolve()
    result = common.EvalConfig(
        label=args.recipe, step=step,
        base_model=args.model_path.expanduser().resolve(),
        run_dir=run_dir,
        test_file=args.data_dir.resolve() / f"{args.split}.jsonl",
        output_dir=args.output_dir.expanduser().resolve(),
        protocol_root=ROOT, entry_file=Path(__file__),
        image_root=args.image_root.expanduser().resolve(),
        judge_model=args.judge_model.expanduser().resolve(),
        split=args.split,
        base_summary=args.base_summary.expanduser().resolve() if args.base_summary else None,
        judge_url=args.judge_url, num_samples=args.num_samples,
        batch_questions=args.batch_questions,
    )
    result.validate()
    return result


def checkpoint_files(config):
    if config.label == "base":
        return []
    run_file = config.checkpoint / "run_config.json"
    run = json.loads(run_file.read_text(encoding="utf-8"))
    algorithm, actor = run["algorithm"], run["worker"]["actor"]
    dual, grpo = config.label.endswith("_v"), config.label.startswith("grpo")
    if Path(actor["model"]["model_path"]).resolve() != config.base_model:
        raise ValueError("Checkpoint base differs from --model-path")
    if (algorithm["adv_estimator"] != "grpo"
            or algorithm["norm_adv_by_std"] != (not dual)
            or algorithm["ithv_dual"]["enabled"] != dual
            or algorithm["online_filtering"] != (not grpo)
            or algorithm["disable_kl"] != (not grpo)
            or algorithm["use_kl_loss"] != grpo
            or algorithm.get("ithv_grpo_variant") != (config.label if grpo else None)):
        raise ValueError("Checkpoint algorithm does not match the requested main recipe")
    reward = run["worker"]["reward"]["reward_function_kwargs"]
    if reward["reward_lambda"] != 0 or reward["reward_mode"] != ("correctness" if dual else "fixed"):
        raise ValueError("Checkpoint uses an unsupported reward recipe")
    adapter = common.validate_adapter(config.checkpoint / "actor/lora_adapter", config.base_model)
    files = [run_file, adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
    state = config.checkpoint / "ithv_dual_state.json"
    if dual:
        content = json.loads(state.read_text(encoding="utf-8"))
        if content["global_step"] != config.step or content["config"] != algorithm["ithv_dual"]:
            raise ValueError("Checkpoint dual state is inconsistent with its step/config")
        files.append(state)
    elif state.exists():
        raise ValueError("A baseline checkpoint must not contain adaptive dual state")
    if grpo:
        recipe_file = config.checkpoint / "ithv_grpo_recipe.json"
        recipe = json.loads(recipe_file.read_text(encoding="utf-8"))
        if (recipe["variant"] != config.label or recipe["norm_adv_by_std"] != (not dual)
                or Path(recipe["model_path"]).resolve() != config.base_model
                or recipe["kl_coef"] != algorithm["kl_coef"]):
            raise ValueError("GRPO checkpoint recipe differs from its run config")
        files.append(recipe_file)
    return files


def load_model(config):
    import torch

    from verl.models.llava_onevision15 import configure_generation, configure_processor, load_policy_model
    from verl.utils.tokenizer import get_processor

    processor = get_processor(str(config.base_model), local_files_only=True, trust_remote_code=True)
    configure_processor(processor, config.min_pixels, config.max_pixels)
    processor.tokenizer.padding_side = "left"
    model = load_policy_model(str(config.base_model), dtype=torch.bfloat16,
                              device_map={"": "cuda:0"}, local_files_only=True)
    configure_generation(model, processor)
    if config.label != "base":
        model = common.attach_adapter(model, config.checkpoint / "actor/lora_adapter")
    return model.eval(), processor


def main(argv=None):
    args = parser().parse_args(argv)
    config = configuration(args)
    validate_splits(args.data_dir)
    if args.dry_run:
        print(json.dumps(asdict(config), default=str, indent=2))
        return
    if args.check_only:
        _, risk = common.load_protocol(config)
        rows = common.load_rows(config, risk)
        files = checkpoint_files(config)
        print(json.dumps({"questions": len(rows), "checkpoint_files": list(map(str, files))}, indent=2))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return common.run_evaluation(config, checkpoint_files, load_model, "ithv_dual_state.json")


if __name__ == "__main__":
    main()
