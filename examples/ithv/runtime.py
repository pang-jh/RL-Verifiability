
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_identity(args, overrides):
    settings = copy.deepcopy(overrides)
    settings.pop("trainer.find_last_checkpoint")
    settings.pop("trainer.logger")
    settings["worker.reward.reward_function_kwargs"].pop("judge_url")
    settings["worker.reward.reward_function_kwargs"].pop("audit_path")
    return {
        "version": 1, "recipe": args.recipe, "backbone": args.backbone,
        "overrides": settings,
        "config_sha256": hashlib.sha256((ROOT / "configs/main.yaml").read_bytes()).hexdigest(),
        "split_sha256": {name: hashlib.sha256((args.data_dir / f"{name}.jsonl").read_bytes()).hexdigest()
                         for name in ("train", "validation", "test")},
    }


def validate_resume(args):
    checkpoints = args.output_dir.expanduser().resolve() / "checkpoints"
    tracker = json.loads((checkpoints / "checkpoint_tracker.json").read_text(encoding="utf-8"))
    step = tracker.get("last_global_step")
    if type(step) is not int or step < 1:
        raise ValueError("Invalid checkpoint tracker; refusing to restart from base")
    folder = checkpoints / f"global_step_{step}"
    required = [folder / "run_config.json", folder / "dataloader.pt"]
    actor = folder / "actor"
    manifest = json.loads((actor / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    world_size = len(args.gpus.split(","))
    if manifest.get("world_size") != world_size:
        raise ValueError("Optimizer resume requires the original FSDP world size")
    required += [actor / "lora_adapter/adapter_config.json", actor / "lora_adapter/adapter_model.safetensors"]
    for rank in range(world_size):
        required += [actor / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
                     for kind in ("lora", "optim", "extra_state")]
    if args.recipe.endswith("_v"):
        required.append(folder / "ithv_dual_state.json")
    if args.recipe.startswith("grpo"):
        required.append(folder / "ithv_grpo_recipe.json")
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Incomplete resume checkpoint: {path}")


def prepare_run(args, overrides):
    """Return an inherited output lock; called only for a real training launch."""
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fd = os.open(output / ".train.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another training process owns this output directory") from error
        path = output / "release_run.json"
        identity = run_identity(args, overrides)
        if args.resume:
            if json.loads(path.read_text(encoding="utf-8")) != identity:
                raise ValueError("Resume recipe/data/settings differ from this release run")
            validate_resume(args)
        else:
            # Exclusive creation also prevents silent overwrite of an interrupted run.
            with path.open("x", encoding="utf-8") as stream:
                json.dump(identity, stream, indent=2, allow_nan=False)
                stream.write("\n")
        os.set_inheritable(fd, True)
        return fd
    except BaseException:
        os.close(fd)
        raise
