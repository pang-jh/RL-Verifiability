
import hashlib
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed._tensor import DTensor


FORMAT = "easy_r1_lora_resume_v1"
MANIFEST = "checkpoint_manifest.json"


def is_lora_key(key: str) -> bool:
    return ".lora_A.default.weight" in key or ".lora_B.default.weight" in key


def lora_shard(state: dict) -> dict:
    result = {key: value for key, value in state.items() if is_lora_key(key)}
    if not result:
        raise ValueError("No LoRA tensors found in FSDP model state")
    return result


def peft_key(key: str) -> str:
    if not is_lora_key(key):
        raise ValueError(f"Not a LoRA tensor: {key}")
    return key.replace(".default.weight", ".weight")


def base_config_hash(base_path: str) -> str:
    config_path = Path(base_path).resolve(strict=True) / "config.json"
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def export_adapter(model: PeftModel, state: dict, path: str) -> int:
    """Collect FSDP LoRA shards on all ranks; rank zero writes PEFT files."""
    import torch.distributed as dist

    shards = lora_shard(state)
    weights = {}
    for key, shard in sorted(shards.items()):
        full = shard.to(torch.device("cuda")).full_tensor() if isinstance(shard, DTensor) else shard
        if dist.get_rank() == 0:
            weights[peft_key(key)] = full.detach().cpu().contiguous()

    count = len(shards)
    if dist.get_rank() == 0:
        adapter_path = Path(path)
        adapter_path.mkdir(parents=True, exist_ok=True)
        model.peft_config["default"].save_pretrained(str(adapter_path))
        target = adapter_path / "adapter_model.safetensors"
        temporary = adapter_path / "adapter_model.safetensors.tmp"
        save_file(weights, str(temporary))
        os.replace(temporary, target)
        with safe_open(str(target), framework="pt", device="cpu") as saved:
            if len(saved.keys()) != count:
                raise RuntimeError("LoRA export produced an incomplete adapter")
    return count


def write_manifest(path: str, *, base_path: str, world_size: int, tensor_count: int) -> None:
    root = Path(path)
    adapter_path = root / "lora_adapter" / "adapter_model.safetensors"
    with safe_open(str(adapter_path), framework="pt", device="cpu") as saved:
        if len(saved.keys()) != tensor_count:
            raise RuntimeError("Adapter is missing tensors; refusing to finalize checkpoint")
    manifest = {
        "format": FORMAT,
        "base_model_path": str(Path(base_path).resolve(strict=True)),
        "base_config_sha256": base_config_hash(base_path),
        "world_size": world_size,
        "adapter_tensor_count": tensor_count,
    }
    temporary = root / (MANIFEST + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, root / MANIFEST)


def validate_lora_config(path: str, current_config) -> None:
    """Reject same-shaped LoRA checkpoints with different scaling/target rules."""
    saved_path = Path(path) / "lora_adapter" / "adapter_config.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    fields = (
        "r", "lora_alpha", "lora_dropout", "target_modules",
        "exclude_modules", "bias", "use_rslora", "fan_in_fan_out",
        "rank_pattern", "alpha_pattern", "modules_to_save",
    )
    for field in fields:
        expected = saved.get(field)
        actual = getattr(current_config, field, None)
        if isinstance(expected, list):
            expected = sorted(expected)
        if isinstance(actual, (set, list, tuple)):
            actual = sorted(actual)
        if actual != expected:
            raise ValueError(f"LoRA config mismatch for {field}: checkpoint={expected}, current={actual}")


def read_manifest(path: str, *, base_path: str, world_size: int) -> dict:
    manifest = json.loads((Path(path) / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported LoRA resume checkpoint format")
    if manifest.get("world_size") != world_size:
        raise ValueError("LoRA optimizer shards require the original FSDP world size")
    if str(Path(base_path).resolve(strict=True)) != manifest.get("base_model_path"):
        raise ValueError("Checkpoint base model path differs from the loaded base model")
    if base_config_hash(base_path) != manifest.get("base_config_sha256"):
        raise ValueError("Base model config changed since the checkpoint was created")
    return manifest
