
import json
import math
from pathlib import Path


GRPO_RECIPE_FILE = "ithv_grpo_recipe.json"
GRPO_VARIANTS = {"grpo", "grpo_v"}


def validate_ithv_grpo_config(config):
    variant = getattr(config.algorithm, "ithv_grpo_variant", None)
    if variant is None:
        return
    if variant not in GRPO_VARIANTS:
        raise ValueError(f"Unknown ITHV GRPO variant: {variant!r}")
    algorithm, actor = config.algorithm, config.worker.actor
    dual = variant == "grpo_v"
    requirements = {
        "GRPO advantage estimator": algorithm.adv_estimator == "grpo",
        "variant-specific standard-deviation normalization": algorithm.norm_adv_by_std == (variant == "grpo"),
        "variant-specific dual control": algorithm.ithv_dual.enabled == dual,
        "no DAPO dynamic filtering": not algorithm.online_filtering,
        "fixed reference KL in the actor loss": not algorithm.disable_kl
        and algorithm.use_kl_loss and algorithm.kl_type == "fixed"
        and math.isfinite(algorithm.kl_coef) and algorithm.kl_coef > 0
        and algorithm.kl_penalty == "low_var_kl",
        "skip invalid judge responses in KL loss": actor.mask_kl_with_reward_valid,
        "sequence-mean clipped loss": actor.loss_avg_mode == "seq" and actor.loss_type == "default",
        "no extra dual clipping": not actor.enable_dual_clip,
        "symmetric clipping": math.isfinite(actor.clip_ratio_low)
        and 0 < actor.clip_ratio_low < 1 and actor.clip_ratio_low == actor.clip_ratio_high,
        "multiple responses per prompt": config.worker.rollout.n > 1,
    }
    reward = config.worker.reward.reward_function_kwargs
    requirements["variant-specific correctness reward"] = reward.get("reward_mode") == (
        "correctness" if dual else "fixed"
    ) and reward.get("reward_lambda") == 0
    if dual:
        requirements["matched no-std advantage scale"] = algorithm.ithv_dual.advantage_scale == 2.0
        requirements["monitor within the unfiltered training batch"] = (
            algorithm.ithv_dual.monitor_prompts <= config.data.rollout_batch_size
        )
    failures = [name for name, passed in requirements.items() if not passed]
    if failures:
        raise ValueError("ITHV GRPO recipe requires: " + "; ".join(failures))


def validate_ithv_dual_kl(config):
    """Keep legacy DAPO-V KL-free; opt-in GRPO-V supports separate actor KL."""
    if not config.algorithm.ithv_dual.enabled:
        return
    if config.algorithm.disable_kl and not config.algorithm.use_kl_loss:
        return
    if getattr(config.algorithm, "ithv_grpo_variant", None) != "grpo_v":
        raise ValueError("ITHV dual KL is only supported by the explicit grpo_v recipe; legacy DAPO is unchanged")
    validate_ithv_grpo_config(config)


def grpo_recipe(config):
    """Checkpoint-local identity prevents silently changing algorithms on resume."""
    variant = getattr(config.algorithm, "ithv_grpo_variant", None)
    if variant is None:
        return None
    validate_ithv_grpo_config(config)
    actor, rollout = config.worker.actor, config.worker.rollout
    return {
        "version": 2,
        "variant": variant,
        "model_path": actor.model.model_path,
        "lora_rank": actor.model.lora.rank,
        "lora_alpha": actor.model.lora.alpha,
        "lora_target_modules": actor.model.lora.target_modules,
        "lora_exclude_modules": actor.model.lora.exclude_modules,
        "train_files": config.data.train_files,
        "val_files": config.data.val_files,
        "seed": config.data.seed,
        "rollout_n": rollout.n,
        "temperature": rollout.temperature,
        "top_p": rollout.top_p,
        "rollout_batch_size": config.data.rollout_batch_size,
        "mini_rollout_batch_size": config.data.mini_rollout_batch_size,
        "actor_global_batch_size": actor.global_batch_size,
        "actor_ppo_epochs": actor.ppo_epochs,
        "actor_learning_rate": actor.optim.lr,
        "max_prompt_length": config.data.max_prompt_length,
        "max_response_length": config.data.max_response_length,
        "max_pixels": config.data.max_pixels,
        "norm_adv_by_std": config.algorithm.norm_adv_by_std,
        "loss_avg_mode": actor.loss_avg_mode,
        "clip_ratio": actor.clip_ratio_low,
        "enable_dual_clip": actor.enable_dual_clip,
        "kl_coef": config.algorithm.kl_coef,
        "disable_kl": config.algorithm.disable_kl,
        "use_kl_loss": config.algorithm.use_kl_loss,
        "kl_penalty": config.algorithm.kl_penalty,
        "kl_type": config.algorithm.kl_type,
        "mask_kl_with_reward_valid": actor.mask_kl_with_reward_valid,
        # Server addresses, cache sizes and audit paths are operational settings,
        # not algorithm identity; a resumed job may use an equivalent judge endpoint.
        "reward": {key: config.worker.reward.reward_function_kwargs.get(key)
                   for key in ("reward_mode", "reward_lambda", "expected_judge_model")},
    }


def check_ithv_grpo_fresh_output(config):
    """A fresh GRPO launch must not overwrite a completed run's checkpoints."""
    if getattr(config.algorithm, "ithv_grpo_variant", None) is None:
        return
    trainer = config.trainer
    if trainer.find_last_checkpoint or trainer.load_checkpoint_path is not None or trainer.save_checkpoint_path is None:
        return
    folder = Path(trainer.save_checkpoint_path)
    if folder.is_dir() and any(path.is_dir() for path in folder.glob("global_step_*")):
        raise ValueError("ITHV GRPO output already contains checkpoints; resume or use a new output directory")


def save_ithv_grpo_recipe(folder, config):
    recipe = grpo_recipe(config)
    if recipe is not None:
        (Path(folder) / GRPO_RECIPE_FILE).write_text(
            json.dumps(recipe, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def check_ithv_grpo_resume(folder, config):
    expected = grpo_recipe(config)
    path = Path(folder) / GRPO_RECIPE_FILE
    if expected is None:
        if path.is_file():
            raise ValueError("Cannot resume an ITHV GRPO checkpoint without its explicit algorithm variant")
        return
    if not path.is_file():
        raise FileNotFoundError("Cannot resume ITHV GRPO from a checkpoint without its GRPO recipe")
    if json.loads(path.read_text(encoding="utf-8")) != expected:
        raise ValueError("ITHV GRPO checkpoint recipe differs from this run; use a separate output directory")
