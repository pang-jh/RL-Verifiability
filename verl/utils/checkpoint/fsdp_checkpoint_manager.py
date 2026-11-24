
import os
from typing import Optional, Union

import torch
import torch.distributed as dist
from peft import PeftModel
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

from .checkpoint_manager import BaseCheckpointManager
from .lora_checkpoint import MANIFEST, export_adapter, is_lora_key, lora_shard, read_manifest, validate_lora_config, write_manifest


class FSDPCheckpointManager(BaseCheckpointManager):

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
    ):
        super().__init__(model, optimizer, lr_scheduler, processing_class)

    def load_checkpoint(self, path: Optional[str] = None):
        if path is None:
            return

        if os.path.isfile(os.path.join(path, MANIFEST)):
            if not isinstance(self.model._fsdp_wrapped_module, PeftModel):
                raise ValueError("LoRA resume checkpoint requires a PEFT actor")
            peft_config = self.model._fsdp_wrapped_module.peft_config["default"]
            manifest = read_manifest(
                path, base_path=peft_config.base_model_name_or_path, world_size=self.world_size
            )
            validate_lora_config(path, peft_config)
            lora_path = os.path.join(path, f"lora_world_size_{self.world_size}_rank_{self.rank}.pt")
            optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
            extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
            model_state_dict = torch.load(lora_path, weights_only=False, map_location="cpu")
            if len(model_state_dict) != manifest["adapter_tensor_count"] or not all(
                is_lora_key(key) for key in model_state_dict
            ):
                raise ValueError("LoRA shard keys do not match the checkpoint manifest")
            options = StateDictOptions(cpu_offload=True, ignore_frozen_params=True, strict=False)
            incompatible = set_model_state_dict(self.model, model_state_dict, options=options)
            if incompatible.unexpected_keys or any(is_lora_key(key) for key in incompatible.missing_keys):
                raise RuntimeError(f"LoRA FSDP restore mismatch: {incompatible}")
            optim_state_dict = torch.load(optim_path, weights_only=False, map_location="cpu")
            set_optimizer_state_dict(self.model, self.optimizer, optim_state_dict, options=options)
            extra_state_dict = torch.load(extra_path, weights_only=False, map_location="cpu")
            self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])
            if "rng" in extra_state_dict:
                self.load_rng_state(extra_state_dict["rng"])
            print(f"[rank-{self.rank}]: Restored LoRA, optimizer, scheduler, and RNG from {path}.")
            return

        # every rank download its own checkpoint
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        print(f"[rank-{self.rank}]: Loading model from {os.path.abspath(model_path)}.")
        print(f"[rank-{self.rank}]: Loading optimizer from {os.path.abspath(optim_path)}.")
        print(f"[rank-{self.rank}]: Loading extra_state from {os.path.abspath(extra_path)}.")
        model_state_dict = torch.load(model_path, weights_only=False)
        optim_state_dict = torch.load(optim_path, weights_only=False)
        extra_state_dict = torch.load(extra_path, weights_only=False)

        state_dict_options = StateDictOptions(cpu_offload=True)
        set_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optim_state_dict,
            options=state_dict_options,
        )
        self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])

        # recover random state
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

    def save_checkpoint(self, path: str, save_model_only: bool = False, save_adapter_only: bool = False):
        if save_adapter_only and (save_model_only or not isinstance(self.model._fsdp_wrapped_module, PeftModel)):
            raise ValueError("Adapter-only resumable saves require a PEFT actor and optimizer state")
        path = self.local_mkdir(path)
        dist.barrier()

        # every rank will save its own model and optim shard
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

        state_dict_options = StateDictOptions(cpu_offload=True, ignore_frozen_params=save_adapter_only)
        if save_model_only:
            model_state_dict = get_model_state_dict(self.model, options=state_dict_options)
            print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}.")
            torch.save(model_state_dict, model_path)
        else:
            model_state_dict, optim_state_dict = get_state_dict(self.model, self.optimizer, options=state_dict_options)
            if save_adapter_only:
                lora_state_dict = lora_shard(model_state_dict)
                # torch.distributed may retain the tied, frozen lm_head alias
                # even with ignore_frozen_params=True. Do not serialize it.
                unexpected = set(model_state_dict) - set(lora_state_dict)
                if unexpected - {"base_model.model.lm_head.weight"}:
                    raise ValueError(f"Adapter-only save found non-LoRA state: {unexpected}")
                model_state_dict = lora_state_dict
                model_path = os.path.join(path, f"lora_world_size_{self.world_size}_rank_{self.rank}.pt")
            extra_state_dict = {
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "rng": self.get_rng_state(),
            }
            print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}.")
            print(f"[rank-{self.rank}]: Saving optimizer to {os.path.abspath(optim_path)}.")
            print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}.")
            torch.save(model_state_dict, model_path)
            torch.save(optim_state_dict, optim_path)
            torch.save(extra_state_dict, extra_path)

        # wait for everyone to dump to local
        dist.barrier()

        if self.rank == 0:
            hf_path = os.path.join(path, "huggingface")
            os.makedirs(hf_path, exist_ok=True)
            assert isinstance(self.model._fsdp_wrapped_module, (PreTrainedModel, PeftModel))
            self.model._fsdp_wrapped_module.config.save_pretrained(hf_path)
            self.model._fsdp_wrapped_module.generation_config.save_pretrained(hf_path)
            self.processing_class.save_pretrained(hf_path)

        if isinstance(self.model._fsdp_wrapped_module, PeftModel):
            lora_path = os.path.join(path, "lora_adapter")
            tensor_count = export_adapter(self.model._fsdp_wrapped_module, model_state_dict, lora_path)
            torch.cuda.empty_cache()
            dist.barrier()
            if self.rank == 0:
                if save_adapter_only:
                    base_path = self.model._fsdp_wrapped_module.peft_config["default"].base_model_name_or_path
                    write_manifest(path, base_path=base_path, world_size=self.world_size, tensor_count=tensor_count)
                print(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_path}")

        dist.barrier()
