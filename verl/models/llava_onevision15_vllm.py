
from collections.abc import Iterable

import torch
from torch import nn
from transformers import Qwen2_5_VLProcessor
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsMultiModal
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLDummyInputsBuilder,
    Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .llava_onevision15 import prepare_hf_model, register_vllm_model


class LlavaOneVision15ProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_hf_processor(self, **kwargs):
        return self.ctx.get_hf_processor(Qwen2_5_VLProcessor, use_fast=kwargs.pop("use_fast", True), **kwargs)

    def get_supported_mm_limits(self):
        return {"image": None}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": self.get_max_image_tokens()}


class LlavaWorkerExtension:
    """Importing this module registers the architecture in spawned workers."""

    @torch.inference_mode()
    def ithv_vision_features(self, pixels, grid):
        model = self.model_runner.model
        device = next(model.visual.parameters()).device
        pixels = torch.as_tensor(pixels, device=device, dtype=model.visual.dtype)
        grid = torch.as_tensor(grid, device=device, dtype=torch.long)
        return model.visual(pixels, grid_thw=grid).cpu().tolist()


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2VLMultiModalProcessor,
    info=LlavaOneVision15ProcessingInfo,
    dummy_inputs=Qwen2VLDummyInputsBuilder,
)
class LlavaOneVision15ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "model.language_model.": "language_model.model.",
        "model.visual.": "visual.",
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    })

    @classmethod
    def get_placeholder_str(cls, modality, i):
        if modality != "image":
            raise ValueError("LLaVA-OneVision-1.5 adapter currently supports images only")
        return "<|vision_start|><|image_pad|><|vision_end|>"

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("LLaVA-OneVision-1.5 RICE backend currently requires tensor_parallel_size=1")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("LLaVA-OneVision-1.5 pipeline parallelism is not validated")
        if not vllm_config.model_config.trust_remote_code:
            raise ValueError("LLaVA-OneVision-1.5 requires trust_remote_code=True")
        if vllm_config.quant_config is not None:
            raise ValueError("Quantized LLaVA-OneVision-1.5 is not validated")
        self.config = vllm_config.model_config.hf_config
        cls = prepare_hf_model(vllm_config.model_config.model)
        import sys

        module = sys.modules[cls.__module__]
        self.config.vision_config._attn_implementation = "flash_attention_2"
        self.visual = module.RiceTransformerPretrainedModel._from_config(self.config.vision_config)
        self.language_model = init_vllm_registered_model(
            vllm_config, hf_config=self.config.text_config, architectures=["Qwen3ForCausalLM"],
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def get_language_model(self):
        return self.language_model

    def get_multimodal_embeddings(self, **kwargs):
        if any(k in kwargs for k in ("pixel_values_videos", "video_embeds")):
            raise ValueError("Video inputs are not validated for this adapter")
        if "image_grid_thw" not in kwargs:
            return []
        grid = kwargs["image_grid_thw"]
        if isinstance(grid, list):
            grid = torch.cat(grid)
        grid = grid.reshape(-1, 3)
        if torch.any(grid[:, 0] != 1):
            raise ValueError("Only temporal_patch_size=1 still images are validated")
        if "image_embeds" in kwargs:
            embeds = kwargs["image_embeds"]
        else:
            pixels = kwargs["pixel_values"]
            if isinstance(pixels, list):
                pixels = torch.cat(pixels)
            pixels = pixels.reshape(-1, pixels.shape[-1]).to(self.visual.dtype)
            embeds = self.visual(pixels, grid_thw=grid)
        sizes = (grid.prod(-1) // self.config.vision_config.spatial_merge_size**2).tolist()
        return tuple(embeds.split(sizes))

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None):
        embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings:
            embeds = merge_multimodal_embeddings(
                input_ids, embeds, multimodal_embeddings, self.config.image_token_id
            )
        return embeds

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if positions.ndim != 1:
            raise ValueError(f"LLaVA-OneVision-1.5 uses 1D flattened text positions, got {positions.shape}")
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids=input_ids, positions=positions,
            intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states):
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self):
        return MultiModelKeys.from_string_field(
            language_model="language_model", connector="visual.merger.", tower_model="visual."
        )


register_vllm_model()
