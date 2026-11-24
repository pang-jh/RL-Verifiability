
import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F


ARCHITECTURE = "LLaVAOneVision1_5_ForConditionalGeneration"


def is_llava_onevision15(path: str) -> bool:
    if not path:
        return False
    config = Path(path) / "config.json"
    if not config.is_file():
        return False
    data = json.loads(config.read_text())
    return ARCHITECTURE in data.get("architectures", []) or data.get("model_type") == "llavaonevision1_5"


def rice_forward(self, hidden_states, grid_thw):
    hidden_states = self.patch_embed(hidden_states)
    rotary = self.rot_pos_emb(grid_thw)
    lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    sizes = lengths.tolist()
    hidden_parts, rotary_parts, patch_indices = [], [], []
    offset = enlarged_offset = 0
    for size in sizes:
        hidden_parts.extend((self.class_embedding.to(hidden_states.dtype)[None], hidden_states[offset:offset + size]))
        rotary_parts.extend((self.class_pos_emb.to(rotary.dtype), rotary[offset:offset + size]))
        patch_indices.append(torch.arange(enlarged_offset + 1, enlarged_offset + size + 1, device=hidden_states.device))
        offset += size
        enlarged_offset += size + 1
    hidden_states = self.pre_layernorm(torch.cat(hidden_parts))
    rotary = torch.cat(rotary_parts)
    embeddings = torch.cat((rotary, rotary), dim=-1)
    position_embeddings = (embeddings.cos(), embeddings.sin())
    cu = F.pad((lengths + 1).cumsum(0, dtype=torch.int32), (1, 0)).to(hidden_states.device)
    for block in self.blocks:
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                block.__call__, hidden_states, cu, None, position_embeddings
            )
        else:
            hidden_states = block(hidden_states, cu_seqlens=cu, position_embeddings=position_embeddings)
    return self.merger(hidden_states.index_select(0, torch.cat(patch_indices)))


@lru_cache
def prepare_hf_model(path: str):
    """Load the trusted local implementation and patch only its RICE class."""
    import transformers.modeling_flash_attention_utils as hf_flash
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    if not hasattr(hf_flash, "flash_attn_varlen_func"):
        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError as exc:
            raise RuntimeError("LLaVA-OneVision-1.5 requires the installed flash-attn CUDA kernel") from exc
        hf_flash.flash_attn_varlen_func = flash_attn_varlen_func
    data = json.loads((Path(path) / "config.json").read_text())
    model_class = get_class_from_dynamic_module(
        data["auto_map"]["AutoModelForCausalLM"], path, local_files_only=True
    )
    import sys

    module = sys.modules[model_class.__module__]
    module.RiceTransformerPretrainedModel.forward = rice_forward
    return model_class


def packed_sequence_metadata(position_ids):
    if position_ids.ndim != 2 or position_ids.shape[0] != 1 or position_ids.numel() == 0:
        raise ValueError("LLaVA packing requires nonempty text position_ids of shape (1, tokens)")
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("LLaVA packed position_ids must be integers")
    positions = position_ids[0]
    starts = torch.nonzero(positions == 0, as_tuple=False).flatten()
    if starts.numel() == 0 or starts[0].item() != 0:
        raise ValueError("Each packed LLaVA sequence must start at position zero")
    valid = (positions[1:] == positions[:-1] + 1) | (positions[1:] == 0)
    if not bool(valid.all()):
        raise ValueError("LLaVA packed positions must increase by one within each sequence")
    cu = torch.cat((starts, starts.new_tensor([positions.numel()]))).to(torch.int32)
    maximum = int((cu[1:] - cu[:-1]).max().item())
    return cu, maximum


def llava_packed_attention_forward(
    self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
    output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None,
):
    """Scoped FA2 adapter; cached generation and padded batches retain HF code.

    Keep the checkpoint's Q/K normalization, RoPE, GQA expansion, dtype cast,
    dropout and sliding-window settings. Only supply the missing varlen bounds.
    This function also runs during gradient-checkpoint recomputation, so bounds
    are derived from its tensor arguments, NOT mutable module/global context.
    """
    if (attention_mask is not None or past_key_value is not None or use_cache
            or position_ids is None or hidden_states.shape[0] != 1):
        return type(self)._verl_dense_forward(
            self, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions,
            use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings,
        )
    if hidden_states.shape[0] != 1 or position_ids.shape[-1] != hidden_states.shape[1]:
        raise ValueError("LLaVA padding-free attention requires one flattened batch with aligned text positions")
    if output_attentions:
        raise ValueError("LLaVA packed FlashAttention does not return attention weights")
    import sys

    module = sys.modules[type(self).__module__]
    cu, maximum = packed_sequence_metadata(position_ids)
    shape = hidden_states.shape[:-1]
    head_shape = (*shape, -1, self.head_dim)
    q = self.q_norm(self.q_proj(hidden_states).view(head_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)
    q, k = module.apply_rotary_pos_emb(q, k, *position_embeddings)
    k = module.repeat_kv(k, self.num_key_value_groups)
    v = module.repeat_kv(v, self.num_key_value_groups)
    if q.dtype == torch.float32:
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = getattr(self.config, "_pre_quantization_dtype", self.q_proj.weight.dtype)
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    window = None
    if self.config.use_sliding_window and getattr(self.config, "sliding_window", None) is not None:
        if self.layer_idx >= self.config.max_window_layers:
            window = self.config.sliding_window
    output = module._flash_attention_forward(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None, shape[1],
        is_causal=self.is_causal, dropout=self.attention_dropout if self.training else 0.0,
        sliding_window=window, use_top_left_mask=self._flash_attn_uses_top_left_mask,
        cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=maximum, max_length_k=maximum,
    )
    return self.o_proj(output.reshape(*shape, -1).contiguous()), None, past_key_value


def enable_padding_free(path: str, ulysses_size: int = 1):
    """Patch only this checkpoint's language FA2 class; Qwen/vLLM stay intact."""
    if ulysses_size != 1:
        raise ValueError("LLaVA padding-free currently requires ulysses_size=1; sequence parallelism is not validated")
    import sys

    cls = prepare_hf_model(path)
    attention = sys.modules[cls.__module__].LLaVAOneVision1_5_FlashAttention2
    if not hasattr(attention, "_verl_dense_forward"):
        attention._verl_dense_forward = attention.forward
        attention.forward = llava_packed_attention_forward


def configure_processor(processor, min_pixels, max_pixels):
    """Use identical processor limits in the actor, dataset and vLLM."""
    if processor is None:
        return
    is_qwen3_vl = "Qwen3VLProcessor" in processor.__class__.__name__
    if getattr(processor, "_verl_position_ids_type", None) != "text" and not is_qwen3_vl:
        return
    image_processor = processor.image_processor
    for key, value, size_key in (("min_pixels", min_pixels, "shortest_edge"),
                                 ("max_pixels", max_pixels, "longest_edge")):
        if value is not None:
            setattr(image_processor, key, value)
            if isinstance(getattr(image_processor, "size", None), dict):
                image_processor.size[size_key] = value


def configure_generation(model, processor):

    config = model.config
    if getattr(config, "model_type", None) != "llavaonevision1_5":
        return
    vocab = processor.tokenizer.get_vocab()
    resolved = {}
    for field, token in (
        ("vision_start_token_id", "<|vision_start|>"),
        ("image_token_id", "<|image_pad|>"),
        ("video_token_id", "<|video_pad|>"),
    ):
        token_id = vocab.get(token)
        if token_id is None:
            raise ValueError(f"LLaVA generation tokenizer is missing {token!r}")
        existing = getattr(config, field, None)
        if existing is not None and existing != token_id:
            raise ValueError(
                f"LLaVA {field}={existing} disagrees with tokenizer {token!r}={token_id}"
            )
        resolved[field] = token_id
    # Validate all IDs before mutating config; no partial repair on mismatch.
    for field, token_id in resolved.items():
        setattr(config, field, token_id)


def register_vllm_model():
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.models:
        ModelRegistry.register_model(
            ARCHITECTURE, "verl.models.llava_onevision15_vllm:LlavaOneVision15ForConditionalGeneration"
        )


def load_policy_model(path, **kwargs):
    """Shared evaluation loader, retaining the native Qwen loading path."""
    if is_llava_onevision15(path):
        cls = prepare_hf_model(path)
        return cls.from_pretrained(path, **kwargs)
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(path, **kwargs)
