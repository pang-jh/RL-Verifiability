
from typing import Optional

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin


def get_tokenizer(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> PreTrainedTokenizer:
    """Create a huggingface pretrained tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    if override_chat_template is not None:
        with open(override_chat_template) as f:
            tokenizer.chat_template = f.read()

        print(f"New chat template: {tokenizer.chat_template}")

    if tokenizer.bos_token == "<bos>" and tokenizer.eos_token == "<eos>":
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    if tokenizer.pad_token_id is None:
        print("Pad token is None. Set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_processor(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> Optional[ProcessorMixin]:
    """Create a huggingface pretrained processor."""
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    from ..models.llava_onevision15 import is_llava_onevision15

    if processor is not None and is_llava_onevision15(model_path):
        processor._verl_position_ids_type = "text"
    if override_chat_template is not None:
        with open(override_chat_template) as f:
            processor.chat_template = f.read()

        print(f"New chat template: {processor.chat_template}")

    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor
