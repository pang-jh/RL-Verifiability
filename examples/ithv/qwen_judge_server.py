
import argparse
import base64
import hashlib
import importlib.util
import io
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForImageTextToText, AutoProcessor


REWARD_FUNCTION_DIR = Path(__file__).resolve().parents[1] / "reward_function"
reward_spec = importlib.util.spec_from_file_location("ithv_reward_for_judge", REWARD_FUNCTION_DIR / "ithv.py")
if reward_spec is None or reward_spec.loader is None:
    raise RuntimeError("Unable to load the ITHV reward protocol")
reward_module = importlib.util.module_from_spec(reward_spec)
reward_spec.loader.exec_module(reward_module)
DEFAULT_JUDGE_MODEL = reward_module.DEFAULT_JUDGE_MODEL
JUDGE_SYSTEM_PROMPT = reward_module.JUDGE_SYSTEM_PROMPT
PROMPT_SHA256 = reward_module.PROMPT_SHA256
make_judge_error = reward_module.make_judge_error
parse_verdict = reward_module.parse_verdict


class JudgeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_base64: str
    question: str
    reference: str
    candidate: str
    task_type: str


class JudgeRequest(BaseModel):
    items: list[JudgeItem] = Field(min_length=1, max_length=32)


def build_messages(item: dict, image: Image.Image) -> list[dict]:
    quoted = {key: item[key] for key in ("question", "reference", "candidate", "task_type")}
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": json.dumps(quoted, ensure_ascii=False)},
            ],
        },
    ]


class FrozenQwenJudge:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        batch_size: int = 1,
        max_new_tokens: int = 16,
        audit_path: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
    ) -> None:
        if batch_size < 1 or max_new_tokens < 1:
            raise ValueError("Batch size and output token budget must be positive")
        if not 0 < temperature <= 2 or not 0 < top_p <= 1 or top_k < 0:
            raise ValueError("Sampling settings out of range")
        path = Path(model_path).resolve(strict=True)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for the ITHV judge")

        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        closing_brace_ids = self.processor.tokenizer.encode("}", add_special_tokens=False)
        if len(closing_brace_ids) != 1:
            raise RuntimeError("Judge tokenizer must encode the JSON closing brace as one token")
        self.stop_token_ids = [self.processor.tokenizer.eos_token_id, closing_brace_ids[0]]
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(path),
            dtype={"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype],
            device_map={"": device},
            local_files_only=True,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.lock = threading.Lock()
        self.audit_path = Path(audit_path) if audit_path else None
        self.metadata = {
            "model_path": str(path),
            "prompt_sha256": PROMPT_SHA256,
            "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
            "dtype": dtype,
            "enable_thinking": False,
            "do_sample": True,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop_on_json_close": True,
            "verdict_schema": {"correct": "boolean"},
            "grading_policy": "human-reviewed-reference-v4-correct-only",
            "frozen": not self.model.training
            and not any(parameter.requires_grad for parameter in self.model.parameters()),
        }
        self.metadata["judge_id"] = hashlib.sha256(
            json.dumps(self.metadata, sort_keys=True).encode()
        ).hexdigest()

    @torch.inference_mode()
    def _generate(self, conversations: list[list[dict]]) -> list[str]:
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            enable_thinking=False,
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            eos_token_id=self.stop_token_ids,
            return_dict_in_generate=False,
        )
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)

    def _audit_failure(self, item: dict, raw: str, error: Exception) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "judge_id": self.metadata["judge_id"],
            "image_sha256": hashlib.sha256(base64.b64decode(item["image_base64"])).hexdigest(),
            **{key: item[key] for key in ("question", "reference", "candidate", "task_type")},
            "event": "judge_failure",
            "raw": raw,
            "error": str(error),
        }
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _grade_candidate(self, item: dict, raw: str) -> dict:
        try:
            return parse_verdict(raw.strip())
        except ValueError as error:
            self._audit_failure(item, raw, error)
            return make_judge_error()

    @torch.inference_mode()
    def judge(self, items: list[dict]) -> list[dict]:
        verdicts = []
        decoded_images: dict[str, Image.Image] = {}
        with self.lock:
            for start in range(0, len(items), self.batch_size):
                batch = items[start : start + self.batch_size]
                conversations = []
                for item in batch:
                    image_key = hashlib.sha256(item["image_base64"].encode("ascii")).hexdigest()
                    if image_key not in decoded_images:
                        payload = base64.b64decode(item["image_base64"], validate=True)
                        with Image.open(io.BytesIO(payload)) as opened:
                            image = opened.convert("RGB")
                            image.load()
                        decoded_images[image_key] = image
                    image = decoded_images[image_key]
                    conversations.append(build_messages(item, image))
                decoded = self._generate(conversations)
                verdicts.extend(
                    self._grade_candidate(item, raw) for item, raw in zip(batch, decoded, strict=True)
                )
        return verdicts


def create_app(judge: FrozenQwenJudge) -> FastAPI:
    app = FastAPI(title="Frozen ITHV Qwen judge")

    @app.get("/health")
    def health() -> dict:
        return judge.metadata

    @app.post("/judge")
    def evaluate(request: JudgeRequest) -> dict:
        try:
            verdicts = judge.judge([item.model_dump() for item in request.items])
            return {"judge_id": judge.metadata["judge_id"], "verdicts": verdicts}
        except (ValueError, RuntimeError, OSError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_JUDGE_MODEL, required=DEFAULT_JUDGE_MODEL is None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--audit-path", default="outputs/judge_failures.jsonl")
    args = parser.parse_args()

    judge = FrozenQwenJudge(
        args.model_path,
        args.device,
        args.dtype,
        args.batch_size,
        args.max_new_tokens,
        audit_path=args.audit_path,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    print(json.dumps(judge.metadata, ensure_ascii=False), flush=True)
    uvicorn.run(create_app(judge), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
