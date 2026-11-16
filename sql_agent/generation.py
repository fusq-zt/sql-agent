"""Raw token completion interface. Missing token IDs are fatal, never re-created."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Protocol, Any
from uuid import uuid4

from .contracts import ContractError
from .trajectory import require_ids


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


@dataclass(frozen=True)
class Generated:
    prompt_ids: list[int]
    token_ids: list[int]
    text: str
    response_id: str
    finish_reason: str
    policy_version: str


class TokenCodec(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, ids: list[int], *, skip_special_tokens: bool = True,
               clean_up_tokenization_spaces: bool = False) -> str: ...


class Generator(Protocol):
    async def generate(self, prompt_ids: list[int], *, max_new_tokens: int,
                       sampling: Sampling, policy_version: str) -> Generated: ...


def sampled_text(result: Generated, codec: TokenCodec) -> str:
    """Validate server text against atomic sampled IDs without editing those IDs.

    vLLM 0.11's check_stop_strings truncates display text at a character boundary,
    while retaining the complete last token. A closing-tag token can also encode
    a newline or the start of another tag (e.g. Qwen token 1472 is `></`). Accept
    only a display cutoff inside the token that first completes the stop string.
    Source: vllm/v1/engine/detokenizer.py, v0.11.0, check_stop_strings/update.
    """
    decoded = codec.decode(result.token_ids, skip_special_tokens=True,
                           clean_up_tokenization_spaces=False)
    if decoded == result.text:
        return decoded
    stop = "</intermediate_sql>"
    if (result.finish_reason == "stop" and result.text.endswith(stop)
            and decoded.startswith(result.text)):
        before_last = codec.decode(result.token_ids[:-1], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
        if stop not in before_last:
            return decoded
    # Preserve the actual failing response, rather than losing it when AGL stops
    # a failed batch. This diagnostic does not repair, resample or alter tokens.
    if directory := os.environ.get("REEX_COMPLETION_DIAGNOSTICS"):
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination/f"mismatch-{uuid4().hex}.json").write_text(
            json.dumps({"generated":asdict(result), "decoded":decoded}, ensure_ascii=False),
            encoding="utf-8")
    raise ContractError("Returned completion text differs from the actual sampled IDs")


class RawTokenCompletionClient:
    """AGL rollout-routed /completions candidate transport, capability-tested before use."""
    def __init__(self, *, client: Any, base_url: str, model: str, api_key: str | None,
                 eos_token_ids: list[int]):
        if not eos_token_ids:
            raise ContractError("EOS IDs must be verified from the model tokenizer/config")
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.eos_token_ids = eos_token_ids

    async def generate(self, prompt_ids: list[int], *, max_new_tokens: int,
                       sampling: Sampling, policy_version: str) -> Generated:
        require_ids(prompt_ids)
        if max_new_tokens <= 0:
            raise ContractError("No generation budget remains")
        payload = {
            # A singleton token batch is semantically identical to one token prompt.
            # LiteLLM 1.80 text_completion accepts list[list[int]], not list[int].
            "model": self.model, "prompt": [prompt_ids], "max_tokens": max_new_tokens,
            "temperature": sampling.temperature, "top_p": sampling.top_p,
            "top_k": sampling.top_k, "repetition_penalty": sampling.repetition_penalty,
            "stop": ["</intermediate_sql>"], "stop_token_ids": self.eos_token_ids,
            "include_stop_str_in_output": True, "return_token_ids": True,
            "add_special_tokens": False, "stream": False, "n": 1,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = await self.client.post(self.base_url + "/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if len(choices) != 1:
            raise ContractError("Expected exactly one completion fragment per call")
        choice = choices[0]
        # vLLM v0.11.0 CompletionResponseChoice carries BOTH ID arrays.
        # Source: b8b302cde434df8c9289a2b465406b47ebab1c2d,
        # vllm/entrypoints/openai/protocol.py (CompletionResponseChoice).
        prompt = choice.get("prompt_token_ids")
        generated = choice.get("token_ids")
        require_ids(prompt)
        require_ids(generated)
        if prompt != prompt_ids:
            raise ContractError("Completion service rewrote the requested token prefix")
        if choice.get("finish_reason") not in {"stop", "length"}:
            raise ContractError("Unsupported finish reason from completion service")
        if not isinstance(choice.get("text"), str) or not data.get("id"):
            raise ContractError("Missing completion text or request identity")
        return Generated(prompt, generated, choice["text"], data["id"], choice["finish_reason"], policy_version)
