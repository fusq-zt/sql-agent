"""Token-origin ledger, exact causal-prefix checks, and trajectory batching."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal

from .contracts import ContractError, SETTINGS


def require_ids(ids: list[int], *, allow_empty: bool = False) -> None:
    if not isinstance(ids, list) or (not ids and not allow_empty):
        raise ContractError("Token IDs must be a nonempty list")
    if any(type(v) is not int or v < 0 for v in ids):
        raise ContractError("Token IDs must be nonnegative integers")


@dataclass
class Segment:
    origin: Literal["policy", "environment"]
    token_ids: list[int]
    text: str
    call_index: int


@dataclass
class PolicyCall:
    index: int
    prompt_ids: list[int]
    response_ids: list[int]
    response_id: str
    finish_reason: str
    policy_version: str


@dataclass
class Trajectory:
    rollout_id: str
    attempt_id: str
    group_id: str
    policy_version: str
    prompt_ids: list[int]
    segments: list[Segment] = field(default_factory=list)
    calls: list[PolicyCall] = field(default_factory=list)
    interactions: int = 0
    termination: str | None = None
    final_sql: str | None = None
    reward: float | None = None
    reward_protocol: str | None = None
    length_contract: dict[str, Any] = field(default_factory=dict)
    reward_components: dict[str, Any] = field(default_factory=dict)

    @property
    def context_ids(self) -> list[int]:
        return self.prompt_ids + [token for segment in self.segments for token in segment.token_ids]

    def append_policy(self, *, prompt_ids: list[int], response_ids: list[int], text: str,
                      response_id: str, finish_reason: str, policy_version: str) -> None:
        require_ids(prompt_ids)
        require_ids(response_ids)
        if self.termination is not None:
            raise ContractError("Cannot append to a terminated trajectory")
        if prompt_ids != self.context_ids:
            raise ContractError("Causal token prefix mismatch; no retokenization or repair is allowed")
        if policy_version != self.policy_version:
            raise ContractError("Policy version changed within a rollout")
        if not response_id or any(c.response_id == response_id for c in self.calls):
            raise ContractError("Missing or duplicate policy response ID")
        if self.segments and self.segments[-1].origin != "environment":
            raise ContractError("A resumed call must follow a real environment observation")
        index = len(self.calls)
        self.calls.append(PolicyCall(index, list(prompt_ids), list(response_ids), response_id,
                                     finish_reason, policy_version))
        self.segments.append(Segment("policy", list(response_ids), text, index))

    def append_environment(self, token_ids: list[int], text: str) -> None:
        require_ids(token_ids)
        if self.termination is not None:
            raise ContractError("Cannot append observations after termination")
        if not self.segments or self.segments[-1].origin != "policy":
            raise ContractError("An observation must follow a policy action")
        if self.interactions >= SETTINGS.max_interactions:
            raise ContractError("Interaction limit exceeded")
        self.segments.append(Segment("environment", list(token_ids), text, len(self.calls) - 1))
        self.interactions += 1

    def attach_reward(self, value: float, protocol: str) -> None:
        if self.termination is None or self.reward is not None:
            raise ContractError("Exactly one terminal reward is required")
        if not math.isfinite(value) or not protocol:
            raise ContractError("Reward must be finite and name its protocol")
        self.reward = float(value)
        self.reward_protocol = protocol

    def validate(self) -> None:
        require_ids(self.prompt_ids)
        if not all((self.rollout_id, self.attempt_id, self.group_id, self.policy_version)):
            raise ContractError("Missing trajectory identity")
        if self.termination is None or not self.calls or len(self.calls) > SETTINGS.max_interactions:
            raise ContractError("Incomplete or empty trajectory cannot enter training")
        context = list(self.prompt_ids)
        policy_count = 0
        environment_count = 0
        response_ids: set[str] = set()
        previous_origin: str | None = None
        for segment in self.segments:
            require_ids(segment.token_ids)
            if segment.origin == "policy":
                if previous_origin == "policy" or policy_count >= len(self.calls):
                    raise ContractError("Invalid policy/environment alternation")
                call = self.calls[policy_count]
                if (call.index != policy_count or segment.call_index != policy_count
                        or call.prompt_ids != context or call.response_ids != segment.token_ids
                        or call.policy_version != self.policy_version
                        or not call.response_id or call.response_id in response_ids):
                    raise ContractError("Stored call does not reproduce the exact causal token history")
                response_ids.add(call.response_id)
                policy_count += 1
            elif segment.origin == "environment":
                if previous_origin != "policy" or segment.call_index != policy_count - 1:
                    raise ContractError("Orphan environment segment")
                environment_count += 1
            else:
                raise ContractError("Unrecognized token origin")
            context.extend(segment.token_ids)
            previous_origin = segment.origin
        if (policy_count != len(self.calls) or environment_count != self.interactions
                or self.interactions > SETTINGS.max_interactions):
            raise ContractError("Trace counters do not match token provenance")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Trajectory:
        data = dict(payload)
        data["segments"] = [Segment(**s) for s in data["segments"]]
        data["calls"] = [PolicyCall(**c) for c in data["calls"]]
        result = cls(**data)
        result.validate()
        return result


@dataclass
class TrainingRow:
    rollout_id: str
    group_id: str
    policy_version: str
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    reward: float
    omitted_terminal_observation_tokens: int


def training_row(trajectory: Trajectory) -> TrainingRow:
    trajectory.validate()
    if trajectory.reward is None or not math.isfinite(trajectory.reward) or not trajectory.reward_protocol:
        raise ContractError("Missing terminal reward; zero-fill is forbidden")
    segments = trajectory.segments
    omitted = 0
    if segments[-1].origin == "environment":
        omitted = len(segments[-1].token_ids)
        segments = segments[:-1]  # No later model action conditioned on this observation.
    ids: list[int] = []
    mask: list[int] = []
    for segment in segments:
        ids.extend(segment.token_ids)
        mask.extend([int(segment.origin == "policy")] * len(segment.token_ids))
    return TrainingRow(trajectory.rollout_id, trajectory.group_id, trajectory.policy_version,
                       list(trajectory.prompt_ids), ids, mask, trajectory.reward, omitted)


def validate_groups(rows: list[TrainingRow]) -> None:
    if not rows or len({r.rollout_id for r in rows}) != len(rows):
        raise ContractError("Empty batch or duplicate rollout IDs")
    counts = Counter(r.group_id for r in rows)
    if any(count != SETTINGS.group_size for count in counts.values()):
        raise ContractError(f"Every group must contain exactly {SETTINGS.group_size} independent rollouts: {dict(counts)}")
    if len({r.policy_version for r in rows}) != 1:
        raise ContractError("Training batch mixes policy versions")
    for group_id in counts:
        prompts = {tuple(r.prompt_ids) for r in rows if r.group_id == group_id}
        if len(prompts) != 1:
            raise ContractError("A GRPO group must share the same input tokens")


def pack_rows(rows: list[TrainingRow], *, max_prompt_length: int, max_response_length: int,
              pad_token_id: int) -> dict[str, Any]:
    """Pure Python packing, independently testable without torch/VERL. Never truncates."""
    validate_groups(rows)
    if max_prompt_length <= 0 or max_response_length <= 0:
        raise ContractError("Packing capacity must be positive")
    data: dict[str, Any] = {key: [] for key in (
        "prompts", "responses", "input_ids", "attention_mask", "position_ids", "response_mask",
        "token_level_scores", "is_drop_mask", "data_id_list", "rollout_id_list", "policy_version")}
    for row in rows:
        require_ids(row.prompt_ids)
        require_ids(row.response_ids)
        if (len(row.response_ids) != len(row.response_mask) or not row.response_mask[-1]
                or any(x not in (0, 1) for x in row.response_mask) or not math.isfinite(row.reward)):
            raise ContractError("Invalid response mask or terminal score")
        if len(row.prompt_ids) > max_prompt_length or len(row.response_ids) > max_response_length:
            raise ContractError("Scored trajectory exceeds packing capacity; refusing post-score truncation")
        lp = max_prompt_length - len(row.prompt_ids)
        rp = max_response_length - len(row.response_ids)
        prompt = [pad_token_id] * lp + row.prompt_ids
        response = row.response_ids + [pad_token_id] * rp
        attention = [0] * lp + [1] * (len(row.prompt_ids) + len(row.response_ids)) + [0] * rp
        position: list[int] = []
        cursor = 0
        for visible in attention:
            cursor += visible
            position.append(max(0, cursor - 1))
        scores = [0.0] * max_response_length
        scores[len(row.response_ids) - 1] = row.reward
        for key, value in {
            "prompts": prompt, "responses": response, "input_ids": prompt + response,
            "attention_mask": attention, "position_ids": position,
            "response_mask": row.response_mask + [0] * rp, "token_level_scores": scores,
            "is_drop_mask": False, "data_id_list": row.group_id,
            "rollout_id_list": row.rollout_id, "policy_version": row.policy_version,
        }.items():
            data[key].append(value)
    return data
