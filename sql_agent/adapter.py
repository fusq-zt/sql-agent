"""Agent Lightning integration through its supported LitAgent/trace adapter APIs."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import asyncio
import json
import time
from typing import Any

from agentlightning import LitAgent
from agentlightning.adapter.triplet import TraceToTripletBase
from agentlightning.types import Span, SpanCoreFields, TraceStatus, Triplet

from .contracts import ContractError
from .graph import InteractiveSQLGraph
from .trajectory import Trajectory, training_row


TRAJECTORY_SPAN = "reex.finalized_trajectory"
TRAJECTORY_ATTRIBUTE = "reex.trajectory.json"


def trajectory_spans(trajectory: Trajectory) -> list[SpanCoreFields]:
    trajectory.validate()
    if trajectory.reward is None:
        raise ContractError("A training trajectory must have one terminal reward")
    spans: list[SpanCoreFields] = []
    now = time.time()
    # These diagnostic spans contain no reward. The adapter consumes exactly one final ledger.
    for index, segment in enumerate(trajectory.segments):
        attributes: dict[str, Any] = {
            "reex.origin": segment.origin, "reex.call_index": segment.call_index,
            "reex.token_ids": segment.token_ids, "reex.text": segment.text,
        }
        if segment.origin == "policy":
            call = trajectory.calls[segment.call_index]
            attributes.update({"reex.prompt_token_ids": call.prompt_ids,
                               "reex.response_id": call.response_id,
                               "reex.policy_version": call.policy_version})
        spans.append(SpanCoreFields(name=f"reex.{segment.origin}", status=TraceStatus(status_code="OK"),
                                    attributes=attributes, start_time=now, end_time=now))
    spans.append(SpanCoreFields(
        name=TRAJECTORY_SPAN, status=TraceStatus(status_code="OK"), start_time=now, end_time=now,
        attributes={TRAJECTORY_ATTRIBUTE: json.dumps(trajectory.to_dict(), ensure_ascii=False),
                    "reex.terminal_reward": trajectory.reward,
                    "reex.reward_protocol": trajectory.reward_protocol or ""},
    ))
    return spans


class FullTrajectoryAdapter(TraceToTripletBase):
    """One finalized ledger -> one triplet with its original alternating mask."""
    def adapt(self, source: Sequence[Span], /) -> list[Triplet]:
        final = [s for s in source if s.name == TRAJECTORY_SPAN]
        if len(final) != 1:
            raise ContractError(f"Expected one finalized trajectory, found {len(final)}")
        span = final[0]
        trajectory = Trajectory.from_dict(json.loads(span.attributes[TRAJECTORY_ATTRIBUTE]))
        if trajectory.rollout_id != span.rollout_id or trajectory.attempt_id != span.attempt_id:
            raise ContractError("Trace identity differs from the Lightning rollout/attempt")
        row = training_row(trajectory)
        return [Triplet(
            prompt={"token_ids": row.prompt_ids},
            response={"token_ids": row.response_ids, "loss_mask": row.response_mask},
            reward=row.reward,
            metadata={"reex_trajectory": trajectory.to_dict(), "group_id": row.group_id,
                      "policy_version": row.policy_version, "call_count": len(trajectory.calls),
                      "omitted_terminal_observation_tokens": row.omitted_terminal_observation_tokens},
        )]


class ReExSQLAgent(LitAgent[dict[str, Any]]):
    """Graph and evaluator are injected explicitly; no hidden reward/parser defaults."""
    def __init__(self, *, graph_factory: Callable[..., InteractiveSQLGraph],
                 terminal_evaluator: Callable[[dict[str, Any], Trajectory], tuple[float, str]],
                 trajectory_sink: Callable[[Trajectory], None] | None = None):
        super().__init__()
        self.graph_factory = graph_factory
        self.terminal_evaluator = terminal_evaluator
        self.trajectory_sink = trajectory_sink

    async def rollout_async(self, task: dict[str, Any], resources: Any, rollout: Any):
        attempt = getattr(rollout, "attempt", None)
        if attempt is None or not rollout.resources_id:
            raise ContractError("AGL rollout attempt and resource version are required")
        group_id = (rollout.metadata or {}).get("data_id")
        if not group_id:
            raise ContractError("Missing AGL group/data ID")
        # The graph gets only the public task subset. Gold is confined to the evaluator below.
        public_task = {key: task[key] for key in ("question", "database_path", "prompt_ids")}
        graph = self.graph_factory(public_task, resources, rollout)
        trajectory = Trajectory(
            rollout_id=rollout.rollout_id, attempt_id=attempt.attempt_id, group_id=group_id,
            policy_version=rollout.resources_id, prompt_ids=list(public_task["prompt_ids"]),
        )
        trajectory = await graph.run(trajectory)
        if self.trajectory_sink is not None:
            await asyncio.to_thread(self.trajectory_sink, trajectory)
        reward, protocol = await asyncio.to_thread(self.terminal_evaluator, task, trajectory)
        trajectory.attach_reward(reward, protocol)
        return trajectory_spans(trajectory)
