"""Real LangGraph policy/executor loop implementing Algorithm 1 control flow."""
from __future__ import annotations

from dataclasses import asdict
import re
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .contracts import ContractError, LengthContract, SETTINGS
from .executor import Observation, SQLiteExecutor
from .generation import Generator, Sampling, TokenCodec, sampled_text
from .trajectory import Trajectory


def single_sql_fence_or_bare_v1(text: str, tag: str) -> str | None:
    """Versioned candidate parser. Production use requires an explicit protocol choice."""
    opening, closing = f"<{tag}>", f"</{tag}>"
    if opening not in text and closing not in text:
        return None
    if text.count(opening) != 1 or text.count(closing) != 1:
        raise ContractError(f"Ambiguous or incomplete {tag} markup")
    match = re.search(re.escape(opening) + r"(.*?)" + re.escape(closing), text, re.S)
    if match is None:
        raise ContractError(f"Invalid {tag} tag order")
    body = match.group(1).strip()
    fences = re.findall(r"```(?:sql)?\s*\n?(.*?)```", body, re.S | re.I)
    if "```" in body:
        if len(fences) != 1 or body.count("```") != 2:
            raise ContractError("Ambiguous SQL code fence")
        body = fences[0].strip()
    if not body:
        raise ContractError("Empty SQL payload")
    return body


class GraphState(TypedDict):
    trajectory: Trajectory
    pending_sql: str | None


class InteractiveSQLGraph:
    def __init__(self, *, generator: Generator, codec: TokenCodec, executor: SQLiteExecutor,
                 length: LengthContract, sampling: Sampling, training: bool,
                 extract_sql: Callable[[str, str], str | None],
                 render_feedback: Callable[[Observation], str]):
        expected_temperature = SETTINGS.train_temperature if training else SETTINGS.eval_temperature
        if sampling.temperature != expected_temperature:
            raise ContractError("Unexpected sampling temperature")
        self.generator, self.codec, self.executor = generator, codec, executor
        self.length, self.sampling = length, sampling
        self.extract_sql, self.render_feedback = extract_sql, render_feedback
        graph = StateGraph(GraphState)
        graph.add_node("policy", self._policy)
        graph.add_node("execute", self._execute)
        graph.add_edge(START, "policy")
        graph.add_conditional_edges("policy", self._after_policy, {"execute": "execute", "end": END})
        graph.add_conditional_edges("execute", self._after_execute, {"policy": "policy", "end": END})
        self.compiled = graph.compile()

    def _remaining(self, trajectory: Trajectory) -> int:
        return self.length.remaining(
            len(trajectory.prompt_ids),
            sum(len(s.token_ids) for s in trajectory.segments if s.origin == "policy"),
            sum(len(s.token_ids) for s in trajectory.segments if s.origin == "environment"),
        )

    async def _policy(self, state: GraphState) -> GraphState:
        trajectory = state["trajectory"]
        remaining = self._remaining(trajectory)
        if remaining <= 0:
            trajectory.termination = "sequence_limit"
            return {"trajectory": trajectory, "pending_sql": None}
        result = await self.generator.generate(trajectory.context_ids, max_new_tokens=remaining,
                                               sampling=self.sampling, policy_version=trajectory.policy_version)
        if len(result.token_ids) > remaining:
            raise ContractError("Generator exceeded the requested token budget")
        decoded = sampled_text(result, self.codec)
        trajectory.append_policy(prompt_ids=result.prompt_ids, response_ids=result.token_ids,
                                 text=decoded, response_id=result.response_id,
                                 finish_reason=result.finish_reason, policy_version=result.policy_version)
        # Parse only the new policy fragment; an observation cannot create an action.
        try:
            intermediate = self.extract_sql(decoded, "intermediate_sql")
        except ContractError:
            trajectory.termination = "invalid_intermediate_markup"
            return {"trajectory": trajectory, "pending_sql": None}
        if intermediate is not None:
            # sampled_text already verified this display boundary against the
            # complete atomic token; the ledger above retains its entire suffix.
            if not result.text.rstrip().endswith("</intermediate_sql>"):
                raise ContractError("Generation failed to stop exactly at the intermediate SQL boundary")
            return {"trajectory": trajectory, "pending_sql": intermediate}
        trajectory.termination = "sequence_limit" if result.finish_reason == "length" else "model_end"
        return {"trajectory": trajectory, "pending_sql": None}

    async def _execute(self, state: GraphState) -> GraphState:
        trajectory = state["trajectory"]
        sql = state["pending_sql"]
        if sql is None:
            raise ContractError("Executor has no policy action")
        observation = await self.executor.execute_async(sql)
        feedback = self.render_feedback(observation)
        trajectory.append_environment(self.codec.encode(feedback, add_special_tokens=False), feedback)
        if trajectory.interactions == SETTINGS.max_interactions:
            trajectory.termination = "interaction_limit"
        elif self._remaining(trajectory) <= 0:
            trajectory.termination = "sequence_limit"
        return {"trajectory": trajectory, "pending_sql": None}

    @staticmethod
    def _after_policy(state: GraphState) -> str:
        return "execute" if state["pending_sql"] is not None else "end"

    @staticmethod
    def _after_execute(state: GraphState) -> str:
        return "end" if state["trajectory"].termination is not None else "policy"

    async def run(self, trajectory: Trajectory) -> Trajectory:
        if trajectory.calls or trajectory.segments or trajectory.termination is not None:
            raise ContractError("A graph run must start from a fresh trajectory")
        trajectory.length_contract = asdict(self.length)
        result = await self.compiled.ainvoke(
            {"trajectory": trajectory, "pending_sql": None},
            # Engineering recursion capacity; the actual algorithm cap remains N=10.
            config={"recursion_limit": 2 * SETTINGS.max_interactions + 2},
        )
        completed = result["trajectory"]
        policy_text = "".join(s.text for s in completed.segments if s.origin == "policy")
        try:
            completed.final_sql = self.extract_sql(policy_text, "final_sql")
        except ContractError:
            completed.termination = "invalid_final_markup"
            completed.final_sql = None
        completed.validate()
        return completed
