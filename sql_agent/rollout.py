"""AGL-managed rollout batch for SQL questions, full trajectories and reward.

The service holds one fixed policy for all G=8 samples of all questions. The
separate update process starts only after collection finishes and GPU is freed.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

import httpx
from agentlightning import Trainer, setup_logging
from agentlightning.algorithm.fast import FastAlgorithm
from agentlightning.algorithm.utils import with_store
from agentlightning.execution.client_server import ClientServerExecutionStrategy
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.otel import OtelTracer
from transformers import AutoTokenizer

from .contracts import SETTINGS, LengthContract
from .executor import SQLiteExecutor, json_feedback_v1
from .generation import RawTokenCompletionClient, Sampling
from .graph import InteractiveSQLGraph, single_sql_fence_or_bare_v1
from .adapter import FullTrajectoryAdapter, ReExSQLAgent
from .rewards import terminal_evaluator
from .trajectory import Trajectory, training_row, validate_groups


class Generator:
    def __init__(self, base_url, eos_ids):
        self.base_url, self.eos_ids = base_url, eos_ids

    async def generate(self, prompt_ids, **kwargs):
        async with httpx.AsyncClient(timeout=600) as client:
            transport = RawTokenCompletionClient(client=client, base_url=self.base_url,
                model=SETTINGS.model_id, api_key=None, eos_token_ids=self.eos_ids)
            return await transport.generate(prompt_ids, **kwargs)


class GraphFactory:
    def __init__(self, backend_url, tokenizer, eos_ids):
        self.backend_url, self.tokenizer, self.eos_ids = backend_url, tokenizer, eos_ids

    def __call__(self, task, resources, rollout):
        # AGL owns rollout IDs, queue, runners, policy resource version and traces.
        # The immutable per-update vLLM service is called directly: the full token
        # ledger already records each call, so no LiteLLM proxy rewrite is needed.
        return InteractiveSQLGraph(generator=Generator(self.backend_url, self.eos_ids),
            codec=self.tokenizer, executor=SQLiteExecutor(task["database_path"], timeout_seconds=30),
            length=LengthContract("total_context", "project-v1: prompt+policy+environment <=4096"),
            sampling=Sampling(1., 1., -1, 1.), training=True,
            extract_sql=single_sql_fence_or_bare_v1, render_feedback=json_feedback_v1)


class TrajectoryJournal:
    """Keep original sampled tokens even if the terminal evaluator fails."""
    def __init__(self, output):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)

    def __call__(self, trajectory):
        (self.output/f"{trajectory.rollout_id}.json").write_text(
            json.dumps(trajectory.to_dict(), ensure_ascii=False), encoding="utf-8")


class BatchAlgorithm(FastAlgorithm):
    def __init__(self, output, model_path):
        super().__init__()
        self.output, self.model_path = output, model_path
        self.trajectories = []

    @with_store
    async def run(self, store, train_dataset=None, val_dataset=None):
        resource = await store.add_resources({})
        queued = []
        for task in train_dataset:
            for _ in range(8):
                queued.append(await store.enqueue_rollout(input=task, mode="train",
                    resources_id=resource.resources_id, metadata={"data_id":task["data_id"]}))
        started = time.monotonic()
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output/"trajectory-progress.jsonl").open("w", encoding="utf-8") as journal:
            for item in queued:
                while True:
                    current = await store.get_rollout_by_id(item.rollout_id)
                    if current.status in {"failed", "cancelled"}:
                        raise RuntimeError(f"Rollout failed: {current.rollout_id}; batch not updated")
                    if current.status == "succeeded":
                        break
                    await asyncio.sleep(.2)
                spans = await store.query_spans(item.rollout_id, attempt_id="latest")
                triplets = self.get_adapter().adapt(spans)
                trajectory = Trajectory.from_dict(triplets[0].metadata["reex_trajectory"])
                self.trajectories.append(trajectory)
                journal.write(json.dumps(trajectory.to_dict(), ensure_ascii=False)+"\n")
                journal.flush()
                count = len(self.trajectories)
                progress = {"completed":count, "total":len(queued), "model":str(self.model_path),
                    "elapsed_seconds":time.monotonic()-started, "updated_unix":time.time()}
                (self.output/"progress.json").write_text(json.dumps(progress))
                if count % 8 == 0:
                    print(json.dumps(progress), flush=True)
        validate_groups([training_row(t) for t in self.trajectories])
        (self.output/"trajectories.json").write_text(
            json.dumps([t.to_dict() for t in self.trajectories], ensure_ascii=False), encoding="utf-8")


def run(model, tasks, output, backend_url, runners):
    setup_logging(level="INFO", console=True)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["REEX_COMPLETION_DIAGNOSTICS"] = str(output/"completion-errors")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    eos_ids = json.loads((model/"generation_config.json").read_text())["eos_token_id"]
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    store = InMemoryLightningStore(thread_safe=True)
    algorithm = BatchAlgorithm(output, model)
    trainer = Trainer(n_runners=runners, algorithm=algorithm, adapter=FullTrajectoryAdapter(),
        tracer=OtelTracer(), store=store,
        strategy=ClientServerExecutionStrategy(n_runners=runners, main_process="algorithm",
            server_host="127.0.0.1", server_port=4747))
    agent = ReExSQLAgent(graph_factory=GraphFactory(backend_url, tokenizer, eos_ids),
        terminal_evaluator=terminal_evaluator, trajectory_sink=TrajectoryJournal(output/"unscored"))
    trainer.dev(agent, train_dataset=json.loads(tasks.read_text()))
    trajectories = algorithm.trajectories
    report = {"rollouts":len(trajectories), "questions":len({t.group_id for t in trajectories}),
        "multi_turn_rollouts":sum(len(t.calls)>1 for t in trajectories),
        "mean_reward":sum(t.reward for t in trajectories)/len(trajectories),
        "result_correct":sum(t.reward_components["result"] for t in trajectories),
        "format_correct":sum(t.reward_components["format"] for t in trajectories),
        "model":str(model), "exploration_reward":False,
        "policy_versions":sorted({t.policy_version for t in trajectories})}
    (output/"rollout-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["model", "tasks", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--runners", type=int, default=4)
    args = parser.parse_args()
    run(args.model, args.tasks, args.output, args.backend_url, args.runners)
