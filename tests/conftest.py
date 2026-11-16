from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sql_agent.contracts import LengthContract
from sql_agent.generation import Generated, Sampling
from sql_agent.trajectory import Trajectory


class ByteCodec:
    """Deterministic test codec, never used as a research model tokenizer."""
    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return [b + 1 for b in text.encode('utf-8')]

    def decode(self, ids, *, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return bytes(i - 1 for i in ids).decode('utf-8')


class ScriptedGenerator:
    def __init__(self, scripts, codec=None):
        self.scripts = scripts
        self.codec = codec or ByteCodec()
        self.requests = []

    async def generate(self, prompt_ids, *, max_new_tokens, sampling, policy_version):
        index = len(self.requests)
        self.requests.append(list(prompt_ids))
        script = self.scripts[index]
        text = script(self.codec.decode(prompt_ids)) if callable(script) else script
        ids = self.codec.encode(text)
        assert len(ids) <= max_new_tokens
        return Generated(list(prompt_ids), ids, text, f'fixture-call-{index}', 'stop', policy_version)


@pytest.fixture
def database(tmp_path):
    path = tmp_path/'fixture.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE employee (id INTEGER, department TEXT, salary INTEGER)')
        db.executemany('INSERT INTO employee VALUES (?, ?, ?)',
                       [(1,'sales',100),(2,'sales',200),(3,'dev',300),(4,'dev',400),(5,'hr',500)])
    return path


@pytest.fixture
def codec():
    return ByteCodec()


@pytest.fixture
def length():
    return LengthContract('total_context', 'DIAGNOSTIC_FIXTURE_ONLY; not a research protocol choice')


@pytest.fixture
def sampling():
    # Unspecified sampler fields are test-fixture inputs, not production defaults.
    return Sampling(temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0)


def fresh(codec, index=0):
    return Trajectory(f'rollout-{index}',f'attempt-{index}','fixture-group','fixture-policy',codec.encode('Question'))


def ledger(index=0, reward=None):
    t=Trajectory(f'rollout-{index}',f'attempt-{index}','fixture-group','fixture-policy',[101,102])
    t.append_policy(prompt_ids=[101,102],response_ids=[11,12],text='action1',
                    response_id=f'r{index}-0',finish_reason='stop',policy_version='fixture-policy')
    t.append_environment([91,92,93],'observation')
    t.append_policy(prompt_ids=t.context_ids,response_ids=[13,14],text='action2',
                    response_id=f'r{index}-1',finish_reason='stop',policy_version='fixture-policy')
    t.termination='model_end'
    t.final_sql='SELECT 1'
    t.attach_reward(float(index % 2) if reward is None else reward,'DIAGNOSTIC_FIXTURE_BINARY')
    return t
