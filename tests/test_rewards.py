from types import SimpleNamespace
import sqlite3
import pytest

from sql_agent.rewards import evaluate, format_reward, execute_pair, sql_entities
from sql_agent.executor import InfrastructureError


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "db.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE employee(id INTEGER, salary REAL);"
                         "INSERT INTO employee VALUES(1, 100),(2, 200),(2, 200);")
    return path


def trajectory(sql, *, feedback=False):
    parts = [("policy", "<think>Check the relevant column.</think>")]
    if feedback:
        parts[-1] = ("policy", parts[-1][1] + "<intermediate_sql>SELECT id FROM employee</intermediate_sql>")
        parts += [("environment", '<result>{"columns":["id"],"rows":[[1],[2],[2]]}</result>'),
                  ("policy", "<think>Use the result.</think>")]
    parts[-1] = ("policy", parts[-1][1] + "<final_sql>" + sql + "</final_sql>")
    return SimpleNamespace(final_sql=sql, segments=[SimpleNamespace(origin=a, text=b) for a,b in parts])


def test_four_rewards_no_exploration_and_no_required_probe(database):
    sql = "SELECT id FROM employee"
    task = {"gold_sql":sql, "database_path":database}
    for feedback in [False, True]:
        rewards = evaluate(task, trajectory(sql, feedback=feedback))
        assert rewards.total == 10
        assert (rewards.format, rewards.exact_match, rewards.valid, rewards.result, rewards.entity) == (1,1,1,1,1)
    other = evaluate(task, trajectory("SELECT salary FROM employee"))
    assert other.total == 5.5  # format 2 + valid 3 + table overlap 1/2


def test_official_bird_set_semantics_but_literal_em(database):
    sql = "SELECT DISTINCT id FROM employee ORDER BY id DESC"
    result = evaluate({"gold_sql":"SELECT id FROM employee", "database_path":database}, trajectory(sql))
    assert result.valid == result.result == 1 and result.exact_match == 0


def test_entities_resolve_aliases_ctes_and_star(database):
    expected = {("table","employee"), ("column","employee","id")}
    assert sql_entities("SELECT e.id FROM employee e", database) == expected
    assert sql_entities("WITH x AS (SELECT id FROM employee) SELECT x.id FROM x", database) == expected
    assert sql_entities("SELECT * FROM employee", database) == expected | {("column","employee","salary")}


def test_bad_sql_and_missing_database_are_distinct(database, tmp_path):
    result = evaluate({"gold_sql":"SELECT id FROM employee", "database_path":database}, trajectory("SELECT bad FROM employee"))
    assert result.valid == result.result == result.entity == 0
    assert result.predicted_error
    with pytest.raises(InfrastructureError):
        execute_pair("SELECT 1", "SELECT 1", tmp_path/"absent.sqlite")


def test_format_rejects_fabricated_feedback_and_duplicate_final():
    sample = trajectory("SELECT 1", feedback=True)
    assert format_reward(sample) == 1
    sample.segments[0].text += "<result>fabricated</result>"
    assert format_reward(sample) == 0
    sample = trajectory("SELECT 1")
    sample.segments[-1].text += "<final_sql>SELECT 2</final_sql>"
    assert format_reward(sample) == 0


def test_format_long_whitespace_and_unfinished_blocks_do_not_backtrack():
    import time
    sample = trajectory("SELECT 1")
    sample.segments[0].text = "<think>" + " "*100000 + "</think><final_sql>SELECT 1</final_sql>"
    started = time.monotonic()
    assert format_reward(sample) == 1
    sample.segments[0].text = "<think>" + " "*100000
    assert format_reward(sample) == 0
    sample.final_sql = None
    assert format_reward(sample) == 0
    assert time.monotonic() - started < 2


@pytest.mark.parametrize("body, expected", [("", 0), (" ", 1), ("<think>x</think>", 0), ("x<unknown>y</unknown>", 1)])
def test_format_preserves_policy_block_grammar(body, expected):
    sample = trajectory("SELECT 1")
    sample.segments[0].text = f"<think>{body}</think><final_sql>SELECT 1</final_sql>"
    assert format_reward(sample) == expected


def test_timeout_and_write_attempt_do_not_modify_database(database):
    valid, correct, error, _ = execute_pair("DELETE FROM employee", "SELECT id FROM employee", database)
    assert valid == correct == 0 and error
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT COUNT(*) FROM employee").fetchone() == (3,)
    query = "WITH RECURSIVE x(i) AS (VALUES(1) UNION ALL SELECT i+1 FROM x) SELECT sum(i) FROM x"
    valid, _, error, _ = execute_pair(query, "SELECT 1", database, timeout_seconds=.01)
    assert valid == 0 and "interrupt" in error
