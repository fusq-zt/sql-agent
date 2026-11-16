"""Terminal rewards for SQL trajectories."""
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sqlite3
import time

from .executor import InfrastructureError


@dataclass(frozen=True)
class RewardBreakdown:
    format: float
    exact_match: float
    valid: float
    result: float
    entity: float
    predicted_error: str | None = None
    gold_error: str | None = None

    @property
    def total(self):
        return 2*self.format + self.exact_match + 3*(self.valid+self.result) + self.entity


CONTROL_TAG = re.compile(r"</?(?:think|intermediate_sql|final_sql|result)>")


def format_reward(trajectory):
    if trajectory.final_sql is None:
        return 0.0
    # A fabricated result in a policy segment cannot satisfy the grammar.
    if any("<result>" in s.text or "</result>" in s.text
           for s in trajectory.segments if s.origin == "policy"):
        return 0.0
    text = "".join(s.text for s in trajectory.segments)
    position = 0

    def skip_space():
        nonlocal position
        while position < len(text) and text[position].isspace():
            position += 1

    def take(tag):
        nonlocal position
        skip_space()
        opening, closing = f"<{tag}>", f"</{tag}>"
        if not text.startswith(opening, position):
            return False
        start = position + len(opening)
        end = text.find(closing, start)
        if end < 0:
            return False
        # Same nonempty policy blocks and forbidden nested control tags as the
        # old grammar. Real environment payloads already escape control tags.
        if tag != "result" and (end == start or CONTROL_TAG.search(text, start, end)):
            return False
        position = end + len(closing)
        return True

    if not take("think"):
        return 0.0
    skip_space()
    while text.startswith("<intermediate_sql>", position):
        if not (take("intermediate_sql") and take("result") and take("think")):
            return 0.0
        skip_space()
    if not take("final_sql"):
        return 0.0
    skip_space()
    return float(position == len(text))


def connection(database_path, timeout_seconds):
    path = Path(database_path).resolve()
    if not path.is_file():
        raise InfrastructureError(f"Missing reward database: {path}")
    db = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    db.set_authorizer(lambda action, *args: sqlite3.SQLITE_DENY
                      if action in {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH} else sqlite3.SQLITE_OK)
    deadline = time.monotonic() + timeout_seconds
    db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
    return db


def sql_entities(sql, database_path, *, timeout_seconds=30):
    entities = set()
    def authorizer(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_READ:
            entities.add(("table", table.casefold()))
            if column:
                entities.add(("column", table.casefold(), column.casefold()))
        if action in {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    db = connection(database_path, timeout_seconds)
    try:
        db.set_authorizer(authorizer)
        # Prepare only. SQLite resolves aliases, correlated subqueries, CTEs,
        # quoted identifiers and SELECT * to the actual referenced base schema.
        db.execute("EXPLAIN " + sql).fetchall()
        return entities
    finally:
        db.close()


def execute_pair(predicted_sql, gold_sql, database_path, *, timeout_seconds=30):
    db = connection(database_path, timeout_seconds)
    def execute(sql):
        try:
            return set(db.execute(sql).fetchall()), None
        except sqlite3.Error as error:
            code = getattr(error, "sqlite_errorcode", 0) & 0xff
            if code in {sqlite3.SQLITE_IOERR, sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB,
                        sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_FULL}:
                raise InfrastructureError("Reward database failure") from error
            return None, str(error)
    try:
        predicted, predicted_error = execute(predicted_sql)
        gold, gold_error = execute(gold_sql)
        valid = float(predicted_error is None)
        result = float(valid and gold_error is None and predicted == gold)
        return valid, result, predicted_error, gold_error
    finally:
        db.close()


def evaluate(task, trajectory):
    sql = trajectory.final_sql
    if not sql:
        return RewardBreakdown(format_reward(trajectory), 0., 0., 0., 0.)
    gold = task["gold_sql"]
    valid, result, predicted_error, gold_error = execute_pair(sql, gold, task["database_path"])
    try:
        expected = sql_entities(gold, task["database_path"])
        actual = sql_entities(sql, task["database_path"])
        entity = len(expected & actual)/len(expected) if expected else float(not actual)
    except sqlite3.Error:
        entity = 0.0
    return RewardBreakdown(format_reward(trajectory), float(sql.strip() == gold.strip()),
                           valid, result, entity, predicted_error, gold_error)


def terminal_evaluator(task, trajectory):
    breakdown = evaluate(task, trajectory)
    trajectory.reward_components = asdict(breakdown)
    return breakdown.total, "reex-four-components-project-v1"
