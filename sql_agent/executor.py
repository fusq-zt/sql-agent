"""Read-only SQLite observations. Reward evaluation is deliberately separate."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time

from .contracts import ContractError, SETTINGS


class InfrastructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class Observation:
    sql: str
    columns: list[str]
    rows: list[list]
    error: str | None
    elapsed_seconds: float


def json_feedback_v1(observation: Observation) -> str:
    """Explicit candidate serialization; not asserted to be the author's serializer."""
    if observation.error is not None:
        body = json.dumps({"error": observation.error}, ensure_ascii=False)
    else:
        body = json.dumps({"columns": observation.columns, "rows": observation.rows},
                          ensure_ascii=False, default=lambda x: {"sqlite_blob_hex": x.hex()})
    # Prevent database cell contents from manufacturing control markup.
    body = body.replace("<", "\\u003c").replace(">", "\\u003e")
    return "<result>" + body + "</result>"


class SQLiteExecutor:
    def __init__(self, database_path: str | Path, *, timeout_seconds: float):
        self.database_path = Path(database_path).resolve()
        if not self.database_path.is_file():
            raise InfrastructureError(f"Database file is missing: {self.database_path}")
        if timeout_seconds <= 0:
            raise ContractError("SQL execution timeout must be explicit and positive")
        self.timeout_seconds = timeout_seconds

    def execute(self, sql: str) -> Observation:
        if not isinstance(sql, str) or not sql.strip():
            raise ContractError("SQL action is empty")
        started = time.monotonic()
        try:
            connection = sqlite3.connect(self.database_path.as_uri() + "?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise InfrastructureError("Unable to open database") from exc
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.set_authorizer(lambda action, *args: sqlite3.SQLITE_DENY
                if action in {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH} else sqlite3.SQLITE_OK)
            connection.set_progress_handler(lambda: int(time.monotonic() - started > self.timeout_seconds), 1000)
            try:
                cursor = connection.execute(sql)
                columns = [c[0] for c in cursor.description] if cursor.description else []
                rows = [list(row) for row in cursor.fetchmany(SETTINGS.feedback_rows)]
                return Observation(sql, columns, rows, None, time.monotonic() - started)
            except sqlite3.Error as exc:
                code = getattr(exc, "sqlite_errorcode", 0) & 0xFF
                if code in {sqlite3.SQLITE_IOERR, sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB,
                            sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_FULL}:
                    raise InfrastructureError("Database infrastructure failure") from exc
                return Observation(sql, [], [], str(exc), time.monotonic() - started)
        finally:
            connection.close()

    async def execute_async(self, sql: str) -> Observation:
        return await asyncio.to_thread(self.execute, sql)
