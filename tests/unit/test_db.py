"""Database session-scope tests against SQLite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import text

from ner_pipeline import db as db_module


@pytest.fixture()
def sqlite_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    url = f"sqlite:///{tmp_path / 'db.db'}"
    monkeypatch.setenv("NER_DATABASE_URL", url)
    # Force settings + engine reload
    from ner_pipeline.settings import get_settings

    get_settings.cache_clear()
    db_module.reset_engine_for_tests()
    yield url
    db_module.reset_engine_for_tests()
    get_settings.cache_clear()


def test_engine_singleton(sqlite_url: str) -> None:
    e1 = db_module.get_engine()
    e2 = db_module.get_engine()
    assert e1 is e2


def test_session_scope_commits(sqlite_url: str) -> None:
    db_module.get_engine()
    with db_module.session_scope() as session:
        session.execute(text("CREATE TABLE t (x INTEGER)"))
        session.execute(text("INSERT INTO t VALUES (1)"))

    with db_module.session_scope() as session:
        row = session.execute(text("SELECT COUNT(*) FROM t")).scalar()
        assert row == 1


def test_session_scope_rolls_back_on_exception(sqlite_url: str) -> None:
    db_module.get_engine()
    with db_module.session_scope() as session:
        session.execute(text("CREATE TABLE t (x INTEGER)"))

    with pytest.raises(RuntimeError, match="boom"), db_module.session_scope() as session:
        session.execute(text("INSERT INTO t VALUES (42)"))
        raise RuntimeError("boom")

    with db_module.session_scope() as session:
        row = session.execute(text("SELECT COUNT(*) FROM t")).scalar()
        assert row == 0
