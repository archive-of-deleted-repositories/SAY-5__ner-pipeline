"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ner_pipeline.models import Base


@pytest.fixture()
def sqlite_engine(tmp_path: Path) -> Iterator[Engine]:
    """A throwaway SQLite engine for unit tests that need persistence."""
    url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture()
def session(sqlite_engine: Engine) -> Iterator[Session]:
    sm = sessionmaker(bind=sqlite_engine, autoflush=False, expire_on_commit=False)
    s = sm()
    try:
        yield s
    finally:
        s.close()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_integration = os.getenv("RUN_INTEGRATION") == "1"
    run_eval_smoke = os.getenv("RUN_EVAL_SMOKE") == "1"
    skip_integration = pytest.mark.skip(reason="set RUN_INTEGRATION=1 to enable")
    skip_eval_smoke = pytest.mark.skip(reason="set RUN_EVAL_SMOKE=1 to enable")
    for item in items:
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)
        if "eval_smoke" in item.keywords and not run_eval_smoke:
            item.add_marker(skip_eval_smoke)
