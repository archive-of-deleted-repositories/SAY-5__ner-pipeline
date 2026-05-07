"""Settings smoke tests."""

from __future__ import annotations

import pytest

from ner_pipeline.settings import Settings, get_settings


def test_settings_defaults() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.model_name.startswith("dslim/")
    assert "postgresql" in s.database_url
    assert s.log_level == "INFO"


def test_get_settings_is_cached() -> None:
    a = get_settings()
    b = get_settings()
    assert a is b


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NER_MODEL_NAME", "foo/bar")
    monkeypatch.setenv("NER_LOG_LEVEL", "DEBUG")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.model_name == "foo/bar"
    assert s.log_level == "DEBUG"
