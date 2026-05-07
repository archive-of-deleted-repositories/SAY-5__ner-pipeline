"""FastAPI ingestion endpoint."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text as sql_text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ner_pipeline.calibration import (
    CalibrationParams,
    calibrated_confidence,
)
from ner_pipeline.db import get_sessionmaker
from ner_pipeline.extractor import Extractor, TransformersExtractor
from ner_pipeline.pipeline import ingest
from ner_pipeline.settings import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=255)
    text: str = Field(..., min_length=1)
    raw_confidence: bool = Field(
        default=False,
        description=(
            "When true, return the raw model softmax score on each mention "
            "instead of the per-entity-type calibrated probability."
        ),
    )


class MentionOut(BaseModel):
    surface_form: str
    char_start: int
    char_end: int
    entity_type: str
    confidence: float


class IngestResponse(BaseModel):
    document_id: int
    entity_count: int
    created: bool
    mentions: list[MentionOut]


class ErrorEnvelope(BaseModel):
    code: str
    message: str
    retryable: bool
    request_id: str


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


_extractor_singleton: Extractor | None = None
_calibration_singleton: CalibrationParams | None = None


def _default_extractor() -> Extractor:
    global _extractor_singleton
    if _extractor_singleton is None:
        s = get_settings()
        _extractor_singleton = TransformersExtractor(
            model_name=s.model_name,
            revision=s.model_revision,
            device=s.model_device,
        )
    return _extractor_singleton


def get_extractor() -> Extractor:
    """Dependency injection seam — tests override this with a MockExtractor."""
    return _default_extractor()


_DEFAULT_CALIBRATION_PATH = Path("artifacts/calibration_platt.json")


def _default_calibration() -> CalibrationParams | None:
    global _calibration_singleton
    if _calibration_singleton is None and _DEFAULT_CALIBRATION_PATH.exists():
        _calibration_singleton = CalibrationParams.load(_DEFAULT_CALIBRATION_PATH)
    return _calibration_singleton


def get_calibration() -> CalibrationParams | None:
    """Dependency seam for tests; returns None if no calibration is loaded."""
    return _default_calibration()


def get_session() -> Any:
    sm = get_sessionmaker()
    session = sm()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    extractor_factory: Callable[[], Extractor] | None = None,
) -> FastAPI:
    """Build the FastAPI app. Tests pass an ``extractor_factory`` to swap models."""
    app = FastAPI(
        title="ner-pipeline",
        version="0.1.0",
        description="Pretrained transformer NER pipeline ingesting into Postgres.",
    )

    if extractor_factory is not None:
        app.dependency_overrides[get_extractor] = extractor_factory

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Any) -> Any:
        rid = request.headers.get(get_settings().request_id_header) or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers[get_settings().request_id_header] = rid
        return response

    @app.exception_handler(SQLAlchemyError)
    async def _sa_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        rid = getattr(request.state, "request_id", "unknown")
        logger.exception("database error", extra={"request_id": rid})
        return JSONResponse(
            status_code=503,
            content=ErrorEnvelope(
                code="database_error",
                message="database is unavailable",
                retryable=True,
                request_id=rid,
            ).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def _http_handler(request: Request, exc: HTTPException) -> JSONResponse:
        rid = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorEnvelope(
                code=f"http_{exc.status_code}",
                message=str(exc.detail),
                retryable=exc.status_code >= 500,
                request_id=rid,
            ).model_dump(),
        )

    @app.get("/healthz")
    def healthz(session: Session = Depends(get_session)) -> dict[str, Any]:
        try:
            session.execute(sql_text("SELECT 1")).scalar()
            db_ok = True
        except SQLAlchemyError:
            db_ok = False
        return {
            "status": "ok" if db_ok else "degraded",
            "db": db_ok,
            "version": "0.1.0",
        }

    @app.post("/v1/ingest", response_model=IngestResponse)
    def ingest_endpoint(
        body: IngestRequest,
        session: Session = Depends(get_session),
        extractor: Extractor = Depends(get_extractor),
        calibration: CalibrationParams | None = Depends(get_calibration),
    ) -> IngestResponse:
        # When the caller asks for raw confidence, we explicitly skip the
        # calibration step on the persisted side. Otherwise the persisted
        # confidence is calibrated.
        applied_cal = None if body.raw_confidence else calibration
        result = ingest(
            source=body.source,
            text=body.text,
            extractor=extractor,
            session=session,
            calibration=applied_cal,
        )
        # Re-extract for the response payload only when we actually ingested
        # fresh; on idempotent hits we return an empty mentions list with the
        # existing count so callers can detect the no-op.
        if result.created:
            mentions = []
            for m in extractor.extract(body.text):
                conf = m.confidence
                if applied_cal is not None:
                    conf = calibrated_confidence(conf, m.entity_type, applied_cal)
                mentions.append(
                    MentionOut(
                        surface_form=m.surface_form,
                        char_start=m.char_start,
                        char_end=m.char_end,
                        entity_type=m.entity_type,
                        confidence=conf,
                    )
                )
        else:
            mentions = []
        return IngestResponse(
            document_id=result.document_id,
            entity_count=result.entity_count,
            created=result.created,
            mentions=mentions,
        )

    return app


app = create_app()
