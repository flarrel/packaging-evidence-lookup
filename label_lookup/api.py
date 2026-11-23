"""Loopback HTTP API for an existing local package index."""

from base64 import b64decode
from binascii import Error
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .answering import Lookup
from .images import load_image
from .retrieval import Index


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=400)
    image_base64: str | None = Field(default=None, max_length=5_600_000)


def create_app(work: Path) -> FastAPI:
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        index = Index(work)
        try:
            app.state.lookup = Lookup(index)
            yield
        finally:
            index.close()

    app = FastAPI(title="Packaging Evidence Lookup", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ready"}

    @app.post("/query")
    def query(body: Question) -> dict:
        if not body.question.strip():
            raise HTTPException(422, "Question is blank.")
        image = None
        if body.image_base64 is not None:
            try:
                image = load_image(b64decode(body.image_base64, validate=True))
            except (Error, OSError, ValueError) as error:
                raise HTTPException(422, "Expected a supported image encoded as base64.") from error
        with lock:
            return app.state.lookup.ask(body.question, image)

    return app
