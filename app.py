"""Minimal FastAPI for observational v0 TabUF episodes.

DiscoSCM-aligned factor law on Unit×Feature grids. Observational only;
not a Layer-3 paper reimplementation.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from generator import sample_episodes

VERSION = "v0"

app = FastAPI(title="TabUF observational episode API", version=VERSION)


class EpisodeRequest(BaseModel):
    n_units: int = Field(default=64, ge=8, le=512)
    n_features: int = Field(default=8, ge=2, le=64)
    unit_dim: int = Field(default=4, ge=1, le=32)
    query_frac: float = Field(default=0.15, gt=0.0, lt=1.0)
    sigma: float = Field(default=0.3, ge=0.0, le=10.0)
    seed: int | None = 0
    n_episodes: int = Field(default=1, ge=1)
    debug: bool = False

    @field_validator("n_episodes")
    @classmethod
    def cap_episodes(cls, v: int) -> int:
        return max(1, min(int(v), 32))


class HealthResponse(BaseModel):
    ok: bool
    version: str


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    return {"ok": True, "version": VERSION}


@app.post("/v0/episodes")
def create_episodes(req: EpisodeRequest) -> dict[str, Any]:
    episodes = sample_episodes(
        n_units=req.n_units,
        n_features=req.n_features,
        unit_dim=req.unit_dim,
        query_frac=req.query_frac,
        sigma=req.sigma,
        seed=req.seed,
        n_episodes=req.n_episodes,
        debug=req.debug,
    )
    return {"episodes": episodes}
