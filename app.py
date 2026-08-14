"""TabUF observational episode API (v0)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from generator import sample_episodes

VERSION = "v0"

DESCRIPTION = """
一次返回 n 条 episode，每条自己抽一个总体。

列类型权重 `type_weights` 和独立列概率 `independent_frac` 可传入；
默认 70/5/10/5/5（数值/有序/二值/多值/超多值）与 0.05。权重不必归一化。
"""

app = FastAPI(
    title="TabUF Episode API",
    version=VERSION,
    description=DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={"name": "tabuf-episode-api", "url": "https://github.com/1587causalai/tabuf-episode-api"},
    license_info={"name": "MIT"},
)


class TypeWeights(BaseModel):
    numeric: float = Field(default=70, ge=0, description="数值列权重")
    ordinal: float = Field(default=5, ge=0, description="有序列权重")
    binary: float = Field(default=10, ge=0, description="二值类别权重")
    categorical: float = Field(default=5, ge=0, description="多值类别权重")
    high_cardinality: float = Field(default=5, ge=0, description="超多值类别权重")

    @model_validator(mode="after")
    def positive_sum(self) -> "TypeWeights":
        if self.numeric + self.ordinal + self.binary + self.categorical + self.high_cardinality <= 0:
            raise ValueError("type_weights 之和必须为正，将按此比例抽样")
        return self


class EpisodeRequest(BaseModel):
    n_units: int = Field(default=64, ge=8, le=512, examples=[16])
    n_features: int = Field(default=8, ge=2, le=64, examples=[8])
    unit_dim: int = Field(default=4, ge=1, le=32, examples=[4])
    query_frac: float = Field(default=0.15, gt=0.0, lt=1.0)
    missing_frac: float = Field(default=0.05, ge=0.0, lt=0.95)
    sigma: float = Field(default=0.3, ge=0.0, le=10.0)
    seed: int | None = Field(default=0)
    n_episodes: int = Field(default=1, ge=1)
    type_weights: TypeWeights = Field(default_factory=TypeWeights, description="列类型抽样权重，不必归一化")
    independent_frac: float = Field(default=0.05, ge=0.0, le=1.0, description="每列与其它特征独立的概率")
    return_mechanism: bool = Field(default=False)
    debug: bool = Field(default=False)

    @field_validator("n_episodes")
    @classmethod
    def cap_episodes(cls, v: int) -> int:
        return max(1, min(int(v), 32))


class TableShapes(BaseModel):
    values: list[int]
    missing_mask: list[int]
    query_mask: list[int]
    n_missing: int
    n_query: int
    n_query_and_missing: int


class Table(BaseModel):
    n_units: int
    n_features: int
    values: list[list[float | int]]
    missing_mask: list[list[bool]]
    query_mask: list[list[bool]]
    column_types: list[str] = Field(description="numeric | ordinal | binary | categorical | high_cardinality")
    n_classes: list[int | None]
    shapes: TableShapes


class Episode(BaseModel):
    seed: int
    table: Table
    population: dict[str, Any] | None = None
    response_law: dict[str, Any] | None = None
    S: list[list[float]] | None = None


class EpisodeResponse(BaseModel):
    n_episodes: int
    episodes: list[Episode]


class HealthResponse(BaseModel):
    ok: bool
    version: str


@app.get("/health", response_model=HealthResponse, tags=["ops"], summary="存活检查")
def health() -> dict[str, Any]:
    return {"ok": True, "version": VERSION}


@app.post("/v0/episodes", response_model=EpisodeResponse, response_model_exclude_none=True, tags=["episodes"], summary="抽取 n 条观测 episode")
def create_episodes(req: EpisodeRequest) -> dict[str, Any]:
    try:
        episodes = sample_episodes(
            n_units=req.n_units,
            n_features=req.n_features,
            unit_dim=req.unit_dim,
            query_frac=req.query_frac,
            missing_frac=req.missing_frac,
            sigma=req.sigma,
            seed=req.seed,
            n_episodes=req.n_episodes,
            debug=req.debug,
            return_mechanism=req.return_mechanism,
            type_weights=req.type_weights.model_dump(),
            independent_frac=req.independent_frac,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"n_episodes": len(episodes), "episodes": episodes}
