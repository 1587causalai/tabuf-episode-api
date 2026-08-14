"""TabUF observational episode API (v0)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from generator import sample_episodes

VERSION = "v0"

DESCRIPTION = """
一次返回 n 条 episode，每条自己抽一个总体。

默认 source=discoscm。也可 sklearn_synthetic / sklearn_real / scm；openml 与 recsys 已占位。
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
    query_frac: float | None = Field(default=None, description="缺省则用来源画像默认")
    missing_frac: float | None = Field(default=None, description="缺省则用来源画像默认")
    query_mode: str | None = Field(default=None, description="cells | label_column | observed_cells；缺省用来源画像")
    sigma: float = Field(default=0.3, ge=0.0, le=10.0)
    seed: int | None = Field(default=0)
    n_episodes: int = Field(default=1, ge=1)
    type_weights: TypeWeights = Field(default_factory=TypeWeights, description="列类型抽样权重，不必归一化")
    independent_frac: float = Field(default=0.05, ge=0.0, le=1.0, description="每列与其它特征独立的概率")
    source: str = Field(
        default="discoscm",
        description="discoscm | sklearn_synthetic | sklearn_real | scm | openml | recsys",
        examples=["discoscm"],
    )
    source_name: str | None = Field(
        default=None,
        description="子名称：make_classification / iris / OpenML id 等。缺省则该 source 内随机抽。",
    )
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
    query_mode: str | None = None


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


@app.get("/v0/sources", tags=["ops"], summary="来源画像")
def list_sources() -> dict[str, Any]:
    from sources import SOURCE_PROFILES
    return {"sources": SOURCE_PROFILES}


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
            source=req.source,
            source_name=req.source_name,
            query_mode=req.query_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return {"n_episodes": len(episodes), "episodes": episodes}
