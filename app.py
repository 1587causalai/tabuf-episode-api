"""TabUF observational episode API (v0)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from generator import sample_episodes

VERSION = "v0"

DESCRIPTION = """
一次返回 n 条 episode，每条自己抽一个总体。

表 `values` 是完整格子。`missing_mask` 与 `query_mask` **可以重合**：
重合处是缺失填补，query 其余是普通预测。没有单独的 y 标签。
列类型按默认比例混合（数值 / 有序 / 多值类别 / 超多值类别），约 10% 列独立于其它特征。
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


class EpisodeRequest(BaseModel):
    n_units: int = Field(default=64, ge=8, le=512, examples=[16])
    n_features: int = Field(default=8, ge=2, le=64, examples=[8])
    unit_dim: int = Field(default=4, ge=1, le=32, examples=[4])
    query_frac: float = Field(default=0.15, gt=0.0, lt=1.0, description="要预测的格子占全表比例，与 missing 独立可重合")
    missing_frac: float = Field(default=0.05, ge=0.0, lt=0.95, description="世界缺失占全表比例，与 query 独立可重合")
    sigma: float = Field(default=0.3, ge=0.0, le=10.0)
    seed: int | None = Field(default=0)
    n_episodes: int = Field(default=1, ge=1, description="条数；每条各自抽总体，硬顶 32")
    return_mechanism: bool = Field(default=False, description="true 才带 population 与 response_law")
    debug: bool = Field(default=False, description="额外潜变量分数 S；隐含 return_mechanism")

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
    values: list[list[float | int]] = Field(description="完整表。数值为 float，离散为 int 编码。不因 mask 挖空。")
    missing_mask: list[list[bool]]
    query_mask: list[list[bool]]
    column_types: list[str] = Field(description="numeric | ordinal | categorical | high_cardinality")
    n_classes: list[int | None] = Field(description="离散列的水平数；数值列为 null")
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
    )
    return {"n_episodes": len(episodes), "episodes": episodes}
