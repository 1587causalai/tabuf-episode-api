"""TabUF observational episode API (v0).

Each response is n episodes. Each episode is one table with three disjoint
masks (missing / context / query). Population and unit-specific response law
are omitted unless return_mechanism=true.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from generator import sample_episodes

VERSION = "v0"

DESCRIPTION = """
观测第零版 episode 数据服务。

一次返回 **n 条 episode**。每条是一张 Unit×Feature 表，带三套互不相交的 mask：
世界缺失 `missing_mask`、上下文 `context_mask`、查询 `query_mask`。

表背后有总体（每个 unit 有表征 \(u_i\)）和 unit-specific response law。
这两样默认 **不序列化**；`return_mechanism: true` 才带上。
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
    n_units: int = Field(default=64, ge=8, le=512, description="这一集总体切片里有多少 unit", examples=[16])
    n_features: int = Field(default=8, ge=2, le=64, description="列数", examples=[4])
    unit_dim: int = Field(default=4, ge=1, le=32, description="个体表征维 k", examples=[4])
    query_frac: float = Field(default=0.15, gt=0.0, lt=1.0, description="查询格占全表比例（value mask）", examples=[0.15])
    missing_frac: float = Field(default=0.0, ge=0.0, lt=0.9, description="世界缺失占全表比例。与 query 互不相交。默认 0，但响应里始终带 missing_mask。", examples=[0.0])
    sigma: float = Field(default=0.3, ge=0.0, le=10.0, description="观测噪声标准差", examples=[0.3])
    seed: int | None = Field(default=0, description="基种子。第 e 集用 seed+e。", examples=[0])
    n_episodes: int = Field(default=1, ge=1, description="一次返回多少条 episode，硬顶 32", examples=[2])
    return_mechanism: bool = Field(default=False, description="true 时每集附带 population 与 response_law。训练必须 false。")
    debug: bool = Field(default=False, description="true 时额外附带 Y_full，并隐含 return_mechanism。训练必须 false。")

    @field_validator("n_episodes")
    @classmethod
    def cap_episodes(cls, v: int) -> int:
        return max(1, min(int(v), 32))

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "n_units": 16,
                    "n_features": 4,
                    "unit_dim": 4,
                    "query_frac": 0.15,
                    "missing_frac": 0.1,
                    "sigma": 0.3,
                    "seed": 0,
                    "n_episodes": 2,
                    "return_mechanism": False,
                    "debug": False,
                }
            ]
        }
    }


class TableShapes(BaseModel):
    values: list[int]
    missing_mask: list[int]
    context_mask: list[int]
    query_mask: list[int]
    y_query: list[int]
    n_missing: int
    n_context: int
    n_query: int


class Table(BaseModel):
    n_units: int
    n_features: int
    values: list[list[float | None]] = Field(description="仅 context 格有值；缺失和查询为 null")
    missing_mask: list[list[bool]] = Field(description="World 级缺失。与 C、Q 不相交")
    context_mask: list[list[bool]] = Field(description="网络可见的上下文 C")
    query_mask: list[list[bool]] = Field(description="Compiler 出题 Q（value mask）")
    y_query: list[float]
    shapes: TableShapes


class Episode(BaseModel):
    seed: int
    table: Table
    population: dict[str, Any] | None = Field(default=None, description="仅 return_mechanism。含每个 unit 的表征")
    response_law: dict[str, Any] | None = Field(default=None, description="仅 return_mechanism。unit-specific 填格律")
    Y_full: list[list[float]] | None = Field(default=None, description="仅 debug")


class EpisodeResponse(BaseModel):
    n_episodes: int
    episodes: list[Episode]


class HealthResponse(BaseModel):
    ok: bool
    version: str


@app.get("/health", response_model=HealthResponse, tags=["ops"], summary="存活检查")
def health() -> dict[str, Any]:
    return {"ok": True, "version": VERSION}


@app.post(
    "/v0/episodes",
    response_model=EpisodeResponse,
    response_model_exclude_none=True,
    tags=["episodes"],
    summary="抽取 n 条观测 episode",
)
def create_episodes(req: EpisodeRequest) -> dict[str, Any]:
    """返回 n 张带 mask 的表。总体和响应律默认不给。"""
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
