"""TabUF observational episode API (v0).

DiscoSCM-aligned factor law on Unit x Feature grids. Observational only.
Interactive docs: /docs  (Swagger)  and  /redoc.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from generator import sample_episodes

VERSION = "v0"

DESCRIPTION = """
观测第零版 episode 数据服务。

生成律：`Y[i,j] = <U[i], W[j]> + E[i,j]`。先抽总体再填格子，再切 C/Q。
训练默认 **不返回** 背后的 DiscoSCM。需要时设 `return_mechanism: true`，按 `<U, E, V, F>` 整份给出。语义以 `docs/data-generation.pdf` 为准；
本服务的字段名是那份文档的临时投影，会跟着文档改。
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
    n_units: int = Field(
        default=64, ge=8, le=512,
        description="这一集实现多少 unit（行）。不是 row-id 个数，是总体切片大小。",
        examples=[16],
    )
    n_features: int = Field(
        default=8, ge=2, le=64,
        description="feature（列）个数。",
        examples=[4],
    )
    unit_dim: int = Field(
        default=4, ge=1, le=32,
        description="个体潜空间维 k。U 形状 (n_units, unit_dim)。",
        examples=[4],
    )
    query_frac: float = Field(
        default=0.15, gt=0.0, lt=1.0,
        description="查询格子 Q 占总格子的比例。其余为上下文 C。C ∩ Q = ∅。",
        examples=[0.15],
    )
    sigma: float = Field(
        default=0.3, ge=0.0, le=10.0,
        description="观测噪声标准差。第零版只抽一次 factual noise。",
        examples=[0.3],
    )
    seed: int | None = Field(
        default=0,
        description="基种子。第 e 集使用 seed+e。null 则随机抽一个基种子，仍写入每集 seed。",
        examples=[0],
    )
    n_episodes: int = Field(
        default=1, ge=1,
        description="一次返回多少集。服务端硬顶 32。",
        examples=[1],
    )
    return_mechanism: bool = Field(
        default=False,
        description="true 时每集附带 DiscoSCM 元组 <U,E,V,F>（含 U、E、结构方程和 W）。训练路径必须 false。",
        examples=[False],
    )
    debug: bool = Field(
        default=False,
        description="true 时额外附带顶层 U、W、Y_full，并隐含 return_mechanism=true。训练路径必须 false。",
        examples=[False],
    )

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
                    "sigma": 0.3,
                    "seed": 0,
                    "n_episodes": 1,
                    "return_mechanism": False,
                    "debug": False,
                }
            ]
        }
    }


class Shapes(BaseModel):
    n_units: int
    n_features: int
    unit_dim: int
    n_query: int
    n_context: int
    y_input: list[int]
    context_mask: list[int]
    query_mask: list[int]
    y_query: list[int]


class Episode(BaseModel):
    context_mask: list[list[bool]] = Field(description="M^C，true 表示该格属于上下文 C")
    query_mask: list[list[bool]] = Field(description="M^Q，true 表示该格属于查询 Q")
    y_input: list[list[float | None]] = Field(
        description="输入表。C 位置是观测值，Q 位置是 null"
    )
    y_query: list[float] = Field(description="Q 上的真值，按行优先拉直，长度 = n_query")
    shapes: Shapes
    seed: int
    mechanism: dict[str, Any] | None = Field(
        default=None,
        description="DiscoSCM 元组。仅 return_mechanism=true（或 debug=true）时出现",
    )
    U: list[list[float]] | None = Field(default=None, description="仅 debug=true，兼容字段；正式通道是 mechanism.U")
    W: list[list[float]] | None = Field(default=None, description="仅 debug=true，兼容字段；正式通道是 mechanism.F.W")
    Y_full: list[list[float]] | None = Field(default=None, description="仅 debug=true 的满表世界，不是机制本身")


class EpisodeResponse(BaseModel):
    episodes: list[Episode]


class HealthResponse(BaseModel):
    ok: bool
    version: str


@app.get("/health", response_model=HealthResponse, tags=["ops"], summary="存活检查")
def health() -> dict[str, Any]:
    """进程活着且生成器可导入时返回 ok。不含鉴权。"""
    return {"ok": True, "version": VERSION}


@app.post(
    "/v0/episodes",
    response_model=EpisodeResponse,
    response_model_exclude_none=True,
    tags=["episodes"],
    summary="抽取观测 episode",
)
def create_episodes(req: EpisodeRequest) -> dict[str, Any]:
    """按 DiscoSCM 第零版因子律生成 Unit×Feature episode。

    默认只给挖空后的表和查询真值。设 return_mechanism=true 才返回该集背后的 DiscoSCM。
    """
    episodes = sample_episodes(
        n_units=req.n_units,
        n_features=req.n_features,
        unit_dim=req.unit_dim,
        query_frac=req.query_frac,
        sigma=req.sigma,
        seed=req.seed,
        n_episodes=req.n_episodes,
        debug=req.debug,
        return_mechanism=req.return_mechanism,
    )
    return {"episodes": episodes}
