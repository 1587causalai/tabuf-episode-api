"""TabUF observational episode API (v0)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from generator import group_batches, sample_episodes

VERSION = "v0"

DESCRIPTION = """
一次返回 n_episodes 条不同的 episode（默认 8），按 batch_size（默认 1）打包；设成 8 则组内同形状；下一组会重抽 d、k。source 默认 discoscm。

各来源语义不同：只有 discoscm 把行当成带潜变量的 unit。
其它来源（scm / sklearn_* / openml / recsys）各有自己的行含义。
共享的只是线上信封：完整 values + missing_mask + query_mask。
目录见 GET /v0/sources。
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
    n_units: int = Field(default=1000, ge=8, le=4096, examples=[1000], description="行数旋钮。只有 discoscm 把行解释成 unit；scm/sklearn 里是 n_samples")
    n_features: int | None = Field(default=None, ge=2, le=1000, description="null 则从先验抽样（众数约 20，支持 2–1000）")
    unit_dim: int | None = Field(default=None, ge=2, le=1024, description="discoscm-only：个体表征维 k。null 则从先验抽样（众数约 16，支持 2–1024）")
    query_frac: float | None = Field(default=None, description="any_cell：全表比例；label_cell：目标列行比例。缺省 0.15")
    missing_frac: float | None = Field(default=None, description="缺省 0.05，所有来源相同")
    query_mode: str | None = Field(default=None, description="any_cell | label_cell。旧名 cells/observed_cells、label_column 仍接受。缺省用来源画像")
    query_column: int | None = Field(default=None, ge=0, description="label_cell 时只在该列按 query_frac 抽行；缺省最后一列")
    sigma: float = Field(default=0.3, ge=0.0, le=10.0)
    seed: int | None = Field(default=0)
    batch_size: int = Field(default=1, ge=1, le=32, examples=[1], description="一组同形状的条数，默认 1（每条自己抽 d、k）；设 8 则可堆 tensor")
    n_episodes: int = Field(default=8, ge=1, le=32, examples=[8], description="不同生成世界的条数；可以大于 batch_size")
    type_weights: TypeWeights = Field(default_factory=TypeWeights, description="discoscm-only：列类型抽样权重，不必归一化")
    independent_frac: float = Field(default=0.05, ge=0.0, le=1.0, description="discoscm-only：每列与其它特征独立的概率")
    dag_edge_p: float = Field(default=0.3, ge=0.0, le=1.0, description="discoscm-only：遗留字段，仍写入 response_law；新 DAG 不再按 Bernoulli(p) 连边")
    max_parents: int | None = Field(default=None, ge=1, le=512, description="discoscm-only：None 时普通图父节点上限 6；显式整数为硬上限。星形枢纽是小概率混合臂，不是默认")
    graph_family: str | None = Field(default=None, description="discoscm-only：null 则 85% sparse / 15% star；可强制 sparse | star")
    token_heritability: float = Field(default=0.75, ge=0.05, le=0.95, description="discoscm-only：子 token 与父信号的余弦目标 α")
    beta_min: float = Field(default=0.5, description="discoscm-only：相对混合权重 |β| 下界（L1 归一之前）")
    beta_max: float = Field(default=2.0, description="discoscm-only：相对混合权重 |β| 上界（L1 归一之前）")
    source: str = Field(
        default="discoscm",
        description=(
            "canonical: discoscm | scm | sklearn_make_classification | sklearn_make_regression "
            "| sklearn_friedman1 | sklearn_low_rank | sklearn_iris | sklearn_wine "
            "| sklearn_breast_cancer | sklearn_diabetes | openml | recsys. "
            "aliases: sklearn_synthetic | sklearn_real (+ source_name)"
        ),
        examples=["discoscm"],
    )
    source_name: str | None = Field(
        default=None,
        description="别名用的子名称：make_classification / iris / OpenML id 等。canonical 名已自带，此项忽略。",
    )
    return_mechanism: bool = Field(default=False)
    debug: bool = Field(default=False)

    @model_validator(mode="after")
    def beta_range(self) -> "EpisodeRequest":
        if not (self.beta_max > self.beta_min):
            raise ValueError("beta_max must be greater than beta_min")
        return self


class TableShapes(BaseModel):
    values: list[int]
    missing_mask: list[int]
    query_mask: list[int]
    n_missing: int
    n_query: int
    n_query_and_missing: int


class Table(BaseModel):
    n_units: int
    n_rows: int | None = None
    n_features: int
    unit_dim: int | None = None
    source: str | None = None
    values: list[list[float | int]]
    missing_mask: list[list[bool]]
    query_mask: list[list[bool]]
    column_types: list[str] = Field(description="numeric | ordinal | binary | categorical | high_cardinality")
    n_classes: list[int | None]
    shapes: TableShapes
    query_mode: str | None = None
    query_column: int | None = None


class Episode(BaseModel):
    seed: int
    table: Table
    population: dict[str, Any] | None = None
    response_law: dict[str, Any] | None = None
    mechanism: dict[str, Any] | None = None
    S: list[list[float]] | None = None


class BatchEnvelope(BaseModel):
    batch_index: int
    n_units: int
    n_features: int
    unit_dim: int | None = None
    shape: list[int]
    n_episodes: int
    episodes: list[Episode]


class EpisodeResponse(BaseModel):
    n_episodes: int
    batch_size: int
    n_batches: int
    n_units: int
    n_features: int | None = None
    unit_dim: int | None = None
    shape: list[int] | None = None
    batches: list[BatchEnvelope]
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
            batch_size=req.batch_size,
            debug=req.debug,
            return_mechanism=req.return_mechanism,
            type_weights=req.type_weights.model_dump(),
            independent_frac=req.independent_frac,
            dag_edge_p=req.dag_edge_p,
            max_parents=req.max_parents,
            token_heritability=req.token_heritability,
            beta_min=req.beta_min,
            beta_max=req.beta_max,
            graph_family=req.graph_family,
            source=req.source,
            source_name=req.source_name,
            query_mode=req.query_mode,
            query_column=req.query_column,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    batches = group_batches(episodes, req.batch_size)
    payload: dict[str, Any] = {
        "n_episodes": len(episodes),
        "batch_size": req.batch_size,
        "n_batches": len(batches),
        "n_units": req.n_units,
        "batches": batches,
        "episodes": episodes,
    }
    if len(batches) == 1:
        payload["n_features"] = batches[0]["n_features"]
        payload["unit_dim"] = batches[0].get("unit_dim")
        payload["shape"] = batches[0]["shape"]
    return payload
