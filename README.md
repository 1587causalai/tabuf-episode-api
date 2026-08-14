# tabuf-episode-api

生成语义以 [docs/data-generation.pdf](docs/data-generation.pdf) 为准（活文档）。API 是这份文档的临时投影，会跟着文档改。

TabUF 第零版 **episode 数据 API**：先从总体抽 unit，再在特征 token 上抽单位球 SCM，再用 unit-specific response law 填 Unit×Feature 格子，最后盖查询掩码。观测数据 only，不是 DiscoSCM Layer 3 的论文复现。

DiscoSCM（默认 `source`）生成律：

```
s_raw = sum_{p in pa(j)} beta_{jp} t_p
s_j   = normalize( φ_j(s_raw) )                               # φ ∈ {id, tanh, leaky_relu, sin}
t_j   = normalize( α s_j + sqrt(1-α²) η_j ),  η_j ⊥ s_j       # 单位球 SEM
Y[i,j] = g_j( <U[i], t_j>, E[i,j] )                           # 原类型特异 g，w_j 换成 t_j
```

- `U`：一次实现的总体，形状 `(n_units, unit_dim)`；行 `i` 是 unit `u_i`。混合数 `M` 在 `1..floor(sqrt(n))` 上按 `1/M` 抽，每个分量独立 Gaussian 或 Cauchy；`M=1` 且高斯即原来的 `N(0,I)`
- `n_features`：默认 `null`，从先验抽样（众数约 20，支持 2–1000）。80% lognormal 绕 20（clip [4,64]），15% log-uniform [20,200]，5% log-uniform [200,1000]
- `t_j`：特征 token，形状 `(unit_dim,)`，L2 单位化。DAG：随机置换为拓扑序；稀疏骨架 Poisson λ=2.2、典型父节点上限 6、preferential attachment（大概率）。约 15%（或 `graph_family=star`）再叠 1 个 out-hub + 1 个 in-hub。`graph_family=sparse` 强制普通图。`dag_edge_p` 是遗留字段，不再按 Bernoulli(p) 连边
- β 是带符号的混合权重，Σ|β|=1；相对 |β| 先 Unif[0.5,2] 再 L1 归一（独立随机符号 ±1）。α=`token_heritability` 默认 0.75，φ=identity 时 ⟨t_j, s_j⟩ ≈ α。φ 只作用在 token SEM 的 s_raw 上
- 噪声族：每列 gaussian / student_t / cauchy（0.50 / 0.30 / 0.20），用于根 token、正交创新、以及 numeric/binary/ordinal 观测噪声。填格对 u 线性：数值 affine、二值 latent threshold、有序 ordered probit、$K$ 类 linear softmax（不再 Gumbel-max）
- `independent_frac` 列不进 DAG（无父、也不被当父），保持原来的类型边缘
- 训练默认 **不返回** `U` / DAG / `t_j`（`debug` 或 `return_mechanism` 才带）

线上信封：完整 `values` + `missing_mask` + `query_mask`。没有 `y_input` / `y_query`。

## 运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8787
```

Docker：

```bash
docker build -t tabuf-episode-api .
docker run --rm -p 8787:8787 tabuf-episode-api
```

## 文档

- 生成语义（活文档）：[docs/data-generation.pdf](docs/data-generation.pdf)
- 接口文档：[docs/api.pdf](docs/api.pdf)
- OpenAPI：[docs/openapi.json](docs/openapi.json)
- 交互式：服务起来后打开 `/docs`（Swagger）或 `/redoc`

## API

`GET /health` → `{"ok": true, "version": "v0"}`

`POST /v0/episodes`

```json
{
  "n_units": 1000,
  "n_features": null,
  "unit_dim": null,
  "query_frac": 0.15,
  "sigma": 0.3,
  "seed": 0,
  "n_episodes": 1,
  "debug": false
}
```

`n_units` 默认 1000（上限 4096）。下面 curl 用 16 只是为了拷贝轻量。

返回每个 episode：`table.values`（完整表）、`table.missing_mask`、`table.query_mask`、`shapes`、`seed`。两张 mask 独立抽取，允许重合。

```bash
curl -s -X POST http://127.0.0.1:8787/v0/episodes \
  -H 'Content-Type: application/json' \
  -d '{"n_units":16,"n_features":4,"unit_dim":4,"query_frac":0.15,"seed":0,"n_episodes":1}'
```

## 测试

```bash
PYTHONPATH=. python -m pytest tests -q
```

## 范围（v0）

不做：干预、反事实噪声重抽、自然缺失、S 臂、curriculum。下一步才加。
