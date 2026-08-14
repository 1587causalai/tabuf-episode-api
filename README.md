# tabuf-episode-api

生成语义以 [docs/data-generation.pdf](docs/data-generation.pdf) 为准（活文档）。API 是这份文档的临时投影，会跟着文档改。

TabUF 第零版 **episode 数据 API**：先从总体抽 unit，再在特征 token 上抽单位球 SCM，再用 unit-specific response law 填 Unit$\times$Feature 格子，最后盖查询掩码。观测数据 only，不是 DiscoSCM Layer 3 的论文复现。

DiscoSCM（默认 `source`）生成律：

$$
\begin{aligned}
\mathbf{s}^{\mathrm{raw}}_j &= \sum_{p\in\mathrm{pa}(j)}\beta_{jp}\mathbf{t}_p \\
\mathbf{s}_j &= \mathrm{normalize}\bigl(\varphi_j(\mathbf{s}^{\mathrm{raw}}_j)\bigr)
\qquad \varphi\in\{\mathrm{id},\tanh,\mathrm{lReLU},\sin\} \\
\mathbf{t}_j &= \mathrm{normalize}\bigl(\alpha\mathbf{s}_j+\sqrt{1-\alpha^2}\,\eta_j\bigr),\quad \eta_j\perp\mathbf{s}_j \\
Y_{ij} &= g_j\bigl(\langle\mathbf{u}_i,\mathbf{t}_j\rangle,e_{ij}\bigr)
\end{aligned}
$$

- $U$：一次实现的总体，形状 `(n_units, unit_dim)`；行 $i$ 是 unit $\mathbf{u}_i$。混合数 $M$ 在 $\{1,\ldots,\lfloor\sqrt{n}\rfloor\}$ 上按 $P(M=m)\propto 1/m$ 抽，每个分量独立 Gaussian 或 Cauchy；$M=1$ 且高斯即原来的 $\mathcal{N}(0,I_k)$
- `n_features`：默认 `null`，从先验抽样（众数约 $20$，支持 $[2,1000]$）。$80\%$ $\mathrm{LogNormal}(\log 20,0.5^2)$ clip $[4,64]$，$15\%$ $\mathrm{LogUnif}[20,200]$，$5\%$ $\mathrm{LogUnif}[200,1000]$
- `unit_dim` $k$：默认 `null`，从先验抽样（众数约 $16$，支持 $[2,1024]$）。$80\%$ $\mathrm{LogNormal}(\log 16,0.45^2)$ clip $[2,64]$，$15\%$ $\mathrm{LogUnif}[32,256]$，$5\%$ $\mathrm{LogUnif}[256,1024]$
- $\mathbf{t}_j$：特征 token，形状 `(unit_dim,)`，L2 单位化。DAG：随机置换为拓扑序；稀疏骨架 $\mathrm{Poisson}(\lambda=2.2)$、典型父节点上限 $6$、preferential attachment（大概率）。约 $15\%$（或 `graph_family=star`）再叠 $1$ 个 out-hub + $1$ 个 in-hub。`graph_family=sparse` 强制普通图。`dag_edge_p` 是遗留字段，不再按 $\mathrm{Bernoulli}(p)$ 连边
- $\beta$ 是带符号的混合权重，$\sum|\beta|=1$；相对 $|\beta|$ 先 $\mathrm{Unif}[0.5,2]$ 再 L1 归一（独立随机符号 $\pm 1$）。$\alpha=$`token_heritability` 默认 $0.75$，$\varphi=\mathrm{identity}$ 时 $\langle\mathbf{t}_j,\mathbf{s}_j\rangle\approx\alpha$。$\varphi$ 只作用在 token SEM 的 $\mathbf{s}^{\mathrm{raw}}$ 上，概率 $0.50/0.20/0.20/0.10$
- 列类型 $\mathrm{Mult}(0.70,0.05,0.10,0.05,0.05)$（数值 / 有序 / 二值 / 多值 / 超多值）。`independent_frac` $=0.05$ 的列不进 DAG（无父、也不被当父），保持原来的类型边缘
- 噪声族：每列 gaussian / student_t / cauchy（$0.50/0.30/0.20$），用于根 token、正交创新、以及 numeric / binary / ordinal 观测噪声。填格对 $u$ 线性：数值 affine、二值 latent threshold、有序 ordered probit、$K$ 类 linear softmax（不再 Gumbel-max）
- 训练默认 **不返回** $U$ / DAG / $\mathbf{t}_j$（`debug` 或 `return_mechanism` 才带）

线上信封：完整 `values` + `missing_mask` + `query_mask`。默认 $P(M_{ij}=1)=0.05$，DiscoSCM 和监督表相同。`query_mode` 两种：`any_cell`（默认，全表抽 $0.15$）或 `label_cell`（只在目标列上抽 $0.15$ 行）。两张 mask 独立，允许重合。没有 `y_input` / `y_query`。

`source=openml` 走 OpenML-CTR23 回归表（默认 `source_name=44970` QSAR_fish_toxicity）。行是实体行，最后一列是 $y$，默认 `label_cell`。`n_features` 为 `null` 时保留原表宽度，不垫到 20 列；表比 `n_units` 短则用全表，不重复填充。`source=recsys` 仍是 501。

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
  "n_episodes": 8,
  "batch_size": 1,
  "debug": false
}
```

`n_episodes` 是几个不同的生成世界（默认 8），不是 batch。`batch_size` 默认 1：每条自己抽列数 $d$ 和 $k$，不能直接 stack。只有写成 `"batch_size": 8` 时，这 8 张表才共用 `(n, d, k)`，可以堆成 `(8, n, d)`；它们仍然是 8 个世界（总体 / DAG / 列类型都重抽），只锁尺寸。`n_units` 默认 1000，不抽。细节见 [生成手册](docs/data-generation.pdf) 和 [API 文档里的三个例子](docs/api.pdf)。下面 curl 用 16 行、1 条只是为了拷贝轻量。

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
