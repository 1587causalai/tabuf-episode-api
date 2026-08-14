# tabuf-episode-api

生成语义以 [docs/data-generation.pdf](docs/data-generation.pdf) 为准（活文档）。API 是这份文档的临时投影，会跟着文档改。

TabUF 第零版 **episode 数据 API**：先从总体抽 unit，再用 unit-specific response law 填 Unit×Feature 格子，最后盖查询掩码。观测数据 only，不是 DiscoSCM Layer 3 的论文复现。

生成律：

```
Y[i, j] = <U[i], W[j]> + E[i, j]
```

- `U`：一次实现的总体，形状 `(n_units, unit_dim)`
- `W`：共享特征方向，形状 `(n_features, unit_dim)`
- 行 `i` 是已经实现的 unit `u_i` 的一次观测，不是「每一行一个新总体」
- 训练默认 **不返回** `U` / `W`（`debug: true` 才带）

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
  "n_units": 16,
  "n_features": 4,
  "unit_dim": 4,
  "query_frac": 0.15,
  "sigma": 0.3,
  "seed": 0,
  "n_episodes": 1,
  "debug": false
}
```

返回每个 episode：`y_input`（查询格为 null）、`context_mask`、`query_mask`、`y_query`、`shapes`、`seed`。`C ∩ Q = ∅`，并覆盖全表。

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

不做：干预、反事实噪声重抽、类别列、自然缺失、S 臂、curriculum。下一步才加。
