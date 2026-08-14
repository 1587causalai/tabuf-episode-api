#!/usr/bin/env python3
"""Render DiscoSCM generation pipeline with defaults (Chinese)."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

stages = [
    ("1  尺寸", "n_units = 1000\nd 未指定则抽样：80% 约 20（4–64），\n15% 对数均匀 20–200，5% 到 200–1000"),
    ("2  Prior 总体", "混合：M ~ 1/m 于 1…⌊√n⌋\n分量独立 Gaussian / Cauchy\nk 未指定则抽样（众数约 16）"),
    ("3  列类型", "权重 70/5/10/5/5\n（数值 / 有序 / 二值 / 多值 / 超多值）\nindependent_frac = 0.05，独立列不进 DAG"),
    ("4  Token DAG", "稀疏骨架 λ=2.2、最多 6 个父节点\n再叠 1 个 out-hub + 1 个 in-hub\nfrac ~ Unif[0.45, 0.85]"),
    ("5  Token SEM", "β：符号混合权重，Σ|β|=1\nφ：id 50% / tanh 20% / leakyReLU 20% / sin 10%\nα = 0.75，t = α s + √(1-α²) η，η ⊥ s"),
    ("6  填格", "Y_ij = g_j(⟨u_i, t_j⟩, e_ij) 对 u 线性\nK 类：线性 softmax\n噪声族 高斯 50% / t 30% / Cauchy 20%，σ = 0.3"),
    ("7  掩码 → 信封", "P(M)=5%，P(Q)=15%，独立抽，允许重合\nquery_mode = cells\n线上只有 values + missing + query"),
]

fig, ax = plt.subplots(figsize=(11.2, 13.2))
ax.set_xlim(0, 11.2)
ax.set_ylim(0, 13.4)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.text(5.6, 12.95, "DiscoSCM 观测 episode 怎么生成", ha="center", va="center", fontsize=16, fontweight="bold", color="#1a1a1a")
ax.text(5.6, 12.52, "空请求走这条流水线。旋钮都挂在对应步骤上，不另开一张表。", ha="center", va="center", fontsize=10, color="#555555")

y = 11.55
box_h = 1.38
gap = 0.22
left_x, left_w = 0.35, 2.35
right_x, right_w = 3.0, 7.7

colors = ["#1f4e79", "#2e6b9e", "#2e6b9e", "#c45c26", "#c45c26", "#2e6b9e", "#1f4e79"]

for i, ((title, body), c) in enumerate(zip(stages, colors)):
    y0 = y - box_h
    left = FancyBboxPatch((left_x, y0), left_w, box_h, boxstyle="round,pad=0.04,rounding_size=0.12",
                          facecolor=c, edgecolor="none")
    right = FancyBboxPatch((right_x, y0), right_w, box_h, boxstyle="round,pad=0.04,rounding_size=0.12",
                           facecolor="#f7f4ef", edgecolor="#d9d0c4", linewidth=1.0)
    ax.add_patch(left)
    ax.add_patch(right)
    ax.text(left_x + left_w / 2, y0 + box_h / 2, title, ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    ax.text(right_x + 0.22, y0 + box_h / 2, body, ha="left", va="center", fontsize=10, color="#222222", linespacing=1.45)
    if i < len(stages) - 1:
        ax.annotate("", xy=(left_x + left_w / 2, y0 - 0.02), xytext=(left_x + left_w / 2, y0 - gap + 0.02),
                    arrowprops=dict(arrowstyle="-", color="#b0b0b0", lw=1.4))
        ax.plot(left_x + left_w / 2, y0 - gap / 2, "v", color="#888888", markersize=7)
    y = y0 - gap

ax.text(5.6, 0.28, "因果图在特征 token 上，不在格子值上。训练默认不返回 U / DAG / t_j。", ha="center", va="center", fontsize=9, color="#666666")

out = Path("/workspace/tabuf-episode-api/docs/discoscm-pipeline.png")
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("wrote", out)
