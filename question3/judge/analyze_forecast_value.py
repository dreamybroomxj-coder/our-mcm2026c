#!/usr/bin/env python3
"""问题三：8种预报/调单组合的经济价值分析。

唯一正式复现命令（在 C题 根目录执行）：
F:\\miniconda3\\envs\\fond\\python.exe 附件\\question3\\judge\\analyze_forecast_value.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from itertools import combinations, permutations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


SEED = 20260912
PLAYERS = ("6", "12", "18")
STRATEGIES = {
    frozenset(): "H0",
    frozenset({"6"}): "H0_6",
    frozenset({"12"}): "H0_12",
    frozenset({"18"}): "H0_18",
    frozenset({"6", "12"}): "H0_6_12",
    frozenset({"6", "18"}): "H0_6_18",
    frozenset({"12", "18"}): "H0_12_18",
    frozenset({"6", "12", "18"}): "H0_6_12_18",
}
LABELS = {
    "H0": "仅0点", "H0_6": "0+6点", "H0_12": "0+12点", "H0_18": "0+18点",
    "H0_6_12": "0+6+12点", "H0_6_18": "0+6+18点",
    "H0_12_18": "0+12+18点", "H0_6_12_18": "0+6+12+18点",
}
COLORS = {"6": "#0072B2", "12": "#E69F00", "18": "#009E73"}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_main = here.parent / "第三问_完整计算" / "第三问_完整计算" / "results" / "main"
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=default_main)
    p.add_argument("--output-dir", type=Path, default=here)
    p.add_argument("--test-start", default="2025-02-25")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--blocks", type=int, nargs="+", default=[7, 14])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--smoke", action="store_true", help="只验证读取、配对和核心公式，不正式出图")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_daily(input_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frames, hashes = [], {}
    for coalition, strategy in STRATEGIES.items():
        path = input_dir / strategy / "daily.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        required = {"strategy", "window_date", "cash_cost_yuan", "emergency_kwh"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} 缺列: {sorted(required - set(df.columns))}")
        df["window_date"] = pd.to_datetime(df["window_date"])
        df["strategy"] = strategy
        df["coalition"] = "+".join(sorted(coalition, key=int)) or "none"
        frames.append(df)
        hashes[str(path.resolve())] = sha256(path)
    all_df = pd.concat(frames, ignore_index=True)
    counts = all_df.groupby("strategy")["window_date"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"八种方案日期数不一致: {counts.to_dict()}")
    pivot = all_df.pivot(index="window_date", columns="strategy", values="cash_cost_yuan")
    if pivot.isna().any().any() or len(pivot.columns) != 8:
        raise ValueError("八种方案不能按日期完整配对")
    return all_df.sort_values(["window_date", "strategy"]), hashes


def cost_map(pivot: pd.DataFrame, indices: np.ndarray | None = None) -> dict[frozenset[str], float]:
    x = pivot if indices is None else pivot.iloc[indices]
    return {coalition: float(x[strategy].sum()) for coalition, strategy in STRATEGIES.items()}


def shapley(costs: dict[frozenset[str], float]) -> dict[str, float]:
    """以费用下降为收益；返回各更新时刻对总节省的Shapley分摊。"""
    out = {p: 0.0 for p in PLAYERS}
    perms = list(permutations(PLAYERS))
    for order in perms:
        joined = frozenset()
        for p in order:
            new = joined | {p}
            out[p] += costs[joined] - costs[new]
            joined = new
    return {p: v / len(perms) for p, v in out.items()}


def marginal_table(costs: dict[frozenset[str], float]) -> pd.DataFrame:
    rows = []
    for p in PLAYERS:
        others = [x for x in PLAYERS if x != p]
        for n in range(3):
            for context_tuple in combinations(others, n):
                context = frozenset(context_tuple)
                rows.append({
                    "新增时刻": f"{p}:00", "已有时刻": "+".join(sorted(context, key=int)) or "无",
                    "边际节省_元": costs[context] - costs[context | {p}],
                })
    return pd.DataFrame(rows)


def interaction_table(costs: dict[frozenset[str], float]) -> pd.DataFrame:
    rows = []
    base = costs[frozenset()]
    for a, b in combinations(PLAYERS, 2):
        ca, cb, cab = costs[frozenset({a})], costs[frozenset({b})], costs[frozenset({a, b})]
        avg_benchmark_gain = (ca + cb) / 2 - cab
        redundancy = (base - ca) + (base - cb) - (base - cab)
        rows.append({
            "组合": f"{a}:00+{b}:00", "单时刻费用均值_元": (ca + cb) / 2,
            "联合方案费用_元": cab, "联合相对单方案均值节省_元": avg_benchmark_gain,
            "加法冗余_元": redundancy,
            "解释": "边际收益重叠（替代）" if redundancy > 0 else "正协同（互补）",
        })
    return pd.DataFrame(rows)


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if block < 1 or block > n:
        raise ValueError(f"非法区块长度 {block}, n={n}")
    starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def bootstrap(pivot: pd.DataFrame, reps: int, block: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + block * 1009)
    rows = []
    for r in range(reps):
        idx = moving_block_indices(len(pivot), block, rng)
        costs = cost_map(pivot, idx)
        phi = shapley(costs)
        row = {
            "replicate": r, "block_days": block,
            "all_vs_h0_saving_yuan_per_day": (costs[frozenset()] - costs[frozenset(PLAYERS)]) / len(idx),
            "add6_given_12_18_yuan_per_day": (costs[frozenset({"12", "18"})] - costs[frozenset(PLAYERS)]) / len(idx),
            "shapley_6_yuan_per_day": phi["6"] / len(idx),
            "shapley_12_yuan_per_day": phi["12"] / len(idx),
            "shapley_18_yuan_per_day": phi["18"] / len(idx),
            "shapley_12_minus_18_yuan_per_day": (phi["12"] - phi["18"]) / len(idx),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def ci_table(boot: pd.DataFrame) -> pd.DataFrame:
    names = {
        "all_vs_h0_saving_yuan_per_day": "全时刻相对仅0点的日均节省",
        "add6_given_12_18_yuan_per_day": "已有12/18点后增加6点的日均节省",
        "shapley_6_yuan_per_day": "6点Shapley日均贡献",
        "shapley_12_yuan_per_day": "12点Shapley日均贡献",
        "shapley_18_yuan_per_day": "18点Shapley日均贡献",
        "shapley_12_minus_18_yuan_per_day": "12点减18点Shapley日均贡献",
    }
    rows = []
    for block, g in boot.groupby("block_days"):
        for col, label in names.items():
            q = g[col].quantile([0.025, 0.5, 0.975])
            rows.append({"区块长度_日": block, "指标": label, "均值": g[col].mean(),
                         "中位数": q.loc[0.5], "CI下限": q.loc[0.025], "CI上限": q.loc[0.975],
                         "区间跨0": bool(q.loc[0.025] <= 0 <= q.loc[0.975])})
    return pd.DataFrame(rows)


def setup_plot_style() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [x for x in candidates if x in available] + ["DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "svg.fonttype": "none", "pdf.fonttype": 42, "figure.dpi": 120, "savefig.dpi": 320,
    })


def savefig(fig: plt.Figure, figures: Path, stem: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"{stem}.{ext}", bbox_inches="tight", dpi=320 if ext == "png" else None)
    preview_dir = figures / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    gray = preview_dir / f"{stem}_gray.png"
    color_preview = preview_dir / f".{stem}_preview_tmp.png"
    fig.savefig(color_preview, bbox_inches="tight", dpi=160)
    with Image.open(color_preview) as im:
        ImageOps.grayscale(im).save(gray, dpi=(160, 160))
    color_preview.unlink()
    plt.close(fig)


def make_figures(all_df: pd.DataFrame, pivot_test: pd.DataFrame, costs: dict, marg: pd.DataFrame,
                 inter: pd.DataFrame, shap: dict, boot: pd.DataFrame, ci: pd.DataFrame, figures: Path) -> None:
    setup_plot_style()
    order = list(STRATEGIES.values())
    labels = [LABELS[x] for x in order]

    # raw 1: 八方案日费用分布
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    vals = [all_df.loc[all_df.strategy == s, "cash_cost_yuan"].to_numpy() / 1000 for s in order]
    ax.boxplot(vals, tick_labels=labels, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#D9EAF7", "edgecolor": "#3B6D8C"}, medianprops={"color": "#D55E00"})
    ax.set_ylabel("单合同现金费用（千元）"); ax.set_title("八种预报组合的日费用分布")
    ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.22)
    savefig(fig, figures, "raw_q3_01_daily_cost_distribution")

    # raw 2: 月度相对H0节省热力图
    tmp = all_df.copy(); tmp["month"] = tmp.window_date.dt.to_period("M").astype(str)
    mon = tmp.groupby(["month", "strategy"]).cash_cost_yuan.sum().unstack()
    saving = mon["H0"].to_numpy()[:, None] - mon[order].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    im = ax.imshow(saving / 1000, aspect="auto", cmap="viridis")
    ax.set_xticks(range(8), labels, rotation=25, ha="right"); ax.set_yticks(range(len(mon)), mon.index)
    ax.set_title("各月相对仅0点方案的费用节省"); ax.set_ylabel("合同月份")
    cb = fig.colorbar(im, ax=ax); cb.set_label("节省（千元）")
    savefig(fig, figures, "raw_q3_02_monthly_saving_heatmap")

    # raw 3: 全时刻相对H0的逐日差异及7日均线
    d = (pivot_test["H0"] - pivot_test["H0_6_12_18"]) / 1000
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.axhline(0, color="#666666", lw=.8); ax.plot(d.index, d, color="#9ECAE1", lw=.6, label="逐日")
    ax.plot(d.index, d.rolling(7, min_periods=1).mean(), color="#0072B2", lw=1.5, label="7日移动平均")
    ax.set_ylabel("相对H0节省（千元/日）"); ax.set_title("全时刻方案逐日经济收益"); ax.legend(frameon=False)
    savefig(fig, figures, "raw_q3_03_daily_saving_timeseries")

    # process 1: 联盟价值
    coalition_savings = [(LABELS[STRATEGIES[c]], (costs[frozenset()] - costs[c]) / 1000) for c in STRATEGIES]
    coalition_savings.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    ax.barh([x[0] for x in coalition_savings], [x[1] for x in coalition_savings], color="#56B4E9")
    ax.set_xlabel("相对仅0点方案节省（千元）"); ax.set_title("不同预报联盟的经济价值"); ax.grid(axis="x", alpha=.2)
    savefig(fig, figures, "process_q3_01_coalition_value")

    # process 2: 边际收益矩阵
    contexts = ["无", "6", "12", "18", "6+12", "6+18", "12+18"]
    mat = np.full((3, len(contexts)), np.nan)
    for i, p in enumerate(PLAYERS):
        for _, row in marg[marg["新增时刻"] == f"{p}:00"].iterrows():
            mat[i, contexts.index(row["已有时刻"])] = row["边际节省_元"] / 1000
    fig, ax = plt.subplots(figsize=(7.2, 2.8), constrained_layout=True)
    masked = np.ma.masked_invalid(mat); im = ax.imshow(masked, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(contexts)), contexts); ax.set_yticks(range(3), [f"新增{p}点" for p in PLAYERS])
    for i in range(3):
        for j in range(len(contexts)):
            if np.isfinite(mat[i, j]): ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", fontsize=7)
    cb = fig.colorbar(im, ax=ax); cb.set_label("边际节省（千元）")
    ax.set_xlabel("已有补充时刻"); ax.set_title("新增预报在不同信息背景下的边际收益")
    savefig(fig, figures, "process_q3_02_marginal_benefit_heatmap")

    # process 3: 用户指定的均值基准与标准加法冗余
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    x = np.arange(len(inter)); w = .36
    ax.bar(x-w/2, inter["联合相对单方案均值节省_元"]/1000, w, label="相对两个单方案费用均值的节省", color="#0072B2")
    ax.bar(x+w/2, inter["加法冗余_元"]/1000, w, label="单项收益的加法冗余", color="#E69F00")
    ax.set_xticks(x, inter["组合"]); ax.set_ylabel("金额（千元）"); ax.set_title("不同预报时刻的组合效应")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    savefig(fig, figures, "process_q3_03_combination_effect")

    # result 1: Shapley点估计 + 7日CI
    ci7 = ci[(ci["区块长度_日"] == 7) & ci["指标"].str.contains("Shapley日均贡献")].copy()
    point = np.array([shap[p] / len(pivot_test) for p in PLAYERS])
    lo = ci7.set_index("指标").loc[[f"{p}点Shapley日均贡献" for p in PLAYERS], "CI下限"].to_numpy()
    hi = ci7.set_index("指标").loc[[f"{p}点Shapley日均贡献" for p in PLAYERS], "CI上限"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 3.7), constrained_layout=True)
    ax.bar(PLAYERS, point, color=[COLORS[p] for p in PLAYERS], alpha=.85)
    ax.errorbar(range(3), point, yerr=np.vstack([point-lo, hi-point]), fmt="none", color="black", capsize=4)
    ax.set_xlabel("补充预报时刻"); ax.set_ylabel("Shapley贡献（元/日）"); ax.set_title("各时刻经济贡献分解（测试期）")
    ax.grid(axis="y", alpha=.2)
    savefig(fig, figures, "result_q3_01_shapley_contribution")

    # result 2: Bootstrap森林图
    metrics = ["全时刻相对仅0点的日均节省", "已有12/18点后增加6点的日均节省", "12点减18点Shapley日均贡献"]
    c7 = ci[(ci["区块长度_日"] == 7) & ci["指标"].isin(metrics)].set_index("指标").loc[metrics]
    y = np.arange(len(metrics))[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    ax.axvline(0, color="#555555", lw=.8)
    ax.errorbar(c7["中位数"], y, xerr=np.vstack([c7["中位数"]-c7["CI下限"], c7["CI上限"]-c7["中位数"]]),
                fmt="o", color="#0072B2", capsize=4)
    ax.set_yticks(y, metrics); ax.set_xlabel("日均费用差（元/日，正值表示左述方案更优）")
    ax.set_title("7日配对移动区块Bootstrap的95%区间")
    savefig(fig, figures, "result_q3_02_bootstrap_forest")

    # result 3: 数据获取成本阈值
    vals = point; orderp = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    ax.barh(np.array(PLAYERS)[orderp], vals[orderp], color=[COLORS[PLAYERS[i]] for i in orderp])
    for j, i in enumerate(orderp): ax.text(vals[i] + max(vals)*.015, j, f"{vals[i]:.0f}元/日", va="center", fontsize=8)
    ax.set_xlabel("补充数据的平均经济价值上限（元/日）"); ax.set_ylabel("预报时刻")
    ax.set_title("基于Shapley价值的数据获取成本阈值"); ax.grid(axis="x", alpha=.2)
    savefig(fig, figures, "result_q3_03_data_cost_threshold")


def fmt_money(x: float) -> str:
    return f"{x:,.2f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """不依赖可选 tabulate 包的轻量 Markdown 表格导出。"""
    def cell(v: object) -> str:
        if isinstance(v, (float, np.floating)):
            return f"{float(v):,.2f}"
        return str(v).replace("|", "\\|")
    header = "|" + "|".join(map(str, df.columns)) + "|"
    separator = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["|" + "|".join(cell(v) for v in row) + "|" for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_report(out: Path, costs_full: dict, costs_test: dict, shap_full: dict, shap_test: dict,
                 marg: pd.DataFrame, inter: pd.DataFrame, ci: pd.DataFrame, n_test: int) -> None:
    total = costs_test[frozenset()] - costs_test[frozenset(PLAYERS)]
    lines = [
        "# 问题三补充预报价值研究", "",
        "## 1. 研究口径", "",
        f"以6:00、12:00和18:00是否启用为三个二元因素，八种方案构成完整的 $2^3$ 组合。主结论采用2025-02-25起的冻结测试期（{n_test}个合同），全年334个合同仅作回溯稳健性对照。每日费用按合同起始日配对。",
        "移动区块Bootstrap对已经连续运行得到的逐日费用差重采样，并未重新求解被抽中的SOC轨迹，因此它用于衡量时间区段上的经验稳定性，不等同于重新模拟新的年度路径。", "",
        "## 2. Shapley经济贡献", "",
        f"全时刻方案在测试期相对仅0点方案共节省 **{fmt_money(total)}元**。Shapley值将该总收益按每个时刻在6种加入顺序中的平均边际贡献进行唯一分摊：", "",
        "|时刻|测试期贡献（元）|占比|全年贡献（元）|", "|---|---:|---:|---:|",
    ]
    for p in PLAYERS:
        lines.append(f"|{p}:00|{fmt_money(shap_test[p])}|{shap_test[p]/total:.2%}|{fmt_money(shap_full[p])}|")
    lines += ["", "12点贡献最大，18点次之，6点最小；三者贡献均为正。若只能补充一个时刻，现有证据优先支持12点。", "",
              "![Shapley贡献](figures/result_q3_01_shapley_contribution.png)", "", "## 3. 增加预报的边际效益", "",
              "边际效益取决于已经启用了哪些时刻，不能只报告单独加入的结果。", "",
              dataframe_to_markdown(marg), "",
              "所有情境下新增时刻的边际节省均为正，但随着其他时刻已经启用，新增价值普遍下降，说明信息之间存在覆盖。", "",
              "![边际效益](figures/process_q3_02_marginal_benefit_heatmap.png)", "", "## 4. 组合效应", "",
              "按题意同时报告两种口径：第一种比较两个单时刻方案费用的平均值与联合方案；第二种使用标准的加法收益分解。‘加法冗余’为正表示两条单项收益存在重叠，即替代关系，而不是负面作用。", "",
              dataframe_to_markdown(inter), "",
              "三个二时刻联合方案均优于对应两个单时刻方案的费用均值；与此同时，加法冗余均为正，表明这些时刻总体属于部分替代而非超加性互补。其中12点与18点的收益重叠最大。", "",
              "![组合效应](figures/process_q3_03_combination_effect.png)", "", "## 5. 配对移动区块Bootstrap", "",
              "同一次重采样对八种方案使用完全相同的连续日期区块，保留同日可比性和一部分跨日相关。主结果使用7日区块以保留周周期，14日区块用于稳健性复核；每种长度重复5000次。", "",
              dataframe_to_markdown(ci), "",
              "若95%区间不跨0，可表述为该方向在不同连续时间区段的重采样中较稳定；本文不把它夸大为独立同分布假设下的传统显著性检验。", "",
              "![Bootstrap区间](figures/result_q3_02_bootstrap_forest.png)", "", "## 6. 对开放性问题的回答", "",
              "在不计补充数据获取成本时，三个时刻全部启用的测试期总费用最低，且三个时刻Shapley贡献均为正，因此补充日内光伏预报具有经济价值。优先级为12点、18点、6点。",
              "是否实际增设某个数据源还应比较其获取、通信和维护成本与相应边际收益。Shapley日均贡献可以作为长期平均成本阈值；若系统已有12点和18点接口，是否再增加6点则应使用“已有12/18点后增加6点”的条件边际收益，而不是6点的独立贡献。", "",
              "![成本阈值](figures/result_q3_03_data_cost_threshold.png)", "", "## 7. 图表索引", "",
              "其余数据分布、月度稳定性、逐日差异、联盟价值图均位于 `figures/`。PNG用于论文预览，SVG保留可编辑矢量文字，`*_gray.png`用于灰度可读性检查。", ""]
    (out / "问题三_补充预报价值分析.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args(); out = args.output_dir.resolve(); results = out / "results"; figures = out / "figures"
    results.mkdir(parents=True, exist_ok=True)
    all_df, hashes = load_daily(args.input_dir.resolve())
    pivot_full = all_df.pivot(index="window_date", columns="strategy", values="cash_cost_yuan").sort_index()
    pivot_test = pivot_full.loc[pd.Timestamp(args.test_start):]
    if len(pivot_test) == 0: raise ValueError("测试期为空")
    costs_full, costs_test = cost_map(pivot_full), cost_map(pivot_test)
    shap_full, shap_test = shapley(costs_full), shapley(costs_test)
    if not np.isclose(sum(shap_test.values()), costs_test[frozenset()] - costs_test[frozenset(PLAYERS)], atol=1e-6):
        raise AssertionError("Shapley效率公理校验失败")
    marg = marginal_table(costs_test); inter = interaction_table(costs_test)
    if args.smoke:
        smoke = {"status": "PASS", "n_full": len(pivot_full), "n_test": len(pivot_test), "shapley_test": shap_test,
                 "shapley_sum": sum(shap_test.values()), "total_saving": costs_test[frozenset()] - costs_test[frozenset(PLAYERS)]}
        (results / "smoke_result.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(smoke, ensure_ascii=False, indent=2)); return
    boots = [bootstrap(pivot_test, args.bootstrap, b, args.seed) for b in args.blocks]
    boot = pd.concat(boots, ignore_index=True); ci = ci_table(boot)
    pd.DataFrame([{"时刻": f"{p}:00", "测试期Shapley_元": shap_test[p], "全年Shapley_元": shap_full[p],
                   "测试期占比": shap_test[p]/sum(shap_test.values())} for p in PLAYERS]).to_csv(results / "shapley_value.csv", index=False, encoding="utf-8-sig")
    marg.to_csv(results / "marginal_benefit.csv", index=False, encoding="utf-8-sig")
    inter.to_csv(results / "combination_effect.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(results / "bootstrap_replicates.csv", index=False, encoding="utf-8-sig")
    ci.to_csv(results / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    make_figures(all_df, pivot_test, costs_test, marg, inter, shap_test, boot, ci, figures)
    write_report(out, costs_full, costs_test, shap_full, shap_test, marg, inter, ci, len(pivot_test))
    manifest = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "matplotlib": mpl.__version__, "seed": args.seed, "bootstrap_repetitions": args.bootstrap,
                "block_lengths_days": args.blocks, "test_start": args.test_start, "input_sha256": hashes,
                "command": "F:\\miniconda3\\envs\\fond\\python.exe 附件\\question3\\judge\\analyze_forecast_value.py"}
    (results / "复现清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{out}")


if __name__ == "__main__":
    main()
