from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from config import OUT_DIR, FIGURE_DIR, PROJECT_ROOT, ensure_project_dirs

# =========================
# 基础配置
# =========================
OUT_SUBDIR = OUT_DIR / "q3_single_factor_visualization"
FIG_SUBDIR = FIGURE_DIR / "q3_single_factor_visualization"

SUMMARY_CANDIDATES = [
    OUT_DIR / "q3_intervention_optimization" / "q3_patient_optimal_summary.csv",
    OUT_DIR / "q3_intervention_optimization_base" / "q3_patient_optimal_summary.csv",
    OUT_DIR / "q3_patient_optimal_summary.csv",
    PROJECT_ROOT / "q3_patient_optimal_summary.csv",
    Path(__file__).resolve().parent.parent / "q3_patient_optimal_summary.csv",
]

FEATURE_RULE_CANDIDATES = [
    OUT_DIR / "q3_intervention_optimization" / "q3_feature_rule_summary.csv",
    OUT_DIR / "q3_intervention_optimization_base" / "q3_feature_rule_summary.csv",
    OUT_DIR / "q3_feature_rule_summary.csv",
    PROJECT_ROOT / "q3_feature_rule_summary.csv",
    Path(__file__).resolve().parent.parent / "q3_feature_rule_summary.csv",
]

AGE_ORDER = ["40-59", "60-79", "80-89"]
ACTIVITY_ORDER = ["<40", "40-59", ">=60"]
SCORE_ORDER = ["55-58", "59-61", "62-65"]


# =========================
# 工具函数
# =========================
def ensure_dirs() -> None:
    ensure_project_dirs()
    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    FIG_SUBDIR.mkdir(parents=True, exist_ok=True)



def first_existing(paths: Iterable[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    checked = "\n".join(str(p) for p in paths)
    raise FileNotFoundError(f"未找到目标文件。已检查：\n{checked}")



def setup_chinese_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["font.sans-serif"] = candidates
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 220



def ordered_categorical(series: pd.Series, order: list[str]) -> pd.Series:
    return pd.Categorical(series.astype(str), categories=order, ordered=True)



def load_base_patient_summary() -> pd.DataFrame:
    path = first_existing(SUMMARY_CANDIDATES)
    df = pd.read_csv(path)
    required = [
        "年龄分层",
        "活动能力分层",
        "初始痰湿积分分层",
        "初始痰湿积分",
        "6个月末痰湿积分",
        "6个月总成本",
        "积分降幅(%)",
        "首月活动强度",
        "首月每周频次",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"q3_patient_optimal_summary.csv 缺少字段：{missing}")

    df = df.copy()
    df["年龄分层"] = ordered_categorical(df["年龄分层"], AGE_ORDER)
    df["活动能力分层"] = ordered_categorical(df["活动能力分层"], ACTIVITY_ORDER)
    df["初始痰湿积分分层"] = ordered_categorical(df["初始痰湿积分分层"], SCORE_ORDER)
    return df



def build_single_factor_summary(df: pd.DataFrame, group_col: str, order: list[str]) -> pd.DataFrame:
    group = (
        df.groupby(group_col, observed=False)
        .agg(
            患者数=(group_col, "size"),
            平均初始积分=("初始痰湿积分", "mean"),
            平均期末积分=("6个月末痰湿积分", "mean"),
            平均总成本=("6个月总成本", "mean"),
            平均降幅百分比=("积分降幅(%)", "mean"),
            平均首月强度=("首月活动强度", "mean"),
            平均首月频次=("首月每周频次", "mean"),
        )
        .reset_index()
    )
    group[group_col] = ordered_categorical(group[group_col], order)
    group = group.sort_values(group_col).reset_index(drop=True)
    return group



def save_single_factor_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {
        "年龄分层": build_single_factor_summary(df, "年龄分层", AGE_ORDER),
        "活动能力分层": build_single_factor_summary(df, "活动能力分层", ACTIVITY_ORDER),
        "初始痰湿积分分层": build_single_factor_summary(df, "初始痰湿积分分层", SCORE_ORDER),
    }
    name_map = {
        "年龄分层": "age",
        "活动能力分层": "activity",
        "初始痰湿积分分层": "initial_score",
    }
    for col, t in tables.items():
        t.to_csv(OUT_SUBDIR / f"q3_single_factor_summary_{name_map[col]}.csv", index=False, encoding="utf-8-sig")
    return tables



def add_value_labels(ax: plt.Axes, fmt: str = "{:.1f}") -> None:
    for patch in ax.patches:
        h = patch.get_height()
        if np.isnan(h):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            h,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
        )



def plot_metric_bar(table: pd.DataFrame, label_col: str, metric_col: str, title: str, ylabel: str, filename: str, fmt: str = "{:.1f}") -> None:
    x = table[label_col].astype(str)
    y = table[metric_col].astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(label_col)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2, value, fmt.format(value), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_SUBDIR / filename, bbox_inches="tight")
    plt.close(fig)



def plot_dual_axis_overview(table: pd.DataFrame, label_col: str, title: str, filename: str) -> None:
    labels = table[label_col].astype(str).tolist()
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(8.8, 5.8))
    width = 0.32

    y1 = table["平均期末积分"].astype(float).to_numpy()
    y2 = table["平均总成本"].astype(float).to_numpy()

    bars1 = ax1.bar(
        x - width / 2,
        y1,
        width=width,
        label="平均期末积分",
        zorder=3,
    )
    ax1.set_ylabel("平均期末积分")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel(label_col)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, max(y1) * 1.18)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        y2,
        width=width,
        alpha=0.75,
        label="平均6个月总成本",
        zorder=2,
    )
    ax2.set_ylabel("平均6个月总成本")
    ax2.set_ylim(0, max(y2) * 1.14)

    # 数值标签
    for bar in bars1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(y1) * 0.015,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for bar in bars2:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(y2) * 0.008,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # 标题放到整张图顶部
    fig.suptitle(title, fontsize=15, y=0.98)

    # 图例放到标题下方，单独一层
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=True,
    )

    # 给标题和图例留空间
    fig.subplots_adjust(top=0.84)

    fig.savefig(FIG_SUBDIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_three_metric_panel(table: pd.DataFrame, label_col: str, main_title: str, filename: str) -> None:
    labels = table[label_col].astype(str).tolist()
    metrics = [
        ("平均期末积分", "平均期末痰湿积分"),
        ("平均总成本", "平均6个月总成本"),
        ("平均降幅百分比", "平均积分降幅(%)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (col, title) in zip(axes, metrics):
        bars = ax.bar(labels, table[col].astype(float))
        ax.set_title(title)
        ax.set_xlabel(label_col)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)
        for bar, value in zip(bars, table[col].astype(float)):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(main_title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_SUBDIR / filename, bbox_inches="tight")
    plt.close(fig)



def plot_first_month_strategy(table: pd.DataFrame, label_col: str, title: str, filename: str) -> None:
    labels = table[label_col].astype(str).tolist()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x, table["平均首月强度"].astype(float), marker="o", linewidth=2, label="平均首月强度")
    ax.plot(x, table["平均首月频次"].astype(float), marker="s", linewidth=2, label="平均首月频次")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(label_col)
    ax.set_ylabel("首月方案强度 / 频次")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    for xi, yi in zip(x, table["平均首月强度"].astype(float)):
        ax.text(xi, yi, f"{yi:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, yi in zip(x, table["平均首月频次"].astype(float)):
        ax.text(xi, yi, f"{yi:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_SUBDIR / filename, bbox_inches="tight")
    plt.close(fig)



def build_markdown_report(tables: dict[str, pd.DataFrame]) -> None:
    lines: list[str] = []
    lines.append("# Q3 单因素分层可视化结果说明")
    lines.append("")
    lines.append("本结果基于**基础版第三问最优方案输出**（未做现实优化），从年龄分层、活动能力分层、初始痰湿积分分层三个维度，对平均期末积分、平均总成本、平均降幅以及首月方案特征进行重新聚合与可视化。")
    lines.append("")

    for name, table in tables.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(table.to_markdown(index=False))
        lines.append("")

    (OUT_SUBDIR / "q3_single_factor_visualization_summary.md").write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    ensure_dirs()
    setup_chinese_font()
    df = load_base_patient_summary()
    tables = save_single_factor_tables(df)

    key_map = {
        "年龄分层": "age",
        "活动能力分层": "activity",
        "初始痰湿积分分层": "initial_score",
    }

    for label_col, table in tables.items():
        tag = key_map[label_col]
        plot_metric_bar(
            table,
            label_col,
            "平均期末积分",
            title=f"基础版模型：{label_col}的平均期末痰湿积分",
            ylabel="平均期末痰湿积分",
            filename=f"q3_{tag}_avg_final_score_bar.png",
        )
        plot_metric_bar(
            table,
            label_col,
            "平均总成本",
            title=f"基础版模型：{label_col}的平均6个月总成本",
            ylabel="平均6个月总成本（元）",
            filename=f"q3_{tag}_avg_total_cost_bar.png",
        )
        plot_metric_bar(
            table,
            label_col,
            "平均降幅百分比",
            title=f"基础版模型：{label_col}的平均积分降幅",
            ylabel="平均积分降幅（%）",
            filename=f"q3_{tag}_avg_drop_pct_bar.png",
        )
        plot_dual_axis_overview(
            table,
            label_col,
            title=f"基础版模型：{label_col}的疗效—成本对照",
            filename=f"q3_{tag}_effect_cost_dual_axis.png",
        )
        plot_three_metric_panel(
            table,
            label_col,
            main_title=f"基础版模型：{label_col}的单因素统计三联图",
            filename=f"q3_{tag}_three_metric_panel.png",
        )
        plot_first_month_strategy(
            table,
            label_col,
            title=f"基础版模型：{label_col}的首月方案特征",
            filename=f"q3_{tag}_first_month_strategy.png",
        )

    build_markdown_report(tables)

    print("第三问基础版单因素分层可视化已完成。")
    print(f"输出表格目录：{OUT_SUBDIR}")
    print(f"输出图像目录：{FIG_SUBDIR}")


if __name__ == "__main__":
    main()
