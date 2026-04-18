
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
问题一：双分支结果交叉合并
--------------------------------
作用：
1. 读取痰湿分支 / 高血脂分支的多变量筛选结果
2. 将两个分支的排序结果合并为一张总表
3. 生成“风险核心 / 风险辅助 / 痰湿辅助 / 交叉候选”的分层结果
4. 输出汇总表、分层表和图表

说明：
- 本脚本沿用 config.py 的路径配置
- 交叉合并以“风险分支优先、痰湿分支辅助”为原则
- 由于当前痰湿分支整体解释力较弱（Elastic Net 全零、R² < 0），
  因此痰湿分支仅作为“辅助表征”而非“核心判别”依据
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import re
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 路径配置
# =========================
try:
    from config import OUT_DIR, FIGURE_DIR  # type: ignore
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    OUT_DIR = PROJECT_ROOT / "out"
    FIGURE_DIR = PROJECT_ROOT / "figure"

MULTI_OUT_DIR = Path(OUT_DIR) / "q1_multivariable"
MERGE_OUT_DIR = Path(OUT_DIR) / "q1_cross_merge"
MERGE_FIG_DIR = Path(FIGURE_DIR) / "q1_cross_merge"

MERGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
MERGE_FIG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 输入文件
# =========================
PH_RANK_CSV = MULTI_OUT_DIR / "q1_phlegm_multivariable_rank.csv"
RISK_RANK_CSV = MULTI_OUT_DIR / "q1_risk_multivariable_rank.csv"
PH_ENET_CSV = MULTI_OUT_DIR / "q1_phlegm_elasticnet_coefficients.csv"
PH_RF_CSV = MULTI_OUT_DIR / "q1_phlegm_rf_permutation_importance.csv"
RISK_LOGIT_CSV = MULTI_OUT_DIR / "q1_risk_logit_coefficients.csv"
RISK_RF_CSV = MULTI_OUT_DIR / "q1_risk_rf_permutation_importance.csv"
SUMMARY_TXT = MULTI_OUT_DIR / "q1_multivariable_summary.txt"

# =========================
# 输出文件
# =========================
MERGE_TABLE_CSV = MERGE_OUT_DIR / "q1_branch_cross_merge.csv"
LAYER_TABLE_CSV = MERGE_OUT_DIR / "q1_indicator_layers.csv"
SUMMARY_OUT_TXT = MERGE_OUT_DIR / "q1_cross_merge_summary.txt"
META_JSON = MERGE_OUT_DIR / "q1_cross_merge_metadata.json"

FIG_JOINT_SCORE = MERGE_FIG_DIR / "q1_joint_score_top.png"
FIG_RANK_MAP = MERGE_FIG_DIR / "q1_rank_map.png"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"缺少输入文件：{path}")


def load_required_files() -> Dict[str, pd.DataFrame]:
    for p in [
        PH_RANK_CSV, RISK_RANK_CSV,
        PH_ENET_CSV, PH_RF_CSV,
        RISK_LOGIT_CSV, RISK_RF_CSV,
    ]:
        require_file(p)

    data = {
        "ph_rank": pd.read_csv(PH_RANK_CSV),
        "risk_rank": pd.read_csv(RISK_RANK_CSV),
        "ph_enet": pd.read_csv(PH_ENET_CSV),
        "ph_rf": pd.read_csv(PH_RF_CSV),
        "risk_logit": pd.read_csv(RISK_LOGIT_CSV),
        "risk_rf": pd.read_csv(RISK_RF_CSV),
    }
    return data


def safe_minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    s_min = s.min()
    s_max = s.max()
    if math.isclose(float(s_min), float(s_max), abs_tol=1e-12):
        return pd.Series(np.ones(len(s)), index=s.index) if s_max > 0 else pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s_min) / (s_max - s_min)


def rank_to_score(rank_series: pd.Series) -> pd.Series:
    """把排名转换为 0-1 分数：名次越高分数越大"""
    r = rank_series.astype(float)
    n = len(r)
    if n <= 1:
        return pd.Series(np.ones(len(r)), index=r.index)
    return 1 - (r - 1) / (n - 1)


def parse_summary_metrics(summary_path: Path) -> Dict[str, Optional[float]]:
    metrics = {
        "ph_r2_mean": None,
        "risk_auc_mean": None,
    }
    if not summary_path.exists():
        return metrics

    text = summary_path.read_text(encoding="utf-8", errors="ignore")

    m1 = re.search(r"R²\s*=\s*([\-0-9.]+)", text)
    if m1:
        metrics["ph_r2_mean"] = float(m1.group(1))

    m2 = re.search(r"AUC\s*=\s*([\-0-9.]+)", text)
    if m2:
        metrics["risk_auc_mean"] = float(m2.group(1))

    return metrics


def build_branch_merge_table(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    ph_rank = data["ph_rank"].copy()
    risk_rank = data["risk_rank"].copy()
    ph_enet = data["ph_enet"].copy()
    ph_rf = data["ph_rf"].copy()
    risk_logit = data["risk_logit"].copy()
    risk_rf = data["risk_rf"].copy()

    # 统一列名
    ph_rank = ph_rank.rename(columns={"平均排名": "痰湿平均排名"})
    risk_rank = risk_rank.rename(columns={"平均排名": "高血脂平均排名"})
    ph_enet = ph_enet.rename(columns={"ElasticNet系数": "痰湿_ElasticNet系数", "|系数|": "痰湿_|系数|"})
    ph_rf = ph_rf.rename(columns={"RF置换重要性均值": "痰湿_RF置换重要性均值", "RF置换重要性标准差": "痰湿_RF置换重要性标准差"})
    risk_logit = risk_logit.rename(columns={"Logit系数": "高血脂_Logit系数", "OR=exp(coef)": "高血脂_OR", "|系数|": "高血脂_|系数|"})
    risk_rf = risk_rf.rename(columns={"RF置换重要性均值": "高血脂_RF置换重要性均值", "RF置换重要性标准差": "高血脂_RF置换重要性标准差"})

    # 合并
    df = ph_rank[["指标", "痰湿平均排名"]].merge(
        risk_rank[["指标", "高血脂平均排名"]],
        on="指标", how="outer"
    )
    df = df.merge(ph_enet[["指标", "痰湿_ElasticNet系数", "痰湿_|系数|"]], on="指标", how="left")
    df = df.merge(ph_rf[["指标", "痰湿_RF置换重要性均值", "痰湿_RF置换重要性标准差"]], on="指标", how="left")
    df = df.merge(risk_logit[["指标", "高血脂_Logit系数", "高血脂_OR", "高血脂_|系数|"]], on="指标", how="left")
    df = df.merge(risk_rf[["指标", "高血脂_RF置换重要性均值", "高血脂_RF置换重要性标准差"]], on="指标", how="left")

    # 缺失值填充
    rank_fill = len(df)
    df["痰湿平均排名"] = df["痰湿平均排名"].fillna(rank_fill)
    df["高血脂平均排名"] = df["高血脂平均排名"].fillna(rank_fill)

    for col in [
        "痰湿_ElasticNet系数", "痰湿_|系数|",
        "痰湿_RF置换重要性均值", "痰湿_RF置换重要性标准差",
        "高血脂_Logit系数", "高血脂_OR", "高血脂_|系数|",
        "高血脂_RF置换重要性均值", "高血脂_RF置换重要性标准差",
    ]:
        df[col] = df[col].fillna(0.0)

    # 分支内分数
    df["痰湿分支得分"] = rank_to_score(df["痰湿平均排名"])
    df["高血脂分支得分"] = rank_to_score(df["高血脂平均排名"])

    # 由于痰湿分支整体较弱，采用偏保守加权
    # 风险分支 0.65，痰湿分支 0.35
    df["联合得分"] = 0.35 * df["痰湿分支得分"] + 0.65 * df["高血脂分支得分"]

    # 额外信息：线性模型是否保留 / 树模型是否有贡献
    df["痰湿线性保留"] = (df["痰湿_|系数|"] > 1e-12).astype(int)
    df["高血脂线性保留"] = (df["高血脂_|系数|"] > 1e-12).astype(int)
    df["痰湿树模型有贡献"] = (df["痰湿_RF置换重要性均值"] > 1e-12).astype(int)
    df["高血脂树模型有贡献"] = (df["高血脂_RF置换重要性均值"] > 1e-12).astype(int)

    # 最终总排序
    df = df.sort_values(["联合得分", "高血脂分支得分", "痰湿分支得分"], ascending=[False, False, False]).reset_index(drop=True)
    df["联合排名"] = np.arange(1, len(df) + 1)

    return df


def classify_layer(row: pd.Series) -> str:
    risk_rank = float(row["高血脂平均排名"])
    ph_rank = float(row["痰湿平均排名"])
    risk_score = float(row["高血脂分支得分"])
    ph_score = float(row["痰湿分支得分"])

    # 分层规则（按当前项目阶段的保守口径）
    # 1) 风险前两名：高血脂风险核心指标
    if risk_rank <= 2:
        return "高血脂风险核心指标"

    # 2) 两个分支都靠前：双分支交叉候选指标
    if risk_rank <= 5 and ph_rank <= 4:
        return "双分支交叉候选指标"

    # 3) 痰湿分支非常靠前：痰湿辅助表征指标
    if ph_rank <= 3:
        return "痰湿辅助表征指标"

    # 4) 高血脂分支前五：高血脂辅助指标
    if risk_rank <= 5:
        return "高血脂辅助指标"

    return "一般候选指标"


def build_layer_table(df_merge: pd.DataFrame) -> pd.DataFrame:
    layer_df = df_merge.copy()
    layer_df["分层类别"] = layer_df.apply(classify_layer, axis=1)
    layer_df = layer_df[
        [
            "指标",
            "痰湿平均排名",
            "高血脂平均排名",
            "痰湿分支得分",
            "高血脂分支得分",
            "联合得分",
            "联合排名",
            "分层类别",
            "痰湿_ElasticNet系数",
            "痰湿_RF置换重要性均值",
            "高血脂_Logit系数",
            "高血脂_OR",
            "高血脂_RF置换重要性均值",
        ]
    ].copy()
    return layer_df


def plot_joint_score(df: pd.DataFrame) -> None:
    top_df = df.sort_values("联合得分", ascending=False).head(8).sort_values("联合得分", ascending=True)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.barh(top_df["指标"], top_df["联合得分"])
    plt.title("问题一双分支交叉合并：联合得分 Top 指标")
    plt.xlabel("joint score")
    plt.tight_layout()
    plt.savefig(FIG_JOINT_SCORE, bbox_inches="tight")
    plt.close()


def plot_rank_map(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6), dpi=150)
    color_map = {
        "高血脂风险核心指标": "#d62728",
        "双分支交叉候选指标": "#ff7f0e",
        "痰湿辅助表征指标": "#2ca02c",
        "高血脂辅助指标": "#1f77b4",
        "一般候选指标": "#7f7f7f",
    }

    temp = df.copy()
    temp["分层类别"] = temp.apply(classify_layer, axis=1)

    for layer, g in temp.groupby("分层类别"):
        plt.scatter(
            g["高血脂平均排名"],
            g["痰湿平均排名"],
            s=80,
            label=layer,
            alpha=0.85,
            color=color_map.get(layer, "#333333"),
        )
        for _, row in g.iterrows():
            plt.text(
                float(row["高血脂平均排名"]) + 0.05,
                float(row["痰湿平均排名"]) + 0.05,
                str(row["指标"]),
                fontsize=9,
            )

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xlabel("高血脂分支排名（越靠左越重要）")
    plt.ylabel("痰湿分支排名（越靠上越重要）")
    plt.title("双分支排名映射图")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_RANK_MAP, bbox_inches="tight")
    plt.close()


def write_summary(df_merge: pd.DataFrame, df_layer: pd.DataFrame, metrics: Dict[str, Optional[float]]) -> None:
    layer_counts = df_layer["分层类别"].value_counts().to_dict()

    core = df_layer[df_layer["分层类别"] == "高血脂风险核心指标"]["指标"].tolist()
    cross = df_layer[df_layer["分层类别"] == "双分支交叉候选指标"]["指标"].tolist()
    ph_aux = df_layer[df_layer["分层类别"] == "痰湿辅助表征指标"]["指标"].tolist()
    risk_aux = df_layer[df_layer["分层类别"] == "高血脂辅助指标"]["指标"].tolist()

    lines = []
    lines.append("问题一：双分支结果交叉合并摘要")
    lines.append("=" * 60)
    lines.append(f"输入目录: {MULTI_OUT_DIR}")
    lines.append(f"输出目录: {MERGE_OUT_DIR}")
    lines.append(f"图片目录: {MERGE_FIG_DIR}")
    lines.append("")

    lines.append("[一] 合并原则")
    lines.append("1. 以‘高血脂分支优先、痰湿分支辅助’为总原则进行交叉合并。")
    lines.append("2. 联合得分 = 0.35 × 痰湿分支得分 + 0.65 × 高血脂分支得分。")
    lines.append("3. 由于当前痰湿分支 Elastic Net 全零，且回归 R² 为负，因此痰湿分支只作为辅助表征依据，不作为核心判别依据。")
    lines.append("")

    lines.append("[二] 分支性能背景")
    lines.append(f"痰湿分支 R²（来自上一阶段摘要）: {metrics.get('ph_r2_mean')}")
    lines.append(f"高血脂分支 AUC（来自上一阶段摘要）: {metrics.get('risk_auc_mean')}")
    lines.append("")

    lines.append("[三] 分层类别统计")
    for k, v in layer_counts.items():
        lines.append(f"{k}: {v} 个")

    lines.append("")
    lines.append("[四] 推荐保留结果")
    lines.append(f"高血脂风险核心指标: {', '.join(core) if core else '无'}")
    lines.append(f"双分支交叉候选指标: {', '.join(cross) if cross else '无'}")
    lines.append(f"痰湿辅助表征指标: {', '.join(ph_aux) if ph_aux else '无'}")
    lines.append(f"高血脂辅助指标: {', '.join(risk_aux) if risk_aux else '无'}")
    lines.append("")

    lines.append("[五] 说明")
    lines.append("1. 本次交叉合并仍然属于问题一前半段的‘候选变量整合’，不是最终论文结论。")
    lines.append("2. 若后续需要更严格的保留标准，可继续加入 Bootstrap 稳定性选择。")
    lines.append("3. 体质贡献分析尚未纳入本次结果。")

    SUMMARY_OUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def write_metadata(df_merge: pd.DataFrame, metrics: Dict[str, Optional[float]]) -> None:
    payload = {
        "phlegm_branch_weight": 0.35,
        "risk_branch_weight": 0.65,
        "phlegm_r2_mean": metrics.get("ph_r2_mean"),
        "risk_auc_mean": metrics.get("risk_auc_mean"),
        "n_indicators": int(len(df_merge)),
        "joint_score_formula": "0.35 * phlegm_branch_score + 0.65 * risk_branch_score",
        "layer_rules": {
            "高血脂风险核心指标": "高血脂平均排名 <= 2",
            "双分支交叉候选指标": "高血脂平均排名 <= 5 且 痰湿平均排名 <= 4",
            "痰湿辅助表征指标": "痰湿平均排名 <= 3 且未进入前两类",
            "高血脂辅助指标": "高血脂平均排名 <= 5 且未进入前几类",
        },
    }
    META_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    data = load_required_files()
    metrics = parse_summary_metrics(SUMMARY_TXT)

    df_merge = build_branch_merge_table(data)
    df_layer = build_layer_table(df_merge)

    df_merge.to_csv(MERGE_TABLE_CSV, index=False, encoding="utf-8-sig")
    df_layer.to_csv(LAYER_TABLE_CSV, index=False, encoding="utf-8-sig")

    plot_joint_score(df_merge)
    plot_rank_map(df_merge)

    write_summary(df_merge, df_layer, metrics)
    write_metadata(df_merge, metrics)

    print("双分支交叉合并完成。")
    print(f"合并总表：{MERGE_TABLE_CSV}")
    print(f"分层结果表：{LAYER_TABLE_CSV}")
    print(f"摘要说明：{SUMMARY_OUT_TXT}")
    print(f"图片目录：{MERGE_FIG_DIR}")


if __name__ == "__main__":
    main()
