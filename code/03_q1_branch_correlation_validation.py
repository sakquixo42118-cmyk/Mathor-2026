from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu, pointbiserialr, spearmanr
from sklearn.metrics import roc_auc_score

from config import OUT_DIR, FIGURE_DIR, ensure_project_dirs


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False



# =========================
# 路径配置
# =========================
INPUT_PATH = OUT_DIR / "c_q1_main_simple.csv"
RESULT_DIR = OUT_DIR / "q1_branch_validation"
FIG_DIR = FIGURE_DIR / "q1_branch_validation"

# =========================
# 分析配置
# =========================
TARGET_PHLEGM = "痰湿质"
TARGET_RISK = "高血脂症二分类标签"
CONSTITUTION_LABEL_COL = "体质标签"
PHLEGM_LABEL_VALUE = 5
ACTIVITY_TOTAL_COL = "活动量表总分（ADL总分+IADL总分）"

# 为了和后续主模型保持一致，这里默认只保留“活动总分”作为活动能力主变量，
# 不同时放入 ADL总分 / IADL总分 / 活动总分 三者。
BASE_FEATURES: List[str] = [
    "TG（甘油三酯）",
    "TC（总胆固醇）",
    "LDL-C（低密度脂蛋白）",
    "HDL-C（高密度脂蛋白）",
    "空腹血糖",
    "血尿酸",
    "BMI",
    ACTIVITY_TOTAL_COL,
]

# 如需把工程变量一起纳入单变量验证，可改为 True。
INCLUDE_ENGINEERED_FEATURES = False
ENGINEERED_FEATURES: List[str] = [
    "血脂异常项数",
    "代谢异常项数",
]

# 对“高血脂风险”来说，下列变量是“越大风险越高”；
# 其余变量默认按“越小风险越高”处理（用于单变量 AUC 方向统一）。
RISK_HIGHER_IS_WORSE = {
    "TG（甘油三酯）",
    "TC（总胆固醇）",
    "LDL-C（低密度脂蛋白）",
    "空腹血糖",
    "血尿酸",
    "BMI",
    "血脂异常项数",
    "代谢异常项数",
}

PLOT_TOP_N = 8
DPI = 160


# =========================
# 工具函数
# =========================
def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取文件: {path}\n最后一次报错: {last_error}")


def ensure_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")


def bh_fdr(pvalues: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR 校正，不依赖 statsmodels。"""
    n = len(pvalues)
    if n == 0:
        return []

    indexed = sorted(enumerate(pvalues), key=lambda x: (math.inf if pd.isna(x[1]) else x[1]))
    adjusted = [np.nan] * n
    prev = 1.0

    for rank_rev, (idx, p) in enumerate(reversed(indexed), start=1):
        rank = n - rank_rev + 1
        if pd.isna(p):
            adj = np.nan
        else:
            adj = min(prev, p * n / rank)
            prev = adj
        adjusted[idx] = adj
    return adjusted


def safe_spearman(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    valid = pd.concat([x, y], axis=1).dropna()
    if valid.shape[0] < 3 or valid.iloc[:, 0].nunique() <= 1 or valid.iloc[:, 1].nunique() <= 1:
        return {"rho": np.nan, "p_value": np.nan}
    rho, p = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    return {"rho": float(rho), "p_value": float(p)}


def safe_pointbiserial(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    valid = pd.concat([x, y], axis=1).dropna()
    if valid.shape[0] < 3 or valid.iloc[:, 0].nunique() <= 1 or valid.iloc[:, 1].nunique() <= 1:
        return {"r_pb": np.nan, "p_value": np.nan}
    r_pb, p = pointbiserialr(valid.iloc[:, 1], valid.iloc[:, 0])
    return {"r_pb": float(r_pb), "p_value": float(p)}


def safe_mannwhitney(x: pd.Series, y_binary: pd.Series) -> Dict[str, float]:
    valid = pd.concat([x, y_binary], axis=1).dropna()
    if valid.shape[0] < 3:
        return {"u_stat": np.nan, "p_value": np.nan}
    g0 = valid.loc[valid.iloc[:, 1] == 0, valid.columns[0]]
    g1 = valid.loc[valid.iloc[:, 1] == 1, valid.columns[0]]
    if len(g0) == 0 or len(g1) == 0:
        return {"u_stat": np.nan, "p_value": np.nan}
    u, p = mannwhitneyu(g0, g1, alternative="two-sided")
    return {"u_stat": float(u), "p_value": float(p)}


def safe_auc(x: pd.Series, y_binary: pd.Series, feature_name: str) -> float:
    valid = pd.concat([x, y_binary], axis=1).dropna()
    if valid.shape[0] < 3 or valid.iloc[:, 0].nunique() <= 1 or valid.iloc[:, 1].nunique() <= 1:
        return np.nan

    score = valid.iloc[:, 0].copy()
    if feature_name not in RISK_HIGHER_IS_WORSE:
        score = -score

    try:
        return float(roc_auc_score(valid.iloc[:, 1], score))
    except Exception:
        return np.nan


def make_quartile_labels(series: pd.Series) -> pd.Series:
    return pd.qcut(series, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")


def plot_barh(df_plot: pd.DataFrame, value_col: str, label_col: str, title: str, out_path: Path) -> None:
    if df_plot.empty:
        return

    plt.figure(figsize=(8, 5), dpi=DPI)
    ordered = df_plot.sort_values(value_col, ascending=True)
    plt.barh(ordered[label_col], ordered[value_col])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# =========================
# 痰湿分支
# =========================
def run_phlegm_branch(df: pd.DataFrame, features: List[str]) -> Dict[str, pd.DataFrame]:
    phlegm_df = df[df[CONSTITUTION_LABEL_COL] == PHLEGM_LABEL_VALUE].copy()
    if phlegm_df.empty:
        raise ValueError("痰湿分支没有找到体质标签=5的样本。")

    # 1) Spearman 相关
    spearman_rows = []
    for feat in features:
        res = safe_spearman(phlegm_df[feat], phlegm_df[TARGET_PHLEGM])
        spearman_rows.append({
            "指标": feat,
            "rho": res["rho"],
            "|rho|": np.abs(res["rho"]),
            "p_value": res["p_value"],
        })
    spearman_df = pd.DataFrame(spearman_rows)
    spearman_df["p_fdr_bh"] = bh_fdr(spearman_df["p_value"].tolist())
    spearman_df = spearman_df.sort_values(["|rho|", "p_value"], ascending=[False, True]).reset_index(drop=True)

    # 2) 痰湿积分四分位验证：Kruskal-Wallis
    quartile_df = phlegm_df[[TARGET_PHLEGM] + features].copy()
    quartile_df["痰湿四分位组"] = make_quartile_labels(quartile_df[TARGET_PHLEGM])

    kruskal_rows = []
    mean_rows = []
    for feat in features:
        sub = quartile_df[[feat, "痰湿四分位组"]].dropna()
        groups = [g[feat].values for _, g in sub.groupby("痰湿四分位组", observed=False)]

        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            h_stat, p_val = kruskal(*groups)
        else:
            h_stat, p_val = np.nan, np.nan

        kruskal_rows.append({
            "指标": feat,
            "H_stat": float(h_stat) if not pd.isna(h_stat) else np.nan,
            "p_value": float(p_val) if not pd.isna(p_val) else np.nan,
        })

        grp_mean = sub.groupby("痰湿四分位组", observed=False)[feat].mean().to_dict()
        row_mean = {"指标": feat}
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            row_mean[q] = grp_mean.get(q, np.nan)
        mean_rows.append(row_mean)

    kruskal_df = pd.DataFrame(kruskal_rows)
    kruskal_df["p_fdr_bh"] = bh_fdr(kruskal_df["p_value"].tolist())
    kruskal_df = kruskal_df.sort_values(["p_value", "H_stat"], ascending=[True, False]).reset_index(drop=True)

    quartile_mean_df = pd.DataFrame(mean_rows)

    return {
        "phlegm_subset": phlegm_df,
        "spearman": spearman_df,
        "kruskal": kruskal_df,
        "quartile_mean": quartile_mean_df,
    }


# =========================
# 高血脂分支
# =========================
def run_risk_branch(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        x = df[feat]
        y = df[TARGET_RISK]

        pb = safe_pointbiserial(x, y)
        mw = safe_mannwhitney(x, y)
        auc = safe_auc(x, y, feat)

        mean_neg = pd.concat([x, y], axis=1).dropna().loc[y == 0, feat].mean()
        mean_pos = pd.concat([x, y], axis=1).dropna().loc[y == 1, feat].mean()

        rows.append({
            "指标": feat,
            "未确诊组均值": mean_neg,
            "确诊组均值": mean_pos,
            "组间差值(确诊-未确诊)": mean_pos - mean_neg,
            "point_biserial_r": pb["r_pb"],
            "|point_biserial_r|": np.abs(pb["r_pb"]),
            "pb_p_value": pb["p_value"],
            "u_stat": mw["u_stat"],
            "mw_p_value": mw["p_value"],
            "单变量AUC": auc,
            "AUC偏离0.5程度": np.abs(auc - 0.5) if not pd.isna(auc) else np.nan,
        })

    risk_df = pd.DataFrame(rows)
    risk_df["pb_p_fdr_bh"] = bh_fdr(risk_df["pb_p_value"].tolist())
    risk_df["mw_p_fdr_bh"] = bh_fdr(risk_df["mw_p_value"].tolist())
    risk_df = risk_df.sort_values(
        ["AUC偏离0.5程度", "|point_biserial_r|", "mw_p_value"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return risk_df


# =========================
# 摘要输出
# =========================
def build_summary_text(
    df: pd.DataFrame,
    features: List[str],
    phlegm_results: Dict[str, pd.DataFrame],
    risk_df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    phlegm_n = len(phlegm_results["phlegm_subset"])
    total_n = len(df)

    lines.append("问题一：痰湿分支 + 高血脂分支 单变量验证摘要")
    lines.append("=" * 52)
    lines.append(f"输入主分析表: {INPUT_PATH}")
    lines.append(f"总样本量: {total_n}")
    lines.append(f"痰湿体质子样本量(体质标签=5): {phlegm_n}")
    lines.append(f"纳入验证的候选指标数: {len(features)}")
    lines.append("")

    lines.append("[一] 痰湿分支：Spearman 相关 Top 5")
    lines.append(phlegm_results["spearman"].head(5).round(4).to_string(index=False))
    lines.append("")

    lines.append("[二] 痰湿分支：Kruskal-Wallis 显著性 Top 5")
    lines.append(phlegm_results["kruskal"].head(5).round(4).to_string(index=False))
    lines.append("")

    lines.append("[三] 高血脂分支：单变量 AUC Top 5")
    auc_top = risk_df.sort_values("AUC偏离0.5程度", ascending=False).head(5)
    lines.append(auc_top[["指标", "单变量AUC", "AUC偏离0.5程度"]].round(4).to_string(index=False))
    lines.append("")

    lines.append("[四] 高血脂分支：点二列相关 |r| Top 5")
    pb_top = risk_df.sort_values("|point_biserial_r|", ascending=False).head(5)
    lines.append(pb_top[["指标", "point_biserial_r", "|point_biserial_r|", "pb_p_value"]].round(4).to_string(index=False))
    lines.append("")

    lines.append("[五] 说明")
    lines.append("1. 痰湿分支仅在体质标签=5的子样本内部进行，以避免全样本结构分离带来的解释偏差。")
    lines.append("2. 高血脂分支中的单变量 AUC 与相关性结果，仅用于‘单变量验证’，不代表最终多变量模型中的稳定重要性。")
    lines.append("3. 当前脚本只完成前期统计验证，不包含交叉合并、Bootstrap 稳定性选择与九种体质贡献分析。")
    return "\n".join(lines)


# =========================
# 主程序
# =========================
def main() -> None:
    ensure_project_dirs()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"未找到主分析表: {INPUT_PATH}\n"
            "请先运行 02_build_q1_main_table_simple.py 生成 c_q1_main_simple.csv"
        )

    print(f"[INFO] 读取主分析表: {INPUT_PATH}")
    df = load_csv_with_fallback(INPUT_PATH)

    features = BASE_FEATURES.copy()
    if INCLUDE_ENGINEERED_FEATURES:
        features += ENGINEERED_FEATURES

    required_cols = [TARGET_PHLEGM, TARGET_RISK, CONSTITUTION_LABEL_COL] + features
    ensure_columns(df, required_cols)

    # 数值化保护
    for col in [TARGET_PHLEGM, TARGET_RISK, CONSTITUTION_LABEL_COL] + features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 仅删除本脚本相关字段中的缺失
    df = df.dropna(subset=[TARGET_PHLEGM, TARGET_RISK, CONSTITUTION_LABEL_COL] + features).copy()

    phlegm_results = run_phlegm_branch(df, features)
    risk_df = run_risk_branch(df, features)

    # 输出表格
    phlegm_results["spearman"].to_csv(RESULT_DIR / "q1_phlegm_branch_spearman.csv", index=False, encoding="utf-8-sig")
    phlegm_results["kruskal"].to_csv(RESULT_DIR / "q1_phlegm_branch_kruskal.csv", index=False, encoding="utf-8-sig")
    phlegm_results["quartile_mean"].to_csv(RESULT_DIR / "q1_phlegm_branch_quartile_means.csv", index=False, encoding="utf-8-sig")
    risk_df.to_csv(RESULT_DIR / "q1_risk_branch_single_factor_validation.csv", index=False, encoding="utf-8-sig")

    # 输出图形
    phlegm_plot_df = phlegm_results["spearman"].head(PLOT_TOP_N)[["指标", "|rho|"]].copy()
    plot_barh(
        phlegm_plot_df,
        value_col="|rho|",
        label_col="指标",
        title="痰湿分支：Spearman |rho| Top 指标",
        out_path=FIG_DIR / "q1_phlegm_branch_spearman_top.png",
    )

    risk_auc_plot = risk_df.sort_values("AUC偏离0.5程度", ascending=False).head(PLOT_TOP_N)[["指标", "单变量AUC"]].copy()
    plot_barh(
        risk_auc_plot,
        value_col="单变量AUC",
        label_col="指标",
        title="高血脂分支：单变量 AUC Top 指标",
        out_path=FIG_DIR / "q1_risk_branch_auc_top.png",
    )

    risk_pb_plot = risk_df.sort_values("|point_biserial_r|", ascending=False).head(PLOT_TOP_N)[["指标", "|point_biserial_r|"]].copy()
    plot_barh(
        risk_pb_plot,
        value_col="|point_biserial_r|",
        label_col="指标",
        title="高血脂分支：点二列相关 |r| Top 指标",
        out_path=FIG_DIR / "q1_risk_branch_pointbiserial_top.png",
    )

    summary_text = build_summary_text(df, features, phlegm_results, risk_df)
    summary_path = RESULT_DIR / "q1_branch_validation_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    print("\n[INFO] 已完成问题一前期单变量验证：")
    print(f"- 痰湿分支 Spearman：{RESULT_DIR / 'q1_phlegm_branch_spearman.csv'}")
    print(f"- 痰湿分支 Kruskal：{RESULT_DIR / 'q1_phlegm_branch_kruskal.csv'}")
    print(f"- 痰湿四分位均值表：{RESULT_DIR / 'q1_phlegm_branch_quartile_means.csv'}")
    print(f"- 高血脂分支单变量验证：{RESULT_DIR / 'q1_risk_branch_single_factor_validation.csv'}")
    print(f"- 摘要：{summary_path}")
    print(f"- 图形输出目录：{FIG_DIR}")


if __name__ == "__main__":
    main()
