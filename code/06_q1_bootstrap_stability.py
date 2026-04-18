from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from config import OUT_DIR, FIGURE_DIR


# =========================================================
# 0. 路径与常量
# =========================================================
INPUT_CSV = OUT_DIR / "c_q1_main_simple.csv"

OUT_SUBDIR = OUT_DIR / "q1_bootstrap_stability"
FIG_SUBDIR = FIGURE_DIR / "q1_bootstrap_stability"

OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
FIG_SUBDIR.mkdir(parents=True, exist_ok=True)

FEATURES: List[str] = [
    "TG（甘油三酯）",
    "TC（总胆固醇）",
    "LDL-C（低密度脂蛋白）",
    "HDL-C（高密度脂蛋白）",
    "空腹血糖",
    "血尿酸",
    "BMI",
    "活动量表总分（ADL总分+IADL总分）",
]

PHLEGM_MASK_COL = "体质标签"
PHLEGM_MASK_VALUE = 5
PHLEGM_TARGET = "痰湿质"
RISK_TARGET = "高血脂症二分类标签"

N_BOOT = 200
SELECTION_EPS = 1e-8
RANDOM_STATE = 42

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 1. 数据读取
# =========================================================
def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"未找到输入文件：{INPUT_CSV}\n"
            "请先确保已经成功生成 c_q1_main_simple.csv。"
        )

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    required_cols = FEATURES + [PHLEGM_MASK_COL, PHLEGM_TARGET, RISK_TARGET]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"输入主分析表缺少必要列：{missing_cols}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    na_summary = df[required_cols].isna().sum()
    na_summary = na_summary[na_summary > 0]
    if len(na_summary) > 0:
        raise ValueError(
            "关键字段在转换数值后出现缺失，请先检查输入数据。\n"
            + na_summary.to_string()
        )

    return df


# =========================================================
# 2. 超参数预选
# =========================================================
def fit_phlegm_cv(X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        cv=5,
        random_state=RANDOM_STATE,
        max_iter=20000,
    )
    model_cv.fit(X_scaled, y)
    return float(model_cv.alpha_), float(model_cv.l1_ratio_)


def fit_risk_cv(X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_cv = LogisticRegressionCV(
        Cs=np.logspace(-2, 2, 10),
        cv=5,
        penalty="elasticnet",
        solver="saga",
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=5000,
        l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        n_jobs=1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    model_cv.fit(X_scaled, y)

    chosen_c = float(model_cv.C_[0])
    # 兼容不同 sklearn 版本返回类型
    if np.ndim(model_cv.l1_ratio_) == 0:
        chosen_l1 = float(model_cv.l1_ratio_)
    else:
        chosen_l1 = float(np.ravel(model_cv.l1_ratio_)[0])

    return chosen_c, chosen_l1


# =========================================================
# 3. Bootstrap 核心函数
# =========================================================
def bootstrap_phlegm(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    l1_ratio: float,
    n_boot: int = N_BOOT,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X)
    coef_matrix = np.zeros((n_boot, X.shape[1]), dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_b = X.iloc[idx]
        y_b = y.iloc[idx]

        scaler = StandardScaler()
        X_b_scaled = scaler.fit_transform(X_b)

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=20000,
            random_state=RANDOM_STATE + b,
        )
        model.fit(X_b_scaled, y_b)
        coef_matrix[b, :] = model.coef_

        if (b + 1) % 50 == 0:
            print(f"[INFO] 痰湿分支 Bootstrap 进度：{b + 1}/{n_boot}")

    return summarize_bootstrap_coefficients(
        coef_matrix=coef_matrix,
        feature_names=X.columns.tolist(),
        coef_name="ElasticNet系数",
    )


def bootstrap_risk(
    X: pd.DataFrame,
    y: pd.Series,
    C: float,
    l1_ratio: float,
    n_boot: int = N_BOOT,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X)
    coef_matrix = np.zeros((n_boot, X.shape[1]), dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_b = X.iloc[idx]
        y_b = y.iloc[idx]

        scaler = StandardScaler()
        X_b_scaled = scaler.fit_transform(X_b)

        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            class_weight="balanced",
            max_iter=5000,
            n_jobs=1,
            random_state=RANDOM_STATE + b,
        )
        model.fit(X_b_scaled, y_b)
        coef_matrix[b, :] = model.coef_.ravel()

        if (b + 1) % 50 == 0:
            print(f"[INFO] 高血脂分支 Bootstrap 进度：{b + 1}/{n_boot}")

    result = summarize_bootstrap_coefficients(
        coef_matrix=coef_matrix,
        feature_names=X.columns.tolist(),
        coef_name="Logit系数",
    )
    result["OR均值"] = np.exp(result["系数均值"])
    result["OR中位数"] = np.exp(result["系数中位数"])
    return result


def summarize_bootstrap_coefficients(
    coef_matrix: np.ndarray,
    feature_names: List[str],
    coef_name: str,
) -> pd.DataFrame:
    selected = (np.abs(coef_matrix) > SELECTION_EPS).astype(int)
    pos = (coef_matrix > SELECTION_EPS).astype(int)
    neg = (coef_matrix < -SELECTION_EPS).astype(int)

    rows = []
    for j, name in enumerate(feature_names):
        coef_j = coef_matrix[:, j]
        sel_j = selected[:, j]
        pos_j = pos[:, j]
        neg_j = neg[:, j]

        sel_rate = sel_j.mean()
        pos_rate = pos_j.mean()
        neg_rate = neg_j.mean()

        if sel_rate > 0:
            sign_consistency = max(pos_rate, neg_rate) / sel_rate
        else:
            sign_consistency = 0.0

        rows.append(
            {
                "指标": name,
                "选择频率": sel_rate,
                "正向频率": pos_rate,
                "负向频率": neg_rate,
                "符号一致性": sign_consistency,
                "系数均值": coef_j.mean(),
                "系数标准差": coef_j.std(ddof=1),
                "系数中位数": np.median(coef_j),
                "2.5%分位数": np.quantile(coef_j, 0.025),
                "97.5%分位数": np.quantile(coef_j, 0.975),
            }
        )

    out = pd.DataFrame(rows)
    out["稳定性等级"] = out["选择频率"].apply(label_stability)
    out = out.sort_values(
        by=["选择频率", "符号一致性", "系数均值"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    out.insert(1, coef_name, out["系数均值"])
    return out


def label_stability(x: float) -> str:
    if x >= 0.80:
        return "高稳定"
    if x >= 0.50:
        return "中等稳定"
    if x >= 0.20:
        return "低稳定"
    return "不稳定"


# =========================================================
# 4. 出图与摘要
# =========================================================
def plot_selection_frequency(df: pd.DataFrame, title: str, save_path: Path) -> None:
    plot_df = df.sort_values(by="选择频率", ascending=True).copy()

    plt.figure(figsize=(10, 5), dpi=150)
    plt.barh(plot_df["指标"], plot_df["选择频率"])
    plt.xlim(0, 1.0)
    plt.xlabel("Bootstrap 选择频率")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_summary_text(
    df: pd.DataFrame,
    phlegm_result: pd.DataFrame,
    risk_result: pd.DataFrame,
    ph_alpha: float,
    ph_l1: float,
    risk_c: float,
    risk_l1: float,
) -> str:
    lines: List[str] = []
    lines.append("问题一：Bootstrap 稳定性选择摘要")
    lines.append("=" * 60)
    lines.append(f"输入主分析表: {INPUT_CSV}")
    lines.append(f"总样本量: {len(df)}")
    lines.append(f"痰湿体质子样本量(体质标签=5): {int((df[PHLEGM_MASK_COL] == PHLEGM_MASK_VALUE).sum())}")
    lines.append(f"候选指标数: {len(FEATURES)}")
    lines.append(f"Bootstrap 次数: {N_BOOT}")
    lines.append("")

    lines.append("[一] 痰湿分支：预选超参数")
    lines.append(f"ElasticNet alpha = {ph_alpha:.6f}")
    lines.append(f"ElasticNet l1_ratio = {ph_l1:.4f}")
    lines.append("")
    lines.append("[二] 痰湿分支：选择频率 Top 5")
    lines.append(
        phlegm_result[["指标", "选择频率", "符号一致性", "稳定性等级"]]
        .head(5)
        .to_string(index=False)
    )
    lines.append("")

    lines.append("[三] 高血脂分支：预选超参数")
    lines.append(f"Logistic C = {risk_c:.6f}")
    lines.append(f"Logistic l1_ratio = {risk_l1:.4f}")
    lines.append("")
    lines.append("[四] 高血脂分支：选择频率 Top 5")
    lines.append(
        risk_result[["指标", "选择频率", "符号一致性", "稳定性等级", "OR均值"]]
        .head(5)
        .to_string(index=False)
    )
    lines.append("")

    lines.append("[五] 说明")
    lines.append("1. Bootstrap 稳定性选择的目的是检验变量保留结果对抽样波动是否敏感，而不是替代前面的主模型结论。")
    lines.append("2. 痰湿分支若整体选择频率偏低，通常意味着当前候选变量对痰湿积分的稳定解释能力有限。")
    lines.append("3. 高血脂分支若 TG、TC 等变量选择频率长期居前，则说明其作为风险核心指标具有较强稳健性。")
    return "\n".join(lines)


# =========================================================
# 5. 主流程
# =========================================================
def main() -> None:
    print("[INFO] 正在读取主分析表...")
    df = load_data()

    # -------------------------
    # 痰湿分支
    # -------------------------
    df_ph = df[df[PHLEGM_MASK_COL] == PHLEGM_MASK_VALUE].copy()
    X_ph = df_ph[FEATURES]
    y_ph = df_ph[PHLEGM_TARGET]

    print("[INFO] 正在为痰湿分支预选 ElasticNet 超参数...")
    ph_alpha, ph_l1 = fit_phlegm_cv(X_ph, y_ph)

    print("[INFO] 正在执行痰湿分支 Bootstrap 稳定性选择...")
    ph_result = bootstrap_phlegm(X_ph, y_ph, ph_alpha, ph_l1, n_boot=N_BOOT)

    ph_csv = OUT_SUBDIR / "q1_phlegm_bootstrap_stability.csv"
    ph_result.to_csv(ph_csv, index=False, encoding="utf-8-sig")

    plot_selection_frequency(
        ph_result,
        title="痰湿分支 Bootstrap 选择频率",
        save_path=FIG_SUBDIR / "q1_phlegm_bootstrap_frequency.png",
    )

    # -------------------------
    # 高血脂分支
    # -------------------------
    X_risk = df[FEATURES]
    y_risk = df[RISK_TARGET].astype(int)

    print("[INFO] 正在为高血脂分支预选 Logistic 超参数...")
    risk_c, risk_l1 = fit_risk_cv(X_risk, y_risk)

    print("[INFO] 正在执行高血脂分支 Bootstrap 稳定性选择...")
    risk_result = bootstrap_risk(X_risk, y_risk, risk_c, risk_l1, n_boot=N_BOOT)

    risk_csv = OUT_SUBDIR / "q1_risk_bootstrap_stability.csv"
    risk_result.to_csv(risk_csv, index=False, encoding="utf-8-sig")

    plot_selection_frequency(
        risk_result,
        title="高血脂分支 Bootstrap 选择频率",
        save_path=FIG_SUBDIR / "q1_risk_bootstrap_frequency.png",
    )

    # -------------------------
    # 摘要与元数据
    # -------------------------
    summary_txt = OUT_SUBDIR / "q1_bootstrap_stability_summary.txt"
    summary_txt.write_text(
        build_summary_text(df, ph_result, risk_result, ph_alpha, ph_l1, risk_c, risk_l1),
        encoding="utf-8",
    )

    metadata = {
        "input_csv": str(INPUT_CSV),
        "features": FEATURES,
        "n_boot": N_BOOT,
        "selection_eps": SELECTION_EPS,
        "random_state": RANDOM_STATE,
        "phlegm_cv_alpha": ph_alpha,
        "phlegm_cv_l1_ratio": ph_l1,
        "risk_cv_C": risk_c,
        "risk_cv_l1_ratio": risk_l1,
    }
    (OUT_SUBDIR / "q1_bootstrap_stability_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[INFO] Bootstrap 稳定性选择完成。")
    print(f"[INFO] 输出目录: {OUT_SUBDIR}")
    print(f"[INFO] 图片目录: {FIG_SUBDIR}")


if __name__ == "__main__":
    main()
