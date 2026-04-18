from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 路径配置（优先复用 code/config.py）
# =========================
def resolve_paths():
    current_file = Path(__file__).resolve()
    code_dir = current_file.parent
    project_root = code_dir.parent

    out_dir_default = project_root / "out"
    fig_dir_default = project_root / "figure"

    try:
        import config  # type: ignore

        out_dir = getattr(config, "OUT_DIR", out_dir_default)
        fig_dir = getattr(config, "FIGURE_DIR", fig_dir_default)
        if not isinstance(out_dir, Path):
            out_dir = Path(out_dir)
        if not isinstance(fig_dir, Path):
            fig_dir = Path(fig_dir)
        return project_root, out_dir, fig_dir
    except Exception:
        return project_root, out_dir_default, fig_dir_default


PROJECT_ROOT, OUT_DIR, FIGURE_DIR = resolve_paths()

INPUT_CSV = OUT_DIR / "c_q1_main_simple.csv"
OUT_SUBDIR = OUT_DIR / "q1_constitution_contribution"
FIG_SUBDIR = FIGURE_DIR / "q1_constitution_contribution"

OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
FIG_SUBDIR.mkdir(parents=True, exist_ok=True)


# =========================
# 字段名配置
# =========================
TARGET_COL = "高血脂症二分类标签"
CONSTITUTION_LABEL_COL = "体质标签"
CONSTITUTION_NAME_COL = "体质名称"
CONTROL_COLS = ["年龄组", "性别", "吸烟史", "饮酒史"]

LABEL_MAP = {
    1: "平和质",
    2: "气虚质",
    3: "阳虚质",
    4: "阴虚质",
    5: "痰湿质",
    6: "湿热质",
    7: "血瘀质",
    8: "气郁质",
    9: "特禀质",
}


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = [TARGET_COL, CONSTITUTION_LABEL_COL] + CONTROL_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    if CONSTITUTION_NAME_COL not in df.columns:
        df[CONSTITUTION_NAME_COL] = df[CONSTITUTION_LABEL_COL].map(LABEL_MAP)

    allowed_labels = set(LABEL_MAP.keys())
    observed = set(df[CONSTITUTION_LABEL_COL].dropna().astype(int).unique().tolist())
    bad = sorted(observed - allowed_labels)
    if bad:
        raise ValueError(f"体质标签存在非法取值: {bad}")
    return df


def build_prevalence_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    overall_rate = float(df[TARGET_COL].mean())
    overall_n = int(len(df))

    ctab = pd.crosstab(df[CONSTITUTION_NAME_COL], df[TARGET_COL])
    chi2, p_value, dof, expected = chi2_contingency(ctab)

    total_cases = int(df[TARGET_COL].sum())

    rows = []
    for label in range(1, 10):
        name = LABEL_MAP[label]
        sub = df[df[CONSTITUTION_LABEL_COL] == label]
        n = int(len(sub))
        cases = int(sub[TARGET_COL].sum())
        non_cases = n - cases
        rate = cases / n if n > 0 else np.nan
        rr_vs_overall = (rate / overall_rate) if overall_rate > 0 and n > 0 else np.nan
        cd = (cases / total_cases) if total_cases > 0 else np.nan
        rows.append(
            {
                "体质标签": label,
                "体质名称": name,
                "样本数": n,
                "确诊数": cases,
                "未确诊数": non_cases,
                "发病率": rate,
                "相对风险RR_相对总体": rr_vs_overall,
                "绝对贡献度CD": cd,
            }
        )

    prevalence_df = pd.DataFrame(rows)
    prevalence_df["发病率排序"] = prevalence_df["发病率"].rank(ascending=False, method="min")
    prevalence_df["绝对贡献度排序"] = prevalence_df["绝对贡献度CD"].rank(ascending=False, method="min")

    chi2_info = {
        "列联表卡方统计量": float(chi2),
        "p_value": float(p_value),
        "自由度": int(dof),
        "总体样本量": overall_n,
        "总体发病率": overall_rate,
    }
    return prevalence_df, chi2_info


def fit_adjusted_logit(df: pd.DataFrame) -> pd.DataFrame:
    X = df[[CONSTITUTION_LABEL_COL] + CONTROL_COLS].copy()
    X[CONSTITUTION_LABEL_COL] = X[CONSTITUTION_LABEL_COL].astype("category")

    const_dummies = pd.get_dummies(X[CONSTITUTION_LABEL_COL], prefix="体质", drop_first=True)
    controls = X[CONTROL_COLS].copy()
    design = pd.concat([const_dummies, controls], axis=1)
    design = sm.add_constant(design, has_constant="add")
    design = design.astype(float)

    y = df[TARGET_COL].astype(int)

    model = sm.Logit(y, design)
    result = model.fit(disp=False, maxiter=500)

    conf = result.conf_int()
    conf.columns = ["CI_lower", "CI_upper"]
    coef = result.params
    pvals = result.pvalues

    rows = []
    for label in range(2, 10):
        col = f"体质_{label}"
        if col not in coef.index:
            continue
        beta = float(coef[col])
        or_val = float(np.exp(beta))
        ci_low = float(np.exp(conf.loc[col, "CI_lower"]))
        ci_high = float(np.exp(conf.loc[col, "CI_upper"]))
        rows.append(
            {
                "体质标签": label,
                "体质名称": LABEL_MAP[label],
                "回归系数beta": beta,
                "OR": or_val,
                "OR_CI_lower": ci_low,
                "OR_CI_upper": ci_high,
                "p_value": float(pvals[col]),
                "相对基准组": "平和质",
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["OR显著性"] = np.where(
            (out["OR_CI_lower"] > 1) | (out["OR_CI_upper"] < 1),
            "是",
            "否",
        )
        out = out.sort_values("OR", ascending=False).reset_index(drop=True)
    return out


def plot_prevalence(prevalence_df: pd.DataFrame) -> None:
    plot_df = prevalence_df.copy().sort_values("发病率", ascending=False)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.bar(plot_df["体质名称"], plot_df["发病率"])
    plt.title("九种体质对应的高血脂发病率")
    plt.xlabel("体质类型")
    plt.ylabel("发病率")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_SUBDIR / "q1_constitution_prevalence.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6), dpi=150)
    plt.bar(plot_df["体质名称"], plot_df["绝对贡献度CD"])
    plt.title("九种体质对总体高血脂病例的绝对贡献度")
    plt.xlabel("体质类型")
    plt.ylabel("绝对贡献度 CD")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_SUBDIR / "q1_constitution_contribution_cd.png", bbox_inches="tight")
    plt.close()


def plot_or_forest(or_df: pd.DataFrame) -> None:
    if or_df.empty:
        return

    plot_df = or_df.sort_values("OR", ascending=True).reset_index(drop=True)

    y_pos = np.arange(len(plot_df))
    err_left = plot_df["OR"] - plot_df["OR_CI_lower"]
    err_right = plot_df["OR_CI_upper"] - plot_df["OR"]

    plt.figure(figsize=(9, 6), dpi=150)
    plt.errorbar(
        plot_df["OR"],
        y_pos,
        xerr=[err_left, err_right],
        fmt="o",
        capsize=4,
    )
    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.yticks(y_pos, plot_df["体质名称"])
    plt.xlabel("OR（相对平和质）")
    plt.title("九种体质对高血脂风险的调整后 OR 森林图")
    plt.tight_layout()
    plt.savefig(FIG_SUBDIR / "q1_constitution_adjusted_or_forest.png", bbox_inches="tight")
    plt.close()


def write_summary(prevalence_df: pd.DataFrame, chi2_info: dict, or_df: pd.DataFrame) -> None:
    lines = []
    lines.append("问题一：九种体质贡献度分析摘要")
    lines.append("=" * 60)
    lines.append(f"输入主分析表: {INPUT_CSV}")
    lines.append(f"总样本量: {chi2_info['总体样本量']}")
    lines.append(f"总体发病率: {chi2_info['总体发病率']:.4f}")
    lines.append("")
    lines.append("[一] 整体体质差异的列联表卡方检验")
    lines.append(f"卡方统计量 = {chi2_info['列联表卡方统计量']:.4f}")
    lines.append(f"p_value = {chi2_info['p_value']:.6f}")
    lines.append(f"自由度 = {chi2_info['自由度']}")
    lines.append("")

    lines.append("[二] 原始发病率 Top 5")
    top_rate = prevalence_df.sort_values("发病率", ascending=False).head(5)
    lines.append(top_rate[["体质名称", "样本数", "确诊数", "发病率", "相对风险RR_相对总体"]].to_string(index=False))
    lines.append("")

    lines.append("[三] 绝对贡献度 CD Top 5")
    top_cd = prevalence_df.sort_values("绝对贡献度CD", ascending=False).head(5)
    lines.append(top_cd[["体质名称", "样本数", "确诊数", "绝对贡献度CD"]].to_string(index=False))
    lines.append("")

    lines.append("[四] 调整后 OR（相对平和质）Top 5")
    if or_df.empty:
        lines.append("未成功拟合 Logistic 模型。")
    else:
        lines.append(or_df.head(5)[["体质名称", "OR", "OR_CI_lower", "OR_CI_upper", "p_value", "OR显著性"]].to_string(index=False))
    lines.append("")
    lines.append("[五] 说明")
    lines.append("1. 本阶段仅完成九种体质对高血脂风险的贡献差异分析，不包含 Bootstrap 稳定性选择。")
    lines.append("2. 原始发病率、相对风险 RR、绝对贡献度 CD 与调整后 OR 分别从不同角度解释体质贡献差异。")
    lines.append("3. 调整后 OR 以平和质为基准组，并控制年龄组、性别、吸烟史、饮酒史。")

    (OUT_SUBDIR / "q1_constitution_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"未找到输入文件: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df = validate_columns(df)

    prevalence_df, chi2_info = build_prevalence_table(df)
    prevalence_df.to_csv(OUT_SUBDIR / "q1_constitution_prevalence_rr_cd.csv", index=False, encoding="utf-8-sig")

    chi2_df = pd.DataFrame([chi2_info])
    chi2_df.to_csv(OUT_SUBDIR / "q1_constitution_chi2_overall.csv", index=False, encoding="utf-8-sig")

    or_df = fit_adjusted_logit(df)
    or_df.to_csv(OUT_SUBDIR / "q1_constitution_adjusted_or.csv", index=False, encoding="utf-8-sig")

    plot_prevalence(prevalence_df)
    plot_or_forest(or_df)

    metadata = {
        "input_csv": str(INPUT_CSV),
        "out_dir": str(OUT_SUBDIR),
        "fig_dir": str(FIG_SUBDIR),
        "baseline_group": "平和质",
        "controls": CONTROL_COLS,
        "constitution_name_map": LABEL_MAP,
    }
    (OUT_SUBDIR / "q1_constitution_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_summary(prevalence_df, chi2_info, or_df)

    print("九种体质贡献度分析完成。")
    print(f"结果输出目录: {OUT_SUBDIR}")
    print(f"图片输出目录: {FIG_SUBDIR}")


if __name__ == "__main__":
    main()
