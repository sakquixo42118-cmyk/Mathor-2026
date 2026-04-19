
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


def get_chinese_font() -> FontProperties:
    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for fp in font_candidates:
        if Path(fp).exists():
            return FontProperties(fname=fp)
    return FontProperties()


ZH_FONT = get_chinese_font()
plt.rcParams["axes.unicode_minus"] = False


def apply_zh_font(ax, title: Optional[str] = None,
                  xlabel: Optional[str] = None,
                  ylabel: Optional[str] = None):
    if title is not None:
        ax.set_title(title, fontproperties=ZH_FONT)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontproperties=ZH_FONT)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontproperties=ZH_FONT)
    for label in ax.get_xticklabels():
        label.set_fontproperties(ZH_FONT)
    for label in ax.get_yticklabels():
        label.set_fontproperties(ZH_FONT)
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontproperties(ZH_FONT)
        title_obj = leg.get_title()
        if title_obj is not None:
            title_obj.set_fontproperties(ZH_FONT)


def _normalize_col_name(name: str) -> str:
    s = str(name).strip()
    s = s.replace(" ", "").replace("\u3000", "")
    return s


def _strip_bracket_suffix(name: str) -> str:
    s = _normalize_col_name(name)
    s = re.sub(r"[（(].*?[）)]", "", s)
    return s


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    columns = list(df.columns)
    exact_map = {_normalize_col_name(c): c for c in columns}
    base_map: Dict[str, List[str]] = {}
    for c in columns:
        base = _strip_bracket_suffix(c)
        base_map.setdefault(base, []).append(c)

    for cand in candidates:
        key = _normalize_col_name(cand)
        if key in exact_map:
            return exact_map[key]

    for cand in candidates:
        key = _strip_bracket_suffix(cand)
        matched = base_map.get(key, [])
        if len(matched) == 1:
            return matched[0]

    for cand in candidates:
        key = _strip_bracket_suffix(cand)
        fuzzy = [c for c in columns if key and (key in _normalize_col_name(c) or key in _strip_bracket_suffix(c))]
        if len(fuzzy) == 1:
            return fuzzy[0]

    raise KeyError(f"未找到字段，候选为：{candidates}。现有字段：{list(df.columns)}")


def project_root_from_script() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, here.parent.parent, Path.cwd()]:
        if (p / "out").exists() and (p / "code").exists():
            return p
    return here.parent.parent


PROJECT_ROOT = project_root_from_script()
OUT_DIR = PROJECT_ROOT / "out"
FIGURE_DIR = PROJECT_ROOT / "figure"

OUT_SUBDIR = OUT_DIR / "q3_model_comparison"
FIG_SUBDIR = FIGURE_DIR / "q3_model_comparison"
OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
FIG_SUBDIR.mkdir(parents=True, exist_ok=True)


MODEL_META = {
    "strict_main": {
        "label": "严格主模型",
        "summary_file": OUT_DIR / "q3_intervention_optimization" / "q3_patient_optimal_summary.csv",
        "feature_file": OUT_DIR / "q3_intervention_optimization" / "q3_feature_rule_summary.csv",
        "monthly_file": OUT_DIR / "q3_intervention_optimization" / "q3_patient_monthly_plan.csv",
        "pareto_file": OUT_DIR / "q3_intervention_optimization" / "q3_sample_pareto_points.csv",
        "kind": "strict",
    },
    "realistic": {
        "label": "现实优化版",
        "summary_file": OUT_DIR / "q3_intervention_optimization_realistic" / "q3_patient_optimal_summary_realistic.csv",
        "feature_file": OUT_DIR / "q3_intervention_optimization_realistic" / "q3_feature_rule_summary_realistic.csv",
        "monthly_file": OUT_DIR / "q3_intervention_optimization_realistic" / "q3_patient_monthly_plan_realistic.csv",
        "pareto_file": OUT_DIR / "q3_intervention_optimization_realistic" / "q3_sample_pareto_points_realistic.csv",
        "kind": "realistic",
    },
    "aux_validation": {
        "label": "帕累托辅助验证",
        "summary_file": OUT_DIR / "q3_aux_validation_pareto" / "q3_patient_pareto_summary_validation.csv",
        "feature_file": OUT_DIR / "q3_aux_validation_pareto" / "q3_feature_rule_summary_validation.csv",
        "monthly_file": OUT_DIR / "q3_aux_validation_pareto" / "q3_patient_pareto_representative_plans_validation.csv",
        "pareto_file": OUT_DIR / "q3_aux_validation_pareto" / "q3_patient_pareto_points_validation.csv",
        "kind": "aux",
    },
}


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def normalize_strict_or_realistic_summary(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    # 兼容两种schema：
    # A. 中文摘要列（主模型）
    # B. 英文摘要列（现实优化版）
    col_id = find_column(df, ["样本ID", "sample_id"])
    col_init = find_column(df, ["初始痰湿积分", "initial_score"])

    if model_key == "realistic":
        col_final = find_column(df, ["realistic_final_score", "6个月末痰湿积分", "期末痰湿积分"])
        col_cost = find_column(df, ["realistic_total_cost", "6个月总成本", "总成本"])
        col_age = find_column(df, ["age_band_name", "年龄分层", "年龄组名称", "年龄组"])
        col_act_bin = find_column(df, ["activity_band_name", "活动能力分层"])
        col_score_bin = find_column(df, ["initial_score_band_name", "初始痰湿积分分层"])
        col_first_level = find_column(df, ["realistic_first_month_intensity", "首月活动强度", "首月强度"])
        col_first_freq = find_column(df, ["realistic_first_month_freq", "首月每周频次", "首月频率"])
    else:
        col_final = find_column(df, ["6个月末痰湿积分", "期末痰湿积分", "strict_final_score"])
        col_cost = find_column(df, ["6个月总成本", "总成本", "strict_total_cost"])
        col_age = find_column(df, ["年龄分层", "年龄组名称", "年龄组", "age_band_name"])
        col_act_bin = find_column(df, ["活动能力分层", "activity_band_name"])
        col_score_bin = find_column(df, ["初始痰湿积分分层", "initial_score_band_name"])
        col_first_level = find_column(df, ["首月活动强度", "首月强度", "strict_first_month_intensity"])
        col_first_freq = find_column(df, ["首月每周频次", "首月频率", "strict_first_month_freq"])

    res = pd.DataFrame({
        "样本ID": df[col_id],
        "模型": MODEL_META[model_key]["label"],
        "方案类型": "最优方案",
        "初始痰湿积分": pd.to_numeric(df[col_init], errors="coerce"),
        "期末痰湿积分": pd.to_numeric(df[col_final], errors="coerce"),
        "6个月总成本": pd.to_numeric(df[col_cost], errors="coerce"),
        "年龄分层": df[col_age].astype(str),
        "活动能力分层": df[col_act_bin].astype(str),
        "初始痰湿积分分层": df[col_score_bin].astype(str),
        "首月强度": pd.to_numeric(df[col_first_level], errors="coerce"),
        "首月频率": pd.to_numeric(df[col_first_freq], errors="coerce"),
    })
    return res


def normalize_aux_summary(df: pd.DataFrame) -> pd.DataFrame:
    col_id = find_column(df, ["样本ID"])
    col_init = find_column(df, ["初始痰湿积分"])
    col_age = find_column(df, ["年龄组名称", "年龄分层", "年龄组"])
    col_act_total = find_column(df, ["活动量表总分", "活动总分", "活动量表总分（ADL总分+IADL总分）"])

    def act_bin(x: float) -> str:
        if x < 40:
            return "<40"
        if x < 60:
            return "40-59"
        return ">=60"

    def score_bin(x: float) -> str:
        if x <= 58:
            return "55-58"
        if x <= 61:
            return "59-61"
        return "62-65"

    plan_specs = [
        ("最低成本", "最低成本_期末痰湿积分", "最低成本_总成本", "最低成本_总频次"),
        ("均衡方案", "均衡方案_期末痰湿积分", "均衡方案_总成本", "均衡方案_总频次"),
        ("疗效最优", "疗效最优_期末痰湿积分", "疗效最优_总成本", "疗效最优_总频次"),
    ]

    rows = []
    for _, row in df.iterrows():
        pid = row[col_id]
        init_score = pd.to_numeric(row[col_init], errors="coerce")
        age = str(row[col_age])
        act_total = pd.to_numeric(row[col_act_total], errors="coerce")
        for ptype, c_final, c_cost, c_freq in plan_specs:
            if c_final in df.columns and c_cost in df.columns:
                rows.append({
                    "样本ID": pid,
                    "模型": MODEL_META["aux_validation"]["label"],
                    "方案类型": ptype,
                    "初始痰湿积分": init_score,
                    "期末痰湿积分": pd.to_numeric(row[c_final], errors="coerce"),
                    "6个月总成本": pd.to_numeric(row[c_cost], errors="coerce"),
                    "年龄分层": age,
                    "活动能力分层": act_bin(float(act_total)),
                    "初始痰湿积分分层": score_bin(float(init_score)),
                    "首月强度": np.nan,
                    "首月频率": pd.to_numeric(row.get(c_freq, np.nan), errors="coerce"),
                })
    return pd.DataFrame(rows)


def load_all_summaries() -> Tuple[pd.DataFrame, List[str]]:
    frames = []
    missing = []

    for key, meta in MODEL_META.items():
        path = meta["summary_file"]
        if not file_exists(path):
            missing.append(str(path))
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        if meta["kind"] in ("strict", "realistic"):
            frames.append(normalize_strict_or_realistic_summary(df, key))
        else:
            frames.append(normalize_aux_summary(df))

    if len(frames) == 0:
        raise FileNotFoundError("三套模型的摘要文件都未找到，无法比较。")

    merged = pd.concat(frames, ignore_index=True)
    return merged, missing


def load_feature_rules() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    out = {}
    missing = []
    for key, meta in MODEL_META.items():
        path = meta["feature_file"]
        if not file_exists(path):
            missing.append(str(path))
            continue
        out[key] = pd.read_csv(path, encoding="utf-8-sig")
    return out, missing


def load_monthly_plans() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    out = {}
    missing = []
    for key, meta in MODEL_META.items():
        path = meta["monthly_file"]
        if not file_exists(path):
            missing.append(str(path))
            continue
        out[key] = pd.read_csv(path, encoding="utf-8-sig")
    return out, missing


def load_pareto_points() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    out = {}
    missing = []
    for key, meta in MODEL_META.items():
        path = meta["pareto_file"]
        if not file_exists(path):
            missing.append(str(path))
            continue
        out[key] = pd.read_csv(path, encoding="utf-8-sig")
    return out, missing


def build_overall_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    res = (
        summary_df.groupby(["模型", "方案类型"], dropna=False)
        .agg(
            患者数=("样本ID", "nunique"),
            平均初始痰湿积分=("初始痰湿积分", "mean"),
            平均期末痰湿积分=("期末痰湿积分", "mean"),
            平均6个月总成本=("6个月总成本", "mean"),
            平均首月强度=("首月强度", "mean"),
            平均首月频率=("首月频率", "mean"),
        )
        .reset_index()
    )
    res["平均积分降幅"] = res["平均初始痰湿积分"] - res["平均期末痰湿积分"]
    res["平均积分降幅(%)"] = 100 * res["平均积分降幅"] / res["平均初始痰湿积分"]
    return res


def build_sample_table(summary_df: pd.DataFrame, sample_ids: List[int]) -> pd.DataFrame:
    sub = summary_df[summary_df["样本ID"].isin(sample_ids)].copy()
    sub = sub.sort_values(["样本ID", "模型", "方案类型"]).reset_index(drop=True)
    return sub


def plot_group_heatmaps(summary_df: pd.DataFrame, plan_type_for_aux: str = "均衡方案"):
    plot_df = summary_df.copy()
    mask_aux = (plot_df["模型"] == MODEL_META["aux_validation"]["label"])
    plot_df = plot_df[(~mask_aux) | (plot_df["方案类型"] == plan_type_for_aux)].copy()

    model_order = [
        MODEL_META["strict_main"]["label"],
        MODEL_META["realistic"]["label"],
        MODEL_META["aux_validation"]["label"],
    ]
    activity_order = ["<40", "40-59", ">=60"]
    score_order = ["55-58", "59-61", "62-65"]

    for value_col, fname, title in [
        ("期末痰湿积分", "q3_compare_group_final_score_heatmap.png", f"三套模型比较：平均期末痰湿积分（辅助验证取{plan_type_for_aux}）"),
        ("6个月总成本", "q3_compare_group_total_cost_heatmap.png", f"三套模型比较：平均6个月总成本（辅助验证取{plan_type_for_aux}）"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=220)
        last_im = None
        for ax, model_name in zip(axes, model_order):
            sub = plot_df[plot_df["模型"] == model_name]
            if len(sub) == 0:
                ax.axis("off")
                continue
            grp = (
                sub.groupby(["初始痰湿积分分层", "活动能力分层"])[value_col]
                .mean()
                .reset_index()
                .pivot(index="初始痰湿积分分层", columns="活动能力分层", values=value_col)
                .reindex(index=score_order, columns=activity_order)
            )
            last_im = ax.imshow(grp.values, aspect="auto")
            apply_zh_font(ax, title=model_name, xlabel="活动能力分层", ylabel="初始积分分层")
            ax.set_xticks(range(len(activity_order)))
            ax.set_xticklabels(activity_order)
            ax.set_yticks(range(len(score_order)))
            ax.set_yticklabels(score_order)
            for i in range(grp.shape[0]):
                for j in range(grp.shape[1]):
                    val = grp.iloc[i, j]
                    if pd.notna(val):
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                                color="white", fontproperties=ZH_FONT, fontsize=12)
        fig.suptitle(title, fontproperties=ZH_FONT, fontsize=16)
        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.85)
            for label in cbar.ax.get_yticklabels():
                label.set_fontproperties(ZH_FONT)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(FIG_SUBDIR / fname, bbox_inches="tight")
        plt.close()


def plot_overall_bar(overall_df: pd.DataFrame, value_col: str, fname: str, title: str):
    df = overall_df.copy()
    df["标签"] = df["模型"] + "｜" + df["方案类型"]
    fig, ax = plt.subplots(figsize=(12, 6), dpi=220)
    x = np.arange(len(df))
    vals = df[value_col].values
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(df["标签"], rotation=35, ha="right")
    apply_zh_font(ax, title=title, ylabel=value_col)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontproperties=ZH_FONT, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_SUBDIR / fname, bbox_inches="tight")
    plt.close()


def plot_sample_scatter(sample_df: pd.DataFrame, sample_id: int):
    sub = sample_df[sample_df["样本ID"] == sample_id].copy()
    if len(sub) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)

    marker_map = {
        "最优方案": "o",
        "最低成本": "s",
        "均衡方案": "D",
        "疗效最优": "*",
    }

    for _, row in sub.iterrows():
        label = f'{row["模型"]}｜{row["方案类型"]}'
        ax.scatter(row["6个月总成本"], row["期末痰湿积分"],
                   s=180 if row["方案类型"] != "疗效最优" else 260,
                   marker=marker_map.get(row["方案类型"], "o"),
                   label=label)
        ax.text(row["6个月总成本"] + 5, row["期末痰湿积分"] + 0.2,
                row["方案类型"], fontproperties=ZH_FONT, fontsize=10)

    apply_zh_font(ax, title=f"样本 {sample_id} 的三套模型方案比较", xlabel="6个月总成本（元）", ylabel="6个月末痰湿积分")
    ax.legend(prop=ZH_FONT, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_SUBDIR / f"q3_sample_{sample_id}_three_model_compare.png", bbox_inches="tight")
    plt.close()


def plot_sample_pareto_overlay(sample_id: int, summary_df: pd.DataFrame, pareto_dict: Dict[str, pd.DataFrame]):
    aux_df = pareto_dict.get("aux_validation")
    if aux_df is None or len(aux_df) == 0:
        return
    pid_col = find_column(aux_df, ["样本ID"])
    cost_col = find_column(aux_df, ["6个月总成本", "总成本"])
    score_col = find_column(aux_df, ["期末痰湿积分", "final_score", "6个月末痰湿积分"])

    front = aux_df[aux_df[pid_col] == sample_id].copy()
    if len(front) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    ax.scatter(front[cost_col], front[score_col], s=45, alpha=0.7, label="辅助验证帕累托前沿")

    sub = summary_df[summary_df["样本ID"] == sample_id].copy()
    for _, row in sub.iterrows():
        ax.scatter(row["6个月总成本"], row["期末痰湿积分"], s=220, label=f'{row["模型"]}｜{row["方案类型"]}')

    apply_zh_font(ax, title=f"样本 {sample_id}：三套模型与帕累托前沿对照", xlabel="6个月总成本（元）", ylabel="6个月末痰湿积分")
    ax.legend(prop=ZH_FONT, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_SUBDIR / f"q3_sample_{sample_id}_pareto_overlay_compare.png", bbox_inches="tight")
    plt.close()


def build_plan_frequency_table(monthly_dict: Dict[str, pd.DataFrame], sample_ids: List[int]) -> pd.DataFrame:
    rows = []

    for key in ["strict_main", "realistic"]:
        df = monthly_dict.get(key)
        if df is None or len(df) == 0:
            continue
        col_id = find_column(df, ["样本ID", "sample_id"])
        col_month = find_column(df, ["月份", "month"])
        col_level = find_column(df, ["活动强度等级", "exercise_level"])
        col_freq = find_column(df, ["每周训练次数", "freq_per_week"])
        for pid in sample_ids:
            sub = df[df[col_id] == pid].copy()
            if len(sub) == 0:
                continue
            sub = sub.sort_values(col_month)
            rows.append({
                "样本ID": pid,
                "模型": MODEL_META[key]["label"],
                "方案类型": "最优方案",
                "首月强度": pd.to_numeric(sub.iloc[0][col_level], errors="coerce"),
                "首月频率": pd.to_numeric(sub.iloc[0][col_freq], errors="coerce"),
                "6个月平均频率": pd.to_numeric(sub[col_freq], errors="coerce").mean(),
                "6个月总频次": pd.to_numeric(sub[col_freq], errors="coerce").sum(),
            })

    df = monthly_dict.get("aux_validation")
    if df is not None and len(df) > 0:
        col_id = find_column(df, ["样本ID"])
        col_plan = find_column(df, ["方案类型"])
        col_month = find_column(df, ["month", "月份"])
        col_level = find_column(df, ["exercise_level", "活动强度等级"])
        col_freq = find_column(df, ["freq_per_week", "每周训练次数"])
        for pid in sample_ids:
            sub_id = df[df[col_id] == pid].copy()
            if len(sub_id) == 0:
                continue
            for ptype, sub in sub_id.groupby(col_plan):
                sub = sub.sort_values(col_month)
                rows.append({
                    "样本ID": pid,
                    "模型": MODEL_META["aux_validation"]["label"],
                    "方案类型": str(ptype),
                    "首月强度": pd.to_numeric(sub.iloc[0][col_level], errors="coerce"),
                    "首月频率": pd.to_numeric(sub.iloc[0][col_freq], errors="coerce"),
                    "6个月平均频率": pd.to_numeric(sub[col_freq], errors="coerce").mean(),
                    "6个月总频次": pd.to_numeric(sub[col_freq], errors="coerce").sum(),
                })

    return pd.DataFrame(rows)


def write_summary_text(overall_df: pd.DataFrame, missing_files: List[str], sample_df: pd.DataFrame):
    lines = []
    lines.append("第三问三套模型比较结果摘要\n")
    lines.append("============================================\n")
    lines.append(f"项目根目录：{PROJECT_ROOT}\n\n")
    lines.append("一、比较对象\n")
    lines.append("1. 严格主模型：字典序动态规划，先最小化6个月末痰湿积分，再在同等疗效下最小化总成本。\n")
    lines.append("2. 现实优化版：在疗效容忍带内优先降低成本与训练负担。\n")
    lines.append("3. 帕累托辅助验证版：保留完整帕累托前沿，并输出最低成本/均衡方案/疗效最优三类代表方案。\n\n")

    lines.append("二、总体平均结果\n")
    for _, row in overall_df.iterrows():
        lines.append(
            f'- {row["模型"]}｜{row["方案类型"]}：患者数={int(row["患者数"])}, '
            f'平均期末积分={row["平均期末痰湿积分"]:.2f}, '
            f'平均总成本={row["平均6个月总成本"]:.2f}, '
            f'平均积分降幅={row["平均积分降幅(%)"]:.2f}%\n'
        )

    if len(sample_df) > 0:
        lines.append("\n三、样本1/2/3结果说明\n")
        for pid, sub in sample_df.groupby("样本ID"):
            lines.append(f"- 样本 {pid}：\n")
            for _, row in sub.iterrows():
                lines.append(
                    f'  * {row["模型"]}｜{row["方案类型"]}：期末积分={row["期末痰湿积分"]:.2f}，总成本={row["6个月总成本"]:.2f}\n'
                )

    if missing_files:
        lines.append("\n四、未找到的文件（对应图/表会自动跳过）\n")
        for p in missing_files:
            lines.append(f"- {p}\n")

    with open(OUT_SUBDIR / "q3_model_comparison_summary.txt", "w", encoding="utf-8-sig") as f:
        f.writelines(lines)


def main():
    summary_df, missing_summary = load_all_summaries()
    feature_dict, missing_feature = load_feature_rules()
    monthly_dict, missing_monthly = load_monthly_plans()
    pareto_dict, missing_pareto = load_pareto_points()

    all_missing = missing_summary + missing_feature + missing_monthly + missing_pareto

    overall_df = build_overall_table(summary_df)
    overall_df.to_csv(OUT_SUBDIR / "q3_model_overall_comparison.csv", index=False, encoding="utf-8-sig")

    sample_ids = [1, 2, 3]
    sample_df = build_sample_table(summary_df, sample_ids)
    sample_df.to_csv(OUT_SUBDIR / "q3_model_sample_123_comparison.csv", index=False, encoding="utf-8-sig")

    plan_freq_df = build_plan_frequency_table(monthly_dict, sample_ids)
    if len(plan_freq_df) > 0:
        plan_freq_df.to_csv(OUT_SUBDIR / "q3_model_sample_123_frequency_comparison.csv", index=False, encoding="utf-8-sig")

    plot_group_heatmaps(summary_df, plan_type_for_aux="均衡方案")
    plot_overall_bar(overall_df, "平均期末痰湿积分",
                     "q3_compare_overall_final_score_bar.png",
                     "三套模型总体比较：平均期末痰湿积分")
    plot_overall_bar(overall_df, "平均6个月总成本",
                     "q3_compare_overall_total_cost_bar.png",
                     "三套模型总体比较：平均6个月总成本")
    plot_overall_bar(overall_df, "平均积分降幅(%)",
                     "q3_compare_overall_drop_pct_bar.png",
                     "三套模型总体比较：平均积分降幅(%)")

    for pid in sample_ids:
        plot_sample_scatter(sample_df, pid)
        plot_sample_pareto_overlay(pid, summary_df, pareto_dict)

    if feature_dict:
        merged_rules = []
        for key, fdf in feature_dict.items():
            tmp = fdf.copy()
            tmp["模型"] = MODEL_META[key]["label"]
            merged_rules.append(tmp)
        merged_rule_df = pd.concat(merged_rules, ignore_index=True)
        merged_rule_df.to_csv(OUT_SUBDIR / "q3_feature_rule_comparison_stacked.csv", index=False, encoding="utf-8-sig")

    write_summary_text(overall_df, all_missing, sample_df)

    metadata = {
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(OUT_SUBDIR),
        "figure_dir": str(FIG_SUBDIR),
        "available_models": sorted(summary_df["模型"].dropna().unique().tolist()),
        "row_count_summary": int(len(summary_df)),
        "missing_files": all_missing,
    }
    with open(OUT_SUBDIR / "q3_model_comparison_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("第三问三套模型比较脚本已运行完成。")
    print(f"输出目录：{OUT_SUBDIR}")
    print(f"图像目录：{FIG_SUBDIR}")


if __name__ == "__main__":
    main()
