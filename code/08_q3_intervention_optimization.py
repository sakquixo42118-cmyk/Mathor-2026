from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# =========================
# Path helpers
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
RAW_DIR = PROJECT_ROOT / "raw"
OUT_DIR = PROJECT_ROOT / "out"
FIGURE_DIR = PROJECT_ROOT / "figure"

# Try to respect existing config.py if it exposes familiar path vars.
try:
    import importlib.util

    config_path = CODE_DIR / "config.py"
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("project_config", config_path)
        if spec and spec.loader:
            project_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(project_config)
            PROJECT_ROOT = Path(getattr(project_config, "PROJECT_ROOT", PROJECT_ROOT))
            CODE_DIR = Path(getattr(project_config, "CODE_DIR", CODE_DIR))
            RAW_DIR = Path(getattr(project_config, "RAW_DIR", RAW_DIR))
            OUT_DIR = Path(getattr(project_config, "OUT_DIR", OUT_DIR))
            FIGURE_DIR = Path(getattr(project_config, "FIGURE_DIR", FIGURE_DIR))
except Exception:
    # Silent fallback is deliberate; the script remains self-contained.
    pass

Q3_OUT_DIR = OUT_DIR / "q3_intervention_optimization"
Q3_FIG_DIR = FIGURE_DIR / "q3_intervention_optimization"
Q3_OUT_DIR.mkdir(parents=True, exist_ok=True)
Q3_FIG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Plot style
# =========================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# Model constants (from Q3 rules)
# =========================
CONSTITUTION_TARGET = 5
BUDGET_CAP = 2000.0
MONTHS = 6
WEEKS_PER_MONTH = 4

TCM_MONTHLY_COST = {
    1: 30.0,
    2: 80.0,
    3: 130.0,
}

ACTIVITY_UNIT_COST = {
    1: 3.0,
    2: 5.0,
    3: 8.0,
}

INTENSITY_MINUTES = {
    1: 10,
    2: 20,
    3: 30,
}


# =========================
# Utilities
# =========================
def read_csv_robust(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    errors = []
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            errors.append(f"{enc}: {exc}")
    raise RuntimeError(f"无法读取文件 {path}\n" + "\n".join(errors))


def _normalize_col_name(name: str) -> str:
    s = str(name).strip()
    s = s.replace(" ", "").replace("　", "")
    return s


def _strip_bracket_suffix(name: str) -> str:
    s = _normalize_col_name(name)
    s = re.sub(r"[（(].*?[）)]", "", s)
    return s


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    columns = list(df.columns)

    exact_map = {_normalize_col_name(c): c for c in columns}
    base_map = {}
    for c in columns:
        base = _strip_bracket_suffix(c)
        base_map.setdefault(base, []).append(c)

    # 1) 先做严格匹配
    for cand in candidates:
        key = _normalize_col_name(cand)
        if key in exact_map:
            return exact_map[key]

    # 2) 再匹配“去掉括号说明后的主字段名”
    for cand in candidates:
        key = _strip_bracket_suffix(cand)
        matched = base_map.get(key, [])
        if len(matched) == 1:
            return matched[0]

    # 3) 最后做包含式匹配，但只接受唯一结果，避免误匹配
    for cand in candidates:
        key = _strip_bracket_suffix(cand)
        fuzzy = [c for c in columns if key and (key in _normalize_col_name(c) or key in _strip_bracket_suffix(c))]
        if len(fuzzy) == 1:
            return fuzzy[0]

    raise KeyError(f"未找到字段，候选为：{candidates}。现有字段：{list(df.columns)}")


def score_to_int(score: float) -> int:
    return int(round(float(score) * 10))


def int_to_score(score_tenth: int) -> float:
    return score_tenth / 10.0


def tcm_grade(score: float) -> int:
    if score < 59:
        return 1
    if score < 62:
        return 2
    return 3


def age_allowed_levels(age_group: int) -> set[int]:
    # 1=40-49, 2=50-59, 3=60-69, 4=70-79, 5=80-89
    if age_group in {1, 2}:
        return {1, 2, 3}
    if age_group in {3, 4}:
        return {1, 2}
    return {1}


def activity_allowed_levels(activity_total: float) -> set[int]:
    if activity_total < 40:
        return {1}
    if activity_total < 60:
        return {1, 2}
    return {1, 2, 3}


def legal_levels(age_group: int, activity_total: float) -> List[int]:
    allowed = age_allowed_levels(age_group) & activity_allowed_levels(activity_total)
    return sorted(allowed)


def monthly_drop_rate(intensity: int, freq_per_week: int) -> float:
    """
    Implements the interpretation consistent with the official wording:
    - if freq < 5 => stable (0 decline)
    - at freq=5, level 1 is baseline, each extra level adds 3%
    - above 5 times/week, each extra session adds 1%

    So:
        r = 0                         if f < 5
        r = 0.03*(k-1) + 0.01*(f-5)  if f >= 5
    """
    if freq_per_week < 5:
        return 0.0
    return 0.03 * (intensity - 1) + 0.01 * (freq_per_week - 5)


def next_score(score: float, intensity: int, freq_per_week: int) -> float:
    r = monthly_drop_rate(intensity, freq_per_week)
    next_s = score * (1 - r)
    return max(0.0, round(next_s, 1))


def monthly_cost(score: float, intensity: int, freq_per_week: int) -> float:
    grade = tcm_grade(score)
    return TCM_MONTHLY_COST[grade] + WEEKS_PER_MONTH * freq_per_week * ACTIVITY_UNIT_COST[intensity]


def initial_score_bin(score: float) -> str:
    if score < 59:
        return "55-58"
    if score < 62:
        return "59-61"
    return "62-65"


def activity_bin(total: float) -> str:
    if total < 40:
        return "<40"
    if total < 60:
        return "40-59"
    return ">=60"


def age_bin(age_group: int) -> str:
    if age_group in {1, 2}:
        return "40-59"
    if age_group in {3, 4}:
        return "60-79"
    return "80-89"


def plan_pattern(month_plan: pd.DataFrame) -> str:
    if month_plan.empty:
        return "未知"
    first_half = month_plan.iloc[:2]
    last_half = month_plan.iloc[-2:]
    i1, i2 = first_half["活动强度等级"].mean(), last_half["活动强度等级"].mean()
    f1, f2 = first_half["每周训练次数"].mean(), last_half["每周训练次数"].mean()
    if (i1 - i2) >= 0.5 or (f1 - f2) >= 1.0:
        return "前高后低"
    if (i2 - i1) >= 0.5 or (f2 - f1) >= 1.0:
        return "前低后高"
    return "稳定维持"


def month_cross_below(series_scores_start: List[float], threshold: float) -> Optional[int]:
    for month_idx, s in enumerate(series_scores_start, start=1):
        if s < threshold:
            return month_idx
    return None


@dataclass
class PatientInfo:
    sample_id: int
    constitution_tag: int
    phlegm_score: float
    age_group: int
    activity_total: float


@dataclass
class DPResult:
    sample_id: int
    best_final_score: float
    total_cost: float
    total_reduction: float
    reduction_pct: float
    legal_levels: List[int]
    monthly_plan: pd.DataFrame
    pareto_df: pd.DataFrame


# =========================
# Core DP solver
# =========================
def optimize_patient(patient: PatientInfo, budget_cap: float = BUDGET_CAP) -> DPResult:
    allowed_levels = legal_levels(patient.age_group, patient.activity_total)
    actions = [(k, f) for k in allowed_levels for f in range(1, 11)]

    start_state = score_to_int(patient.phlegm_score)
    layers: List[Dict[int, float]] = [defaultdict(lambda: math.inf) for _ in range(MONTHS + 2)]
    backptr: List[Dict[int, Tuple[int, int, int]]] = [dict() for _ in range(MONTHS + 2)]
    layers[1][start_state] = 0.0

    for month in range(1, MONTHS + 1):
        if not layers[month]:
            break
        for score_tenth, accum_cost in list(layers[month].items()):
            if accum_cost > budget_cap:
                continue
            score = int_to_score(score_tenth)
            for intensity, freq in actions:
                c = monthly_cost(score, intensity, freq)
                new_cost = accum_cost + c
                if new_cost > budget_cap + 1e-9:
                    continue
                score2 = next_score(score, intensity, freq)
                score2_tenth = score_to_int(score2)
                if new_cost + 1e-9 < layers[month + 1].get(score2_tenth, math.inf):
                    layers[month + 1][score2_tenth] = round(new_cost, 2)
                    backptr[month + 1][score2_tenth] = (score_tenth, intensity, freq)

    terminal_states = layers[MONTHS + 1]
    if not terminal_states:
        raise RuntimeError(f"样本 {patient.sample_id} 在预算约束下没有可行方案。")

    # Lexicographic objective: first min final score, then min cost.
    best_terminal = min(terminal_states.items(), key=lambda kv: (kv[0], kv[1]))
    best_score_tenth, best_cost = best_terminal

    # Backtrack best path.
    path = []
    cur_score_tenth = best_score_tenth
    for month in range(MONTHS + 1, 1, -1):
        prev_score_tenth, intensity, freq = backptr[month][cur_score_tenth]
        path.append((month - 1, prev_score_tenth, intensity, freq, cur_score_tenth))
        cur_score_tenth = prev_score_tenth
    path.reverse()

    month_rows = []
    accum = 0.0
    score_start_list = []
    for month, score_start_tenth, intensity, freq, score_end_tenth in path:
        score_start = int_to_score(score_start_tenth)
        score_end = int_to_score(score_end_tenth)
        score_start_list.append(score_start)
        grade = tcm_grade(score_start)
        tcm_c = TCM_MONTHLY_COST[grade]
        act_c = WEEKS_PER_MONTH * freq * ACTIVITY_UNIT_COST[intensity]
        total_c = tcm_c + act_c
        accum += total_c
        month_rows.append({
            "样本ID": patient.sample_id,
            "月份": month,
            "月初痰湿积分": score_start,
            "中医调理等级": grade,
            "调理月成本": tcm_c,
            "活动强度等级": intensity,
            "单次训练时长(分钟)": INTENSITY_MINUTES[intensity],
            "每周训练次数": freq,
            "活动月成本": act_c,
            "月总成本": total_c,
            "累计成本": round(accum, 2),
            "月下降率": round(monthly_drop_rate(intensity, freq), 4),
            "下月月初痰湿积分": score_end,
        })
    month_plan_df = pd.DataFrame(month_rows)

    # Extract Pareto frontier from terminal states (min cost per final score already stored)
    pareto_candidates = sorted(
        [(int_to_score(s_tenth), cost) for s_tenth, cost in terminal_states.items()],
        key=lambda x: (x[1], x[0])
    )
    frontier = []
    best_score_seen = math.inf
    for final_score, cost in pareto_candidates:
        if final_score < best_score_seen - 1e-9:
            frontier.append((cost, final_score))
            best_score_seen = final_score
    pareto_df = pd.DataFrame(frontier, columns=["总成本", "6个月末痰湿积分"])
    pareto_df.insert(0, "样本ID", patient.sample_id)

    result = DPResult(
        sample_id=patient.sample_id,
        best_final_score=int_to_score(best_score_tenth),
        total_cost=round(float(best_cost), 2),
        total_reduction=round(patient.phlegm_score - int_to_score(best_score_tenth), 2),
        reduction_pct=round((patient.phlegm_score - int_to_score(best_score_tenth)) / patient.phlegm_score * 100, 2)
        if patient.phlegm_score > 0 else 0.0,
        legal_levels=allowed_levels,
        monthly_plan=month_plan_df,
        pareto_df=pareto_df,
    )
    return result


# =========================
# Figures
# =========================
def plot_sample_pareto(pareto_df: pd.DataFrame, result: DPResult, save_path: Path) -> None:
    plt.figure(figsize=(7, 5), dpi=200)
    plt.scatter(pareto_df["总成本"], pareto_df["6个月末痰湿积分"], s=35)
    plt.plot(pareto_df["总成本"], pareto_df["6个月末痰湿积分"], alpha=0.8)
    plt.scatter([result.total_cost], [result.best_final_score], s=80, marker="*", label="最优方案")
    plt.xlabel("6个月总成本（元）")
    plt.ylabel("6个月末痰湿积分")
    plt.title(f"样本 {result.sample_id} 的成本—疗效帕累托前沿")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_group_heatmap(group_df: pd.DataFrame, value_col: str, title: str, save_path: Path) -> None:
    if group_df.empty:
        return
    pivot = group_df.pivot_table(index="初始痰湿积分分层", columns="活动能力分层", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=["55-58", "59-61", "62-65"], columns=["<40", "40-59", ">=60"])

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    im = ax.imshow(pivot.values.astype(float), aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("活动能力分层")
    ax.set_ylabel("初始痰湿积分分层")
    ax.set_title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# =========================
# Main pipeline
# =========================
def locate_main_table() -> Path:
    candidates = [
        OUT_DIR / "c_q1_main_simple.csv",
        OUT_DIR / "c_q1_main.csv",
        RAW_DIR / "C题：附件1：样例数据.csv",
        RAW_DIR / "C题：附件1：样例数据.xlsx",
    ]
    for path in candidates:
        if path.exists() and path.suffix.lower() == ".csv":
            return path
    raise FileNotFoundError(
        "未找到可读取的 CSV 主分析表。请确认 out/c_q1_main_simple.csv 已存在。"
    )


def prepare_patient_df(df: pd.DataFrame) -> pd.DataFrame:
    col_sample = find_column(df, ["样本ID", "样本 ID", "ID", "id", "样本id"])
    col_const = find_column(df, ["体质标签", "中医体质标签", "体质分类标签"])
    col_phlegm = find_column(df, ["痰湿质", "痰湿积分", "痰湿质积分", "痰湿体质积分"])
    col_age = find_column(df, ["年龄组", "年龄分组"])
    col_activity = find_column(df, ["活动量表总分", "活动总分", "ADL总分+IADL总分", "ADL总分＋IADL总分"])

    out = df[[col_sample, col_const, col_phlegm, col_age, col_activity]].copy()
    out.columns = ["样本ID", "体质标签", "痰湿质积分", "年龄组", "活动量表总分"]
    out["样本ID"] = pd.to_numeric(out["样本ID"], errors="coerce").astype("Int64")
    out["体质标签"] = pd.to_numeric(out["体质标签"], errors="coerce")
    out["痰湿质积分"] = pd.to_numeric(out["痰湿质积分"], errors="coerce")
    out["年龄组"] = pd.to_numeric(out["年龄组"], errors="coerce")
    out["活动量表总分"] = pd.to_numeric(out["活动量表总分"], errors="coerce")

    out = out.dropna().copy()
    out["样本ID"] = out["样本ID"].astype(int)
    out["年龄组"] = out["年龄组"].astype(int)
    out = out[out["体质标签"] == CONSTITUTION_TARGET].copy()
    out = out.sort_values("样本ID").reset_index(drop=True)
    return out


def summarize_rules(summary_df: pd.DataFrame, detailed_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    merged = summary_df.copy()
    patterns = detailed_df.groupby("样本ID").apply(plan_pattern).rename("方案形态").reset_index()
    merged = merged.merge(patterns, on="样本ID", how="left")

    def mode_or_na(series: pd.Series):
        s = series.dropna().tolist()
        if not s:
            return np.nan
        return Counter(s).most_common(1)[0][0]

    grp = merged.groupby(["年龄分层", "活动能力分层", "初始痰湿积分分层"], dropna=False)
    rule_df = grp.agg(
        患者数=("样本ID", "count"),
        平均初始积分=("初始痰湿积分", "mean"),
        平均期末积分=("6个月末痰湿积分", "mean"),
        平均总成本=("6个月总成本", "mean"),
        平均降幅=("积分总降幅", "mean"),
        平均降幅百分比=("积分降幅(%)", "mean"),
        最常见首月强度=("首月活动强度", mode_or_na),
        最常见首月频率=("首月每周频次", mode_or_na),
        最常见方案形态=("方案形态", mode_or_na),
    ).reset_index()
    return rule_df


def main() -> None:
    csv_path = locate_main_table()
    df = read_csv_robust(csv_path)
    patient_df = prepare_patient_df(df)

    if patient_df.empty:
        raise RuntimeError("主分析表中没有体质标签=5 的痰湿体质患者。")

    summary_rows = []
    monthly_plans = []
    pareto_rows = []
    results_by_id: Dict[int, DPResult] = {}

    for _, row in patient_df.iterrows():
        patient = PatientInfo(
            sample_id=int(row["样本ID"]),
            constitution_tag=int(row["体质标签"]),
            phlegm_score=float(row["痰湿质积分"]),
            age_group=int(row["年龄组"]),
            activity_total=float(row["活动量表总分"]),
        )
        result = optimize_patient(patient)
        results_by_id[patient.sample_id] = result

        plan = result.monthly_plan.copy()
        monthly_plans.append(plan)
        pareto_rows.append(result.pareto_df)

        score_starts = plan["月初痰湿积分"].tolist() + [plan.iloc[-1]["下月月初痰湿积分"]]
        below_62_month = month_cross_below(score_starts, 62.0)
        below_59_month = month_cross_below(score_starts, 59.0)

        summary_rows.append({
            "样本ID": patient.sample_id,
            "体质标签": patient.constitution_tag,
            "初始痰湿积分": round(patient.phlegm_score, 1),
            "年龄组": patient.age_group,
            "活动量表总分": round(patient.activity_total, 1),
            "合法活动强度集合": "/".join(map(str, result.legal_levels)),
            "首月活动强度": int(plan.iloc[0]["活动强度等级"]),
            "首月每周频次": int(plan.iloc[0]["每周训练次数"]),
            "6个月末痰湿积分": result.best_final_score,
            "积分总降幅": result.total_reduction,
            "积分降幅(%)": result.reduction_pct,
            "6个月总成本": result.total_cost,
            "首次低于62分的月份": below_62_month,
            "首次低于59分的月份": below_59_month,
            "年龄分层": age_bin(patient.age_group),
            "活动能力分层": activity_bin(patient.activity_total),
            "初始痰湿积分分层": initial_score_bin(patient.phlegm_score),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("样本ID").reset_index(drop=True)
    monthly_plan_df = pd.concat(monthly_plans, ignore_index=True)
    pareto_df_all = pd.concat(pareto_rows, ignore_index=True)
    rule_df = summarize_rules(summary_df, monthly_plan_df)

    # Save main outputs.
    summary_path = Q3_OUT_DIR / "q3_patient_optimal_summary.csv"
    monthly_path = Q3_OUT_DIR / "q3_patient_monthly_plan.csv"
    pareto_path = Q3_OUT_DIR / "q3_sample_pareto_points.csv"
    rules_path = Q3_OUT_DIR / "q3_feature_rule_summary.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    monthly_plan_df.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    pareto_df_all.to_csv(pareto_path, index=False, encoding="utf-8-sig")
    rule_df.to_csv(rules_path, index=False, encoding="utf-8-sig")

    # Sample 1/2/3 dedicated outputs.
    for sid in [1, 2, 3]:
        if sid in results_by_id:
            results_by_id[sid].monthly_plan.to_csv(
                Q3_OUT_DIR / f"q3_sample_{sid}_optimal_plan.csv",
                index=False,
                encoding="utf-8-sig",
            )
            plot_sample_pareto(
                results_by_id[sid].pareto_df,
                results_by_id[sid],
                Q3_FIG_DIR / f"q3_sample_{sid}_pareto.png",
            )

    # Group figures.
    plot_group_heatmap(
        summary_df,
        value_col="6个月总成本",
        title="不同初始积分—活动能力分层下的平均6个月最优总成本",
        save_path=Q3_FIG_DIR / "q3_group_avg_cost_heatmap.png",
    )
    plot_group_heatmap(
        summary_df,
        value_col="6个月末痰湿积分",
        title="不同初始积分—活动能力分层下的平均6个月末最优痰湿积分",
        save_path=Q3_FIG_DIR / "q3_group_avg_final_score_heatmap.png",
    )

    # Text summary.
    phlegm_n = len(summary_df)
    mean_init = summary_df["初始痰湿积分"].mean()
    mean_final = summary_df["6个月末痰湿积分"].mean()
    mean_cost = summary_df["6个月总成本"].mean()
    mean_drop_pct = summary_df["积分降幅(%)"].mean()

    lines = []
    lines.append("问题三：痰湿体质患者6个月干预优化结果摘要")
    lines.append("=" * 44)
    lines.append(f"输入数据文件：{csv_path}")
    lines.append(f"痰湿体质患者数量（体质标签=5）：{phlegm_n}")
    lines.append("")
    lines.append("一、模型口径")
    lines.append("1. 仅针对体质标签=5的患者求解。")
    lines.append("2. 采用 6 阶段前向动态规划。")
    lines.append("3. 采用字典序目标：先最小化6个月末痰湿积分，再在同等疗效下最小化总成本。")
    lines.append("4. 活动干预约束、调理分级、频率、成本和预算均按题面规则执行。")
    lines.append("5. 月下降率使用：f<5时为0；f>=5时 r = 3%*(k-1) + 1%*(f-5)。")
    lines.append("")
    lines.append("二、总体结果")
    lines.append(f"平均初始痰湿积分：{mean_init:.2f}")
    lines.append(f"平均6个月末痰湿积分：{mean_final:.2f}")
    lines.append(f"平均积分降幅百分比：{mean_drop_pct:.2f}%")
    lines.append(f"平均6个月总成本：{mean_cost:.2f} 元")
    lines.append("")
    lines.append("三、典型匹配规律（基于最优方案汇总）")
    if not rule_df.empty:
        for _, r in rule_df.sort_values(["年龄分层", "活动能力分层", "初始痰湿积分分层"]).iterrows():
            lines.append(
                f"- 年龄{r['年龄分层']}、活动{r['活动能力分层']}、初始积分{r['初始痰湿积分分层']}："
                f"患者数={int(r['患者数'])}，平均期末积分={r['平均期末积分']:.2f}，"
                f"平均总成本={r['平均总成本']:.2f}，"
                f"最常见首月强度={r['最常见首月强度']}，最常见首月频率={r['最常见首月频率']}，"
                f"最常见方案形态={r['最常见方案形态']}。"
            )
    else:
        lines.append("- 当前分组结果为空。")
    lines.append("")
    lines.append("四、样本1/2/3文件说明")
    for sid in [1, 2, 3]:
        if sid in results_by_id:
            res = results_by_id[sid]
            lines.append(
                f"- 样本{sid}：6个月末痰湿积分={res.best_final_score:.1f}，"
                f"总成本={res.total_cost:.2f} 元，"
                f"最优方案文件=q3_sample_{sid}_optimal_plan.csv"
            )
        else:
            lines.append(f"- 样本{sid}：当前主分析表中未找到或不属于痰湿体质患者。")

    (Q3_OUT_DIR / "q3_intervention_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("Q3 干预优化完成。")
    print(f"输出目录：{Q3_OUT_DIR}")
    print(f"图像目录：{Q3_FIG_DIR}")


if __name__ == "__main__":
    main()
