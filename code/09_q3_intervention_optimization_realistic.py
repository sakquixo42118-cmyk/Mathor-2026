import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd

# =========================
# 用户可调整参数（现实优化版）
# =========================
BUDGET_MAX = 2000.0
MONTHS = 6
WEEKS_PER_MONTH = 4
SCORE_TOLERANCE = 1.0  # 允许比严格最优终点高出的积分容忍带（现实优化版核心）
SCORE_DECIMALS = 1

ACTIVITY_COST = {1: 3.0, 2: 5.0, 3: 8.0}
TCM_COST = {1: 30.0, 2: 80.0, 3: 130.0}

# 负担指数：每次训练按强度加权。1级=1.0，2级=1.5，3级=2.0
INTENSITY_BURDEN_WEIGHT = {1: 1.0, 2: 1.5, 3: 2.0}


# =========================
# 路径与字体
# =========================
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = get_project_root()
OUT_DIR = PROJECT_ROOT / "out" / "q3_intervention_optimization_realistic"
FIG_DIR = PROJECT_ROOT / "figure" / "q3_intervention_optimization_realistic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def setup_chinese_font() -> None:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    chosen = None
    for fp in candidates:
        p = Path(fp)
        if p.exists():
            try:
                fm.fontManager.addfont(str(p))
                chosen = fm.FontProperties(fname=str(p)).get_name()
                break
            except Exception:
                continue
    if chosen is not None:
        plt.rcParams["font.family"] = chosen
        plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False


setup_chinese_font()


# =========================
# 工具函数
# =========================
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
    base_map: Dict[str, List[str]] = defaultdict(list)
    for c in columns:
        base_map[_strip_bracket_suffix(c)].append(c)

    for cand in candidates:
        key = _normalize_col_name(cand)
        if key in exact_map:
            return exact_map[key]

    for cand in candidates:
        key = _strip_bracket_suffix(cand)
        if key in base_map and len(base_map[key]) == 1:
            return base_map[key][0]

    for cand in candidates:
        key = _strip_bracket_suffix(cand)
        fuzzy = [c for c in columns if key and (key in _normalize_col_name(c) or key in _strip_bracket_suffix(c))]
        if len(fuzzy) == 1:
            return fuzzy[0]

    raise KeyError(f"未找到字段，候选为：{candidates}。现有字段：{list(df.columns)}")


def round_score(x: float) -> float:
    return round(float(x), SCORE_DECIMALS)


def tcm_level(score: float) -> int:
    if score <= 58:
        return 1
    if score <= 61:
        return 2
    return 3


def monthly_tcm_cost(score: float) -> float:
    return TCM_COST[tcm_level(score)]


def next_score(score: float, intensity: int, weekly_freq: int) -> float:
    if weekly_freq < 5:
        return round_score(score)
    # 题面口径：每周5次为基准，每提高一级强度 +3%，每再增加1次/周 +1%
    reduction_rate = 0.03 * (intensity - 1) + 0.01 * (weekly_freq - 5)
    score2 = score * (1 - reduction_rate)
    return round_score(max(score2, 0.0))


def monthly_activity_cost(intensity: int, weekly_freq: int) -> float:
    return WEEKS_PER_MONTH * weekly_freq * ACTIVITY_COST[intensity]


def monthly_burden(intensity: int, weekly_freq: int) -> float:
    return WEEKS_PER_MONTH * weekly_freq * INTENSITY_BURDEN_WEIGHT[intensity]


def parse_age_band(age_name: str) -> str:
    text = str(age_name)
    nums = re.findall(r"\d+", text)
    if nums:
        first = int(nums[0])
        if 40 <= first <= 59:
            return "40-59"
        if 60 <= first <= 79:
            return "60-79"
        if 80 <= first <= 89:
            return "80-89"
    # 兜底：按关键词
    if "80" in text:
        return "80-89"
    if "60" in text or "70" in text:
        return "60-79"
    return "40-59"


def activity_band(activity_total: float) -> str:
    if activity_total < 40:
        return "<40"
    if activity_total < 60:
        return "40-59"
    return ">=60"


def initial_score_band(score: float) -> str:
    if score <= 58:
        return "55-58"
    if score <= 61:
        return "59-61"
    return "62-65"


def legal_intensities(age_band_name: str, activity_total: float) -> List[int]:
    # 年龄约束
    if age_band_name == "40-59":
        age_allowed = {1, 2, 3}
    elif age_band_name == "60-79":
        age_allowed = {1, 2}
    else:
        age_allowed = {1}

    # 活动能力约束
    if activity_total < 40:
        act_allowed = {1}
    elif activity_total < 60:
        act_allowed = {1, 2}
    else:
        act_allowed = {1, 2, 3}

    return sorted(age_allowed & act_allowed)


def classify_plan_shape(intensities: List[int], freqs: List[int]) -> str:
    if len(intensities) == 0:
        return "未知"
    loads = [i + 0.1 * f for i, f in zip(intensities, freqs)]
    if max(loads) - min(loads) < 0.35:
        return "稳定维持"
    first_avg = float(np.mean(loads[:2]))
    last_avg = float(np.mean(loads[-2:]))
    if last_avg - first_avg > 0.25:
        return "前低后高"
    if first_avg - last_avg > 0.25:
        return "前高后低"
    return "波动调整"


# =========================
# 动态规划数据结构
# =========================
@dataclass
class Node:
    cost: float
    burden: float
    sessions: int

    def key(self) -> Tuple[float, float, int]:
        return (round(self.cost, 6), round(self.burden, 6), self.sessions)


@dataclass
class PatientResult:
    sample_id: int
    age_band_name: str
    activity_band_name: str
    initial_score_band_name: str
    initial_score: float
    strict_final_score: float
    strict_total_cost: float
    strict_total_burden: float
    realistic_final_score: float
    realistic_total_cost: float
    realistic_total_burden: float
    score_gap_vs_strict: float
    cost_saving_vs_strict: float
    burden_saving_vs_strict: float
    realistic_first_month_intensity: int
    realistic_first_month_freq: int
    realistic_plan_shape: str
    realistic_feasible_states: int


# =========================
# 求解核心
# =========================
def solve_patient(sample_id: int, init_score: float, age_band_name: str, act_total: float) -> Tuple[PatientResult, pd.DataFrame, pd.DataFrame]:
    allowed = legal_intensities(age_band_name, act_total)
    states_by_month: List[Dict[float, Node]] = []
    backptr_by_month: List[Dict[float, Tuple[float, int, int]]] = []

    start_score = round_score(init_score)
    states = {start_score: Node(cost=0.0, burden=0.0, sessions=0)}
    states_by_month.append(states)

    for month in range(1, MONTHS + 1):
        next_states: Dict[float, Node] = {}
        next_back: Dict[float, Tuple[float, int, int]] = {}
        for score, node in states.items():
            tcm_cost_month = monthly_tcm_cost(score)
            for intensity in allowed:
                for freq in range(1, 11):
                    cost_add = tcm_cost_month + monthly_activity_cost(intensity, freq)
                    new_cost = node.cost + cost_add
                    if new_cost > BUDGET_MAX + 1e-9:
                        continue
                    new_burden = node.burden + monthly_burden(intensity, freq)
                    new_sessions = node.sessions + freq
                    score2 = next_score(score, intensity, freq)
                    cand = Node(cost=new_cost, burden=new_burden, sessions=new_sessions)
                    if (score2 not in next_states) or (cand.key() < next_states[score2].key()):
                        next_states[score2] = cand
                        next_back[score2] = (score, intensity, freq)
        states = next_states
        states_by_month.append(states)
        backptr_by_month.append(next_back)

    final_states = states_by_month[-1]
    if not final_states:
        raise RuntimeError(f"样本 {sample_id} 在预算约束下无可行方案。")

    # 严格模型：先最小终点积分，再最小成本/负担
    strict_best_score = min(final_states.keys())
    strict_node = final_states[strict_best_score]

    # 现实模型：允许终点积分在最优 + 容忍带以内，再优先成本，再优先负担，再优先频次，再优先更低积分
    tolerant_scores = [s for s in final_states.keys() if s <= strict_best_score + SCORE_TOLERANCE + 1e-9]
    realistic_best_score = min(
        tolerant_scores,
        key=lambda s: (
            round(final_states[s].cost, 6),
            round(final_states[s].burden, 6),
            final_states[s].sessions,
            round(s, 6),
        ),
    )
    realistic_node = final_states[realistic_best_score]

    def reconstruct(end_score: float) -> pd.DataFrame:
        rows = []
        curr_score = end_score
        for month in range(MONTHS, 0, -1):
            prev_score, intensity, freq = backptr_by_month[month - 1][curr_score]
            month_tcm_level = tcm_level(prev_score)
            month_tcm_cost = monthly_tcm_cost(prev_score)
            month_act_cost = monthly_activity_cost(intensity, freq)
            rows.append(
                {
                    "样本ID": sample_id,
                    "月份": month,
                    "月初痰湿积分": prev_score,
                    "中医调理等级": month_tcm_level,
                    "活动强度等级": intensity,
                    "每周训练次数": freq,
                    "当月中医调理成本": month_tcm_cost,
                    "当月活动成本": month_act_cost,
                    "当月总成本": month_tcm_cost + month_act_cost,
                    "下月月初痰湿积分": curr_score,
                }
            )
            curr_score = prev_score
        rows.reverse()
        df_plan = pd.DataFrame(rows)
        df_plan["累计总成本"] = df_plan["当月总成本"].cumsum()
        df_plan["当月负担指数"] = [monthly_burden(i, f) for i, f in zip(df_plan["活动强度等级"], df_plan["每周训练次数"])]
        df_plan["累计负担指数"] = df_plan["当月负担指数"].cumsum()
        return df_plan

    strict_plan = reconstruct(strict_best_score)
    realistic_plan = reconstruct(realistic_best_score)

    pareto_df = pd.DataFrame(
        {
            "样本ID": sample_id,
            "6个月末痰湿积分": list(final_states.keys()),
            "6个月总成本": [final_states[s].cost for s in final_states.keys()],
            "总负担指数": [final_states[s].burden for s in final_states.keys()],
            "总周频次和": [final_states[s].sessions for s in final_states.keys()],
            "是否严格最优": [1 if s == strict_best_score else 0 for s in final_states.keys()],
            "是否现实最优": [1 if s == realistic_best_score else 0 for s in final_states.keys()],
            "是否在容忍带内": [1 if s in tolerant_scores else 0 for s in final_states.keys()],
        }
    ).sort_values(["6个月总成本", "6个月末痰湿积分"]).reset_index(drop=True)

    intensities = realistic_plan["活动强度等级"].tolist()
    freqs = realistic_plan["每周训练次数"].tolist()
    plan_shape = classify_plan_shape(intensities, freqs)

    result = PatientResult(
        sample_id=int(sample_id),
        age_band_name=age_band_name,
        activity_band_name=activity_band(act_total),
        initial_score_band_name=initial_score_band(init_score),
        initial_score=float(init_score),
        strict_final_score=float(strict_best_score),
        strict_total_cost=float(strict_node.cost),
        strict_total_burden=float(strict_node.burden),
        realistic_final_score=float(realistic_best_score),
        realistic_total_cost=float(realistic_node.cost),
        realistic_total_burden=float(realistic_node.burden),
        score_gap_vs_strict=float(realistic_best_score - strict_best_score),
        cost_saving_vs_strict=float(strict_node.cost - realistic_node.cost),
        burden_saving_vs_strict=float(strict_node.burden - realistic_node.burden),
        realistic_first_month_intensity=int(intensities[0]),
        realistic_first_month_freq=int(freqs[0]),
        realistic_plan_shape=plan_shape,
        realistic_feasible_states=int(len(final_states)),
    )
    return result, realistic_plan, pareto_df


# =========================
# 绘图
# =========================
def save_heatmap(df_pivot: pd.DataFrame, title: str, cbar_label: str, save_path: Path, fmt: str = ".1f") -> None:
    arr = df_pivot.values.astype(float)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(df_pivot.shape[1]))
    ax.set_xticklabels(df_pivot.columns)
    ax.set_yticks(np.arange(df_pivot.shape[0]))
    ax.set_yticklabels(df_pivot.index)
    ax.set_title(title)
    ax.set_xlabel("活动能力分层")
    ax.set_ylabel("初始痰湿积分分层")
    for i in range(df_pivot.shape[0]):
        for j in range(df_pivot.shape[1]):
            val = df_pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", color="white", fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_compare(pareto_df: pd.DataFrame, sample_id: int, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    ax.scatter(
        pareto_df["6个月总成本"],
        pareto_df["6个月末痰湿积分"],
        s=40,
        alpha=0.85,
        label="可行前沿点",
    )
    strict_row = pareto_df.loc[pareto_df["是否严格最优"] == 1].iloc[0]
    realistic_row = pareto_df.loc[pareto_df["是否现实最优"] == 1].iloc[0]
    ax.scatter(
        strict_row["6个月总成本"],
        strict_row["6个月末痰湿积分"],
        marker="*",
        s=180,
        color="tab:red",
        label="严格最优",
        zorder=5,
    )
    ax.scatter(
        realistic_row["6个月总成本"],
        realistic_row["6个月末痰湿积分"],
        marker="D",
        s=90,
        color="tab:orange",
        label="现实优化最优",
        zorder=6,
    )
    ax.set_title(f"样本 {sample_id} 的成本—疗效帕累托前沿（现实优化版对比）")
    ax.set_xlabel("6个月总成本（元）")
    ax.set_ylabel("6个月末痰湿积分")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# =========================
# 主程序
# =========================
def main() -> None:
    csv_path = PROJECT_ROOT / "out" / "c_q1_main_simple.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到输入文件：{csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    col_id = find_column(df, ["样本ID", "样本id", "ID", "id"])
    col_label = find_column(df, ["体质标签"])
    col_score = find_column(df, ["痰湿质"])
    col_activity = find_column(
        df,
        [
            "活动量表总分",
            "活动总分",
            "ADL总分+IADL总分",
            "ADL总分＋IADL总分",
            "活动量表总分（ADL总分+IADL总分）",
        ],
    )
    col_age_name = find_column(df, ["年龄组名称", "年龄名称", "年龄组"])

    work = df.copy()
    work = work.loc[pd.to_numeric(work[col_label], errors="coerce") == 5].copy()
    work[col_score] = pd.to_numeric(work[col_score], errors="coerce")
    work[col_activity] = pd.to_numeric(work[col_activity], errors="coerce")
    work = work.dropna(subset=[col_id, col_score, col_activity, col_age_name]).copy()

    patient_results: List[PatientResult] = []
    plans: List[pd.DataFrame] = []
    pareto_points: List[pd.DataFrame] = []

    sample_targets = {1, 2, 3}
    sample_plan_map: Dict[int, pd.DataFrame] = {}
    sample_pareto_map: Dict[int, pd.DataFrame] = {}

    for _, row in work.iterrows():
        sample_id = int(row[col_id])
        age_band_name = parse_age_band(str(row[col_age_name]))
        result, plan_df, pareto_df = solve_patient(
            sample_id=sample_id,
            init_score=float(row[col_score]),
            age_band_name=age_band_name,
            act_total=float(row[col_activity]),
        )
        patient_results.append(result)
        plans.append(plan_df)
        pareto_points.append(pareto_df)

        if sample_id in sample_targets:
            sample_plan_map[sample_id] = plan_df
            sample_pareto_map[sample_id] = pareto_df

    result_df = pd.DataFrame([r.__dict__ for r in patient_results])
    plan_all_df = pd.concat(plans, ignore_index=True)
    pareto_all_df = pd.concat(pareto_points, ignore_index=True)

    # 输出主表
    result_df.to_csv(OUT_DIR / "q3_patient_optimal_summary_realistic.csv", index=False, encoding="utf-8-sig")
    plan_all_df.to_csv(OUT_DIR / "q3_patient_monthly_plan_realistic.csv", index=False, encoding="utf-8-sig")
    pareto_all_df.to_csv(OUT_DIR / "q3_sample_pareto_points_realistic.csv", index=False, encoding="utf-8-sig")

    # 样本1/2/3输出
    for sid in sorted(sample_targets):
        if sid in sample_plan_map:
            sample_plan_map[sid].to_csv(
                OUT_DIR / f"q3_sample_{sid}_optimal_plan_realistic.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if sid in sample_pareto_map:
            plot_pareto_compare(
                sample_pareto_map[sid],
                sid,
                FIG_DIR / f"q3_sample_{sid}_pareto_compare_realistic.png",
            )

    # 患者特征-方案规律汇总（现实版）
    group_rows = []
    grouped = result_df.groupby(["age_band_name", "activity_band_name", "initial_score_band_name"], dropna=False)
    for keys, g in grouped:
        age_band_name, activity_band_name, initial_score_band_name = keys
        first_intensity_mode = Counter(g["realistic_first_month_intensity"].tolist()).most_common(1)[0][0]
        first_freq_mode = Counter(g["realistic_first_month_freq"].tolist()).most_common(1)[0][0]
        plan_shape_mode = Counter(g["realistic_plan_shape"].tolist()).most_common(1)[0][0]
        group_rows.append(
            {
                "年龄分层": age_band_name,
                "活动能力分层": activity_band_name,
                "初始痰湿积分分层": initial_score_band_name,
                "患者数": int(len(g)),
                "平均期末痰湿积分": round(float(g["realistic_final_score"].mean()), 2),
                "平均6个月总成本": round(float(g["realistic_total_cost"].mean()), 2),
                "平均总负担指数": round(float(g["realistic_total_burden"].mean()), 2),
                "相对严格模型平均积分损失": round(float(g["score_gap_vs_strict"].mean()), 2),
                "相对严格模型平均成本节约": round(float(g["cost_saving_vs_strict"].mean()), 2),
                "相对严格模型平均负担下降": round(float(g["burden_saving_vs_strict"].mean()), 2),
                "最常见首月强度": int(first_intensity_mode),
                "最常见首月频率": int(first_freq_mode),
                "最常见方案形态": plan_shape_mode,
            }
        )
    feature_df = pd.DataFrame(group_rows).sort_values(["年龄分层", "活动能力分层", "初始痰湿积分分层"])
    feature_df.to_csv(OUT_DIR / "q3_feature_rule_summary_realistic.csv", index=False, encoding="utf-8-sig")

    # 热力图：现实版
    pivot_cost = (
        result_df.groupby(["initial_score_band_name", "activity_band_name"], dropna=False)["realistic_total_cost"]
        .mean()
        .unstack()
        .reindex(index=["55-58", "59-61", "62-65"], columns=["<40", "40-59", ">=60"])
    )
    pivot_score = (
        result_df.groupby(["initial_score_band_name", "activity_band_name"], dropna=False)["realistic_final_score"]
        .mean()
        .unstack()
        .reindex(index=["55-58", "59-61", "62-65"], columns=["<40", "40-59", ">=60"])
    )
    save_heatmap(
        pivot_cost,
        "现实优化版：不同初始积分—活动能力分层下的平均6个月总成本",
        "6个月总成本",
        FIG_DIR / "q3_realistic_group_avg_cost_heatmap.png",
    )
    save_heatmap(
        pivot_score,
        "现实优化版：不同初始积分—活动能力分层下的平均6个月末痰湿积分",
        "6个月末痰湿积分",
        FIG_DIR / "q3_realistic_group_avg_final_score_heatmap.png",
    )

    # 对比热力图：相对严格模型的平均成本节约与积分损失
    pivot_save = (
        result_df.groupby(["initial_score_band_name", "activity_band_name"], dropna=False)["cost_saving_vs_strict"]
        .mean()
        .unstack()
        .reindex(index=["55-58", "59-61", "62-65"], columns=["<40", "40-59", ">=60"])
    )
    pivot_gap = (
        result_df.groupby(["initial_score_band_name", "activity_band_name"], dropna=False)["score_gap_vs_strict"]
        .mean()
        .unstack()
        .reindex(index=["55-58", "59-61", "62-65"], columns=["<40", "40-59", ">=60"])
    )
    save_heatmap(
        pivot_save,
        "现实优化版相对严格模型的平均成本节约",
        "成本节约（元）",
        FIG_DIR / "q3_realistic_cost_saving_heatmap.png",
    )
    save_heatmap(
        pivot_gap,
        "现实优化版相对严格模型的平均积分损失",
        "积分损失",
        FIG_DIR / "q3_realistic_score_gap_heatmap.png",
    )

    # 摘要文本
    lines = []
    lines.append("问题三：痰湿体质患者6个月现实优化版干预结果摘要")
    lines.append("=" * 54)
    lines.append(f"输入数据文件：{csv_path}")
    lines.append(f"痰湿体质患者数量（体质标签=5）：{len(result_df)}")
    lines.append("")
    lines.append("一、现实优化版模型口径")
    lines.append("1. 先按严格模型求得可达到的最优6个月末痰湿积分 S7*。")
    lines.append(f"2. 允许现实容忍带：S7 <= S7* + {SCORE_TOLERANCE:.1f}。")
    lines.append("3. 在容忍带内，优先选择总成本更低的方案。")
    lines.append("4. 若成本相同，则优先选择总负担指数更低的方案。")
    lines.append("5. 总负担指数定义为 Σ[4 * 每周频次 * 强度权重]，其中强度1/2/3级权重为1.0/1.5/2.0。")
    lines.append("6. 这样可避免严格模型在疗效极致偏好下过度集中于高频高强度干预。")
    lines.append("")
    lines.append("二、总体结果（现实优化版）")
    lines.append(f"平均初始痰湿积分：{result_df['initial_score'].mean():.2f}")
    lines.append(f"平均6个月末痰湿积分：{result_df['realistic_final_score'].mean():.2f}")
    lines.append(f"平均6个月总成本：{result_df['realistic_total_cost'].mean():.2f} 元")
    lines.append(f"平均总负担指数：{result_df['realistic_total_burden'].mean():.2f}")
    lines.append("")
    lines.append("三、相对严格模型的平均变化")
    lines.append(f"平均积分损失（现实版 - 严格版）：{result_df['score_gap_vs_strict'].mean():.2f}")
    lines.append(f"平均成本节约（严格版 - 现实版）：{result_df['cost_saving_vs_strict'].mean():.2f} 元")
    lines.append(f"平均负担下降（严格版 - 现实版）：{result_df['burden_saving_vs_strict'].mean():.2f}")
    lines.append("")
    lines.append("四、典型匹配规律（现实优化版）")
    for _, r in feature_df.iterrows():
        lines.append(
            f"- 年龄{r['年龄分层']}、活动{r['活动能力分层']}、初始积分{r['初始痰湿积分分层']}："
            f"患者数={r['患者数']}，平均期末积分={r['平均期末痰湿积分']:.2f}，"
            f"平均总成本={r['平均6个月总成本']:.2f}，平均总负担={r['平均总负担指数']:.2f}，"
            f"相对严格模型平均积分损失={r['相对严格模型平均积分损失']:.2f}，"
            f"相对严格模型平均成本节约={r['相对严格模型平均成本节约']:.2f}，"
            f"相对严格模型平均负担下降={r['相对严格模型平均负担下降']:.2f}，"
            f"最常见首月强度={int(r['最常见首月强度'])}，最常见首月频率={int(r['最常见首月频率'])}，"
            f"最常见方案形态={r['最常见方案形态']}。"
        )
    lines.append("")
    lines.append("五、样本1/2/3文件说明（现实优化版）")
    for sid in [1, 2, 3]:
        row = result_df.loc[result_df["sample_id"] == sid]
        if not row.empty:
            row = row.iloc[0]
            lines.append(
                f"- 样本{sid}：严格版期末积分={row['strict_final_score']:.1f}、严格版总成本={row['strict_total_cost']:.2f}；"
                f"现实版期末积分={row['realistic_final_score']:.1f}、现实版总成本={row['realistic_total_cost']:.2f}、"
                f"现实版总负担={row['realistic_total_burden']:.2f}；"
                f"最优方案文件=q3_sample_{sid}_optimal_plan_realistic.csv"
            )

    (OUT_DIR / "q3_intervention_summary_realistic.txt").write_text("\n".join(lines), encoding="utf-8")

    # metadata
    metadata = {
        "input_csv": str(csv_path),
        "output_dir": str(OUT_DIR),
        "figure_dir": str(FIG_DIR),
        "model": "现实优化版：严格最优容忍带 + 成本/负担优先",
        "score_tolerance": SCORE_TOLERANCE,
        "budget_max": BUDGET_MAX,
        "activity_cost": ACTIVITY_COST,
        "tcm_cost": TCM_COST,
        "intensity_burden_weight": INTENSITY_BURDEN_WEIGHT,
        "n_patients": int(len(result_df)),
    }
    with open(OUT_DIR / "q3_intervention_metadata_realistic.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("现实优化版结果已输出：")
    print(f"- 表格目录：{OUT_DIR}")
    print(f"- 图片目录：{FIG_DIR}")


if __name__ == "__main__":
    main()
