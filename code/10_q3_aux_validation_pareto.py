
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


# ============================================================
# 问题三辅助验证版（对应“妈妈杯问题3(1)”与“豆包帕累托思路”）
# 口径说明：
# 1) 不覆盖主模型代码；单独输出到 q3_aux_validation_pareto/
# 2) 采用“完整帕累托前沿 + 三类代表方案”作为辅助验证
# 3) 为了和主模型可比，默认沿用主模型的下降率口径：
#       f < 5  -> r = 0
#       f >= 5 -> r = 3%*(k-1) + 1%*(f-5)
#    若要改为上传思路中更“字面”的口径，可把 RATE_MODE 改为 "literal"
# ============================================================

RATE_MODE = "main_consistent"   # "main_consistent" or "literal"
BUDGET_CAP = 2000.0
MONTHS = 6
WEEKS_PER_MONTH = 4
ROUND_SCORE = 1     # 1 表示保留 1 位小数
TOP_SAMPLE_IDS = [1, 2, 3]

# 负担权重（仅用于代表方案筛选，不改变帕累托定义）
BURDEN_WEIGHT = {1: 1.0, 2: 1.5, 3: 2.0}


def get_chinese_font() -> FontProperties:
    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for fp in font_candidates:
        if Path(fp).exists():
            return FontProperties(fname=fp)
    return FontProperties()  # fallback


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

OUT_SUBDIR = OUT_DIR / "q3_aux_validation_pareto"
FIG_SUBDIR = FIGURE_DIR / "q3_aux_validation_pareto"
OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
FIG_SUBDIR.mkdir(parents=True, exist_ok=True)


def round_score(x: float) -> float:
    return round(float(x), ROUND_SCORE)


def activity_session_cost(level: int) -> float:
    return {1: 3.0, 2: 5.0, 3: 8.0}[int(level)]


def tcm_level(score: float) -> int:
    if score <= 58:
        return 1
    elif score <= 61:
        return 2
    else:
        return 3


def tcm_month_cost(score: float) -> float:
    return {1: 30.0, 2: 80.0, 3: 130.0}[tcm_level(score)]


def monthly_drop_rate(level: int, freq_per_week: int) -> float:
    if freq_per_week < 5:
        return 0.0
    if RATE_MODE == "literal":
        r = 0.03 * level + 0.01 * (freq_per_week - 5)
    else:
        r = 0.03 * (level - 1) + 0.01 * (freq_per_week - 5)
    return max(0.0, min(r, 0.95))


def next_score(score: float, level: int, freq_per_week: int) -> float:
    r = monthly_drop_rate(level, freq_per_week)
    return round_score(score * (1.0 - r))


def age_max_strength(age_name: str) -> int:
    s = str(age_name)
    if "80" in s:
        return 1
    if "60" in s or "70" in s:
        return 2
    return 3


def activity_max_strength(total_score: float) -> int:
    if total_score < 40:
        return 1
    if total_score < 60:
        return 2
    return 3


def allowed_levels(age_name: str, total_score: float) -> List[int]:
    max_level = min(age_max_strength(age_name), activity_max_strength(total_score))
    return list(range(1, max_level + 1))


def burden_of_action(level: int, freq_per_week: int) -> float:
    return WEEKS_PER_MONTH * freq_per_week * BURDEN_WEIGHT[int(level)]


def plan_shape(levels: List[int], freqs: List[int]) -> str:
    if len(levels) == 0:
        return "未知"
    if len(set(levels)) == 1 and len(set(freqs)) == 1:
        return "稳定维持"
    if levels[0] <= levels[-1] and freqs[0] <= freqs[-1] and (levels[0] < levels[-1] or freqs[0] < freqs[-1]):
        return "前低后高"
    if levels[0] >= levels[-1] and freqs[0] >= freqs[-1] and (levels[0] > levels[-1] or freqs[0] > freqs[-1]):
        return "前高后低"
    return "动态调整"


def extract_score_bin(score: float) -> str:
    if score <= 58:
        return "55-58"
    if score <= 61:
        return "59-61"
    return "62-65"


def extract_activity_bin(total_score: float) -> str:
    if total_score < 40:
        return "<40"
    if total_score < 60:
        return "40-59"
    return ">=60"


def dominates(a: Dict, b: Dict) -> bool:
    """双目标：min(final_score), min(total_cost)"""
    better_or_equal = (a["final_score"] <= b["final_score"]) and (a["total_cost"] <= b["total_cost"])
    strictly_better = (a["final_score"] < b["final_score"]) or (a["total_cost"] < b["total_cost"])
    return better_or_equal and strictly_better


def pareto_filter(points: List[Dict]) -> List[Dict]:
    if not points:
        return []
    # 先按 final_score, total_cost 排序
    pts = sorted(points, key=lambda x: (x["final_score"], x["total_cost"], x["burden"], x["total_freq"]))
    kept: List[Dict] = []
    best_cost = math.inf
    for p in pts:
        if p["total_cost"] < best_cost - 1e-9:
            kept.append(p)
            best_cost = p["total_cost"]
    return kept


def choose_representatives(front: List[Dict]) -> Dict[str, Dict]:
    if not front:
        return {}
    low_cost = min(front, key=lambda x: (x["total_cost"], x["final_score"], x["burden"], x["total_freq"]))
    best_effect = min(front, key=lambda x: (x["final_score"], x["total_cost"], x["burden"], x["total_freq"]))

    costs = np.array([p["total_cost"] for p in front], dtype=float)
    scores = np.array([p["final_score"] for p in front], dtype=float)
    burdens = np.array([p["burden"] for p in front], dtype=float)

    def norm(arr):
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    c_norm = norm(costs)
    s_norm = norm(scores)
    b_norm = norm(burdens)

    # 均衡方案：成本/疗效/负担三者都不过于极端
    balanced_idx = np.argmin(c_norm**2 + s_norm**2 + 0.25 * b_norm**2)
    balanced = front[int(balanced_idx)]

    return {
        "lowest_cost": low_cost,
        "best_effect": best_effect,
        "balanced": balanced,
    }


def run_exact_pareto_validation(initial_score: float, age_name: str, total_score: float) -> Tuple[List[Dict], Dict[str, Dict]]:
    legal_levels = allowed_levels(age_name, total_score)
    actions = [(lvl, f) for lvl in legal_levels for f in range(1, 11)]

    # dp[t][score] = best partial path for this month/state in terms of accumulated cost
    frontier: Dict[float, Dict] = {
        round_score(initial_score): {
            "cost": 0.0,
            "burden": 0.0,
            "total_freq": 0,
            "trace": [],
        }
    }

    for month in range(1, MONTHS + 1):
        nxt: Dict[float, Dict] = {}
        for score_now, partial in frontier.items():
            score_now = float(score_now)
            tcm_cost_now = tcm_month_cost(score_now)
            tcm_level_now = tcm_level(score_now)

            for lvl, freq in actions:
                act_cost = WEEKS_PER_MONTH * freq * activity_session_cost(lvl)
                total_cost = partial["cost"] + tcm_cost_now + act_cost
                if total_cost > BUDGET_CAP + 1e-9:
                    continue

                score_next = next_score(score_now, lvl, freq)
                total_burden = partial["burden"] + burden_of_action(lvl, freq)
                total_freq = partial["total_freq"] + freq
                step = {
                    "month": month,
                    "month_start_score": score_now,
                    "tcm_level": tcm_level_now,
                    "tcm_cost": tcm_cost_now,
                    "exercise_level": lvl,
                    "freq_per_week": freq,
                    "exercise_cost": act_cost,
                    "month_total_cost": tcm_cost_now + act_cost,
                    "cumulative_cost": total_cost,
                    "month_end_score": score_next,
                }

                candidate = {
                    "cost": total_cost,
                    "burden": total_burden,
                    "total_freq": total_freq,
                    "trace": partial["trace"] + [step],
                }

                prev = nxt.get(score_next)
                if prev is None:
                    nxt[score_next] = candidate
                else:
                    # 同一状态、同一月份，未来可行空间相同，因此只保留成本更低者；
                    # 若成本相同，保留负担更低者。
                    if (candidate["cost"] < prev["cost"] - 1e-9) or (
                        abs(candidate["cost"] - prev["cost"]) <= 1e-9 and candidate["burden"] < prev["burden"] - 1e-9
                    ) or (
                        abs(candidate["cost"] - prev["cost"]) <= 1e-9 and
                        abs(candidate["burden"] - prev["burden"]) <= 1e-9 and
                        candidate["total_freq"] < prev["total_freq"]
                    ):
                        nxt[score_next] = candidate
        frontier = nxt

    all_terminal = []
    for final_score, partial in frontier.items():
        plan = partial["trace"]
        levels = [x["exercise_level"] for x in plan]
        freqs = [x["freq_per_week"] for x in plan]
        all_terminal.append({
            "final_score": float(final_score),
            "total_cost": round(float(partial["cost"]), 2),
            "burden": round(float(partial["burden"]), 2),
            "total_freq": int(partial["total_freq"]),
            "shape": plan_shape(levels, freqs),
            "trace": plan,
        })

    pareto = pareto_filter(all_terminal)
    reps = choose_representatives(pareto)
    return pareto, reps


def main():
    csv_path = OUT_DIR / "c_q1_main_simple.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到清洗后的主分析表：{csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    col_id = find_column(df, ["样本ID", "样本 Id", "样本id"])
    col_const_label = find_column(df, ["体质标签"])
    col_phlegm_score = find_column(df, ["痰湿质"])
    col_age_name = find_column(df, ["年龄组名称", "年龄组"])
    col_activity_total = find_column(
        df,
        [
            "活动量表总分",
            "活动总分",
            "ADL总分+IADL总分",
            "ADL总分＋IADL总分",
            "活动量表总分（ADL总分+IADL总分）",
        ],
    )

    work = df[df[col_const_label] == 5].copy().reset_index(drop=True)
    work[col_phlegm_score] = pd.to_numeric(work[col_phlegm_score], errors="coerce")
    work[col_activity_total] = pd.to_numeric(work[col_activity_total], errors="coerce")
    work = work.dropna(subset=[col_id, col_phlegm_score, col_activity_total])

    patient_rows = []
    monthly_rows = []
    pareto_rows = []

    for _, row in work.iterrows():
        pid = int(row[col_id])
        s1 = float(row[col_phlegm_score])
        age_name = str(row[col_age_name])
        act_total = float(row[col_activity_total])

        pareto, reps = run_exact_pareto_validation(s1, age_name, act_total)
        if len(pareto) == 0:
            continue

        rep_low = reps["lowest_cost"]
        rep_best = reps["best_effect"]
        rep_bal = reps["balanced"]

        patient_rows.append({
            "样本ID": pid,
            "年龄组名称": age_name,
            "活动量表总分": act_total,
            "初始痰湿积分": s1,
            "帕累托点数量": len(pareto),
            "疗效最优_期末痰湿积分": rep_best["final_score"],
            "疗效最优_总成本": rep_best["total_cost"],
            "疗效最优_总负担": rep_best["burden"],
            "疗效最优_总频次": rep_best["total_freq"],
            "疗效最优_方案形态": rep_best["shape"],
            "最低成本_期末痰湿积分": rep_low["final_score"],
            "最低成本_总成本": rep_low["total_cost"],
            "最低成本_总负担": rep_low["burden"],
            "最低成本_总频次": rep_low["total_freq"],
            "最低成本_方案形态": rep_low["shape"],
            "均衡方案_期末痰湿积分": rep_bal["final_score"],
            "均衡方案_总成本": rep_bal["total_cost"],
            "均衡方案_总负担": rep_bal["burden"],
            "均衡方案_总频次": rep_bal["total_freq"],
            "均衡方案_方案形态": rep_bal["shape"],
        })

        for ptype, p in [("疗效最优", rep_best), ("最低成本", rep_low), ("均衡方案", rep_bal)]:
            for step in p["trace"]:
                monthly_rows.append({
                    "样本ID": pid,
                    "方案类型": ptype,
                    **step
                })

        for p in pareto:
            pareto_rows.append({
                "样本ID": pid,
                "初始痰湿积分": s1,
                "年龄组名称": age_name,
                "活动量表总分": act_total,
                "活动能力分层": extract_activity_bin(act_total),
                "初始积分分层": extract_score_bin(s1),
                "期末痰湿积分": p["final_score"],
                "6个月总成本": p["total_cost"],
                "总负担指数": p["burden"],
                "总训练频次": p["total_freq"],
                "方案形态": p["shape"],
            })

        # 样本 1/2/3 单独作图
        if pid in TOP_SAMPLE_IDS:
            front_df = pd.DataFrame([{
                "final_score": p["final_score"],
                "total_cost": p["total_cost"],
            } for p in pareto])

            fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
            ax.scatter(front_df["total_cost"], front_df["final_score"], s=55, alpha=0.85, label="帕累托前沿")
            ax.scatter(rep_low["total_cost"], rep_low["final_score"], s=220, marker="s", label="最低成本")
            ax.scatter(rep_best["total_cost"], rep_best["final_score"], s=240, marker="*", label="疗效最优")
            ax.scatter(rep_bal["total_cost"], rep_bal["final_score"], s=220, marker="D", label="均衡方案")

            apply_zh_font(
                ax,
                title=f"样本 {pid} 的帕累托辅助验证前沿",
                xlabel="6个月总成本（元）",
                ylabel="6个月末痰湿积分",
            )
            ax.legend(prop=ZH_FONT)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIG_SUBDIR / f"q3_sample_{pid}_pareto_validation.png", bbox_inches="tight")
            plt.close()

    patient_df = pd.DataFrame(patient_rows).sort_values("样本ID").reset_index(drop=True)
    monthly_df = pd.DataFrame(monthly_rows)
    pareto_df = pd.DataFrame(pareto_rows)

    patient_df.to_csv(OUT_SUBDIR / "q3_patient_pareto_summary_validation.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(OUT_SUBDIR / "q3_patient_pareto_representative_plans_validation.csv", index=False, encoding="utf-8-sig")
    pareto_df.to_csv(OUT_SUBDIR / "q3_patient_pareto_points_validation.csv", index=False, encoding="utf-8-sig")

    # 分组规律：这里用“均衡方案”汇总，作为上传思路中的“代表性方案推荐”落地
    group_df = patient_df.copy()
    group_df["初始积分分层"] = group_df["初始痰湿积分"].map(extract_score_bin)
    group_df["活动能力分层"] = group_df["活动量表总分"].map(extract_activity_bin)

    feature_rule = (
        group_df.groupby(["年龄组名称", "活动能力分层", "初始积分分层"], dropna=False)
        .agg(
            患者数=("样本ID", "size"),
            平均帕累托点数量=("帕累托点数量", "mean"),
            平均均衡方案期末积分=("均衡方案_期末痰湿积分", "mean"),
            平均均衡方案总成本=("均衡方案_总成本", "mean"),
            平均疗效最优期末积分=("疗效最优_期末痰湿积分", "mean"),
            平均疗效最优总成本=("疗效最优_总成本", "mean"),
            平均最低成本期末积分=("最低成本_期末痰湿积分", "mean"),
            平均最低成本总成本=("最低成本_总成本", "mean"),
            最常见均衡方案形态=("均衡方案_方案形态", lambda x: x.mode().iat[0] if len(x.mode()) else x.iloc[0]),
        )
        .reset_index()
    )
    feature_rule.to_csv(OUT_SUBDIR / "q3_feature_rule_summary_validation.csv", index=False, encoding="utf-8-sig")

    # 两张热力图：均衡方案下的平均成本、平均期末积分
    for value_col, fname, title, cbar_label in [
        ("平均均衡方案总成本", "q3_validation_group_avg_cost_heatmap.png",
         "帕累托辅助验证：均衡方案平均6个月总成本", "6个月总成本"),
        ("平均均衡方案期末积分", "q3_validation_group_avg_final_score_heatmap.png",
         "帕累托辅助验证：均衡方案平均6个月末痰湿积分", "6个月末痰湿积分"),
    ]:
        pivot = feature_rule.pivot_table(
            index="初始积分分层",
            columns="活动能力分层",
            values=value_col,
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=["55-58", "59-61", "62-65"], columns=["<40", "40-59", ">=60"])

        fig, ax = plt.subplots(figsize=(10, 7), dpi=220)
        im = ax.imshow(pivot.values, aspect="auto")
        apply_zh_font(ax, title=title, xlabel="活动能力分层", ylabel="初始痰湿积分分层")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            color="white", fontproperties=ZH_FONT, fontsize=13)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(cbar_label, fontproperties=ZH_FONT, rotation=90, labelpad=14)
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(ZH_FONT)
        plt.tight_layout()
        plt.savefig(FIG_SUBDIR / fname, bbox_inches="tight")
        plt.close()

    # 汇总文本
    avg_front_size = float(patient_df["帕累托点数量"].mean()) if len(patient_df) else 0.0
    avg_best_score = float(patient_df["疗效最优_期末痰湿积分"].mean()) if len(patient_df) else 0.0
    avg_best_cost = float(patient_df["疗效最优_总成本"].mean()) if len(patient_df) else 0.0
    avg_low_score = float(patient_df["最低成本_期末痰湿积分"].mean()) if len(patient_df) else 0.0
    avg_low_cost = float(patient_df["最低成本_总成本"].mean()) if len(patient_df) else 0.0
    avg_bal_score = float(patient_df["均衡方案_期末痰湿积分"].mean()) if len(patient_df) else 0.0
    avg_bal_cost = float(patient_df["均衡方案_总成本"].mean()) if len(patient_df) else 0.0

    lines = []
    lines.append("问题三辅助验证：完整帕累托前沿 + 三类代表方案\n")
    lines.append("============================================\n")
    lines.append(f"输入数据文件：{csv_path}\n")
    lines.append(f"痰湿体质患者数量（体质标签=5）：{len(work)}\n\n")
    lines.append("一、辅助验证口径\n")
    lines.append("1. 仅针对体质标签=5的患者求解。\n")
    lines.append("2. 采用 6 阶段前向动态规划，但不做字典序塌缩，保留全部终点帕累托前沿。\n")
    lines.append("3. 双目标定义：最小化6个月末痰湿积分、最小化6个月总成本。\n")
    lines.append("4. 每月状态相同则仅保留最小成本路径，因此该实现是精确帕累托辅助验证，不是启发式近似。\n")
    lines.append(f"5. 下降率口径：RATE_MODE = {RATE_MODE}。\n\n")
    lines.append("二、总体结果\n")
    lines.append(f"平均帕累托点数量：{avg_front_size:.2f}\n")
    lines.append(f"平均疗效最优期末积分：{avg_best_score:.2f}，平均疗效最优总成本：{avg_best_cost:.2f} 元\n")
    lines.append(f"平均最低成本期末积分：{avg_low_score:.2f}，平均最低成本总成本：{avg_low_cost:.2f} 元\n")
    lines.append(f"平均均衡方案期末积分：{avg_bal_score:.2f}，平均均衡方案总成本：{avg_bal_cost:.2f} 元\n\n")
    lines.append("三、文件说明\n")
    lines.append("- q3_patient_pareto_summary_validation.csv：每位患者的帕累托代表性方案汇总\n")
    lines.append("- q3_patient_pareto_representative_plans_validation.csv：每位患者三类代表方案的逐月明细\n")
    lines.append("- q3_patient_pareto_points_validation.csv：每位患者完整帕累托点集\n")
    lines.append("- q3_feature_rule_summary_validation.csv：分组后的匹配规律汇总（基于均衡方案）\n")
    lines.append("- q3_sample_1/2/3_pareto_validation.png：样本1/2/3帕累托辅助验证图\n")
    lines.append("- q3_validation_group_avg_cost_heatmap.png / q3_validation_group_avg_final_score_heatmap.png：均衡方案分组热力图\n")

    with open(OUT_SUBDIR / "q3_aux_validation_summary.txt", "w", encoding="utf-8-sig") as f:
        f.writelines(lines)

    metadata = {
        "rate_mode": RATE_MODE,
        "budget_cap": BUDGET_CAP,
        "months": MONTHS,
        "round_score_digits": ROUND_SCORE,
        "weeks_per_month": WEEKS_PER_MONTH,
        "source_file": str(csv_path),
        "patient_count": int(len(work)),
    }
    with open(OUT_SUBDIR / "q3_aux_validation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("第三问辅助验证版（完整帕累托前沿）已运行完成。")
    print(f"输出目录：{OUT_SUBDIR}")
    print(f"图像目录：{FIG_SUBDIR}")


if __name__ == "__main__":
    main()
