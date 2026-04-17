from __future__ import annotations

from typing import Iterable, List, Dict, Any

import pandas as pd

from config import C_Q1_CSV, OUT_DIR, FIGURE_DIR, REFERENCE_DIR, ensure_project_dirs

# =========================
# 路径设置
# =========================
FILE_NAME = C_Q1_CSV.name
INPUT_PATH = C_Q1_CSV
REPORT_PATH = OUT_DIR / "c_q1_data_audit_report.txt"
ISSUES_PATH = OUT_DIR / "c_q1_data_audit_issues.csv"

# =========================
# 列名配置
# =========================
CONSTITUTION_SCORE_COLS = [
    "平和质", "气虚质", "阳虚质", "阴虚质", "痰湿质",
    "湿热质", "血瘀质", "气郁质", "特禀质"
]

ADL_ITEM_COLS = ["ADL用厕", "ADL吃饭", "ADL步行", "ADL穿衣", "ADL洗澡"]
IADL_ITEM_COLS = ["IADL购物", "IADL做饭", "IADL理财", "IADL交通", "IADL服药"]

ADL_TOTAL_COL = "ADL总分"
IADL_TOTAL_COL = "IADL总分"
ACTIVITY_TOTAL_COL = "活动量表总分（ADL总分+IADL总分）"

LAB_COLS = [
    "HDL-C（高密度脂蛋白）",
    "LDL-C（低密度脂蛋白）",
    "TG（甘油三酯）",
    "TC（总胆固醇）",
    "空腹血糖",
    "血尿酸",
    "BMI",
]

CODE_COL_RULES = {
    "体质标签": set(range(1, 10)),
    "高血脂症二分类标签": {0, 1},
    "血脂异常分型标签（确诊病例）": {0, 1, 2, 3},
    "年龄组": {1, 2, 3, 4, 5},
    "性别": {0, 1},
    "吸烟史": {0, 1},
    "饮酒史": {0, 1},
}

REQUIRED_COLUMNS = (
    ["样本ID", "体质标签"]
    + CONSTITUTION_SCORE_COLS
    + ADL_ITEM_COLS
    + [ADL_TOTAL_COL]
    + IADL_ITEM_COLS
    + [IADL_TOTAL_COL, ACTIVITY_TOTAL_COL]
    + LAB_COLS
    + ["高血脂症二分类标签", "血脂异常分型标签（确诊病例）", "年龄组", "性别", "吸烟史", "饮酒史"]
)


# =========================
# 工具函数
# =========================
def load_csv_with_fallback(path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise RuntimeError(f"无法读取文件: {path}\n最后一次报错: {last_error}")


def add_issue(issues: List[Dict[str, Any]], issue_type: str, sample_id: Any, row_index: Any,
              column: str, value: Any, detail: str) -> None:
    issues.append(
        {
            "issue_type": issue_type,
            "sample_id": sample_id,
            "row_index": row_index,
            "column": column,
            "value": value,
            "detail": detail,
        }
    )


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")


def convert_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def check_missing_and_duplicates(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    missing_summary = df.isna().sum()
    duplicate_count = int(df.duplicated().sum())

    na_locs = df[df.isna().any(axis=1)]
    for idx, row in na_locs.iterrows():
        sample_id = row.get("样本ID", None)
        for col in df.columns[df.loc[idx].isna()]:
            add_issue(issues, "missing_value", sample_id, idx, col, None, "该字段存在缺失值")

    dup_rows = df[df.duplicated(keep=False)]
    for idx, row in dup_rows.iterrows():
        add_issue(issues, "duplicate_row", row.get("样本ID", None), idx, "<row>", None, "该行与其他行完全重复")

    return {
        "missing_summary": missing_summary,
        "duplicate_count": duplicate_count,
    }


def check_numeric_ranges(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> None:
    for col in CONSTITUTION_SCORE_COLS:
        bad = df[(df[col] < 0) | (df[col] > 100)]
        for idx, row in bad.iterrows():
            add_issue(issues, "invalid_range", row["样本ID"], idx, col, row[col], "体质积分应在 0-100 之间")

    for col in ADL_ITEM_COLS + IADL_ITEM_COLS:
        bad = df[(df[col] < 0) | (df[col] > 10)]
        for idx, row in bad.iterrows():
            add_issue(issues, "invalid_range", row["样本ID"], idx, col, row[col], "ADL/IADL 单项应在 0-10 之间")

    total_rules = {
        ADL_TOTAL_COL: (0, 50, "ADL总分应在 0-50 之间"),
        IADL_TOTAL_COL: (0, 50, "IADL总分应在 0-50 之间"),
        ACTIVITY_TOTAL_COL: (0, 100, "活动量表总分应在 0-100 之间"),
    }
    for col, (low, high, msg) in total_rules.items():
        bad = df[(df[col] < low) | (df[col] > high)]
        for idx, row in bad.iterrows():
            add_issue(issues, "invalid_range", row["样本ID"], idx, col, row[col], msg)

    positive_cols = [
        "HDL-C（高密度脂蛋白）",
        "LDL-C（低密度脂蛋白）",
        "TG（甘油三酯）",
        "TC（总胆固醇）",
        "空腹血糖",
        "血尿酸",
        "BMI",
    ]
    for col in positive_cols:
        bad = df[df[col] <= 0]
        for idx, row in bad.iterrows():
            add_issue(issues, "hard_invalid_value", row["样本ID"], idx, col, row[col], "该数值按定义应大于 0")


def check_code_columns(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> None:
    for col, valid_set in CODE_COL_RULES.items():
        bad = df[~df[col].isin(valid_set)]
        for idx, row in bad.iterrows():
            add_issue(issues, "invalid_code", row["样本ID"], idx, col, row[col], f"合法取值应为 {sorted(valid_set)}")


def check_logic_consistency(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> Dict[str, int]:
    adl_sum = df[ADL_ITEM_COLS].sum(axis=1)
    bad_adl = df[adl_sum != df[ADL_TOTAL_COL]]
    for idx, row in bad_adl.iterrows():
        add_issue(
            issues,
            "logic_inconsistency",
            row["样本ID"],
            idx,
            ADL_TOTAL_COL,
            row[ADL_TOTAL_COL],
            f"ADL五项之和={adl_sum.loc[idx]}，但记录总分={row[ADL_TOTAL_COL]}",
        )

    iadl_sum = df[IADL_ITEM_COLS].sum(axis=1)
    bad_iadl = df[iadl_sum != df[IADL_TOTAL_COL]]
    for idx, row in bad_iadl.iterrows():
        add_issue(
            issues,
            "logic_inconsistency",
            row["样本ID"],
            idx,
            IADL_TOTAL_COL,
            row[IADL_TOTAL_COL],
            f"IADL五项之和={iadl_sum.loc[idx]}，但记录总分={row[IADL_TOTAL_COL]}",
        )

    activity_sum = df[ADL_TOTAL_COL] + df[IADL_TOTAL_COL]
    bad_activity = df[activity_sum != df[ACTIVITY_TOTAL_COL]]
    for idx, row in bad_activity.iterrows():
        add_issue(
            issues,
            "logic_inconsistency",
            row["样本ID"],
            idx,
            ACTIVITY_TOTAL_COL,
            row[ACTIVITY_TOTAL_COL],
            f"ADL总分+IADL总分={activity_sum.loc[idx]}，但记录活动总分={row[ACTIVITY_TOTAL_COL]}",
        )

    bad_subtype_0 = df[(df["高血脂症二分类标签"] == 0) & (df["血脂异常分型标签（确诊病例）"] != 0)]
    for idx, row in bad_subtype_0.iterrows():
        add_issue(
            issues,
            "logic_inconsistency",
            row["样本ID"],
            idx,
            "血脂异常分型标签（确诊病例）",
            row["血脂异常分型标签（确诊病例）"],
            "未确诊病例的血脂异常分型应为 0",
        )

    bad_subtype_1 = df[(df["高血脂症二分类标签"] == 1) & (~df["血脂异常分型标签（确诊病例）"].isin([1, 2, 3]))]
    for idx, row in bad_subtype_1.iterrows():
        add_issue(
            issues,
            "logic_inconsistency",
            row["样本ID"],
            idx,
            "血脂异常分型标签（确诊病例）",
            row["血脂异常分型标签（确诊病例）"],
            "确诊病例的血脂异常分型应为 1/2/3",
        )

    return {
        "adl_sum_mismatch": int((adl_sum != df[ADL_TOTAL_COL]).sum()),
        "iadl_sum_mismatch": int((iadl_sum != df[IADL_TOTAL_COL]).sum()),
        "activity_sum_mismatch": int((activity_sum != df[ACTIVITY_TOTAL_COL]).sum()),
        "subtype_mismatch_when_negative": int(len(bad_subtype_0)),
        "subtype_mismatch_when_positive": int(len(bad_subtype_1)),
    }


def build_report(df: pd.DataFrame, base_result: Dict[str, Any], logic_result: Dict[str, int], issues_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("C题 问题一：数据读入、基础检查与逻辑一致性检验报告")
    lines.append("=" * 72)
    lines.append(f"输入文件: {INPUT_PATH}")
    lines.append(f"样本量: {df.shape[0]}")
    lines.append(f"字段数: {df.shape[1]}")
    lines.append("")

    lines.append("[1] 字段类型")
    for col, dtype in df.dtypes.items():
        lines.append(f"- {col}: {dtype}")
    lines.append("")

    lines.append("[2] 缺失值统计")
    missing_summary = base_result["missing_summary"]
    total_missing = int(missing_summary.sum())
    lines.append(f"- 全表缺失值总数: {total_missing}")
    for col, count in missing_summary.items():
        lines.append(f"  - {col}: {int(count)}")
    lines.append("")

    lines.append("[3] 重复值统计")
    lines.append(f"- 完全重复行数: {base_result['duplicate_count']}")
    lines.append("")

    lines.append("[4] 逻辑一致性检查")
    lines.append(f"- ADL 五项求和与 ADL总分 不一致条数: {logic_result['adl_sum_mismatch']}")
    lines.append(f"- IADL 五项求和与 IADL总分 不一致条数: {logic_result['iadl_sum_mismatch']}")
    lines.append(f"- ADL总分 + IADL总分 与 活动量表总分 不一致条数: {logic_result['activity_sum_mismatch']}")
    lines.append(f"- 未确诊但分型标签非 0 的条数: {logic_result['subtype_mismatch_when_negative']}")
    lines.append(f"- 已确诊但分型标签不在 1/2/3 的条数: {logic_result['subtype_mismatch_when_positive']}")
    lines.append("")

    lines.append("[5] 问题记录统计")
    if issues_df.empty:
        lines.append("- 未发现需要进一步处理的问题。")
    else:
        issue_counts = issues_df["issue_type"].value_counts()
        for issue_type, count in issue_counts.items():
            lines.append(f"- {issue_type}: {int(count)}")
    lines.append("")

    lines.append("[6] 说明")
    lines.append("- 本脚本只做‘读入 + 检查 + 逻辑一致性检验’，暂不做临床异常值删除或修正。")
    lines.append("- 像 BMI 很低、血糖偏低、TG/TC 偏高或偏低这类值，后续建议做‘保留原值 + 异常标记’，而不是在本阶段直接删除。")
    lines.append("")

    return "\n".join(lines)


# =========================
# 主程序
# =========================
def main() -> None:
    ensure_project_dirs()
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"未找到输入文件: {INPUT_PATH}\n"
            f"请确认你已经把文件放到 raw 文件夹下，并且文件名为: {FILE_NAME}"
        )

    print(f"[INFO] 正在读取文件: {INPUT_PATH}")
    df = load_csv_with_fallback(INPUT_PATH)

    ensure_required_columns(df)
    df = convert_numeric_columns(df, REQUIRED_COLUMNS)

    issues: List[Dict[str, Any]] = []

    base_result = check_missing_and_duplicates(df, issues)
    check_numeric_ranges(df, issues)
    check_code_columns(df, issues)
    logic_result = check_logic_consistency(df, issues)

    issues_df = pd.DataFrame(issues)
    if issues_df.empty:
        issues_df = pd.DataFrame(columns=["issue_type", "sample_id", "row_index", "column", "value", "detail"])
    else:
        issues_df = issues_df.sort_values(by=["issue_type", "row_index", "column"], kind="stable").reset_index(drop=True)

    report_text = build_report(df, base_result, logic_result, issues_df)

    issues_df.to_csv(ISSUES_PATH, index=False, encoding="utf-8-sig")
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print(f"\n[INFO] 检查报告已保存到: {REPORT_PATH}")
    print(f"[INFO] 问题明细已保存到: {ISSUES_PATH}")


if __name__ == "__main__":
    main()
