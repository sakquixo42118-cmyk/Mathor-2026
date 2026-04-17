from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


# =========================
# 0. 路径配置
# =========================
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
RAW_DIR = PROJECT_ROOT / "raw"
OUT_DIR = PROJECT_ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = RAW_DIR / "C题：附件1：样例数据.csv"
RAW_XLSX = RAW_DIR / "C题：附件1：样例数据.xlsx"


# =========================
# 1. 常量配置
# =========================
CONSTITUTION_MAP: Dict[int, str] = {
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

AGE_GROUP_MAP: Dict[int, str] = {
    1: "40-49岁",
    2: "50-59岁",
    3: "60-69岁",
    4: "70-79岁",
    5: "80-89岁",
}

SEX_MAP: Dict[int, str] = {0: "女", 1: "男"}
HISTORY_MAP: Dict[int, str] = {0: "无", 1: "有"}
LIPID_TYPE_MAP: Dict[int, str] = {
    0: "未确诊/无分型",
    1: "高胆固醇型",
    2: "高甘油三酯型",
    3: "混合型",
}

CORE_COLUMNS: List[str] = [
    "样本ID",
    "体质标签",
    "平和质",
    "气虚质",
    "阳虚质",
    "阴虚质",
    "痰湿质",
    "湿热质",
    "血瘀质",
    "气郁质",
    "特禀质",
    "ADL用厕",
    "ADL吃饭",
    "ADL步行",
    "ADL穿衣",
    "ADL洗澡",
    "ADL总分",
    "IADL购物",
    "IADL做饭",
    "IADL理财",
    "IADL交通",
    "IADL服药",
    "IADL总分",
    "活动量表总分（ADL总分+IADL总分）",
    "HDL-C（高密度脂蛋白）",
    "LDL-C（低密度脂蛋白）",
    "TG（甘油三酯）",
    "TC（总胆固醇）",
    "空腹血糖",
    "血尿酸",
    "BMI",
    "高血脂症二分类标签",
    "血脂异常分型标签（确诊病例）",
    "年龄组",
    "性别",
    "吸烟史",
    "饮酒史",
]

NUMERIC_COLUMNS: List[str] = CORE_COLUMNS.copy()


# =========================
# 2. 读取数据
# =========================
def find_input_file() -> Path:
    if RAW_CSV.exists():
        return RAW_CSV
    if RAW_XLSX.exists():
        return RAW_XLSX
    raise FileNotFoundError(
        "未在 raw 文件夹中找到 C题附件1 数据文件。\n"
        f"已检查：\n- {RAW_CSV}\n- {RAW_XLSX}"
    )


def read_input_data(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"不支持的文件类型：{file_path.suffix}")


# =========================
# 3. 构造变量
# =========================
def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["体质名称"] = df["体质标签"].map(CONSTITUTION_MAP)
    df["年龄组名称"] = df["年龄组"].map(AGE_GROUP_MAP)
    df["性别名称"] = df["性别"].map(SEX_MAP)
    df["吸烟史名称"] = df["吸烟史"].map(HISTORY_MAP)
    df["饮酒史名称"] = df["饮酒史"].map(HISTORY_MAP)
    df["血脂异常分型名称"] = df["血脂异常分型标签（确诊病例）"].map(LIPID_TYPE_MAP)
    return df


def add_clinical_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 题面给定的正常范围异常标记
    df["TC异常"] = (df["TC（总胆固醇）"] < 3.1) | (df["TC（总胆固醇）"] > 6.2)
    df["TG异常"] = (df["TG（甘油三酯）"] < 0.56) | (df["TG（甘油三酯）"] > 1.7)
    df["LDL异常"] = (df["LDL-C（低密度脂蛋白）"] < 2.07) | (df["LDL-C（低密度脂蛋白）"] > 3.1)
    df["HDL异常"] = (df["HDL-C（高密度脂蛋白）"] < 1.04) | (df["HDL-C（高密度脂蛋白）"] > 1.55)
    df["血糖异常"] = (df["空腹血糖"] < 3.9) | (df["空腹血糖"] > 6.1)
    df["BMI异常"] = (df["BMI"] < 18.5) | (df["BMI"] > 23.9)

    male_mask = df["性别"] == 1
    female_mask = df["性别"] == 0
    df["尿酸异常"] = False
    df.loc[male_mask, "尿酸异常"] = (df.loc[male_mask, "血尿酸"] < 208) | (df.loc[male_mask, "血尿酸"] > 428)
    df.loc[female_mask, "尿酸异常"] = (df.loc[female_mask, "血尿酸"] < 155) | (df.loc[female_mask, "血尿酸"] > 357)

    df["血脂异常项数"] = df[["TC异常", "TG异常", "LDL异常", "HDL异常"]].astype(int).sum(axis=1)
    df["代谢异常项数"] = df[["血糖异常", "尿酸异常", "BMI异常"]].astype(int).sum(axis=1)
    df["是否存在任一血脂异常"] = (df["血脂异常项数"] >= 1).astype(int)

    return df


def add_activity_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total_col = "活动量表总分（ADL总分+IADL总分）"

    def map_band(x: float) -> str:
        if x < 40:
            return "低活动能力(<40)"
        if x < 60:
            return "中活动能力(40-59)"
        return "高活动能力(>=60)"

    df["活动能力分层"] = df[total_col].apply(map_band)
    return df


# =========================
# 4. 输出结果
# =========================
def build_variable_dictionary() -> pd.DataFrame:
    rows = [
        ["体质名称", "由体质标签映射得到", "类别"],
        ["年龄组名称", "由年龄组映射得到", "类别"],
        ["性别名称", "由性别映射得到", "类别"],
        ["吸烟史名称", "由吸烟史映射得到", "类别"],
        ["饮酒史名称", "由饮酒史映射得到", "类别"],
        ["血脂异常分型名称", "由血脂异常分型标签映射得到", "类别"],
        ["TC异常", "TC 是否超出题面正常范围[3.1, 6.2]", "布尔"],
        ["TG异常", "TG 是否超出题面正常范围[0.56, 1.7]", "布尔"],
        ["LDL异常", "LDL-C 是否超出题面正常范围[2.07, 3.1]", "布尔"],
        ["HDL异常", "HDL-C 是否超出题面正常范围[1.04, 1.55]", "布尔"],
        ["血糖异常", "空腹血糖是否超出题面正常范围[3.9, 6.1]", "布尔"],
        ["尿酸异常", "按性别分别判断血尿酸是否超出正常范围", "布尔"],
        ["BMI异常", "BMI 是否超出题面正常范围[18.5, 23.9]", "布尔"],
        ["血脂异常项数", "TC/TG/LDL/HDL 异常个数", "整数"],
        ["代谢异常项数", "血糖/尿酸/BMI 异常个数", "整数"],
        ["是否存在任一血脂异常", "血脂异常项数是否>=1", "整数(0/1)"],
        ["活动能力分层", "依据活动量表总分划分为低/中/高活动能力", "类别"],
    ]
    return pd.DataFrame(rows, columns=["变量名", "含义", "类型"])


def build_summary_text(df: pd.DataFrame, input_file: Path) -> str:
    lines: List[str] = []
    lines.append("问题一主分析表构建摘要（简化版）")
    lines.append("=" * 36)
    lines.append(f"输入文件：{input_file}")
    lines.append(f"样本量：{len(df)}")
    lines.append(f"字段数：{df.shape[1]}")
    lines.append("")

    lines.append("一、标签分布")
    lines.append("-" * 36)
    lines.append("高血脂症二分类标签：")
    lines.append(df["高血脂症二分类标签"].value_counts(dropna=False).sort_index().to_string())
    lines.append("")
    lines.append("体质标签：")
    lines.append(df["体质名称"].value_counts(dropna=False).to_string())
    lines.append("")

    lines.append("二、异常标记计数")
    lines.append("-" * 36)
    abnormal_cols = [
        "TC异常", "TG异常", "LDL异常", "HDL异常",
        "血糖异常", "尿酸异常", "BMI异常",
        "是否存在任一血脂异常",
    ]
    for col in abnormal_cols:
        lines.append(f"{col}: {int(df[col].astype(int).sum())}")
    lines.append("")

    lines.append("三、连续变量描述统计（核心字段）")
    lines.append("-" * 36)
    desc_cols = [
        "痰湿质",
        "ADL总分",
        "IADL总分",
        "活动量表总分（ADL总分+IADL总分）",
        "HDL-C（高密度脂蛋白）",
        "LDL-C（低密度脂蛋白）",
        "TG（甘油三酯）",
        "TC（总胆固醇）",
        "空腹血糖",
        "血尿酸",
        "BMI",
        "血脂异常项数",
        "代谢异常项数",
    ]
    lines.append(df[desc_cols].describe().round(4).to_string())
    lines.append("")
    lines.append("说明：本简化版主分析表不单独构造极端值标记，仅保留题面正常范围下的异常标记与计数变量，适合先行建模与描述统计。")
    return "\n".join(lines)


# =========================
# 5. 主流程
# =========================
def main() -> None:
    input_file = find_input_file()
    print(f"[INFO] 读取文件：{input_file}")
    df = read_input_data(input_file)

    missing_cols = [col for col in CORE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少关键字段：{missing_cols}")

    df = df[CORE_COLUMNS].copy()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[NUMERIC_COLUMNS].isna().sum().sum() > 0:
        na_summary = df[NUMERIC_COLUMNS].isna().sum()
        na_summary = na_summary[na_summary > 0]
        raise ValueError(
            "关键数值列在转数值后出现缺失，请先检查原始数据。\n"
            + na_summary.to_string()
        )

    df = add_label_columns(df)
    df = add_clinical_flags(df)
    df = add_activity_bands(df)

    main_csv_path = OUT_DIR / "c_q1_main_simple.csv"
    variable_dict_path = OUT_DIR / "c_q1_main_simple_variable_dict.csv"
    summary_txt_path = OUT_DIR / "c_q1_main_simple_summary.txt"

    df.to_csv(main_csv_path, index=False, encoding="utf-8-sig")
    build_variable_dictionary().to_csv(variable_dict_path, index=False, encoding="utf-8-sig")
    summary_txt_path.write_text(build_summary_text(df, input_file), encoding="utf-8")

    print("[INFO] 已生成问题一主分析表（简化版）与说明文件：")
    print(f"- 主分析表：{main_csv_path}")
    print(f"- 变量字典：{variable_dict_path}")
    print(f"- 摘要报告：{summary_txt_path}")
    print("[INFO] 下一步建议：先基于 c_q1_main_simple.csv 做描述统计、单变量分析和可视化。")


if __name__ == "__main__":
    main()
