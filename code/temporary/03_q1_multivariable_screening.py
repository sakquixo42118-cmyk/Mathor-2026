from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =========================
# 路径与配置读取
# =========================
try:
    from config import OUT_DIR, FIGURE_DIR  # type: ignore
except Exception:
    CURRENT_FILE = Path(__file__).resolve()
    PROJECT_ROOT = CURRENT_FILE.parent.parent
    OUT_DIR = PROJECT_ROOT / "out"
    FIGURE_DIR = PROJECT_ROOT / "figure"

OUT_SUBDIR = OUT_DIR / "q1_multivariable"
FIG_SUBDIR = FIGURE_DIR / "q1_multivariable"
OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
FIG_SUBDIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = OUT_DIR / "c_q1_main_simple.csv"


# =========================
# 基础参数
# =========================
RANDOM_STATE = 42
PHLEGM_LABEL = 5
PHLEGM_HIGH_THRESHOLD = 60

FEATURES = [
    "TG（甘油三酯）",
    "TC（总胆固醇）",
    "LDL-C（低密度脂蛋白）",
    "HDL-C（高密度脂蛋白）",
    "空腹血糖",
    "血尿酸",
    "BMI",
    "活动量表总分（ADL总分+IADL总分）",
]

TARGET_PHLEGM = "痰湿质"
TARGET_RISK = "高血脂症二分类标签"
TARGET_CONSTITUTION = "体质标签"


# =========================
# 工具函数
# =========================
def check_required_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")



def save_barh(
    df_plot: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    outpath: Path,
    top_n: int | None = None,
) -> None:
    plot_df = df_plot.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)
    plot_df = plot_df.iloc[::-1]

    plt.figure(figsize=(10, 6), dpi=140)
    plt.barh(plot_df[y_col], plot_df[x_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()



def rank_average(*series_list: pd.Series) -> pd.DataFrame:
    if not series_list:
        raise ValueError("至少需要一个排名序列")

    rank_df = pd.DataFrame(index=series_list[0].index)
    for i, s in enumerate(series_list, start=1):
        rank_df[f"rank_{i}"] = s.rank(ascending=False, method="average")
    rank_df["平均排名"] = rank_df.mean(axis=1)
    return rank_df.sort_values("平均排名")


# =========================
# 读取数据
# =========================
if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"未找到输入文件: {INPUT_CSV}\n"
        "请先运行 02_build_q1_main_table_simple.py 生成主分析表。"
    )

df = pd.read_csv(INPUT_CSV)
check_required_columns(df, FEATURES + [TARGET_PHLEGM, TARGET_RISK, TARGET_CONSTITUTION])


# =========================
# 一、痰湿分支：Elastic Net 回归 + 随机森林回归
# =========================
phlegm_df = df[df[TARGET_CONSTITUTION] == PHLEGM_LABEL].copy()
X_ph = phlegm_df[FEATURES]
y_ph = phlegm_df[TARGET_PHLEGM]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            FEATURES,
        )
    ],
    remainder="drop",
)

elastic_net = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "model",
            ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                alphas=np.logspace(-3, 1, 60),
                cv=5,
                max_iter=20000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)
elastic_net.fit(X_ph, y_ph)

ph_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
ph_cv_result = cross_validate(
    elastic_net,
    X_ph,
    y_ph,
    cv=ph_cv,
    scoring={
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    },
    n_jobs=None,
    return_train_score=False,
)
ph_r2 = float(np.mean(ph_cv_result["test_r2"]))
ph_r2_std = float(np.std(ph_cv_result["test_r2"]))
ph_rmse = float(-np.mean(ph_cv_result["test_rmse"]))
ph_rmse_std = float(np.std(-ph_cv_result["test_rmse"]))
ph_mae = float(-np.mean(ph_cv_result["test_mae"]))
ph_mae_std = float(np.std(-ph_cv_result["test_mae"]))

ph_coef = pd.DataFrame(
    {
        "指标": FEATURES,
        "ElasticNet系数": elastic_net.named_steps["model"].coef_,
    }
)
ph_coef["|系数|"] = ph_coef["ElasticNet系数"].abs()
ph_coef = ph_coef.sort_values("|系数|", ascending=False).reset_index(drop=True)
ph_coef.to_csv(OUT_SUBDIR / "q1_phlegm_elasticnet_coefficients.csv", index=False, encoding="utf-8-sig")

rf_reg = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)
rf_reg.fit(X_ph, y_ph)
rf_reg_perm = permutation_importance(
    rf_reg,
    X_ph,
    y_ph,
    n_repeats=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scoring="r2",
)
ph_rf = pd.DataFrame(
    {
        "指标": FEATURES,
        "RF置换重要性均值": rf_reg_perm.importances_mean,
        "RF置换重要性标准差": rf_reg_perm.importances_std,
    }
).sort_values("RF置换重要性均值", ascending=False).reset_index(drop=True)
ph_rf.to_csv(OUT_SUBDIR / "q1_phlegm_rf_permutation_importance.csv", index=False, encoding="utf-8-sig")

ph_rank = rank_average(
    ph_coef.set_index("指标")["|系数|"],
    ph_rf.set_index("指标")["RF置换重要性均值"],
).reset_index().rename(columns={"index": "指标"})
ph_rank.to_csv(OUT_SUBDIR / "q1_phlegm_multivariable_rank.csv", index=False, encoding="utf-8-sig")

save_barh(
    ph_coef,
    x_col="|系数|",
    y_col="指标",
    title="痰湿分支：Elastic Net |系数| Top 指标",
    xlabel="|coef|",
    outpath=FIG_SUBDIR / "q1_phlegm_elasticnet_top.png",
)

save_barh(
    ph_rf,
    x_col="RF置换重要性均值",
    y_col="指标",
    title="痰湿分支：随机森林置换重要性 Top 指标",
    xlabel="permutation importance (R² drop)",
    outpath=FIG_SUBDIR / "q1_phlegm_rf_permutation_top.png",
)


# =========================
# 二、高血脂分支：惩罚 Logistic + 随机森林分类
# =========================
X_risk = df[FEATURES]
y_risk = df[TARGET_RISK].astype(int)

logit_en = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "model",
            LogisticRegressionCV(
                Cs=np.logspace(-2, 2, 20),
                cv=5,
                penalty="elasticnet",
                solver="saga",
                l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                scoring="roc_auc",
                class_weight="balanced",
                max_iter=20000,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                refit=True,
            ),
        ),
    ]
)
logit_en.fit(X_risk, y_risk)

risk_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
risk_cv_result = cross_validate(
    logit_en,
    X_risk,
    y_risk,
    cv=risk_cv,
    scoring={
        "auc": "roc_auc",
        "brier": "neg_brier_score",
        "balanced_acc": "balanced_accuracy",
        "f1": "f1",
    },
    n_jobs=None,
    return_train_score=False,
)
risk_auc = float(np.mean(risk_cv_result["test_auc"]))
risk_auc_std = float(np.std(risk_cv_result["test_auc"]))
risk_brier = float(-np.mean(risk_cv_result["test_brier"]))
risk_brier_std = float(np.std(-risk_cv_result["test_brier"]))
risk_bal_acc = float(np.mean(risk_cv_result["test_balanced_acc"]))
risk_bal_acc_std = float(np.std(risk_cv_result["test_balanced_acc"]))
risk_f1 = float(np.mean(risk_cv_result["test_f1"]))
risk_f1_std = float(np.std(risk_cv_result["test_f1"]))

logit_model = logit_en.named_steps["model"]
coef = logit_model.coef_.ravel()
risk_coef = pd.DataFrame(
    {
        "指标": FEATURES,
        "Logit系数": coef,
        "OR=exp(coef)": np.exp(coef),
    }
)
risk_coef["|系数|"] = risk_coef["Logit系数"].abs()
risk_coef = risk_coef.sort_values("|系数|", ascending=False).reset_index(drop=True)
risk_coef.to_csv(OUT_SUBDIR / "q1_risk_logit_coefficients.csv", index=False, encoding="utf-8-sig")

rf_clf = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)
rf_clf.fit(X_risk, y_risk)
rf_clf_perm = permutation_importance(
    rf_clf,
    X_risk,
    y_risk,
    n_repeats=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scoring="roc_auc",
)
risk_rf = pd.DataFrame(
    {
        "指标": FEATURES,
        "RF置换重要性均值": rf_clf_perm.importances_mean,
        "RF置换重要性标准差": rf_clf_perm.importances_std,
    }
).sort_values("RF置换重要性均值", ascending=False).reset_index(drop=True)
risk_rf.to_csv(OUT_SUBDIR / "q1_risk_rf_permutation_importance.csv", index=False, encoding="utf-8-sig")

risk_rank = rank_average(
    risk_coef.set_index("指标")["|系数|"],
    risk_rf.set_index("指标")["RF置换重要性均值"],
).reset_index().rename(columns={"index": "指标"})
risk_rank.to_csv(OUT_SUBDIR / "q1_risk_multivariable_rank.csv", index=False, encoding="utf-8-sig")

save_barh(
    risk_coef,
    x_col="|系数|",
    y_col="指标",
    title="高血脂分支：惩罚 Logistic |系数| Top 指标",
    xlabel="|coef|",
    outpath=FIG_SUBDIR / "q1_risk_logit_top.png",
)

save_barh(
    risk_rf,
    x_col="RF置换重要性均值",
    y_col="指标",
    title="高血脂分支：随机森林置换重要性 Top 指标",
    xlabel="permutation importance (AUC drop)",
    outpath=FIG_SUBDIR / "q1_risk_rf_permutation_top.png",
)


# =========================
# 三、摘要输出
# =========================
summary_lines = [
    "问题一：多变量筛选摘要",
    "=" * 60,
    f"输入主分析表: {INPUT_CSV}",
    f"总样本量: {len(df)}",
    f"痰湿体质子样本量(体质标签=5): {len(phlegm_df)}",
    f"候选指标数: {len(FEATURES)}",
    "",
    "[一] 痰湿分支：Elastic Net Top 5",
    ph_coef.head(5).to_string(index=False),
    "",
    "[二] 痰湿分支：随机森林置换重要性 Top 5",
    ph_rf.head(5).to_string(index=False),
    "",
    "[三] 痰湿分支：5x5重复交叉验证回归性能",
    f"R² = {ph_r2:.4f} ± {ph_r2_std:.4f}",
    f"RMSE = {ph_rmse:.4f} ± {ph_rmse_std:.4f}",
    f"MAE = {ph_mae:.4f} ± {ph_mae_std:.4f}",
    "",
    "[四] 高血脂分支：惩罚 Logistic Top 5",
    risk_coef.head(5).to_string(index=False),
    "",
    "[五] 高血脂分支：随机森林置换重要性 Top 5",
    risk_rf.head(5).to_string(index=False),
    "",
    "[六] 高血脂分支：5x5重复分层交叉验证性能",
    f"AUC = {risk_auc:.4f} ± {risk_auc_std:.4f}",
    f"Brier = {risk_brier:.4f} ± {risk_brier_std:.4f}",
    f"Balanced Accuracy = {risk_bal_acc:.4f} ± {risk_bal_acc_std:.4f}",
    f"F1 = {risk_f1:.4f} ± {risk_f1_std:.4f}",
    "",
    "[七] 说明",
    "1. 当前脚本仅完成两个分支的多变量筛选，不包含双分支交叉合并、Bootstrap稳定性选择与九种体质贡献分析。",
    "2. 痰湿分支采用 Elastic Net 回归 + 随机森林置换重要性；高血脂分支采用惩罚 Logistic + 随机森林置换重要性。",
    "3. 结果中的排名用于下一步候选变量保留与图表解释，不直接等同于最终论文结论。",
]

summary_text = "\n".join(summary_lines)
(OUT_SUBDIR / "q1_multivariable_summary.txt").write_text(summary_text, encoding="utf-8")

metadata = {
    "input_csv": str(INPUT_CSV),
    "features": FEATURES,
    "phlegm_branch": {
        "subset_rule": f"{TARGET_CONSTITUTION} == {PHLEGM_LABEL}",
        "target": TARGET_PHLEGM,
        "model_1": "ElasticNetCV",
        "model_2": "RandomForestRegressor + permutation importance",
        "cv": "RepeatedKFold(5x5)",
    },
    "risk_branch": {
        "target": TARGET_RISK,
        "model_1": "LogisticRegressionCV(elasticnet)",
        "model_2": "RandomForestClassifier + permutation importance",
        "cv": "RepeatedStratifiedKFold(5x5)",
    },
}
(OUT_SUBDIR / "q1_multivariable_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("多变量筛选完成。")
print(f"结果输出目录: {OUT_SUBDIR}")
print(f"图片输出目录: {FIG_SUBDIR}")
