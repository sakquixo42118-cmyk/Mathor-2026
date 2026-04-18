from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties


# =========================
# 中文字体：硬指定字体文件
# =========================
def get_chinese_font() -> FontProperties:
    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",   # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf", # 黑体
        r"C:\Windows\Fonts\simsun.ttc", # 宋体
    ]
    for fp in font_candidates:
        if Path(fp).exists():
            return FontProperties(fname=fp)
    raise FileNotFoundError(
        "没有找到可用中文字体。请确认 C:\\Windows\\Fonts 下存在微软雅黑、黑体或宋体。"
    )


ZH_FONT = get_chinese_font()
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', context='talk')


def apply_zh_font(
    ax: plt.Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
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

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontproperties(ZH_FONT)
        if legend.get_title() is not None:
            legend.get_title().set_fontproperties(ZH_FONT)


# =========================
# 路径配置（优先复用 code/config.py）
# =========================
def resolve_paths():
    current_file = Path(__file__).resolve()
    code_dir = current_file.parent
    project_root = code_dir.parent

    out_dir_default = project_root / 'out'
    fig_dir_default = project_root / 'figure'

    try:
        import config  # type: ignore
        out_dir = getattr(config, 'OUT_DIR', out_dir_default)
        fig_dir = getattr(config, 'FIGURE_DIR', fig_dir_default)
        if not isinstance(out_dir, Path):
            out_dir = Path(out_dir)
        if not isinstance(fig_dir, Path):
            fig_dir = Path(fig_dir)
        return project_root, out_dir, fig_dir
    except Exception:
        return project_root, out_dir_default, fig_dir_default


PROJECT_ROOT, OUT_DIR, FIGURE_DIR = resolve_paths()
REDRAW_DIR = FIGURE_DIR / 'q1_publication_redraw'
REDRAW_DIR.mkdir(parents=True, exist_ok=True)

BRANCH_DIR = OUT_DIR / 'q1_branch_validation'
MULTI_DIR = OUT_DIR / 'q1_multivariable'
CROSS_DIR = OUT_DIR / 'q1_cross_merge'
CONS_DIR = OUT_DIR / 'q1_constitution_contribution'
BOOT_DIR = OUT_DIR / 'q1_bootstrap_stability'

# 兼容聊天中单独上传的文件
FALLBACK_DIR = Path('/mnt/data')


# =========================
# 通用工具
# =========================
def find_file(primary_dir: Path, name: str) -> Path:
    c1 = primary_dir / name
    c2 = FALLBACK_DIR / name
    if c1.exists():
        return c1
    if c2.exists():
        return c2
    raise FileNotFoundError(f'未找到文件: {name}\n已检查: {c1}\n{c2}')


def load_csv(primary_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(find_file(primary_dir, name))


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(REDRAW_DIR / name, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =========================
# 1) 单变量验证：三联棒棒糖图
# =========================
def redraw_branch_validation_panel():
    ph = load_csv(BRANCH_DIR, 'q1_phlegm_branch_spearman.csv')
    risk = load_csv(BRANCH_DIR, 'q1_risk_branch_single_factor_validation.csv')

    fig, axes = plt.subplots(1, 3, figsize=(19, 7), dpi=180)

    # 痰湿 Spearman
    ph_plot = ph.sort_values('|rho|', ascending=True)
    y = np.arange(len(ph_plot))
    axes[0].hlines(y=y, xmin=0, xmax=ph_plot['|rho|'], color='#C0C0C0', linewidth=1.5)
    axes[0].scatter(ph_plot['|rho|'], y, s=70, color='#4C78A8')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ph_plot['指标'])
    apply_zh_font(axes[0], '（a）痰湿分支：Spearman 相关强度', '|rho|', None)
    for yi, val in zip(y, ph_plot['|rho|']):
        axes[0].text(val + 0.003, yi, f'{val:.3f}', va='center', fontsize=9, fontproperties=ZH_FONT)

    # 风险 点二列相关
    risk_pb = risk.sort_values('|point_biserial_r|', ascending=True)
    y = np.arange(len(risk_pb))
    axes[1].hlines(y=y, xmin=0, xmax=risk_pb['|point_biserial_r|'], color='#C0C0C0', linewidth=1.5)
    axes[1].scatter(risk_pb['|point_biserial_r|'], y, s=70, color='#F58518')
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(risk_pb['指标'])
    apply_zh_font(axes[1], '（b）高血脂分支：点二列相关', '|r|', None)
    for yi, val in zip(y, risk_pb['|point_biserial_r|']):
        axes[1].text(val + 0.01, yi, f'{val:.3f}', va='center', fontsize=9, fontproperties=ZH_FONT)

    # 风险 AUC
    risk_auc = risk.sort_values('单变量AUC', ascending=True)
    y = np.arange(len(risk_auc))
    axes[2].hlines(y=y, xmin=0.5, xmax=risk_auc['单变量AUC'], color='#C0C0C0', linewidth=1.5)
    axes[2].scatter(risk_auc['单变量AUC'], y, s=70, color='#54A24B')
    axes[2].axvline(0.5, linestyle='--', color='gray', linewidth=1)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(risk_auc['指标'])
    axes[2].set_xlim(0.45, max(0.9, risk_auc['单变量AUC'].max() + 0.03))
    apply_zh_font(axes[2], '（c）高血脂分支：单变量 AUC', 'AUC', None)
    for yi, val in zip(y, risk_auc['单变量AUC']):
        axes[2].text(val + 0.01, yi, f'{val:.3f}', va='center', fontsize=9, fontproperties=ZH_FONT)

    save_fig(fig, 'q1_branch_validation_triptych.png')


# =========================
# 2) 痰湿四分位趋势图
# =========================
def redraw_phlegm_quartile_profile(top_n: int = 4):
    quart = load_csv(BRANCH_DIR, 'q1_phlegm_branch_quartile_means.csv')
    ph = load_csv(BRANCH_DIR, 'q1_phlegm_branch_spearman.csv')
    pick = ph.sort_values('|rho|', ascending=False).head(top_n)['指标'].tolist()
    plot_df = quart[quart['指标'].isin(pick)].copy()

    long_df = plot_df.melt(id_vars='指标', value_vars=['Q1', 'Q2', 'Q3', 'Q4'],
                           var_name='痰湿分位组', value_name='均值')

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    sns.lineplot(data=long_df, x='痰湿分位组', y='均值', hue='指标', marker='o', linewidth=2.5, ax=ax)
    apply_zh_font(ax, '痰湿分支：候选指标在痰湿积分四分位中的均值趋势', '痰湿积分分位组（Q1→Q4）', '组内均值')
    save_fig(fig, 'q1_phlegm_quartile_profile.png')


# =========================
# 3) 高血脂 OR 棒棒糖图（对数轴）
# =========================
def redraw_risk_or_lollipop():
    logit = load_csv(MULTI_DIR, 'q1_risk_logit_coefficients.csv')
    plot_df = logit.copy().sort_values('OR=exp(coef)', ascending=True)
    plot_df['颜色'] = np.where(plot_df['OR=exp(coef)'] >= 1, '#D62728', '#1F77B4')

    y = np.arange(len(plot_df))
    vals = plot_df['OR=exp(coef)'].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    ax.hlines(y=y, xmin=1, xmax=vals, color='#C0C0C0', linewidth=1.5)
    ax.scatter(vals, y, s=90, color=plot_df['颜色'])
    ax.axvline(1.0, linestyle='--', color='gray', linewidth=1)
    ax.set_xscale('log')
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['指标'])
    apply_zh_font(ax, '高血脂分支：惩罚 Logistic 的 OR 棒棒糖图', 'OR（对数坐标，参考线 OR=1）', None)
    for yi, val in zip(y, vals):
        ax.text(val * 1.04, yi, f'{val:.2f}', va='center', fontsize=9, fontproperties=ZH_FONT)
    save_fig(fig, 'q1_risk_or_lollipop.png')


# =========================
# 4) 双分支二维象限图
# =========================
def redraw_cross_quadrant():
    cross = load_csv(CROSS_DIR, 'q1_indicator_layers.csv')
    x = cross['痰湿分支得分']
    y = cross['高血脂分支得分']
    x_mid = float(np.median(x))
    y_mid = float(np.median(y))

    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    sns.scatterplot(data=cross, x='痰湿分支得分', y='高血脂分支得分', hue='分层类别', s=110, ax=ax)
    ax.axvline(x_mid, linestyle='--', color='gray', linewidth=1)
    ax.axhline(y_mid, linestyle='--', color='gray', linewidth=1)
    apply_zh_font(ax, '双分支指标二维象限图', '痰湿分支综合得分', '高血脂分支综合得分')
    for _, row in cross.iterrows():
        ax.text(row['痰湿分支得分'] + 0.01, row['高血脂分支得分'] + 0.01, row['指标'], fontsize=9, fontproperties=ZH_FONT)
    save_fig(fig, 'q1_cross_quadrant.png')


# =========================
# 5) Bootstrap 双图：痰湿全尺度 + 高血脂缩放版
# =========================
def redraw_bootstrap_panel():
    ph_path = BOOT_DIR / 'q1_phlegm_bootstrap_stability.csv'
    risk_path = BOOT_DIR / 'q1_risk_bootstrap_stability.csv'
    if not ph_path.exists() or not risk_path.exists():
        ph_alt = FALLBACK_DIR / 'q1_phlegm_bootstrap_stability.csv'
        risk_alt = FALLBACK_DIR / 'q1_risk_bootstrap_stability.csv'
        if not ph_alt.exists() or not risk_alt.exists():
            return
        ph_path, risk_path = ph_alt, risk_alt

    ph = pd.read_csv(ph_path).sort_values('选择频率', ascending=True)
    risk = pd.read_csv(risk_path).sort_values('选择频率', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=180)

    axes[0].barh(ph['指标'], ph['选择频率'], color='#4C78A8')
    axes[0].set_xlim(0, 1.0)
    apply_zh_font(axes[0], '（a）痰湿分支 Bootstrap 选择频率', 'Bootstrap 选择频率', None)
    for i, v in enumerate(ph['选择频率']):
        axes[0].text(v + 0.015, i, f'{v:.2f}', va='center', fontsize=9, fontproperties=ZH_FONT)

    axes[1].barh(risk['指标'], risk['选择频率'], color='#F58518')
    axes[1].set_xlim(0.6, 1.0)
    apply_zh_font(axes[1], '（b）高血脂分支 Bootstrap 选择频率', 'Bootstrap 选择频率（放大显示）', None)
    for i, v in enumerate(risk['选择频率']):
        axes[1].text(min(v + 0.005, 0.995), i, f'{v:.2f}', va='center', fontsize=9, fontproperties=ZH_FONT)

    save_fig(fig, 'q1_bootstrap_frequency_panel.png')


# =========================
# 6) 九种体质三联图
# =========================
def redraw_constitution_triptych():
    prev_path = CONS_DIR / 'q1_constitution_prevalence_rr_cd.csv'
    or_path = CONS_DIR / 'q1_constitution_adjusted_or.csv'
    if not prev_path.exists() or not or_path.exists():
        prev_alt = FALLBACK_DIR / 'q1_constitution_prevalence_rr_cd.csv'
        or_alt = FALLBACK_DIR / 'q1_constitution_adjusted_or.csv'
        if not prev_alt.exists() or not or_alt.exists():
            return
        prev_path, or_path = prev_alt, or_alt

    prev_df = pd.read_csv(prev_path)
    or_df = pd.read_csv(or_path)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=180)

    p1 = prev_df.sort_values('发病率', ascending=False)
    axes[0].bar(p1['体质名称'], p1['发病率'], color='#4C78A8')
    apply_zh_font(axes[0], '（a）九种体质原始发病率', None, '发病率')
    axes[0].tick_params(axis='x', rotation=30)

    p2 = prev_df.sort_values('绝对贡献度CD', ascending=False)
    axes[1].bar(p2['体质名称'], p2['绝对贡献度CD'], color='#F58518')
    apply_zh_font(axes[1], '（b）九种体质绝对贡献度 CD', None, 'CD')
    axes[1].tick_params(axis='x', rotation=30)

    if not or_df.empty:
        p3 = or_df.sort_values('OR', ascending=True)
        ypos = np.arange(len(p3))
        err_left = p3['OR'] - p3['OR_CI_lower']
        err_right = p3['OR_CI_upper'] - p3['OR']
        axes[2].errorbar(p3['OR'], ypos, xerr=[err_left, err_right], fmt='o', capsize=4, color='#54A24B')
        axes[2].axvline(1.0, linestyle='--', color='gray', linewidth=1)
        axes[2].set_yticks(ypos)
        axes[2].set_yticklabels(p3['体质名称'])
        apply_zh_font(axes[2], '（c）调整后 OR 森林图', 'OR（相对平和质）', None)

    save_fig(fig, 'q1_constitution_triptych.png')


# =========================
# 主程序
# =========================
def main():
    redraw_branch_validation_panel()
    redraw_phlegm_quartile_profile(top_n=4)
    redraw_risk_or_lollipop()
    redraw_cross_quadrant()
    redraw_bootstrap_panel()
    redraw_constitution_triptych()
    print(f'重绘完成，输出目录：{REDRAW_DIR}')


if __name__ == '__main__':
    main()
