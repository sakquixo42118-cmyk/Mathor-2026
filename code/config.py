from __future__ import annotations

from pathlib import Path

# =========================
# 项目路径配置（统一入口）
# =========================
# 约定：config.py 放在 code/ 文件夹中
# 目录结构示例：
# project_root/
# ├─ code/
# │  ├─ config.py
# │  ├─ 01_read_and_audit.py
# │  └─ 02_build_q1_main_table_simple.py
# ├─ raw/
# ├─ out/
# ├─ figure/
# └─ reference/

CONFIG_FILE = Path(__file__).resolve()
CODE_DIR = CONFIG_FILE.parent
PROJECT_ROOT = CODE_DIR.parent

RAW_DIR = PROJECT_ROOT / "raw"
OUT_DIR = PROJECT_ROOT / "out"
FIGURE_DIR = PROJECT_ROOT / "figure"
REFERENCE_DIR = PROJECT_ROOT / "reference"

# =========================
# C题附件1路径配置
# =========================
C_Q1_CSV = RAW_DIR / "C题：附件1：样例数据.csv"
C_Q1_XLSX = RAW_DIR / "C题：附件1：样例数据.xlsx"
C_Q1_CANDIDATES = [C_Q1_CSV, C_Q1_XLSX]


def ensure_project_dirs() -> None:
    """确保项目中常用输出目录存在。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)


def find_existing_file(candidates: list[Path]) -> Path:
    """在候选路径中返回第一个存在的文件。"""
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(f"未找到目标文件。已检查：\n{checked}")
