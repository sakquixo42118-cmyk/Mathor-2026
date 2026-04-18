import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path

def get_chinese_font():
    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for fp in font_candidates:
        if Path(fp).exists():
            return FontProperties(fname=fp)
    raise FileNotFoundError("没有找到可用中文字体。")

ZH_FONT = get_chinese_font()

plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
ax.plot([0, 1], [0, 1])

ax.set_title("痰湿分支 Spearman 相关强度", fontproperties=ZH_FONT)
ax.set_xlabel("相关系数", fontproperties=ZH_FONT)
ax.set_ylabel("指标名称", fontproperties=ZH_FONT)

for label in ax.get_xticklabels():
    label.set_fontproperties(ZH_FONT)
for label in ax.get_yticklabels():
    label.set_fontproperties(ZH_FONT)

plt.tight_layout()
plt.savefig("test_font_force.png", bbox_inches="tight")
print("done")