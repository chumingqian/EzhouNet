import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Load Chinese font
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
my_font = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False

# Cancer data
data = {
    "男性-新发病例": {
        "labels": ['前列腺', '肺和支气管', '结肠直肠', '膀胱', '皮肤黑色素瘤', '肾及肾盂', '非霍奇金淋巴瘤', '口腔和咽部', '白血病', '胰腺'],
        "sizes": [248530, 119100, 79520, 64280, 62260, 48780, 45630, 38800, 35530, 31950],
        "total": 970250
    },
    "女性-新发病例": {
        "labels": ['乳腺', '肺和支气管', '结肠直肠', '膀胱', '皮肤黑色素瘤', '非霍奇金淋巴瘤', '甲状腺', '胰腺', '肾及肾盂', '白血病'],
        "sizes": [281550, 116660, 69980, 66570, 43850, 35930, 32130, 28480, 27300, 25560],
        "total": 927910
    },
    "男性-死亡病例": {
        "labels": ['肺和支气管', '前列腺', '结肠直肠', '胰腺', '肝及肝内胆管', '白血病', '食管', '膀胱', '非霍奇金淋巴瘤', '脑及其他神经系统'],
        "sizes": [69410, 34130, 28520, 25270, 20300, 13900, 12410, 12260, 12170, 10500],
        "total": 319420
    },
    "女性-死亡病例": {
        "labels": ['肺和支气管', '乳腺', '结肠直肠', '胰腺', '卵巢', '子宫体', '肝及肝内胆管', '白血病', '非霍奇金淋巴瘤', '脑及其他神经系统'],
        "sizes": [62470, 43600, 24460, 22950, 22950, 12940, 9930, 9760, 8550, 8100],
        "total": 289150
    }
}

colors = plt.cm.tab20.colors

# Draw 2x2 pie chart
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("2024年疾病估计与死亡病例（按性别）", fontsize=18)

for ax, (title, value) in zip(axs.flat, data.items()):
    wedges, texts, autotexts = ax.pie(
        value["sizes"],
        autopct='%1.0f%%',
        startangle=140,
        colors=colors,
        textprops={'fontsize': 10}
    )
    ax.set_title(f"{title}\n合计 {value['total']:,}", fontsize=12)
    ax.legend(wedges, value["labels"], title="癌症类型", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), prop=my_font)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("cancer_pie_charts2024.png", dpi=300, bbox_inches='tight')
plt.show()
