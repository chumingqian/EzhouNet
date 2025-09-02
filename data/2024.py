
import matplotlib

# 设置中文字体（选择系统支持的字体）
# matplotlib.rcParams['font.family'] = 'SimHei'       # 黑体
# matplotlib.rcParams['axes.unicode_minus'] = False   # 正常显示负号

import matplotlib.pyplot as plt
from matplotlib import font_manager

# 手动加载字体（指定为系统中 NotoSansCJK-Regular）
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
my_font = font_manager.FontProperties(fname=font_path)

# 设置为默认字体
plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False






# 数据定义
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

# 颜色预定义（可自定义调整）
colors = plt.cm.tab20.colors

# 创建图像并绘制子图
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("癌症预估新发与死亡病例（按性别）", fontsize=18)

for ax, (title, value) in zip(axs.flat, data.items()):
    wedges, texts, autotexts = ax.pie(
        value["sizes"], labels=value["labels"], autopct='%1.0f%%',
        startangle=140, colors=colors
    )
    ax.set_title(f"{title}\n合计 {value['total']:,}")

# 自动布局调整 & 保存图像
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("cancer_pie_charts.png", dpi=300, bbox_inches='tight')  # 保存图片
plt.show()
