import os
import matplotlib.pyplot as plt

# 数据
settings = ["Full", "Drop baseline", "Drop structure", "Drop state", "Drop dynamics"]
cindex = [0.807207, 0.809260, 0.815721, 0.757944, 0.808423]

# 确保 figures 文件夹存在
os.makedirs("figures", exist_ok=True)

# 画图
plt.figure(figsize=(8, 5))
plt.bar(settings, cindex)

plt.ylabel("Test C-index")
plt.title("Supplementary Figure S1. Exploratory module ablation performance")

plt.ylim(0.74, 0.83)
plt.xticks(rotation=20)

plt.tight_layout()

# 保存
plt.savefig("figures/Supplementary_Figure_S1_ablation_barplot.png", dpi=300)

plt.show()