import numpy as np
import matplotlib.pyplot as plt

# 假设 x 轴有 5 个点
x = np.arange(1, 6)

# 三组数据的均值和方差（示例）
mean1 = np.array([1.0, 1.3, 1.6, 1.8, 2.0])
var1  = np.array([0.05, 0.08, 0.05, 0.07, 0.06])

# 曲线2：中等水平，线性上升
mean2 = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
var2  = np.array([0.1, 0.15, 0.12, 0.2, 0.18])

# 曲线3：整体偏高，增长更快，波动也大
mean3 = np.array([3.0, 3.8, 4.6, 5.5, 6.5])
var3  = np.array([0.25, 0.3, 0.35, 0.4, 0.5])

# 方差开根号 -> 标准差
std1, std2, std3 = np.sqrt(var1), np.sqrt(var2), np.sqrt(var3)

plt.figure(figsize=(8, 5))

# 第一条曲线
plt.plot(x, mean1, label='Group 1')
plt.fill_between(x, mean1-std1, mean1+std1, alpha=0.2)

# 第二条曲线
plt.plot(x, mean2, color='orange', label='Group 2')
plt.fill_between(x, mean2-std2, mean2+std2, color='orange', alpha=0.2)

# 第三条曲线
plt.plot(x, mean3, color='black', label='Group 3')
plt.fill_between(x, mean3-std3, mean3+std3, color='black', alpha=0.2)

plt.xlabel("X axis")
plt.ylabel("Value")
plt.title("Three curves with uncertainty (±1 std)")
plt.legend()
plt.grid(True)
plt.show()
