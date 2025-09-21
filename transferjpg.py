import cv2
import numpy as np
import os

# 输入 JPG 图像路径
jpg_path = "multi_gt.JPG"

# 输出 NPY 文件路径
npy_path = "multi_gt.npy"

# 读取图像
image = cv2.imread(jpg_path)
if image is None:
    print(f"无法加载图像：{jpg_path}")
    exit(1)

# 可选：将 BGR 转为 RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 保存为 NPY 文件
np.save(npy_path, image_rgb)
print(f"已保存为 NPY 文件：{npy_path}")

# 输出 NumPy 数组
print("图像的 NumPy 数组：")
print(image_rgb)
print("形状：", image_rgb.shape)
print("数据类型：", image_rgb.dtype)
