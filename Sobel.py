import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取灰度图
img = cv2.imread('ORB Matching_screenshot_11.08.2025.png', cv2.IMREAD_GRAYSCALE)

# 计算 x 方向梯度 Ix
Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # ksize = Sobel 核大小

# 计算 y 方向梯度 Iy
Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# 梯度幅值
grad_mag = np.sqrt(Ix**2 + Iy**2)

# 可视化
plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.title('Ix')
plt.imshow(Ix, cmap='gray')
plt.subplot(1,3,2)
plt.title('Iy')
plt.imshow(Iy, cmap='gray')
plt.subplot(1,3,3)
plt.title('Gradient Magnitude')
plt.imshow(grad_mag, cmap='gray')
plt.show()
