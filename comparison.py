import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def process(img):
    img = (cv2.resize(img, (320, 160)) + 50) * 2
    img = img.astype(np.uint8)
    return img

def compare_npy_by_index(folder1, folder2, index):
    # 获取两个文件夹中的文件名并排序
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith('.npy')])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith('.npy')])

    if index < 0 or index >= min(len(files1), len(files2)):
        raise IndexError(f"Index {index} 超出范围")

    # 获取对应的文件路径
    path1 = os.path.join(folder1, files1[index])
    path2 = os.path.join(folder2, files2[index])

    # 读取 npy 数据
    img1 = np.load(path1)
    img2 = np.load(path2)

    img1 = process(img1)
    img2 = process(img2)

    # 做差
    diff = img1.astype(np.float32) - img2.astype(np.float32)

    # 可视化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Image from folder1")
    plt.imshow(img1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Image from folder2")
    plt.imshow(img2, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.imshow(diff, cmap='bwr')  # bwr 显示正负差异
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 使用示例
folderA = "left_denoise_input"
folderB = "Poisson_out/left"
compare_npy_by_index(folderA, folderB, index=0)  # 比较第6张
