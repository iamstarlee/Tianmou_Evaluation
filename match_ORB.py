import os
import numpy as np
import cv2
import re
import torch
import torch.nn.functional as F
from blending import poisson_blending
# 输入文件夹
left_dir = "Poisson_out/left"
right_dir = "Poisson_out/right"

# 文件排序函数
def sorted_npy_list(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    files.sort(key=lambda x: tuple(int(num) for num in re.search(r"idx(\d+)_frame(\d+)", x).groups()))
    return files

left_files = sorted_npy_list(left_dir)
right_files = sorted_npy_list(right_dir)

# assert len(left_files) == len(right_files), "左右目数量不一致"

# ORB 特征提取器
orb = cv2.ORB_create(
    nfeatures=1000,
    scaleFactor=1.1,
    nlevels=10,
    edgeThreshold=20,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=7
)

# BFMatcher（汉明距离）
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

for lf, rf in zip(left_files[::10], right_files[::10]):
    # 读取 npy -> 转 uint8 灰度图
    sdlc = np.load(os.path.join(left_dir, lf))
    sdrc = np.load(os.path.join(right_dir, rf))

    sdlc = (cv2.resize(sdlc, (320, 160)) + 50) * 2
    sdrc = (cv2.resize(sdrc, (320, 160)) + 50) * 2
    sdlc = sdlc.astype(np.uint8)
    sdrc = sdrc.astype(np.uint8)

    # 如果是多通道，转灰度
    if sdlc.ndim == 3:
        sdlc = cv2.cvtColor(sdlc, cv2.COLOR_BGR2GRAY)
    if sdrc.ndim == 3:
        sdrc = cv2.cvtColor(sdrc, cv2.COLOR_BGR2GRAY)

    # ORB 特征点与描述符
    kp1, des1 = orb.detectAndCompute(sdlc, None)
    kp2, des2 = orb.detectAndCompute(sdrc, None)

    if des1 is None or des2 is None:
        print(f"{lf} / {rf} 无法检测到特征点，跳过")
        continue

    # 匹配
    matches = bf.match(des1, des2)
    max_y_diff = 12

    filtered_matches = []
    for m in matches:
        pt1 = kp1[m.queryIdx].pt  # (x,y)
        pt2 = kp2[m.trainIdx].pt
        y_diff = abs(pt1[1] - pt2[1])
        if y_diff <= max_y_diff:
            filtered_matches.append(m)

    # 可视化匹配（取前50个匹配）
    match_img = cv2.drawMatches(sdlc, kp1, sdrc, kp2, filtered_matches[:20], None, flags=2)

    cv2.imshow("ORB Matching", match_img)
    print(f"Number of matches: {len(filtered_matches)}")
    key = cv2.waitKey(0)  # 按任意键看下一帧，按 q 退出
    if key == ord('q'):
        break

cv2.destroyAllWindows()
