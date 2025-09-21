import os
import re
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from blending import poisson_blending


# ========= 文件夹清空/创建 =========
sdlc_dir = "left_denoise_input"
sdrc_dir = "right_denoise_input"

def event_visualization(diff_data, thresh=1.0, gain=1.0, white_bg=True):
    """
    创建事件相机风格的可视化
    
    参数:
        diff_data: 差分数据 (H, W)
        thresh: 阈值，低于此值的变化被忽略
        gain: 增益，控制颜色强度
        white_bg: 是否使用白色背景
    
    返回:
        rgb_image: (H, W, 3) RGB图像
    """
    H, W = diff_data.shape
    
    # 创建RGB图像
    if white_bg:
        # 白色背景
        rgb = np.ones((H, W, 3), dtype=np.float32) * 255
    else:
        # 黑色背景
        rgb = np.zeros((H, W, 3), dtype=np.float32)
    
    # 应用阈值
    pos_mask = diff_data > thresh
    neg_mask = diff_data < -thresh
    
    # 应用增益
    diff_scaled = diff_data * gain
    
    if white_bg:
        # 白色背景：正事件红色，负事件蓝色
        # 正事件（红色）
        rgb[pos_mask, 0] = 255  # R通道
        rgb[pos_mask, 1] = 255 - np.clip(diff_scaled[pos_mask], 0, 255)  # G通道减少
        rgb[pos_mask, 2] = 255 - np.clip(diff_scaled[pos_mask], 0, 255)  # B通道减少
        
        # 负事件（蓝色）
        rgb[neg_mask, 0] = 255 - np.clip(-diff_scaled[neg_mask], 0, 255)  # R通道减少
        rgb[neg_mask, 1] = 255 - np.clip(-diff_scaled[neg_mask], 0, 255)  # G通道减少
        rgb[neg_mask, 2] = 255  # B通道
    else:
        # 黑色背景：正事件红色，负事件蓝色
        # 正事件（红色）
        rgb[pos_mask, 0] = np.clip(diff_scaled[pos_mask], 0, 255)
        
        # 负事件（蓝色）
        rgb[neg_mask, 2] = np.clip(-diff_scaled[neg_mask], 0, 255)
    
    return rgb.astype(np.uint8)

for folder in [sdlc_dir, sdrc_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)  # 删除整个文件夹
    os.makedirs(folder)  # 创建空文件夹


# ========= SD2XY 函数 =========
def SD2XY(sd_raw: torch.Tensor) -> torch.Tensor:
    if len(sd_raw.shape) == 3:
        assert (sd_raw.shape[2] == 2 or sd_raw.shape[0] == 2)
        if sd_raw.shape[2] == 2:
            sd = sd_raw.permute(2, 0, 1).unsqueeze(0)
        else:
            sd = sd_raw.unsqueeze(0)
    else:
        assert (len(sd_raw.shape) == 4 and sd_raw.shape[1] == 2)
        sd = sd_raw

    b, c, h, w = sd.shape
    sdul = sd[:, 0:1, 0::2, ...]
    sdll = sd[:, 0:1, 1::2, ...]
    sdur = sd[:, 1:2, 0::2, ...]
    sdlr = sd[:, 1:2, 1::2, ...]

    target_size = (h, w * 2)
    sdul = F.interpolate(sdul, size=target_size, mode='bilinear', align_corners=False)
    sdll = F.interpolate(sdll, size=target_size, mode='bilinear', align_corners=False)
    sdur = F.interpolate(sdur, size=target_size, mode='bilinear', align_corners=False)
    sdlr = F.interpolate(sdlr, size=target_size, mode='bilinear', align_corners=False)

    sqrt2 = 1.41421356237
    sdx = ((sdul + sdll) / sqrt2 - (sdur + sdlr) / sqrt2) / 2
    sdy = ((sdur - sdlr) / sqrt2 + (sdul - sdll) / sqrt2) / 2

    if len(sd_raw.shape) == 3:
        return sdx.squeeze(0).squeeze(0), sdy.squeeze(0).squeeze(0)
    else:
        return sdx.squeeze(1), sdy.squeeze(1)


# ========= 文件排序 =========
def sorted_npy_list(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    files.sort(key=lambda x: tuple(int(num) for num in re.search(r"idx(\d+)_frame(\d+)", x).groups()))
    return files


# ========= 主循环 =========
left_dir = "rod_output_cam0"
# right_dir = "rod_output_cam1"
left_files = sorted_npy_list(left_dir)
# right_files = sorted_npy_list(right_dir)

for lf in left_files:
    left_path = os.path.join(left_dir, lf)
    # right_path = os.path.join(right_dir, rf)

    left_data = np.load(left_path)
    # right_data = np.load(right_path)

    # 取最后两个通道 -> 转为 torch
    left_last2 = torch.from_numpy(left_data[1:]).float()
    # right_last2 = torch.from_numpy(right_data[1:]).float()

    # SD2XY
    sdlcx, sdlcy = SD2XY(left_last2)
    # sdrcx, sdrcy = SD2XY(right_last2)

    # Poisson blending
    sdlc_gpu = poisson_blending(sdlcx, sdlcy, iteration=15)
    # sdrc_gpu = poisson_blending(sdrcx, sdrcy, iteration=15)

    # 转 CPU numpy
    sdlc = sdlc_gpu.cpu().numpy()
    # sdrc = sdrc_gpu.cpu().numpy()

    # print(f"保存: {lf} -> {os.path.join(sdlc_dir, lf)}")
    # print(f"保存: {rf} -> {os.path.join(sdrc_dir, rf)}")

    # sdlc = sdlcx + sdlcy
    # sdrc = sdrcx + sdrcy

    # event_rgb_l = event_visualization(sdlc, thresh=1.0, gain=1.0, white_bg=True)
    # event_rgb_r = event_visualization(sdlc, thresh=1.0, gain=1.0, white_bg=True)

    # event_rgb_l = cv2.flip(cv2.resize(event_rgb_l, (320, 160)), 1) * 10
    # event_rgb_r = cv2.flip(cv2.resize(event_rgb_r, (320, 160)), 1) * 10

    # sdlc = (sdlc + 50) * 2
    # sdrc = (sdrc + 50) * 2
    

    # 保存到对应文件夹（文件名和原文件一致）
    np.save(os.path.join(sdlc_dir, lf), sdlc)
    # np.save(os.path.join(sdrc_dir, rf), sdrc)

    sdlc = sdlc.astype(np.uint8)
    # sdrc = sdrc.astype(np.uint8)
    
    cv2.imshow("left_image", sdlc)
    # cv2.imshow("right_image", sdrc)
    cv2.waitKey(1)
    # 在这里处理你的双目数据
