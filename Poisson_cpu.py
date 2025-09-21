import numpy as np
import cv2
import scipy.sparse.linalg as sparse_la
from scipy import sparse

def laplacian_blending_1c_cpu(Ix, Iy, gray=None, iteration=50):
    """
    灰度 Poisson 重建 (CPU NumPy 版本)
    Ix, Iy: [H, W] 梯度
    gray: 初始图像 (可为 None)
    """
    if gray is None:
        gray_iter = np.zeros_like(Ix, dtype=np.float32)
    else:
        gray_iter = gray.copy().astype(np.float32)

    div_v = Ix[1:-1, 2:] - Ix[1:-1, 1:-1] + Iy[2:, 1:-1] - Iy[1:-1, 1:-1]

    for _ in range(iteration):
        gray_iter_old = gray_iter.copy()
        gray_iter[1:-1, 1:-1] = 0.25 * (
            gray_iter[2:, 1:-1] +
            gray_iter[0:-2, 1:-1] +
            gray_iter[1:-1, 2:] +
            gray_iter[1:-1, 0:-2] -
            div_v
        )
        if np.sum(np.abs(gray_iter - gray_iter_old)) < 0.1:
            break

    return gray_iter

def smooth_edges_cpu(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    blurred = cv2.GaussianBlur(closing, (5, 5), sigmaX=1)
    return blurred

def genMask_cpu(gray, th=24, maxV=255, minV=0):
    """
    生成过欠曝区域遮罩 (CPU NumPy)
    """
    gap = maxV - minV
    mask_np = (gray < (maxV - th) / gap).astype(np.float32)
    mask_np_gap = (mask_np * gap).astype(np.uint8)

    if gray.ndim == 3:
        mask_np_b = np.zeros_like(mask_np_gap)
        for c in range(3):
            mask_np_b[:, :, c] = smooth_edges_cpu(mask_np_gap[:, :, c])
    else:
        mask_np_b = smooth_edges_cpu(mask_np_gap)

    return mask_np_b.astype(np.float32)

def poisson_blending_cpu(Ix, Iy, srcimg=None, iteration=20, mask_rgb=False, mask_th=24, smooth=True):
    """
    RGB/灰度 HDR 融合重建 (CPU 版本)
    """
    if mask_rgb and srcimg is not None:
        mask = genMask_cpu(srcimg, th=mask_th, maxV=255, minV=0)

    if srcimg is None:
        result = laplacian_blending_1c_cpu(Ix, Iy, gray=None, iteration=iteration)
    elif srcimg.ndim == 2:
        result = laplacian_blending_1c_cpu(Ix, Iy, gray=srcimg, iteration=iteration)
    elif srcimg.ndim == 3:
        result = srcimg.copy()
        for c in range(result.shape[2]):
            result[..., c] = laplacian_blending_1c_cpu(Ix, Iy, gray=srcimg[..., c], iteration=iteration)
    else:
        raise ValueError(f"Unsupported srcimg shape: {srcimg.shape}")

    if mask_rgb and result is not None:
        if smooth:
            mask /= 255.0
            result = (1 - mask) * result + mask * srcimg
        else:
            mask_bool = mask > 0.5
            result[mask_bool] = srcimg[mask_bool]

    return result
