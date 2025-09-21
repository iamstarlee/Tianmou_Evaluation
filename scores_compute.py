import cv2
import numpy as np

# FAST 圆圈16个点的坐标偏移（半径=3）
FAST_OFFSETS = np.array([
    (0, -3), (1, -3), (2, -2), (3, -1),
    (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1),
    (-3, 0), (-3, -1), (-2, -2), (-1, -3)
], dtype=np.int32)

def fast_score(gray, kp, threshold=10):
    """
    计算一个FAST关键点的角点强度分数
    gray: 灰度图 (uint8)
    kp: cv2.KeyPoint
    threshold: FAST阈值
    """
    x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
    if (x < 3 or y < 3 or x >= gray.shape[1]-3 or y >= gray.shape[0]-3):
        return 0.0

    center_val = int(gray[y, x])
    diffs = []
    for dx, dy in FAST_OFFSETS:
        val = int(gray[y+dy, x+dx])
        diff = abs(val - center_val)
        if diff > threshold:
            diffs.append(diff)

    # 用差值的和作为分数（越大越强）
    if len(diffs) == 0:
        return 0.0
    return float(np.mean(diffs))  # 也可以用 sum(diffs) 或 min(diffs)

def compute_fast_scores(gray, keypoints, threshold=10):
    """
    批量计算所有FAST关键点的分数
    """
    scores = []
    for kp in keypoints:
        score = fast_score(gray, kp, threshold)
        kp.response = score  # 写回 KeyPoint
        scores.append(score)
    return scores

def compute_harris_confidence(gray, corners, blockSize=12, ksize=3, k=0.04, normalize=False, positive_only=True):
    """
    计算 Harris 角点的响应值作为 confidence
    """
    # 计算 Harris 响应图
    harris_resp = cv2.cornerHarris(np.float32(gray), blockSize, ksize, k)

    # 只取正值（角点响应），忽略负值（边缘）
    if positive_only:
        harris_resp = np.where(harris_resp > 0, harris_resp, 0)

    # 可选归一化
    if normalize:
        harris_norm = cv2.normalize(harris_resp, None, alpha=0, beta=1,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
        harris_norm = harris_resp

    confidences = []
    for c in corners:
        x, y = int(round(c[0][0])), int(round(c[0][1]))
        if 0 <= x < harris_norm.shape[1] and 0 <= y < harris_norm.shape[0]:
            confidences.append(float(harris_norm[y, x]))
        else:
            confidences.append(0.0)

    return confidences

def compute_shitomasi_confidence(gray, corners,
                                 blockSize=3, ksize=3,
                                 ratio_thresh=0.4,
                                 normalize=True):
    """
    计算 Shi-Tomasi / Harris 角点的 min-eigenvalue 作为 confidence。
    只对角点本身归一化，保持原始角点的相对强度。
    """
    if corners is None or len(corners) == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    # 转为 Nx2
    c = np.asarray(corners)
    if c.ndim == 3 and c.shape[1] == 1:
        pts = c.reshape(-1, 2)
    else:
        pts = c

    h, w = gray.shape[:2]
    gray_f = np.float32(gray)
    eig = cv2.cornerEigenValsAndVecs(gray_f, blockSize, ksize)  # (h,w,6)

    min_eigs = []
    ratios = []

    for x_f, y_f in pts:
        x = int(round(x_f)); y = int(round(y_f))
        if x < 0 or x >= w or y < 0 or y >= h:
            min_eigs.append(0.0)
            ratios.append(0.0)
            continue
        lam1 = float(eig[y, x, 0])
        lam2 = float(eig[y, x, 1])
        mn = min(lam1, lam2)
        mx = max(lam1, lam2)
        ratio = mn / (mx + 1e-12)
        min_eigs.append(mn)
        ratios.append(ratio)

    min_eigs = np.array(min_eigs, dtype=float)
    ratios = np.array(ratios, dtype=float)

    # trackable: ratio >= ratio_thresh
    trackable = ratios >= ratio_thresh

    if normalize and min_eigs.size > 0:
        # 只归一化角点 min-eig 数组，而不是整张图
        mnv, mxv = min_eigs.min(), min_eigs.max()
        if mxv > mnv:
            min_eigs = (min_eigs - mnv) / (mxv - mnv)
        else:
            min_eigs = np.ones_like(min_eigs)  # 全部相等时置 1

    return min_eigs, trackable


def safe_normalize(arr, clip_percentiles=(2, 98), eps=1e-12):
    """
    Robust normalization to [0,1].
    - arr: 1D array-like
    - clip_percentiles: tuple (low_p, high_p) for percentile clipping before scaling
    - eps: small number to avoid division by zero
    Returns numpy array of dtype float64.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr  # empty
    # replace nan/inf
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # if all nearly equal -> return neutral 0.5 (or zeros if you prefer)
    if np.allclose(arr, arr.ravel()[0], atol=1e-15):
        return np.full_like(arr, 0.5, dtype=np.float64)

    # percentile clipping then min-max
    low = np.percentile(arr, clip_percentiles[0])
    high = np.percentile(arr, clip_percentiles[1])
    if (high - low) > eps:
        clipped = np.clip(arr, low, high)
        return (clipped - low) / (high - low)

    # fallback to plain min-max
    minv = arr.min()
    maxv = arr.max()
    if (maxv - minv) > eps:
        return (arr - minv) / (maxv - minv)

    # fallback to log-scale (shift so positive)
    shifted = arr - arr.min() + eps
    loged = np.log(shifted + eps)
    if (loged.max() - loged.min()) > eps:
        return (loged - loged.min()) / (loged.max() - loged.min())

    # last resort
    return np.full_like(arr, 0.5, dtype=np.float64)