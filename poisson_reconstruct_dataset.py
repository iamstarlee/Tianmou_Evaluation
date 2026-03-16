import os
import cv2
import torch
import numpy as np
from scipy.fftpack import dst, idst
from tqdm import tqdm
from samples.denoised_poisson import SD2XY
from samples.blending import poisson_blending


def poisson_reconstruct_image(img_bgr1, img_bgr2):
    """
    对 BGR 彩色图像做泊松重建。
    """
    t1 = torch.from_numpy(img_bgr1).float()
    t2 = torch.from_numpy(img_bgr2).float()
    sd = torch.stack([t1, t2], dim=0)

    Ix, Iy = SD2XY(sd)

    sd_recon_gpu = poisson_blending(Ix, Iy, iteration=15)
    sd_recon = sd_recon_gpu.cpu().numpy()
    return ((sd_recon + 150) * 0.75).astype(np.uint8)


def event_visualization_white_bg(diff_data, thresh=8.0, gain=2.2):
    """
    按 rod_event_visualization.py 的白底思路可视化：
    - 背景全白
    - 正值区域：白 -> 深蓝
    - 负值区域：白 -> 黑
    输入:
        diff_data: float32/float64, shape [H, W]
    输出:
        BGR uint8 图
    """
    H, W = diff_data.shape
    rgb = np.ones((H, W, 3), dtype=np.float32) * 255

    pos_mask = diff_data > thresh
    neg_mask = diff_data < -thresh

    diff_scaled = np.abs(diff_data) * gain
    diff_scaled = np.clip(diff_scaled, 0, 255)

    white = np.array([255, 255, 255], dtype=np.float32)
    base_blue = np.array([180, 120, 80], dtype=np.float32)  # BGR

    rgb[pos_mask] = white - (white - base_blue) * (
        diff_scaled[pos_mask][:, None] / 255.0
    )

    rgb[neg_mask] = 255 - diff_scaled[neg_mask][:, None]

    return rgb.astype(np.uint8)


def gray_to_white_bg(gray_img, thresh=8.0, gain=2.2, ref_mode="midgray"):
    """
    把灰度图转成白底事件风格图。
    ref_mode:
        - midgray: 以 128 为参考
        - mean:    以整张图均值为参考
    """
    gray = gray_img.astype(np.float32)

    if ref_mode == "midgray":
        diff = gray - 128.0
    elif ref_mode == "mean":
        diff = gray - float(gray.mean())
    else:
        raise ValueError(f"Unsupported ref_mode: {ref_mode}")

    return event_visualization_white_bg(diff, thresh=thresh, gain=gain)


def build_sdl_sdr_pairs(sdl_dir, sdr_dir):
    """
    在 sdl_dir 和 sdr_dir 中按文件名配对：
    xxx_sdl.png  <->  xxx_sdr.png

    返回:
        [
            (sdl_path, sdr_path, rel_key),
            ...
        ]
    其中 rel_key 例如:
        subdir1/subdir2/xxx
    """
    sdl_map = {}

    for root, _, files in os.walk(sdl_dir):
        for f in files:
            if not f.lower().endswith("_sdl.png"):
                continue

            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, sdl_dir)   # 例如 sub/a_sdl.png
            rel_stem = os.path.splitext(rel_path)[0]         # 例如 sub/a_sdl

            if not rel_stem.endswith("_sdl"):
                continue

            key = rel_stem[:-4]  # 去掉 "_sdl"，得到 sub/a
            sdl_map[key] = full_path

    pairs = []

    for key, sdl_path in sdl_map.items():
        sdr_rel = key + "_sdr.png"
        sdr_path = os.path.join(sdr_dir, sdr_rel)

        if os.path.exists(sdr_path):
            pairs.append((sdl_path, sdr_path, key))

    pairs.sort(key=lambda x: x[2])
    return pairs


def process_folder(
    sdl_dir,
    sdr_dir,
    output_dir,
    save_whitebg=True,
    whitebg_subdir_name="images_whitebg",
    thresh=8.0,
    gain=2.2,
    ref_mode="midgray",
):
    """
    直接从 sdl_dir / sdr_dir 读取成对图片：
        xxx_sdl.png <-> xxx_sdr.png

    输出：
        output_dir/key.png
        whitebg_dir/key_whitebg.png
    """
    pairs = build_sdl_sdr_pairs(sdl_dir, sdr_dir)

    if not pairs:
        print(f"[Warning] No sdl/sdr pairs found in:\n  {sdl_dir}\n  {sdr_dir}")
        return

    whitebg_dir = None
    if save_whitebg:
        parent_dir = os.path.dirname(output_dir)
        whitebg_dir = os.path.join(parent_dir, whitebg_subdir_name)

    for sdl_path, sdr_path, rel_key in tqdm(pairs, desc=f"Processing {os.path.basename(os.path.dirname(sdl_dir))}"):
        out_path = os.path.join(output_dir, rel_key + ".png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img_l = cv2.imread(sdl_path, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(sdr_path, cv2.IMREAD_GRAYSCALE)

        if img_l is None or img_r is None:
            print(f"[Skip] Failed to read: {sdl_path} or {sdr_path}")
            continue

        try:
            # 这里调用你已有的重建函数
            recon = poisson_reconstruct_image(img_l, img_r)
            cv2.imwrite(out_path, recon)

            if save_whitebg:
                whitebg = gray_to_white_bg(
                    recon,
                    thresh=thresh,
                    gain=gain,
                    ref_mode=ref_mode,
                )

                whitebg_path = os.path.join(
                    whitebg_dir,
                    rel_key + "_whitebg.png"
                )
                os.makedirs(os.path.dirname(whitebg_path), exist_ok=True)
                cv2.imwrite(whitebg_path, whitebg)

        except Exception as e:
            print(f"[Error] {sdl_path} or {sdr_path}: {e}")


def process_one_image(
    input_path_l,
    input_path_r,
    output_path,
    save_whitebg=True,
    whitebg_output_path=None,
    thresh=8.0,
    gain=2.2,
    ref_mode="midgray",
):
    img_l = cv2.imread(input_path_l, cv2.IMREAD_GRAYSCALE) 
    img_r = cv2.imread(input_path_r, cv2.IMREAD_GRAYSCALE) 
    if img_l is None or img_r is None:
        print(f"[Skip] Failed to read: {input_path_l} or {input_path_r}")
        return

    try:
        recon = poisson_reconstruct_image(img_l, img_r)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, recon)

        if save_whitebg:
            whitebg = gray_to_white_bg(
                recon,
                thresh=thresh,
                gain=gain,
                ref_mode=ref_mode,
            )

            if whitebg_output_path is None:
                whitebg_output_path = os.path.splitext(output_path)[0] + "_whitebg.png"

            os.makedirs(os.path.dirname(whitebg_output_path), exist_ok=True)
            cv2.imwrite(whitebg_output_path, whitebg)
            

    except Exception as e:
        print(f"[Error] {input_path_l} or {input_path_r}: {e}")


def contrast_between_images(path1, path2):
    """
    计算两张图像的平均绝对误差（MAE）。
    输入:
        img1, img2: uint8, shape [H, W] 或 [H, W, C]
    输出:
        MAE 值
    """
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    mae = np.mean(np.abs(img1 - img2))
    mse = np.mean((img1 - img2) ** 2)
    psnr = 10 * np.log10((255 ** 2) / (mse + 1e-8))
    print("MAE =", mae)
    print("PSNR =", psnr)


def main():
    # Folder-level processing
    dataset_root="./dataset_tm"
    output_root="./dataset_pos"
    splits = ["train", "valid", "test"]
    save_whitebg=False
    thresh=8.0
    gain=2.2
    ref_mode="midgray"

    for split in splits:
        sdl_dir = os.path.join(dataset_root, split, "sdl")
        sdr_dir = os.path.join(dataset_root, split, "sdr")
        out_dir = os.path.join(output_root, split, "images")

        if not os.path.isdir(sdl_dir):
            print(f"[Skip] Folder not found: {sdl_dir}")
            continue
        if not os.path.isdir(sdr_dir):
            print(f"[Skip] Folder not found: {sdr_dir}")
            continue

        process_folder(
            sdl_dir=sdl_dir,
            sdr_dir=sdr_dir,
            output_dir=out_dir,
            save_whitebg=save_whitebg,
            whitebg_subdir_name="images_whitebg",
            thresh=thresh,
            gain=gain,
            ref_mode=ref_mode,
        )

    print(f"\nDone. Reconstructed images saved to: {output_root}")

    # # Single image processing
    # input_path_l = r"/home/lxx/Projects/Tianmou_Evaluation/output/samples/sdl/imagecone_cam0_idx000_F0_sdl.png"
    # input_path_r = r"/home/lxx/Projects/Tianmou_Evaluation/output/samples/sdr/imagecone_cam0_idx000_F0_sdr.png"
    # output_path = r"./samples/test_recon.jpg"
    # whitebg_output_path = r"./samples/test_whitebg.png"

    # process_one_image(
    #     input_path_l=input_path_l,
    #     input_path_r=input_path_r,
    #     output_path=output_path,
    #     save_whitebg=True,
    #     whitebg_output_path=whitebg_output_path,
    #     thresh=8.0,
    #     gain=2.2,
    #     ref_mode="midgray",
    # )
    # img1 = "./samples/test_recon.jpg"
    # img2 = "./samples/sdl_rod_cam0_idx000_frame0000.png"
    # contrast_between_images(img1, img2)



if __name__ == "__main__":
    main()
