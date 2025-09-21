import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "OFres"
corner_types = ["FAST", "HARRIS", "ORB"]
image_types = ["cone_output_cam0", "Ixy", "td", "Poisson_denoised_out"]

FORM_COLORS = {
    "cone_output_cam0": "#07141e",
    "Ixy": "#ff7f0e",
    "td": "#319cee",
    "Poisson_denoised_out": "#af8ee3"
}

def parse_txt_file(path):
    """读取单个txt，返回平均重投影误差 & 平均轨迹长度"""
    try:
        df = pd.read_csv(path)
        avg_error = df["AvgReprojError"].mean()
        avg_traj = df["AvgTrajectoryLength"].mean()
        return avg_error, avg_traj
    except Exception:
        return np.nan, np.nan

def collect_data():
    """收集两个结果 dict: 
       reproj_results: {corner -> {speed -> {image_type -> avg_error}}}
       traj_results:   {corner -> {speed -> {image_type -> avg_traj}}}
    """
    reproj_results = {c: {} for c in corner_types}
    traj_results = {c: {} for c in corner_types}

    for fname in os.listdir(BASE_DIR):
        if not fname.endswith(".txt"):
            continue
        parts = fname.replace(".txt", "").split("_")
        if len(parts) < 4:
            continue

        corner, _, speed, image_type = parts[0], parts[1], parts[2], "_".join(parts[3:])
        if corner not in corner_types or image_type not in image_types:
            continue

        speed = int(speed)
        path = os.path.join(BASE_DIR, fname)
        avg_error, avg_traj = parse_txt_file(path)

        # --- 公平对比：cone 不变，其余除以 25 ---
        if image_type != "cone_output_cam0":
            avg_traj = avg_traj / 25.0

        if speed not in reproj_results[corner]:
            reproj_results[corner][speed] = {}
            traj_results[corner][speed] = {}

        reproj_results[corner][speed][image_type] = avg_error
        traj_results[corner][speed][image_type] = avg_traj

    return reproj_results, traj_results


def plot_bar(results, ylabel, title_suffix, save_dir="plots_res"):
    os.makedirs(save_dir, exist_ok=True)

    for corner, speed_dict in results.items():
        speeds = sorted(speed_dict.keys())
        x = np.arange(len(speeds))  # x轴位置

        width = 0.15  # 每个条形的宽度
        offset = np.linspace(-1.5*width, 1.5*width, len(image_types))  # 偏移量

        plt.figure(figsize=(5, 4))
        for i, img_type in enumerate(image_types):
            vals = [speed_dict[s].get(img_type, np.nan) for s in speeds]
            plt.bar(
                x + offset[i], 
                vals, 
                width=width,
                color=FORM_COLORS[img_type], 
                label=img_type,
                alpha=0.4,   # 设置透明度
                edgecolor="black"  # 黑色边框更清晰
            )

        # 重投影误差用对数坐标，轨迹长度保持线性
        if "Reproj" in title_suffix:
            plt.yscale("log")

        plt.xticks(x, speeds)
        plt.xlabel("Speed (rpm)")
        plt.ylabel(ylabel)
        plt.title(f"{corner} - {title_suffix}")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        # plt.legend()

        save_path = os.path.join(save_dir, f"{corner}_{title_suffix.replace(' ', '_')}.svg")
        plt.savefig(save_path, format="svg")
        plt.close()
        print(f"Saved: {save_path}")

def main():
    reproj_results, traj_results = collect_data()
    # 绘制重投影误差
    plot_bar(reproj_results, ylabel="Avg Reprojection Error (log scale)", title_suffix="Avg Reproj Error", save_dir="plots_reproj")
    # 绘制轨迹长度
    plot_bar(traj_results, ylabel="Avg Trajectory Length", title_suffix="Avg Trajectory Length", save_dir="plots_traj")

if __name__ == "__main__":
    main()
