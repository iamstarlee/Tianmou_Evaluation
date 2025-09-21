import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "warp_results/multi"
corner_types = ["FAST", "Harris", "ORB"]   # 角点种类
image_types = ["cone_output_cam0", "Ixy", "td", "Poisson_denoised_out"]  # 图像形式

FORM_COLORS = {
    "cone_output_cam0": "#07141e",        # 蓝黑
    "Ixy": "#ff7f0e",                     # 橙
    "td": "#319cee",                      # 绿
    "Poisson_denoised_out": "#af8ee3"     # 紫
}

def parse_txt_file(path):
    """读取单个txt，返回完整的 TP/FP/FN 数据列表"""
    try:
        df = pd.read_csv(path)
        return df[["TP", "FP", "FN"]]
    except Exception:
        return pd.DataFrame(columns=["TP", "FP", "FN"])

def collect_data():
    """返回 DataFrame: [corner, image, speed, metric, value]"""
    records = []

    for speed in sorted(os.listdir(BASE_DIR)):
        if not speed.isdigit():
            continue
        speed_int = int(speed)
        speed_path = os.path.join(BASE_DIR, speed)

        for fname in os.listdir(speed_path):
            if not fname.endswith(".txt"):
                continue

            parts = fname.split("_", 1)
            if len(parts) < 2:
                continue
            corner, image_type = parts[0], parts[1].replace(".txt", "")

            if corner not in corner_types or image_type not in image_types:
                continue

            df = parse_txt_file(os.path.join(speed_path, fname))
            for metric in ["TP", "FP", "FN"]:
                for v in df[metric]:
                    records.append({
                        "corner": corner,
                        "image": image_type,
                        "speed": speed_int,
                        "metric": metric,
                        "value": v
                    })

    return pd.DataFrame(records)

def plot_boxplots(df, save_dir="plots_svg"):
    """绘制箱形图并保存为 SVG"""
    import os
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["TP", "FP", "FN"]

    for corner in corner_types:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
        fig.suptitle(f"{corner} Corner Detection (Boxplots per Speed)", fontsize=14)

        for ax, metric in zip(axes, metrics):
            sub_df = df[(df["corner"] == corner) & (df["metric"] == metric)]

            speeds = sorted(sub_df["speed"].unique())
            data = []
            positions = []
            colors = []

            group_width = len(image_types)  # 每个 speed 内的箱子数
            gap = 2                         # speed 之间的额外间隔
            width = 0.6                     # 箱子宽度

            for i, speed in enumerate(speeds):
                base_pos = i * (group_width + gap)  # 每组起始位置
                for j, img in enumerate(image_types):
                    vals = sub_df[(sub_df["speed"] == speed) & (sub_df["image"] == img)]["value"].values
                    if len(vals) > 0:
                        data.append(vals)
                        positions.append(base_pos + j + 1)
                        colors.append(FORM_COLORS[img])

            # 画箱形图
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=width,
                patch_artist=True,
                showfliers=True
            )

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.4)  # 半透明

            # 竖线分隔组
            for i in range(len(speeds) - 1):
                sep_x = (i + 1) * (group_width + gap) - gap/2
                ax.axvline(sep_x, color="gray", linestyle="--", alpha=0.4)

            # X 轴刻度放在每组中间
            tick_positions = [
                i * (group_width + gap) + group_width / 2 + 0.5
                for i in range(len(speeds))
            ]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(s) for s in speeds])

            # GT=17 参考线
            if metric == "TP":
                ax.axhline(y=17, color="red", linestyle="-", linewidth=1.2, alpha=0.8, label="GT=17")

            ax.set_title(metric)
            ax.set_xlabel("Speed (rpm)")
            ax.set_ylabel("Count")
            ax.grid(True, linestyle="--", alpha=0.6)

        # 图例
        handles = [plt.Rectangle((0, 0), 1, 1, color=FORM_COLORS[img], alpha=0.6) for img in image_types]
        handles.append(plt.Line2D([0], [0], color="red", linewidth=1.2, label="GT=17"))
        # fig.legend(handles, [*image_types, "GT=17"], loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存 SVG
        save_path = os.path.join(save_dir, f"{corner}_boxplots.svg")
        fig.savefig(save_path, format="svg")
        print(f"Saved SVG: {save_path}")

        plt.close(fig)  # 关闭图，节省内存


def main():
    df = collect_data()
    plot_boxplots(df, save_dir="plots_svg")

if __name__ == "__main__":
    main()
