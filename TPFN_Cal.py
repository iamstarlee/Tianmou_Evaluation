import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "warp_results/multi"
corner_types = ["FAST", "Harris", "ORB"]
image_types = ["cone_output_cam0", "Ixy", "td", "Poisson_denoised_out"]

FORM_COLORS = {
    "cone_output_cam0": "#07141e",
    "Ixy": "#ff7f0e",
    "td": "#319cee",
    "Poisson_denoised_out": "#af8ee3"
}

def parse_txt_file(path):
    """读取单个txt，返回 TP/FP/FN DataFrame"""
    try:
        df = pd.read_csv(path)
        return df[["TP", "FP", "FN"]]
    except Exception:
        return pd.DataFrame(columns=["TP", "FP", "FN"])

def compute_metrics(tp, fp, fn):
    acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return acc, precision, recall, f1

def collect_and_summarize():
    """收集所有txt并计算四个指标"""
    summary = []
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
            if df.empty:
                tp_mean, fp_mean, fn_mean = 0, 0, 0
            else:
                tp_mean = df["TP"].mean()
                fp_mean = df["FP"].mean()
                fn_mean = df["FN"].mean()

            acc, precision, recall, f1 = compute_metrics(tp_mean, fp_mean, fn_mean)

            summary.append({
                "corner": corner,
                "image": image_type,
                "speed": speed_int,
                "ACC": acc,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            })

    return pd.DataFrame(summary)

def plot_metrics(df, save_dir="plots_line"):
    os.makedirs(save_dir, exist_ok=True)
    metrics = ["ACC", "Precision", "Recall", "F1"]

    for corner in corner_types:
        for metric in metrics:
            plt.figure(figsize=(8, 5))
            sub_df = df[df["corner"] == corner]

            for img in image_types:
                vals = sub_df[sub_df["image"] == img].sort_values("speed")
                plt.plot(vals["speed"], vals[metric], marker="o",
                         color=FORM_COLORS[img], label=img, alpha=0.4)

            plt.ylim(0, 1.05)
            plt.xlabel("Speed (rpm)")
            plt.ylabel(metric)
            plt.title(f"{corner} - {metric}")
            # plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)

            save_path = os.path.join(save_dir, f"{corner}_{metric}.svg")
            plt.savefig(save_path, format="svg")
            plt.close()
            print(f"Saved: {save_path}")

def main():
    df = collect_and_summarize()
    csv_path = os.path.join(BASE_DIR, "metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")
    plot_metrics(df)

if __name__ == "__main__":
    main()
