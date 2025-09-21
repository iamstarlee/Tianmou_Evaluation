#!/usr/bin/env python3
"""
plot_by_form.py (改进版)

区别：
- mean_conf 在归一化后再计算方差
- 不再使用已有的 var_mean_conf，而是重新基于归一化序列计算 variance
"""

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- config ----------
data_dir = "/result"
if not os.path.isdir(data_dir):
    data_dir = "result"
if not os.path.isdir(data_dir):
    data_dir = "."
PATTERN = os.path.join(data_dir, "*_multi_*.txt")
# ---------- end config ------

# 定义固定颜色（Matplotlib 支持颜色名 / HEX / RGB）
FORM_COLORS = {
    "cone_output_cam0": "#07141e",        # 蓝
    "Ixy": "#ff7f0e",                     # 橙
    "td": "#319cee",                      # 绿
    "Poisson_denoised_out": "#af8ee3"     # 红
}

files = sorted(glob.glob(PATTERN))
if not files:
    raise SystemExit(f"No files found with pattern: {PATTERN}")

KNOWN_FORMS = ["cone_output_cam0", "Ixy", "td", "Poisson_denoised_out"]

def detect_algo_from_basename(basename_low):
    if basename_low.startswith("harris"):
        return "HARRIS"
    if basename_low.startswith("orb"):
        return "ORB"
    if basename_low.startswith("fast"):
        return "FAST"
    return "UNKNOWN"

def detect_form_from_basename(basename_low):
    if "cone_output_cam0" in basename_low:
        return "cone_output_cam0"
    if "ixy" in basename_low:
        return "Ixy"
    if "poisson" in basename_low:
        return "Poisson_denoised_out"
    if "td" in basename_low:
        return "td"
    return None

def safe_read_table(path):
    try:
        df = pd.read_csv(path, sep=r'\s+|\t|,', engine='python')
    except Exception:
        df = pd.read_csv(path, header=None, sep=r'\s+|\t|,', engine='python')
    return df

# --- 收集所有数据 ---
raw_records = []  # 保存所有单帧数据，方便统一归一化
for f in files:
    b = os.path.basename(f)
    b_low = b.lower()

    algo = detect_algo_from_basename(b_low)
    if algo == "UNKNOWN":
        continue

    m = re.search(rf"{algo.lower()}_multi_(\d+)_", b_low)
    if not m:
        print(f"[WARN] skip (cannot parse rpm): {b}")
        continue
    rpm = int(m.group(1))

    form = detect_form_from_basename(b_low)
    if form is None:
        print(f"[WARN] unknown form, skipping file: {b}")
        continue

    df = safe_read_table(f)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "mean_conf" in df.columns:
        conf_vals = df["mean_conf"].astype(float).values
    else:
        conf_vals = df.iloc[:, 1].astype(float).values

    if "num_kp" in df.columns:
        kp_vals = df["num_kp"].astype(float).values
    elif df.shape[1] > 2:
        kp_vals = df.iloc[:, 2].astype(float).values
    else:
        kp_vals = np.zeros_like(conf_vals)

    raw_records.append({
        "algo": algo, "form": form, "rpm": rpm,
        "conf_seq": conf_vals, "kp_seq": kp_vals,
        "file": f
    })

# --- 统一归一化 mean_conf ---
all_conf = np.concatenate([r["conf_seq"] for r in raw_records])
conf_min, conf_max = float(all_conf.min()), float(all_conf.max())
conf_range = conf_max - conf_min if conf_max > conf_min else 1.0

for r in raw_records:
    r["conf_seq_norm"] = (r["conf_seq"] - conf_min) / conf_range

# --- 重新统计 ---
data_all = {}
for r in raw_records:
    algo, form, rpm = r["algo"], r["form"], r["rpm"]
    conf_seq_norm = r["conf_seq_norm"]
    kp_seq = r["kp_seq"]

    mean_conf_avg = conf_seq_norm.mean()
    conf_var = conf_seq_norm.var()
    num_kp_avg = kp_seq.mean()
    kp_var = kp_seq.var()

    data_all.setdefault(algo, {}).setdefault(form, []).append({
        "rpm": rpm,
        "mean_conf": mean_conf_avg,
        "conf_var": conf_var,
        "num_kp": num_kp_avg,
        "kp_var": kp_var,
        "file": r["file"]
    })

# --- 绘图 ---
for algo, data_by_form in data_all.items():
    dfs = {}
    for form, lst in data_by_form.items():
        if lst:
            dfs[form] = pd.DataFrame(lst).sort_values("rpm").reset_index(drop=True)

    if not dfs:
        continue

    # 保存汇总 CSV
    summary_rows = []
    for form, dfk in dfs.items():
        for _, row in dfk.iterrows():
            summary_rows.append({
                "form": form, "rpm": int(row["rpm"]),
                "mean_conf": row["mean_conf"], "num_kp": row["num_kp"],
                "conf_var": row["conf_var"], "kp_var": row["kp_var"],
                "file": row["file"]
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(data_dir, f"{algo}_summary_by_form.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV to {summary_csv}")

    # --- 图1: mean_conf (归一化后再求方差) ---
    plt.figure(figsize=(10, 6))
    for form, dfk in dfs.items():
        x = dfk["rpm"].astype(float)
        y = dfk["mean_conf"]
        yerr = np.sqrt(dfk["conf_var"])
        c = FORM_COLORS.get(form, None)   # 如果 form 不在字典里，用默认颜色
        plt.plot(x, y, marker="o", label=form, color=c)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.18, color=c)

    plt.xlabel("RPM")
    plt.ylabel("Normalized mean_conf")
    plt.title(f"{algo}: normalized mean_conf by image form")
    plt.legend()
    out1 = os.path.join(data_dir, f"{algo}_mean_conf_by_form_normalized.svg")
    plt.savefig(out1, format="svg")
    plt.close()

    # --- 图2: num_kp ---
    plt.figure(figsize=(10, 6))
    for form, dfk in dfs.items():
        x = dfk["rpm"].astype(float)
        y = dfk["num_kp"]
        yerr = np.sqrt(dfk["kp_var"])
        c = FORM_COLORS.get(form, None)
        plt.plot(x, y, marker="o", label=form, color=c)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.18, color=c)

    plt.axhline(y=17, color="black", linewidth=2.5, linestyle="--", label="GT=17")
    plt.xlabel("RPM")
    plt.ylabel("num_kp (average over frames)")
    plt.title(f"{algo}: num_kp by image form")
    # plt.legend()
    out2 = os.path.join(data_dir, f"{algo}_num_kp_by_form.svg")
    plt.savefig(out2, format="svg")
    plt.close()
    print(f"Saved {algo} num_kp figure to {out2}")

print("Done.")
