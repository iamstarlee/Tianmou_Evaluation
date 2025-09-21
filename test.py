import pandas as pd

df = pd.read_csv("warp_results/multi/1200/Harris_cone_output_cam0.txt")
df[["TP", "FP", "FN"]].mean().to_dict()
print(df[["TP", "FP", "FN"]].mean().to_dict())