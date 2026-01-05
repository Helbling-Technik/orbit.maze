import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <path_to_csv>")
    sys.exit(1)

csv_path = sys.argv[1]

# Read CSV
df = pd.read_csv(csv_path)

# Assume first row = mean, second = median, third = max
metrics = ["Mean", "Median"]
checkpoints = list(df.columns)
print("df ", df)
print("df shape: ", df.shape)
print("checkpoints ", checkpoints)

# Convert wide CSV to long form
rows = []
for metric_idx, metric_name in enumerate(metrics):
    for ckpt in checkpoints:
        rows.append({"Checkpoint": os.path.basename(ckpt), "Metric": metric_name, "Score": df.iloc[metric_idx][ckpt]})

plot_df = pd.DataFrame(rows)

# Plot grouped bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=plot_df, x="Checkpoint", y="Score", hue="Metric", palette="Set2", edgecolor="black")

plt.title("Policy Comparison", fontsize=14, weight="bold")
plt.ylabel("Score")
plt.xlabel("Checkpoint")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Metric")
plt.tight_layout()
plt.show()
