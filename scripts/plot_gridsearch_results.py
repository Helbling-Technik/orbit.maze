import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV file
csv_path = "logs/gridsearch/grid_search_results_20250523_154632.csv"
df = pd.read_csv(csv_path)

# Drop columns that are not needed
df = df.drop(columns=["start_time", "end_time"])

# Rename columns
df = df.rename(columns={"frames_per_second": "FPS",
                        "synced_obs_delay": "SyncedObs",
                        "small_delay": "DelaySmall",
                        "small_joint_friction": "JointFrictionSmall",
                        "small_actuator_gains": "ActuatorGainSmall",
                        "use_pid": "PID"})

# Plot
sns.clustermap(df.pivot_table(
    values="result_training",
    index=["FPS", "SyncedObs", "DelaySmall"],
    columns=["JointFrictionSmall", "ActuatorGainSmall", "PID"]
), annot=True)

# Save the plot in the same folder as the CSV
plot_path = os.path.splitext(csv_path)[0] + ".png"
plt.savefig(plot_path)
print(f"Plot saved to: {plot_path}")
