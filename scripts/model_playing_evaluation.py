import pandas as pd
import numpy as np

# Load CSV file
path_to_rewards_log = "logs/sb3/Isaac-Maze-v0/test-scores/test_run_2025-05-20_07-44-38.csv"
df = pd.read_csv(path_to_rewards_log)

# Drop the 'env' column to focus only on reward values
reward_values = df.drop(columns='env')

# Compute per-environment stats
reward_stats = pd.DataFrame()
reward_stats['env'] = df['env']
reward_stats['mean_reward'] = reward_values.mean(axis=1, skipna=True)
reward_stats['std_reward'] = reward_values.std(axis=1, skipna=True)

# Compute overall stats (flatten all values into a single array, ignoring NaNs)
all_rewards = reward_values.values.flatten()
all_rewards = all_rewards[~np.isnan(all_rewards)]  # Remove NaNs

overall_mean = np.mean(all_rewards)
overall_std = np.std(all_rewards)

# Display results
print("Per-environment reward statistics:")
print(reward_stats)
print("\nOverall reward statistics across all environments and runs:")
print(f"Mean reward: {overall_mean:.4f}")
print(f"Standard deviation: {overall_std:.4f}")

# TODO ROV write back to file
# Optionally, save per-env stats to CSV
# reward_stats.to_csv("reward_statistics.csv", index=False)
