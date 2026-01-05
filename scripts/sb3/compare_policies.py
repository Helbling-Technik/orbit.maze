import itertools
import subprocess
import csv
import os
from datetime import datetime

# Define your hyperparameter grid
higher_fps = 50
lower_fps = 30
n_timesteps = 50000000

checkpoints = [
    # "logs/gridsearch/2025-06-13_14-14-52_Gridsearch/2025-06-14_21-53-39/model.zip",
    # "logs/sb3/Isaac-Maze-v0/2025-08-11_11-32-26/model_90112000_steps.zip",
    # "logs/sb3/Isaac-Maze-v0/2025-08-11_11-32-26/model_229376000_steps.zip",
    # "logs/sb3/Isaac-Maze-v0/2025-08-11_11-32-26/model_479232000_steps.zip",
    # "logs/sb3/Isaac-Maze-v0/2025-08-08_15-06-03/model_1531904000_steps.zip"
    # "logs/gridsearch/2025-06-13_14-14-52_Gridsearch/2025-06-15_04-55-14/model.zip"
    "logs/sb3/Isaac-Maze-v0/2025-08-11_11-32-26/model_1056768000_steps.zip"
]

# Output CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"logs/comparisons/policy_comparison_{timestamp}.csv"

# Create directory if needed
os.makedirs("logs/comparisons", exist_ok=True)

fieldnames = checkpoints
with open(output_file, mode="w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

scores = {}
for i, checkpoint in enumerate(checkpoints):
    print(f"\nRunning configuration {i + 1}/{len(checkpoints)}: {checkpoint}")

    # Build the command
    cmd = [
        "python",
        "scripts/sb3/evaluate_policy.py",
        "--task",
        "Isaac-Maze-v0",
        "--num_envs",
        "100",
        "--num_episodes",
        "1",
        "--maze_start_point",
        "-1",
        "--pos_ctrl",
        "--real_maze",
        "--frames_per_second",
        "30",
        "--set_params",
        "--checkpoint",
        checkpoint,
        "--headless",
    ]
    print(cmd)

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        print(output)
        mean_score_line = next((line for line in output.splitlines() if "FINAL_MEAN_SCORE:" in line), None)
        median_score_line = next((line for line in output.splitlines() if "FINAL_MEDIAN_SCORE:" in line), None)
        max_score_line = next((line for line in output.splitlines() if "FINAL_MAX_SCORE:" in line), None)
        if mean_score_line and median_score_line and max_score_line:
            mean_score = float(mean_score_line.split("FINAL_MEAN_SCORE:")[1].strip())
            median_score = float(median_score_line.split("FINAL_MEDIAN_SCORE:")[1].strip())
            max_score = float(max_score_line.split("FINAL_MAX_SCORE:")[1].strip())
        else:
            mean_score = "NaN"
            median_score = "NaN"
            max_score = "NaN"
            end_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    except Exception as e:
        print(f"Error running configuration: {e}")
        end_time = "ERROR"
        mean_score = "ERROR"
        median_score = "ERROR"
        max_score = "ERROR"

    scores[checkpoint] = [mean_score, median_score, max_score]
# Write results to CSV
with open(output_file, mode="a", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    row = {ckpt: scores.get(ckpt, "")[0] for ckpt in checkpoints}
    writer.writerow(row)
    row = {ckpt: scores.get(ckpt, "")[1] for ckpt in checkpoints}
    writer.writerow(row)
    row = {ckpt: scores.get(ckpt, "")[2] for ckpt in checkpoints}
    writer.writerow(row)
print(f"\nComparison complete. Results saved to {output_file}")
