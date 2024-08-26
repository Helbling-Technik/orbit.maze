import numpy as np
from stable_baselines3 import PPO


# Function to evaluate the model
def evaluate_model(model, observation):
    # TODO ROV not sure if we should test in deterministic mode
    action, _ = model.predict(observation, deterministic=True)
    print(f"Action: {action}    shape: {action.shape}")

    return action


"""Define observations"""
# State space observations
joint_pos = np.array([0.0, 0.0])
joint_est_vel = np.array([0.0, 0.0])
sphere_pos = np.array([0.0, 0.0, 0.0])
sphere_est_vel = np.array([0.0, 0.0, 0.0])
target1_pos = np.array([0.0, 0.0])
target2_pos = np.array([0.0, 0.0])
target3_pos = np.array([0.0, 0.0])
# Image observations
image = np.zeros((1, 1, 16, 16))

"""Evaluating the old model"""
# Fill the observation tensor
old_observation = np.concatenate(
    [joint_pos, joint_est_vel, sphere_pos, sphere_est_vel, target1_pos, target2_pos, target3_pos]
)

# Load the trained PPO model
old_model_path = "logs/sb3/Isaac-Maze-v0/2024-07-11_LearnMazeAdaptedActionSpace/model_491520000_steps.zip"
old_model = PPO.load(old_model_path)

evaluate_model(old_model, old_observation)

"""Evaluating the new model"""
# Fill the 'mlp_policy' array with shape (1, 16) and dtype float32
mlp_values = np.concatenate(
    [joint_pos, joint_est_vel, sphere_pos, sphere_est_vel, target1_pos, target2_pos, target3_pos]
)
mlp_policy = mlp_values.reshape(1, -1).astype(np.float32)

# Fill the 'cnn_policy' array with shape (1, 1, 16, 16) and dtype float32
cnn_policy = image.astype(np.float32)

# Define the structure as a dictionary
new_observation = {"mlp_policy": mlp_policy, "cnn_policy": cnn_policy}

# Load the trained PPO model
new_model_path = "logs/sb3/Isaac-Maze-v0/2024-08-26_08-44-35/model_147456000_steps.zip"
new_model = PPO.load(new_model_path)

evaluate_model(new_model, new_observation)
