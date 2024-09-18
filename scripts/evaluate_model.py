import numpy as np
from stable_baselines3 import PPO
import time


# TODO ROV left over for hardware module, only left here for debug purpose
class RL_Agent:
    def __init__(self, model_path, new_model=False):
        self.new_model = new_model
        self.model = PPO.load(model_path)

    def update_status(self, message):
        print(f"Status: {message}")

    def other_nodes_available(self):
        other_nodes_ready = True
        return other_nodes_ready

    def predict_action(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def set_action(self, actions):
        # New model has shape (1,2)
        if self.new_model:
            print(f"Action: {actions}    shape: {actions.shape}")
        # Old model has shape (2,)
        else:
            print(f"Action: {actions}    shape: {actions.shape}")

    def get_observation(self):
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

        if self.new_model:
            # Fill the 'mlp_policy' array with shape (1, 16) and dtype float32
            mlp_values = np.concatenate(
                [joint_pos, joint_est_vel, sphere_pos, sphere_est_vel, target1_pos, target2_pos, target3_pos]
            )
            mlp_policy = mlp_values.reshape(1, -1).astype(np.float32)

            # Fill the 'cnn_policy' array with shape (1, 1, 16, 16) and dtype float32
            cnn_policy = image.astype(np.float32)

            # Define the structure as a dictionary
            observation = {"mlp_policy": mlp_policy, "cnn_policy": cnn_policy}
        else:
            observation = np.concatenate(
                [joint_pos, joint_est_vel, sphere_pos, sphere_est_vel, target1_pos, target2_pos, target3_pos]
            )

        return observation


def main():
    # Path to trained model
    model_path = "logs/sb3/Isaac-Maze-v0/2024-07-11_LearnMazeAdaptedActionSpace/model_491520000_steps.zip"
    new_model = False
    # model_path = "logs/sb3/Isaac-Maze-v0/2024-08-26_08-44-35_pos_Real_MultiInput_Normalized_Uniform_Penalty_JointLimits_Friction_Noise_DistanceToTarget/model.zip"
    # new_model = True

    frequency = 50
    update_period = 1.0 / frequency  # Period in seconds (20ms)
    wait_period = 1.0  # Wait 1s if nothing is happening
    agent = RL_Agent(model_path, new_model)

    while True:
        if agent.other_nodes_available():
            start_time = time.time()  # Record the start time of the loop iteration

            # Update actions
            obs = agent.get_observation()
            if obs is not None:
                actions = agent.predict_action(obs)
                agent.set_action(actions)
                agent.update_status("Calculated action")
            else:
                agent.update_status("Ready, but no observations")

            # Calculate the time taken for the prediction
            elapsed_time = time.time() - start_time
            # Sleep for the remaining time to achieve the desired frequency
            time.sleep(max(0, update_period - elapsed_time))
        else:
            start_time = time.time()  # Record the start time of the loop iteration
            # Update status
            agent.update_status("Waiting for other nodes")

            elapsed_time = time.time() - start_time
            # Sleep for the remaining time to achieve the desired frequency
            time.sleep(max(0, wait_period - elapsed_time))


if __name__ == "__main__":
    # run the main function
    main()
