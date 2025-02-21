import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
import os
from datetime import datetime
import json
import torch
import random


# Directories to save the model and logs
models_dir = "models/custom_StabVel_LR5M"
logdir = "logs_custom_StabVel_LR5M"

# Create directories if they don't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

class CustomWalker2dEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomWalker2dEnv, self).__init__(env)
        # Access the unwrapped environment to get sim
        self.sim = env.unwrapped.model  # Access MuJoCo simulation object
        self.data = env.unwrapped.data  # Access MuJoCo runtime state

        # Flatten the observation space
        original_obs_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(np.prod(original_obs_space.shape),), 
            dtype=np.float32
        )

    def step(self, action):
        # Perform the action in the environment
        result = self.env.step(action)

        # Unpack the result based on the number of returned values
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated


        # access simulation data
        joint_positions= self.data.qpos 
        
        # Stability-based reward calculation
        torso_angle = joint_positions[1]  # Index 1 corresponds to torso angle
        upright_reward = 1.0 - abs(torso_angle)  # Reward for being upright
        upright_reward = max(0, upright_reward)
         

        # Forward progress reward
        forward_progress_reward = info.get("x_velocity", 0)
        if forward_progress_reward is None:
            forward_progress_reward = self.data.qvel[0]

        
        # Adding custom reward to the original reward
        custom_reward = reward + 0.5*upright_reward+forward_progress_reward 
        
        # Flatten the observation for compatibility
        flat_obs = obs.flatten()
        
        # Print the rewards for comparison
        #print(f"Original Reward: {reward}, Upright Reward: {upright_reward}, Forward Reward: {forward_progress_reward}")
        
        return flat_obs, custom_reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Reset the environment and flatten the initial observation
        obs, info = self.env.reset(**kwargs)
        return obs.flatten(), info




# Create the environment
try:
    base_env = gym.make('Walker2d-v5')
    env = CustomWalker2dEnv(base_env)
    env.reset()
    print("Environment successfully created and reset.")
except Exception as e:
    print(f"Failed to create environment: {e}")
    env = None

# Learning rate scheduler function
def learning_rate_schedule(progress_remaining):
    """
    Custom learning rate scheduler.
    Args:
        progress_remaining (float): Fraction of training remaining (1 = start, 0 = end).
    Returns:
        float: Adjusted learning rate.
    """
    initial_lr = 3e-4  # Initial learning rate
    return initial_lr * progress_remaining  # Linear decay



# Initialize the SAC  model if not already initialized
model_path = f"{models_dir}/final_model.zip"

# Check if the model path exists
if os.path.exists(model_path) and env is not None:
    try:
        model = SAC.load(model_path, env=env,learning_rate=learning_rate_schedule)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=logdir,learning_rate=learning_rate_schedule)
else:
    print("Model path does not exist or environment creation failed. Initializing new model.")
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# Training parameters
TIMESTEPS = 5000000

if env is not None:
    # Train the model
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="Stability_reward")
    #Save the model after the full training
    model.save(f"{models_dir}/final_model")

    env.close()
else:
    print("Skipping training as the environment was not created successfully.")