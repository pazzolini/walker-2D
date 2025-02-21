# Customizing OpenAI Gym Environments and Implementing Reinforcement Learning Agents with Stable Baselines

# Overview

This project was developed as part of the "Introduction to Intelligent and Autonomous Systems". We compared several RL algorithms on the Walker2D environment and proposed some modifications, regarding the action space and reward functions.

## Files and Folders

We have included the following files, folders and scripts:

- `BaselineComparison.ipynb`: Initial comparison of various RL algorithms on the Walker2D baseline environment followed by performance optimization of SAC through learning rate scheduling for extended training.
- `LR_5M_costum_EqVel_final.py`: A custom Walker2D environment implementation with a modified reward function that emphasizes stability and forward velocity.
- `test_action.ipynb`: Action space modifications for the Walker2D environment through various wrappers, aiming to achieve more controlled and natural walking movements.
- `viz.py`: Visualization and evaluation script for different trained Walker2D models. It loads a saved model and runs test episodes, collecting metrics. It includes plotting capabilities for tracking velocity or distance over time and outputs performance statistics including confidence intervals.
- `models`: Contains different trained versions of the Walker2D agent,tensorboard logs, as well as the default versions (1M and 5M timesteps).

@FMSCarvalho (Filipe Carvalho), @luanalegi (Luana Letra), @pazzolini (Vítor Ferreira).
