import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
import matplotlib.pyplot as plt



# Load the environment with rendering enabled
# gym.make('Walker2d-v5', render_mode="human") #With visualization
env = gym.make('Walker2d-v5') #Without vizualization
env.reset()

# Path to the saved SAC model
# model_path = "models/default/final_model.zip" # The default model with 1 000 000 Timesteps
# model_path = "models/default/final_model_5M.zip" # The default model with 5 000 000 Timesteps
# model_path = "models/default/SAC_Baseline_5M_LR.zip" # The default model with 5 000 000 Timesteps and LR
# model_path = "models/custom_stability/final_model.zip" # The model with 1*stability with 1 000 000 Timesteps
# model_path = "models/custom_StabVel/final_model.zip" # The model with 0.5*stability and 1*Velocity with 1 000 000 Timesteps
model_path = "models/custom_StabVel_5M/final_model.zip" # The model with 0.5*stability and 1*Velocity with 5 000 000 Timesteps
# model_path = "models/custom_StabVel_LR5M/final_model.zip" # The model with 0.5*stability and 1*Velocity with 5 000 000 Timesteps and RL
# model_path = "models/custom_Stab_Vel_x2/final_model.zip" # The model with 0.5*stability and 2*Velocity with 1 000 000 Timesteps
# model_path = "models/action_model_5M_lr.zip" # The model with a different action space


n_alive=0
distance=[]
velocity=[]
equilibrium=[]
g_vel=[]
g_dist=[]
g_acc=[]
# Load the trained SAC model
model = SAC.load(model_path)
n_episodes=1
# Visualize for 10 episodes
for episode in range(n_episodes):
    obs, info = env.reset()
    done = False

    while not done:
        # Predict the action using the trained SAC model
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        g_vel.append(info['x_velocity'])
        g_dist.append(info['x_position'])

        # Check if the episode has ended
        done = terminated or truncated
    
    distance.append(info['x_position'])
    velocity.append(info['x_velocity'])
    equilibrium.append(info['z_distance_from_origin'])
    # Step through the environment
    
    print(f"Episode {episode}: Terminated={terminated}, Truncated={truncated}, Info={info}")
    if not terminated:
        n_alive+=1

print("Success: ", n_alive, " of ", n_episodes)
print("Mean distance: ", np.mean(distance))
print("Mean velocity: ", np.mean(velocity))
print("Mean equilibrium: ", np.mean(equilibrium))
print("IC 95% for distance: (", np.mean(distance)-1.96*np.std(distance)/np.sqrt(n_episodes),", ", np.mean(distance)+1.96*np.std(distance)/np.sqrt(n_episodes), ")")
print("IC 95% for velocity: (", np.mean(velocity)-1.96*np.std(velocity)/np.sqrt(n_episodes),", ", np.mean(velocity)+1.96*np.std(velocity)/np.sqrt(n_episodes), ")")
# Close the environment after visualization
env.close()

plot_vel=True
plot_dist=False

if plot_vel or plot_dist:
    if plot_vel:
        # Generate the y-values as the array values
        y_values = g_vel
        
        # Generate the x-values as the indices of the array
        x_values = np.arange(len(g_vel))
        # Generate the y-values as the array values
    else:
        # Generate the y-values as the array values
        y_values = g_dist

        # Generate the x-values as the indices of the array
        x_values = np.arange(len(g_dist))

    # Create the plot
    plt.figure(figsize=(8, 6))  # Optional: Set figure size
    plt.plot(x_values, y_values, linestyle='-', color='b')

    # Add labels, title, and legend
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Velocity', fontsize=12)
    plt.title('5M costumized - reward function', fontsize=14)
    plt.legend()

    # Show grid for better readability (optional)
    plt.grid(True)

    # Display the plot
    plt.show()