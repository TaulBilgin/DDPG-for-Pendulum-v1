import torch.nn as nn
import torch
import numpy as np
import gymnasium as gym

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the Actor network class
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()

        # Define the layers for the Actor network
        self.l1 = nn.Linear(state_dim, net_width)  # Input layer
        self.l2 = nn.Linear(net_width, 300)  # Hidden layer
        self.l3 = nn.Linear(300, action_dim)  # Output layer

    # Define forward pass for the Actor network
    def forward(self, state):
        a = torch.relu(self.l1(state))  # Apply ReLU activation to the first layer
        a = torch.relu(self.l2(a))  # Apply ReLU activation to the second layer
        a = torch.tanh(self.l3(a)) * 2  # Apply tanh activation and scale action
        return a
    
# Function to select action from the Actor model given the current state
def select_action(now_state, actor):
    with torch.no_grad():  # No gradient calculation is needed during action selection
        now_state = torch.FloatTensor(now_state).to(device)  # Convert the current state to tensor and move to device
        action = actor(now_state).cpu().numpy()[0]  # Get the action from the actor network and convert it to numpy
        return np.array([action])  # Return the action as a numpy array

# Create the environment with human rendering mode
env = gym.make('Pendulum-v1', render_mode="human")
state_dim = 3  # Dimension of the state space
action_dim = 1  # Dimension of the action space
net_width = 400  # Width of the network (number of units in hidden layers)

# Seed everything for reproducibility
env_seed = 0
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Initialize the Actor network and move it to the device (GPU or CPU)
actor = Actor(state_dim, action_dim, net_width).to(device)

# Load the pre-trained model weights for the Actor network
actor.load_state_dict(torch.load("your model name"))

# Switch the Actor network to evaluation mode (disables dropout, etc.)
actor.eval()

run = 0  # Initialize episode counter
while run < 10:  # Run for 10 episodes
    # Reset the environment and get the initial state
    now_state = env.reset(seed=env_seed)[0]
    step = 0  # Reset step counter for the episode

    # Interact with the environment for 200 steps (or until termination)
    while step < 200:  
        action = select_action(now_state, actor)  # Select action using the actor network
        step += 1  # Increment step counter
        next_state, reward, done, truncated, info = env.step(action)  # Take the selected action in the environment
        now_state = next_state  # Update the current state to the next state
