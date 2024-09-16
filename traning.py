import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import numpy as np
import gymnasium as gym


# Define Actor network for the policy
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()

        # Define layers for the Actor network
        self.l1 = nn.Linear(state_dim, net_width)  # First hidden layer
        self.l2 = nn.Linear(net_width, 300)  # Second hidden layer
        self.l3 = nn.Linear(300, action_dim)  # Output layer for action

    # Forward pass through the network
    def forward(self, state):
        a = torch.relu(self.l1(state))  # Apply ReLU activation
        a = torch.relu(self.l2(a))  # Apply ReLU activation
        a = torch.tanh(self.l3(a)) * 2  # Output scaled by 2 for action
        return a


# Define Critic network for Q-value estimation
class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()

        # Define layers for the Critic network
        self.l1 = nn.Linear(state_dim + action_dim, net_width)  # First hidden layer
        self.l2 = nn.Linear(net_width, 300)  # Second hidden layer
        self.l3 = nn.Linear(300, 1)  # Output layer for Q-value

    # Forward pass through the network
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)  # Concatenate state and action
        q = F.relu(self.l1(sa))  # Apply ReLU activation
        q = F.relu(self.l2(q))  # Apply ReLU activation
        q = self.l3(q)  # Output Q-value
        return q


# Function to select action based on the current state and the actor network
def select_action(now_state, actor, deterministic):
    noise_level = 0.1  # Noise level for exploration
    max_action = 2  # Maximum action value
    action_dim = 1  # Dimension of the action space
    with torch.no_grad():
        now_state = torch.FloatTensor(now_state).to(device)  # Convert state to tensor
        action = actor(now_state).cpu().numpy()[0]  # Get action from actor network
        if deterministic:
            return np.array([action])  # Return deterministic action
        else:
            noise = np.random.normal(0, max_action * noise_level, size=action_dim)  # Add noise for exploration
            return (action + noise).clip(-max_action, max_action)  # Clip action to valid range


# Training function for both actor and critic networks
def train(actor, actor_optimizer, actor_target, q_critic, q_critic_optimizer, q_critic_target, replay_buffer):
    batch_size = 128  # Batch size for sampling from replay buffer
    gamma = 0.99  # Discount factor for future rewards
    tau = 0.01  # Soft update factor for target networks
    
    # Compute the target Q-value
    with torch.no_grad():
        now_state, action, reward, next_state = replay_buffer.sample(batch_size)  # Sample from replay buffer
        target_a_next = actor_target(next_state)  # Get action from target actor
        target_Q = q_critic_target(next_state, target_a_next)  # Get Q-value from target critic
        target_Q = reward + gamma * target_Q  # Calculate target Q-value

    # Get current Q-value estimates from critic
    current_Q = q_critic(now_state, action)

    # Compute critic loss using Mean Squared Error (MSE)
    q_loss = F.mse_loss(current_Q, target_Q)

    # Optimize the critic network
    q_critic_optimizer.zero_grad()
    q_loss.backward()  # Backpropagate the loss
    q_critic_optimizer.step()

    # Update the Actor network
    a_loss = -q_critic(now_state, actor(now_state)).mean()  # Actor loss (maximize Q-value)
    actor_optimizer.zero_grad()
    a_loss.backward()  # Backpropagate the actor loss
    actor_optimizer.step()

    # Soft update for the target networks (Actor and Critic)
    with torch.no_grad():
        for param, target_param in zip(q_critic.parameters(), q_critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)  # Update target critic

        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)  # Update target actor


# Replay Buffer for storing experience tuples (state, action, reward, next_state)
class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size  # Maximum size of the buffer
        self.mem_cntr = 0  # Counter for adding new experiences

        # Allocate memory for storing experiences
        self.now_state = torch.zeros((max_size, state_dim), dtype=torch.float).to(device)
        self.action = torch.zeros((max_size, action_dim), dtype=torch.float).to(device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float).to(device)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float).to(device)

    # Add a new experience to the buffer
    def add(self, now_state, action, reward, next_state):
        index = self.mem_cntr % self.max_size  # Find index to store the experience
        self.now_state[index] = torch.from_numpy(now_state).to(device)
        self.action[index] = torch.from_numpy(action).to(device)
        self.next_state[index] = torch.from_numpy(next_state).to(device)
        self.reward[index] = reward

        self.mem_cntr += 1  # Increment memory counter

    # Sample a batch of experiences from the buffer
    def sample(self, batch_size):
        batch = torch.randint(0, self.max_size, (batch_size,)).to(device)  # Randomly sample indices

        next_state = self.next_state[batch]  # Sample next states
        now_state = self.now_state[batch]  # Sample current states
        action = self.action[batch]  # Sample actions
        reward = self.reward[batch]  # Sample rewards

        return now_state, action, reward, next_state


# Function to test the saved actor and save the model if it performs well
def test_for_save(actor):
    average_save_rewards = -130  # Threshold for saving the model
    env_test = gym.make('Pendulum-v1')  # Create test environment
    run_reward = 0
    step = 0
    now_state = env_test.reset(seed=0)[0]  # Reset environment and get initial state
    while step < 200:
        action = select_action(now_state, actor, deterministic=True)  # Select action
        step += 1
        next_state, reward, done, truncated, info = env_test.step(action)  # Take action in environment
        run_reward += reward  # Accumulate reward
        now_state = next_state  # Update state

    run_reward = int(run_reward)
    print(f"test reward = {run_reward}")  # Print the test reward
    if run_reward >= average_save_rewards:
        average_save_rewards = run_reward  # Update the threshold
        torch.save(actor.state_dict(), f"Pendulum{run_reward}.pt")  # Save the model if reward is high enough
    env_test.close


# Main training loop
def runs():
    # Build Environment
    env = gym.make('Pendulum-v1')  # Create environment
    state_dim = 3  # State dimension for Pendulum-v1
    action_dim = 1  # Action dimension
    net_width = 400  # Network width for the layers

    # Seed Everything for reproducibility
    env_seed = 0
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Initialize Actor network and its optimizer
    actor = Actor(state_dim, action_dim, net_width).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
    actor_target = copy.deepcopy(actor)  # Create a target actor network

    # Initialize Critic network and its optimizer
    q_critic = Q_Critic(state_dim, action_dim, net_width).to(device)
    q_critic_optimizer = torch.optim.Adam(q_critic.parameters(), lr=0.001)
    q_critic_target = copy.deepcopy(q_critic)  # Create a target critic network

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(5e5))

    total_steps = 0  # Total interaction steps
    run = 0  # Number of episodes
    while run < 1000:
        run_reward = 0
        now_state = env.reset(seed=env_seed)[0]  # Reset the environment
        env_seed += 1  # Increment environment seed
        run += 1  # Increment episode counter
        step = 0  # Reset step counter

        # Interact with the environment and train the networks
        while step < 200:
            if total_steps < 50000:
                action = env.action_space.sample()  # Random action for initial exploration
            else:
                action = select_action(now_state, actor, deterministic=False)  # Select action from actor
            step += 1
            next_state, reward, done, truncated, info = env.step(action)  # Step in the environment
            run_reward += reward  # Accumulate reward
            replay_buffer.add(now_state, action, reward, next_state)  # Add experience to replay buffer
            now_state = next_state  # Update current state
            total_steps += 1  # Increment total steps

            # Train the networks after sufficient exploration
            if total_steps >= 50000:
                train(actor, actor_optimizer, actor_target, q_critic, q_critic_optimizer, q_critic_target, replay_buffer)

        print(step, run, run_reward)  # Print episode statistics

        if run_reward > -120:  # Test and save model if reward is high enough
            env.close()
            test_for_save(actor)

    env.close()


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Run the main training loop
runs()

print("runs end")
