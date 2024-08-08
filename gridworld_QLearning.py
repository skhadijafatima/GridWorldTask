import numpy as np
import matplotlib.pyplot as plt
import random


class GridWorld:
    def __init__(self):
        self.grid_size = 5
        self.state = (1, 0)  # Starting state [2,1] in 0-indexed form
        self.goal = (4, 4)  # Terminal state [5,5] in 0-indexed form
        self.obstacles = [(2, 2), (2, 3), (2, 4), (3, 2)]  # Obstacles positions
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}  # North, South, East, West
        self.rewards = np.full((self.grid_size, self.grid_size), -1.0)  # Default reward for each cell
        self.grid_size = 5
        self.state = (1, 0)  # Starting state [2,1] in 0-indexed form
        self.goal = (4, 4)  # Terminal state [5,5] in 0-indexed form
        self.obstacles = [(2, 2), (2, 3), (2, 4), (3, 2)]  # Obstacles positions
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}  # North, South, East, West
        self.rewards = np.full((self.grid_size, self.grid_size), -1.0)  # Default reward for each cell
        self.rewards[4, 4] = 10.0  # Reward for reaching the goal

        # Adjust obstacles and rewards to allow the desired path
        self.rewards[1, 1] = -1.0  # Obstacle
        self.rewards[2, 2] = -1.0  # Obstacle
        self.rewards[3, 2] = -1.0  # Obstacle
        self.rewards[1, 2] = -1.0  # Obstacle
        self.rewards[2, 3] = -1.0  # Obstacle
        self.rewards[2, 4] = -1.0  # Obstacle
        self.rewards[1, 3] = 0.0  # Allow passage on this path
        self.rewards[1, 4] = -1.0  # Keep this as obstacle, or you can remove it
    def reset(self):
        self.state = (1, 0)  # Reset to initial state
        return self.state

    def step(self, action):
        new_state = (self.state[0] + self.actions[action][0], self.state[1] + self.actions[action][1])

        if self.state == (1, 3) and action == 2:  # Assuming action 2 corresponds to 'East'
            self.state = (3, 3)  # Jump to (3, 3)
            reward = 5.0  # Reward for jumping
            done = self.state == self.goal
            return self.state, reward, done

        if (0 <= new_state[0] < self.grid_size and 0 <= new_state[1] < self.grid_size and
                new_state not in self.obstacles):
            self.state = new_state

            # Normal movement
            reward = self.rewards[self.state]
            done = self.state == self.goal
            return self.state, reward, done

        # If the new state is invalid (e.g., hitting an obstacle), stay in place
        reward = -1.0  # Penalize invalid actions
        return self.state, reward, False

    def render(self, path=None):
        grid = np.full((self.grid_size, self.grid_size), ' ')
        for obs in self.obstacles:
            grid[obs] = 'X'  # Mark obstacles
        grid[self.goal] = 'G'  # Goal
        grid[self.state] = 'A'  # Agent's current position

        # Mark the path
        if path:
            for step in path:
                if step != self.state and step != self.goal:
                    grid[step] = '.'  # Mark the path

        print(grid)


class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((env.grid_size, env.grid_size, 4))  # Initialize Q-table
        self.epsilon = 0.1  # Adjusted exploration rate
        self.alpha = 0.1
        self.gamma = 0.95

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # Random action
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_delta


class SARSAAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((env.grid_size, env.grid_size, 4))  # Initialize Q-table
        self.epsilon = 0.1  # Higher exploration rate initially
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon_decay = 0.99  # Decay rate for epsilon

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # Random action
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state, next_action):
        td_target = reward + self.gamma * self.q_table[next_state][next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_delta


def train_agent(agent, episodes):
    rewards_per_episode = []
    for episode in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        path = [state]  # Store the path taken

        for _ in range(50):  # Max steps per episode
            next_state, reward, done = env.step(action)
            total_reward += reward
            path.append(next_state)  # Append the path taken

            if done:
                if isinstance(agent, QLearningAgent):
                    agent.learn(state, action, reward, next_state)
                break

            next_action = agent.choose_action(next_state)

            # For Q-learning
            if isinstance(agent, QLearningAgent):
                agent.learn(state, action, reward, next_state)
            # For SARSA
            elif isinstance(agent, SARSAAgent):
                agent.learn(state, action, reward, next_state, next_action)

            state, action = next_state, next_action

        rewards_per_episode.append(total_reward)

    return rewards_per_episode, path


# Initialize environment and agents
env = GridWorld()

# Train Q-Learning agent
q_agent = QLearningAgent(env)
q_rewards, q_path = train_agent(q_agent, 200)

# Train SARSA agent
sarsa_agent = SARSAAgent(env)
sarsa_rewards, sarsa_path = train_agent(sarsa_agent, 200)

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(q_rewards, label='Q-Learning Reward per Episode', alpha=0.7)
plt.plot(sarsa_rewards, label='SARSA Reward per Episode', alpha=0.7)
plt.xlabel('Episode Number')
plt.ylabel('Reward')
plt.title('Agent Training Rewards')
plt.legend()
plt.grid()

# Save the plot instead of showing it
plt.savefig('agent_training_rewards.png')
plt.close()  # Close the figure to avoid displaying it in some environments

# Display final grid and path taken by agents
print("Final Grid and Path for Q-Learning Agent:")
env.render(q_path)
print("Path Taken:", q_path)

print("\nFinal Grid and Path for SARSA Agent:")
env.render(sarsa_path)
print("Path Taken:", sarsa_path)
