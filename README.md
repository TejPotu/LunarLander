# LunarLander Reinforcement Learning Project

A comprehensive exploration of reinforcement learning techniques for the LunarLander environment using PyTorch. This project demonstrates progressively advanced RL algorithms, from rule-based agents to deep Q-networks.

## 📋 Project Overview

This Jupyter notebook implements multiple approaches to solve the LunarLander-v3 environment from Gymnasium:

1. **Safe Agent** - Rule-based approach preventing ground collision
2. **Stable Agent** - Rule-based approach maintaining stability
3. **Deep Q-Network (DQN)** - Neural network-based learning
4. **Double DQN (DDQN)** - Improved DQN with reduced overestimation
5. **Dueling DQN** - Advanced architecture separating value and advantage streams

## 🎮 LunarLander Environment

The LunarLander-v3 environment challenges an agent to land a lunar module safely on a landing pad. 

### Observation Space (8 dimensions)
- `x` - Horizontal position (-1 to 1)
- `y` - Vertical position (0 to 1)
- `vx` - Horizontal velocity (-5 to 5)
- `vy` - Vertical velocity (-8 to 8)
- `angle` - Rotation angle (-π to π)
- `angular_velocity` - Rotation speed (-5 to 5)
- `left_leg_contact` - Left leg touching ground (0 or 1)
- `right_leg_contact` - Right leg touching ground (0 or 1)

### Action Space (4 discrete actions)
- `0` - Do nothing
- `1` - Fire left engine (rotate counter-clockwise)
- `2` - Fire main engine (move upward)
- `3` - Fire right engine (rotate clockwise)

### Rewards
- Landing smoothly: +100 to +200
- Ground contact: +10
- Fuel consumption: -0.3
- Crashing: -100
- Leg out of bounds: -100

## 🏗️ Project Structure

```
LunarLander/
├── LunarLander.ipynb          # Main Jupyter notebook with all implementations
├── q_network.pth              # Saved DQN Q-network weights
├── target_network.pth         # Saved DQN target network weights
└── README.md                   # This file
```

## 📚 Implementation Details

### 1. Rule-Based Agents

#### SafeAgent
Simple agent that prevents the lander from crashing by applying upward thrust when below a minimum height.

```python
class SafeAgent:
    def act(self, observation):
        MIN_HEIGHT = 1
        if observation[1] < MIN_HEIGHT:
            return 2  # Main engine
        else:
            return 0  # Do nothing
```

#### StableAgent
More sophisticated rule-based agent that maintains stability using heuristic rules for angle, position, and height.

### 2. Deep Q-Network (DQN)

Neural network architecture for learning Q-values:
```
Input (8) → Linear → ReLU → Linear (64) → ReLU → Linear (64) → Output (4 actions)
```

**Key Components:**
- **Q-Network**: Online network that gets updated with gradient descent
- **Target Network**: Stable network used to compute target Q-values
- **Replay Buffer**: Stores transitions for experience replay
- **Epsilon-Greedy Policy**: Balances exploration and exploitation

**Training Algorithm:**
1. Sample random action with probability ε, otherwise select action with max Q-value
2. Store transition (state, action, reward, next_state, done) in replay buffer
3. Sample mini-batch from replay buffer
4. Compute Q-values for current states using Q-network
5. Compute target Q-values using target network
6. Update Q-network using MSE loss
7. Periodically copy Q-network weights to target network

### 3. Double DQN (DDQN)

Addresses DQN's overestimation problem by decoupling action selection from evaluation:
- Q-network selects the best action for next state
- Target network evaluates that action
- Reduces bias in Q-value estimation

**Key Difference:**
```python
# DQN: uses target network for both selection and evaluation
next_q_values = self.target_network(next_states).max(1)[0]

# DDQN: uses q_network for selection, target_network for evaluation
next_actions = self.q_network(next_states).argmax(1)
next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
```

### 4. Dueling DQN

Advanced architecture that separates value and advantage streams (implementation started).

## 🚀 Getting Started

### Prerequisites

```bash
pip install gymnasium torch numpy pygame
```

### Running the Notebook

1. Start Jupyter:
```bash
jupyter notebook LunarLander.ipynb
```

2. Execute cells sequentially:
   - Initialize environment
   - Run rule-based agents
   - Train DQN model (commented out, run to train from scratch)
   - Load pre-trained weights
   - Test the trained agent
   - Explore DDQN variant

### Training a New Model

Uncomment and run this cell to train from scratch:
```python
scores = train_agent(agent, env, n_episodes=2000, epsilon_start=1.0, 
                     epsilon_end=0.01, epsilon_decay=0.995, target_update_freq=10)
```

**Training Parameters:**
- `n_episodes`: Total episodes to train (2000 recommended)
- `epsilon_start`: Initial exploration rate (1.0)
- `epsilon_end`: Final exploration rate (0.01)
- `epsilon_decay`: Decay rate per episode (0.995)
- `target_update_freq`: Frequency of target network updates (every 10 episodes)

## 📊 Key Classes and Methods

### ReplayBuffer
Stores and samples transitions for experience replay.

```python
buffer = ReplayBuffer(buffer_size=10000)
buffer.push(state, action, reward, next_state, done)
states, actions, rewards, next_states, dones = buffer.sample(batch_size=64)
```

### DQNAgent
Main DQN agent with methods for interaction and learning.

**Methods:**
- `act(state, epsilon)` - Select action using ε-greedy policy
- `step(state, action, reward, next_state, done)` - Store transition and update model
- `update_model()` - Train Q-network on sampled batch
- `update_target_network()` - Sync target network with Q-network

### DDQNAgent
Double DQN variant with improved action selection in `update_model()`.

## 📈 Performance Metrics

The agent's performance is tracked using:
- **Total Reward per Episode**: Sum of rewards received
- **Average Score (100-episode window)**: Rolling average for convergence detection
- **Success Threshold**: Average score ≥ 200 indicates environment solved

## 🔍 Key Concepts Explained

### Gather Operation
```python
q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
```
Extracts Q-values for the specific actions taken (not all possible actions).

### Target Q-Value Computation
```python
target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
```
Implements Bellman equation: V(s) = r + γ·V(s')

### Experience Replay
Stores transitions in a buffer and samples randomly to break temporal correlations and reduce variance in gradient updates.

## 💾 Saving and Loading Models

Save trained networks:
```python
torch.save(agent.q_network.state_dict(), "q_network.pth")
torch.save(agent.target_network.state_dict(), "target_network.pth")
```

Load pre-trained networks:
```python
agent.q_network.load_state_dict(torch.load("q_network.pth"))
agent.target_network.load_state_dict(torch.load("target_network.pth"))
```

## 📚 Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Deep Q-Networks Paper](https://arxiv.org/abs/1312.5602)
- [Double DQN Paper](https://arxiv.org/abs/1509.06461)
- [Dueling DQN Paper](https://arxiv.org/abs/1511.06581)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 📝 License

This project is for educational purposes.

**Last Updated:** July 2025
