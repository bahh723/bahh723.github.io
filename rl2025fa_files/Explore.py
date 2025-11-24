## This code is for lunar lander
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import imageio


class BonusStatistics:
    def __init__(self, num_bins, state_bounds):
        self.num_bins = num_bins
        self.visit_counts = {}
        self.bin_bonuses = {}  # bin -> list of bonus values
        self.state_bounds = state_bounds
        
    def _discretize_state(self, state):
        """Convert continuous state to discrete bins"""
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy().flatten()
        elif not isinstance(state, np.ndarray):
            state = np.array(state).flatten()
            
        discrete_state = []
        for i in range(len(state)):
            min_val, max_val = self.state_bounds[i]
            bin_size = (max_val - min_val) / self.num_bins
            bin_idx = int((state[i] - min_val) / bin_size)
            bin_idx = max(0, min(self.num_bins - 1, bin_idx))  # Clamp to valid range
            discrete_state.append(bin_idx)
        
        return tuple(discrete_state)
    
    def update(self, state, bonus_value):
        """Update statistics for a given state and its bonus value"""
        discrete_state = self._discretize_state(state) 
        self.visit_counts[discrete_state] = self.visit_counts.get(discrete_state, 0) + 1
        
        if discrete_state not in self.bin_bonuses:
            self.bin_bonuses[discrete_state] = []
        self.bin_bonuses[discrete_state].append(bonus_value)
    
    
    def get_2d_bonus_map(self):
        bonus_map = np.zeros((self.num_bins, self.num_bins))
        for (pos_bin, vel_bin), bonuses in self.bin_bonuses.items():
            if len(bonuses) > 0:
                bonus_map[vel_bin, pos_bin] = np.mean(bonuses)
                
        return bonus_map
    
    def get_2d_visitation_map(self):
        visitation_map = np.full((self.num_bins, self.num_bins), -20.0)
        total_visits = sum(self.visit_counts.values()) if self.visit_counts else 0
        
        if total_visits > 0:
            for (pos_bin, vel_bin), count in self.visit_counts.items():
                percentage = (count / total_visits) * 100
                log_percentage = np.log(percentage)
                visitation_map[vel_bin, pos_bin] = log_percentage
                
        return visitation_map


class TabularExplorationBonus:
    def __init__(self, state_dim, num_bins, bonus_scale, state_bounds):
        self.state_dim = state_dim
        self.num_bins = num_bins
        self.bonus_scale = bonus_scale
        self.visit_counts = {}
        self.state_bounds = np.array(state_bounds)
        
    def _discretize_state(self, state):
        """Convert continuous state to discrete bins"""
        # Convert to numpy array regardless of input type
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy().flatten()
        else:
            state = np.array(state).flatten()
            
        # Vectorized discretization
        min_vals = self.state_bounds[:, 0]
        max_vals = self.state_bounds[:, 1]
        bin_sizes = (max_vals - min_vals) / self.num_bins
        bin_indices = ((state - min_vals) / bin_sizes).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        return tuple(bin_indices)
    
    def get_bonus(self, state, update_counts=False):
        discrete_state = self._discretize_state(state)   # discretize a d-dimensional state into a d-tuple

        ########################################
        #  TODO: 
        #  1. Calculate the bonus as self.bonus_scale / sqrt( self.visit_counts[discrete_state] )
        #     If discrete_state has not been visited before, initialize self.visit_counts[discrete_state] as 1.
        #  2. If update_counts == True, add 1 to self.visit_counts[discrete_state].
        #  3. Return the bonus. 
        ########################################

        return 0.0 

    def train_and_get_bonus(self, state):
        return self.get_bonus(state, update_counts=True)
    


class RandomNetworkDistillation:
    def __init__(self, state_dim, hidden_dim, bonus_scale, lr, state_bounds):
        self.state_dim = state_dim
        self.bonus_scale = bonus_scale
        self.state_bounds = np.array(state_bounds)
        
        # Target network (fixed, randomly initialized)
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Predictor network (trainable)
        self.predictor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Freeze target network
        for param in self.target_net.parameters():
            param.requires_grad = False
            
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)
        
        
    def _normalize_state(self, state):
        """Normalize the state"""
        if isinstance(state, torch.Tensor):
            state_tensor = state.flatten().unsqueeze(0)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
            
        min_vals = torch.tensor(self.state_bounds[:, 0], dtype=torch.float32)  # [-1.2, -0.07]
        max_vals = torch.tensor(self.state_bounds[:, 1], dtype=torch.float32)  # [0.6, 0.07]
        
        # Normalize to [-1, 1] range
        normalized = 2 * (state_tensor - min_vals) / (max_vals - min_vals) - 1
        return normalized
        
        
    def get_bonus(self, state, train=False):
        state_norm = self._normalize_state(state)   # Normalize the state
        
        ###############################################
        #  TODO: 
        #  1. Calculate the bonus as the squared distance bewteen self.target_net(state_norm) 
        #     and self.predictor_net(state_norm).
        #  2. If train == True, perform one gradient update to minimize the 
        #     distance between self.target(state_norm) and self.predictor_net(state_norm). 
        #     The target_net should be fixed --- only train the predictor_net. 
        #  3. Return the bonus
        ###############################################

        return 0.0
            
    def train_and_get_bonus(self, state):
        return self.get_bonus(state, train=True)
    



def set_seed(seed):
    random.seed(seed)                   # Python random module
    np.random.seed(seed)                 # NumPy
    torch.manual_seed(seed)              # PyTorch CPU
    torch.cuda.manual_seed(seed)         # PyTorch GPU (if used)
    torch.cuda.manual_seed_all(seed)     # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.benchmark = False  # Disable cuDNN auto-tuner to prevent non-determinism


def plot_progress(episode_returns, figure=None):
    running_avg = [np.mean(episode_returns[max(0, i - 100 + 1):i + 1]) for i in range(len(episode_returns))]
    max_avg = max(running_avg)
    plt.figure(1)
    plt.clf()
    plt.title(f'Training... (Max Running Avg: {max_avg:.2f})')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    x = range(0, len(episode_returns))
    plt.plot(x, episode_returns, label='Episode Returns')
    plt.plot(x, running_avg, label=f'Running Average')
    plt.pause(0.001)  
    if len(episode_returns) % 100 == 0:
        if figure:
            output_name = figure.replace('.png', '') + '-return.png'
        else:
            output_name = 'return.png'
        plt.savefig(output_name)
        print(f"Return curve saved to: {output_name}")

def plot_bonus_heatmap(bonus_stats, episode, args):
    """Plot 2D heatmap of bonus values and visitation percentages"""
    if bonus_stats is None:
        return
    
    # Get state bounds from bonus_stats
    state_bounds = bonus_stats.state_bounds
    # Create extent for imshow: [x_min, x_max, y_min, y_max] = [pos_min, pos_max, vel_min, vel_max]
    extent = [state_bounds[0][0], state_bounds[0][1], state_bounds[1][0], state_bounds[1][1]]
    
    plt.figure(2, figsize=(12, 5))
    plt.clf()
    
    # Subplot 1: Bonus values (empty for "none" exploration)
    plt.subplot(1, 2, 1)
    if args.exploration != "none":
        bonus_map = bonus_stats.get_2d_bonus_map()
        im1 = plt.imshow(bonus_map, cmap='viridis', aspect='auto', origin='lower',
                        extent=extent)
        plt.colorbar(im1, label='Average Bonus Value')
        plt.title(f'Bonus Values - Episode {episode}')
    else:
        # Empty plot for "none" exploration
        plt.imshow(np.zeros((20, 20)), cmap='viridis', aspect='auto', origin='lower',
                  extent=extent)
        plt.colorbar(label='Average Bonus Value')
        plt.title(f'Bonus Values - Episode {episode} (no exploration)')
    
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Visitation percentages
    plt.subplot(1, 2, 2)
    visitation_map = bonus_stats.get_2d_visitation_map()
    im2 = plt.imshow(visitation_map, cmap='hot', aspect='auto', origin='lower',
                    extent=extent)
    plt.colorbar(im2, label='Log(Visitation Percentage)')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title(f'State Visitation - Episode {episode}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.pause(0.001)
    
    # Save the figure every 100 episodes
    if episode % 100 == 0:
        if args.figure:
            bonus_name = args.figure.replace('.png', '') + '-bonus.png'
        else:
            bonus_name = 'bonus.png'
        plt.savefig(bonus_name)
        print(f"Bonus and visitation heatmaps saved to: {bonus_name}")



Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminated'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args): 
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        if batch_size <= len(self.memory):
            return random.sample(self.memory, batch_size)
        else: 
            return random.choices(self.memory, k=batch_size)
        
    def __len__(self):
        return len(self.memory)



class MDPModel(nn.Module):
    def __init__(self, dim, n_actions, args):
        super(MDPModel, self).__init__()
        self.dim = dim
        self.n_actions = n_actions
        self.algorithm = args.algorithm
        
        # Policy network
        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        
        # Value network
        self.value_fc1 = nn.Linear(dim, 128)
        self.value_fc2 = nn.Linear(128, 128)
        self.value_fc3 = nn.Linear(128, 1)
        
        # Bonus value network  
        self.bonus_fc1 = nn.Linear(dim, 128)
        self.bonus_fc2 = nn.Linear(128, 128)
        self.bonus_fc3 = nn.Linear(128, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.M = M
        
    def forward(self, x):
        """
        The Q_theta in DQN or pi_theta in PPO
        """
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        if self.algorithm == "PPO": 
            x = nn.Softmax(dim=1)(x)
        return x

    def forward_baseline(self, x): 
        """The V_phi network in the pseudocode of PPO (kept for compatibility)"""
        return self.forward_value(x)
    
    def forward_value(self, x):
        """Value network"""
        x = F.relu(self.value_fc1(x))
        x = F.relu(self.value_fc2(x))
        x = self.value_fc3(x)
        return x
    
    def forward_bonus_value(self, x):
        """Bonus value network"""
        x = F.relu(self.bonus_fc1(x))
        x = F.relu(self.bonus_fc2(x))
        x = self.bonus_fc3(x)
        return x
        
    def get_state_action_values(self, batch_state, batch_action):   
        """
        return Q[s,a]
        """
        q_values = self(batch_state)
        row_index = torch.arange(0, batch_state.shape[0])
        selected_actions_q_values = q_values[row_index, batch_action]
        return selected_actions_q_values
    
    def get_state_values(self, batch_state): 
        """
        return max_a Q[s,a]
        """
        q_values = self(batch_state)
        max_q_values = q_values.max(dim=1).values
        return max_q_values
    
    def get_max_value_actions(self, batch_state): 
        """
        return argmax_a Q[s,a]
        """
        q_values = self(batch_state)
        max_q_actions = q_values.max(dim=1).indices
        return max_q_actions
    

    def update_batch(self, batch_state, batch_action, batch_reward, batch_next_state, batch_terminated, batch_action_prob, memory=None, target_net=None, exploration_bonus=None):
        batch_action_prob = batch_action_prob.detach()
        if self.algorithm == "DQN": 
            for state, action, reward, next_state, terminated in zip(batch_state, batch_action, batch_reward, batch_next_state, batch_terminated):
                memory.push(state, action, reward, next_state, terminated) 
            states, actions, rewards, next_states, terminateds = zip(*memory.sample(MINIBATCH_SIZE))
            batch_state = torch.stack(states)
            batch_action = torch.stack(actions)
            batch_reward = torch.stack(rewards)
            batch_next_state = torch.stack(next_states)
            batch_terminated = torch.stack(terminateds)

            state_action_values = self.get_state_action_values(batch_state, batch_action)
            
            augmented_rewards = batch_reward.clone()
            if exploration_bonus is not None:
                for i, state in enumerate(batch_state):
                    bonus = exploration_bonus.get_bonus(state)
                    augmented_rewards[i] += bonus
            
            with torch.no_grad():
                next_state_values = target_net.get_state_values(batch_next_state) * (1 - batch_terminated)    
                expected_state_action_values = augmented_rewards + GAMMA * next_state_values

            loss = F.mse_loss(state_action_values, expected_state_action_values) 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for target_param, policy_param in zip(target_net.parameters(), self.parameters()):
                target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

        elif self.algorithm == "PPO":
            bonuses = torch.zeros_like(batch_reward)
            if exploration_bonus is not None:
                with torch.no_grad():
                    bonuses_list = []
                    for i, state in enumerate(batch_state):
                        bonus = exploration_bonus.get_bonus(state.cpu().numpy())
                        bonuses_list.append(bonus)
                    bonuses = torch.tensor(bonuses_list, dtype=torch.float32)
            
            with torch.no_grad():
                values = self.forward_value(batch_state).squeeze(-1)
                bonus_values = self.forward_bonus_value(batch_state).squeeze(-1)
                next_values = self.forward_value(batch_next_state).squeeze(-1)
                bonus_next_values = self.forward_bonus_value(batch_next_state).squeeze(-1)

            advantages = torch.zeros_like(batch_reward)
            bonus_advantages = torch.zeros_like(bonuses)
            gae = 0
            bonus_gae = 0
            
            for t in reversed(range(len(batch_reward))):
                # GAE
                delta = batch_reward[t] + GAMMA * next_values[t] * (1.0 - batch_terminated[t]) - values[t]
                gae = delta + GAMMA * LAMBDA * (1.0 - batch_terminated[t]) * gae
                advantages[t] = gae
                
                # Bonus GAE
                bonus_delta = bonuses[t] + BONUS_GAMMA * bonus_next_values[t] * (1.0 - batch_terminated[t]) - bonus_values[t]
                bonus_gae = bonus_delta + BONUS_GAMMA * LAMBDA * (1.0 - batch_terminated[t]) * bonus_gae
                bonus_advantages[t] = bonus_gae
            
            total_advantages = advantages + bonus_advantages
            
            returns = advantages + values.detach()
            bonus_returns = bonus_advantages + bonus_values.detach()
            
            for _ in range(self.M):
                indices = torch.randperm(len(batch_state))
                
                for start in range(0, len(batch_state), MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    mb_indices = indices[start:end]
                    
                    mb_states = batch_state[mb_indices]
                    mb_actions = batch_action[mb_indices]
                    mb_advantages = total_advantages[mb_indices]
                    mb_returns = returns[mb_indices]
                    mb_bonus_returns = bonus_returns[mb_indices]
                    mb_old_log_probs = batch_action_prob[mb_indices]
                    
                    new_probs_all = self(mb_states)
                    new_action_prob = new_probs_all.gather(1, mb_actions.unsqueeze(1))

                    ratios = new_action_prob / (mb_old_log_probs.unsqueeze(1))
                    clipped_ratios = torch.clamp(ratios, min=1-EPS, max=1+EPS)
                    
                    # PPO policy loss with combined advantages
                    ppo_term = torch.mean(torch.min(ratios * mb_advantages.unsqueeze(1), 
                                                  clipped_ratios * mb_advantages.unsqueeze(1)))
                    policy_loss = -ppo_term
                    
                    # Entropy bonus
                    entropy_term = BETA * torch.mean(torch.sum(new_probs_all * torch.log(new_probs_all + 1e-8), dim=1))
                    
                    # Value losses
                    value_pred = self.forward_value(mb_states).squeeze(-1)
                    bonus_value_pred = self.forward_bonus_value(mb_states).squeeze(-1)
                    value_loss = F.mse_loss(value_pred, mb_returns)
                    bonus_value_loss = F.mse_loss(bonus_value_pred, mb_bonus_returns)
                    
                    # Joint loss (policy + value networks + entropy)
                    total_loss = (policy_loss + 
                                 VF_COEF * value_loss + 
                                 BONUS_VF_COEF * bonus_value_loss - 
                                 entropy_term)
                    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()


    def act(self, x, iteration):
        q_values = self(x)
        if self.algorithm == "DQN":   
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * iteration / EPS_DECAY)
            max_index = q_values.argmax(dim=1)
            batchsize = x.shape[0]
            prob = torch.zeros_like(q_values)
            prob[torch.arange(batchsize), max_index] = 1.0
            prob = (1-eps) * prob + eps / self.n_actions * torch.ones_like(q_values)  
            return prob
    
        elif self.algorithm == "PPO":  
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * iteration / EPS_DECAY)
            policy_probs = self(x)
            uniform_probs = torch.ones_like(policy_probs) / self.n_actions
            mixed_probs = (1 - eps) * policy_probs + eps * uniform_probs
            return mixed_probs  
    



def train(args, n_actions, state_dim): 
    set_seed(seed=args.seed) 
    if args.algorithm == "DQN": 
        num_episodes = 3000
    elif args.algorithm == "PPO": 
        num_episodes = 6000

    policy_net = MDPModel(state_dim, n_actions, args)     # the online network in DQN or the policy network in PPO
    if args.algorithm == "DQN": 
        target_net = MDPModel(state_dim, n_actions, args)
        target_net.load_state_dict(policy_net.state_dict())   
        memory = ReplayMemory(1000000)
    else: 
        target_net = None
        memory = None

    exploration_bonus = None
    env_temp = gym.make('MountainCar-v0')
    state_bounds = [[env_temp.observation_space.low[i], env_temp.observation_space.high[i]] for i in range(len(env_temp.observation_space.low))] #[[-1.2, 0.6], [-0.07, 0.07]] in mountain car
    env_temp.close()
    bonus_stats = BonusStatistics(num_bins=20, state_bounds=state_bounds)  # Always create for plotting visitation
    
    if args.exploration == "tabular":
        exploration_bonus = TabularExplorationBonus(state_dim=state_dim, num_bins=20, bonus_scale=0.5, state_bounds=state_bounds)
    elif args.exploration == "rnd":
        exploration_bonus = RandomNetworkDistillation(state_dim=state_dim, hidden_dim=64, bonus_scale=500, lr=1e-3, state_bounds=state_bounds)

    
    batch_state = []
    batch_action = []
    batch_reward = []
    batch_next_state = []
    batch_terminated = []
    batch_action_prob = []
    episode_returns = []
    t = 0

    for iteration in range(num_episodes):
        current_episode_return = 0
        state, _ = env.reset(seed=9876543210*args.seed+iteration*(246+args.seed))
        terminated = truncated = 0

        while not (terminated or truncated):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            action_prob_all = policy_net.act(state, iteration)
            action = torch.multinomial(action_prob_all, num_samples=1).item()
            action_prob = action_prob_all[0,action]
            next_state, reward, terminated, truncated, _ = env.step(action)
            current_episode_return += reward


            action = torch.as_tensor(action)
            reward = torch.as_tensor(reward, dtype=torch.float32)
            next_state = torch.as_tensor(next_state, dtype=torch.float32)
            terminated = torch.as_tensor(terminated).int()
        
            batch_state.append(state)
            batch_action.append(action)
            batch_reward.append(reward)
            batch_next_state.append(next_state)
            batch_terminated.append(terminated)
            batch_action_prob.append(action_prob)
            

            if (t + 1) % N == 0:

 
                batch_state = torch.cat(batch_state, dim=0)
                batch_action = torch.stack(batch_action, dim=0)
                batch_reward = torch.stack(batch_reward, dim=0)
                batch_next_state = torch.stack(batch_next_state, dim=0)
                batch_terminated = torch.stack(batch_terminated, dim=0)
                batch_action_prob = torch.stack(batch_action_prob, dim=0)


                policy_net.update_batch(
                    batch_state = batch_state, 
                    batch_action = batch_action, 
                    batch_reward = batch_reward, 
                    batch_next_state = batch_next_state, 
                    batch_terminated = batch_terminated, 
                    batch_action_prob = batch_action_prob,
                    memory = memory, 
                    target_net = target_net,
                    exploration_bonus = exploration_bonus
                )
                
                                
                # Update exploration bonus and statistics after training on this batch
                for state in batch_state:
                    if exploration_bonus is not None:
                        bonus_value = exploration_bonus.train_and_get_bonus(state)
                    else:
                        bonus_value = 0
                    
                    bonus_stats.update(state, bonus_value)
                batch_state = []
                batch_action = []
                batch_reward = []
                batch_next_state = []
                batch_terminated = []
                batch_action_prob = []
                            
        
            if terminated or truncated:
                episode_returns.append(current_episode_return)
                print('Episode {},  score: {}'.format(iteration, current_episode_return), flush=True)
                
                # Visualize bonus heatmap every 20 episodes
                if (iteration + 1) % 20 == 0:
                    plot_bonus_heatmap(bonus_stats, iteration + 1, args)
                
                plot_progress(episode_returns, figure=args.figure)    
                    
            else: 
                state = next_state

            t = t+1


if __name__ == "__main__": 
    env = gym.make('MountainCar-v0')
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    parser = argparse.ArgumentParser(description="Script with input parameters")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--algorithm", type=str, required=True, default="DQN", help="DQN or PPO")
    parser.add_argument("--exploration", type=str, choices=["none", "tabular", "rnd"], default="none", 
                       help="Exploration strategy: none (no bonus), tabular (discretized state bonus), rnd (Random Network Distillation)")
    parser.add_argument("--figure", type=str, required=False, default=None)
    args = parser.parse_args()

    ##### Shared parameters between DQN and PPO #####
    MINIBATCH_SIZE = 128      # the B in the pseudocode
    GAMMA = 0.99              # Discount for rewards
    BONUS_GAMMA = 0.99        # Discount for bonuses
    if args.algorithm == "DQN": 
       LR = 1e-4
       N = 4                 # Standard DQN: Update every 4 environment steps
       M = 1
    elif args.algorithm == "PPO": 
       LR = 3e-4              # Learning rate
       N = 2048               # Steps per update
       M = 10                 # Epochs per update


    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 500           # decay rate of epsilon (the larger the slower decay)


    ##### DQN-specific parameters #####
    TAU = 0.01                # the update rate of the target network 


    ##### PPO-specific parameters #####
    LAMBDA = 0.99             # GAE lambda
    EPS = 0.1                 # PPO clipping range
    BETA = 0.01               # Entropy coefficient
    VF_COEF = 0.5             # Value function coefficient
    BONUS_VF_COEF = 1.0       # Bonus value function coefficient
    

    train(args, n_actions, state_dim)

