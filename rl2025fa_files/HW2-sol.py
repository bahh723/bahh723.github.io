import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


# Load MNIST dataset
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

X = mnist.data.numpy().reshape(-1, 784).astype(np.float32)
y = mnist.targets.numpy()

# Select 1000 samples per class
X_list, y_list = [], []
for digit in range(4):
    X_d, y_d = X[y == digit][:2500], y[y == digit][:2500]
    X_list.append(X_d)
    y_list.append(y_d)

# Concatenate the selected samples
X = np.vstack(X_list)
y = np.hstack(y_list)

print("X.shape (digit images)", X.shape)
print("y.shape (digit labels)", y.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
train_dataset = TensorDataset(X_tensor, y_tensor)
context_dim = X_tensor.shape[1]
n_actions = len(torch.unique(y_tensor))


class BanditModel(nn.Module):
    def __init__(self, dim, n_actions, args):
        super().__init__()
        self.dim = dim
        self.n_actions = n_actions
        self.algorithm = args.algorithm
        if args.algorithm == "EG": 
            self.eps = args.eps
        elif args.algorithm == "BE" or args.algorithm == "IGW":
            self.ld = args.ld
        elif args.algorithm == "PPO": 
            self.b = args.b
            self.kl_coeff = args.kl_coeff
        elif args.algorithm == "PG": 
            self.b = args.b

        self.fc1 = nn.Linear(dim, 32)
        self.fc2 = nn.Linear(32, n_actions)

        # baseline network
        self.fc3 = nn.Linear(dim, 32)
        self.fc4 = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.N = 256  #8
        self.M = 30   #5


    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        if self.algorithm == "PPO": 
            x = nn.Softmax(dim=1)(x)
        return x

    def forward_baseline(self, x): 
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        return x
        

    def act(self, x, time):
        predicted_reward_all = self(x)
        max_predicted_reward = torch.max(predicted_reward_all, dim=1, keepdim=True)[0]
        gap = predicted_reward_all - max_predicted_reward
        batchsize = gap.shape[0] 

        if self.algorithm == "Rand":
            return 1 / self.n_actions * torch.ones_like(gap)  

        elif self.algorithm == "Greedy":
            max_index = gap.argmax(dim=1)
            prob = torch.zeros_like(gap)
            prob[torch.arange(batchsize), max_index] = 1
            return prob

        elif self.algorithm == "EG":  # Epsilon Greedy
            max_index = gap.argmax(dim=1)
            prob = torch.zeros_like(gap)
            prob[torch.arange(batchsize), max_index] = 1
            prob = (1-self.eps) * prob + self.eps / self.n_actions * torch.ones_like(gap)  
            return prob

        elif self.algorithm == "IGW":  # Inverse Gap Weighting
            max_index = gap.argmax(dim=1)
            prob = 1 / (self.n_actions - self.ld * gap)
            row_sums = prob.sum(dim=1)
            new_values = 1 - (row_sums - prob[torch.arange(batchsize), max_index])
            prob[torch.arange(batchsize), max_index] = new_values
            return prob

        elif self.algorithm == "BE":  # Boltzmann Exploration
            prob = torch.exp(self.ld * gap)
            prob /= torch.sum(prob)
            return prob
        
        elif self.algorithm == "PPO":  # PPO
            return self(x)

        elif self.algorithm == "PG": 
            return torch.softmax(self(x), dim=1)

    def update_batch(self, batch_x, batch_action, batch_reward, batch_action_prob):
        """
        Args:
            batch_x: a batch of contexts
            batch_action: the batch of actions chosen in those contexts
            batch_reward: the batch of rewards observed
            batch_action_prob: the batch of probabilities for only those chosen actions
        """

        
        if self.algorithm == "PPO" or self.algorithm == "PG":
            # calculate baseline
            reward_prediction = self.forward_baseline(batch_x)
            batch_reward_estimator = batch_reward - reward_prediction - self.b

            batch_reward_estimator = batch_reward_estimator.detach()
            batch_reward = batch_reward.detach()
            batch_action_prob = batch_action_prob.detach()

            loss_value = torch.mean((reward_prediction - batch_reward_estimator)**2)
        
            if self.algorithm == "PPO": 
                for _ in range(self.M):
                    new_probs_all = self(batch_x)
                    new_action_prob = new_probs_all.gather(1, batch_action)
                    ratios = new_action_prob / batch_action_prob 
            
                    # payoff = torch.mean( torch.min(ratios * batch_reward_estimator, torch.clamp(ratios, max=1.1) * batch_reward_estimator ) )
                    payoff = torch.mean(ratios * batch_reward_estimator)

                    # ratios = new_action_prob / batch_action_prob
                    kl = torch.mean(ratios - 1 - torch.log(ratios))

                    loss_policy = -payoff + self.kl_coeff * kl

                    # baseline network
                    reward_prediction = self.forward_baseline(batch_x)
                    loss_value = torch.mean((reward_prediction - batch_reward)**2)

                    loss = loss_policy + 0.1 * loss_value

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            elif self.algorithm == "PG": 
                logits_all = self(batch_x)
                logits_action = logits_all.gather(1, batch_action)
                payoff = torch.mean(batch_reward_estimator * logits_action) 
                loss_policy = -payoff

                # baseline network
                reward_prediction = self.forward_baseline(batch_x)
                loss_value = torch.mean((reward_prediction - batch_reward)**2)

                loss = loss_policy + 0.1 * loss_value
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        else:
            for _ in range(self.M):
                all_predicted_rewards = self(batch_x)  
                predicted_rewards = all_predicted_rewards.gather(1, batch_action)
                loss = F.mse_loss(predicted_rewards, batch_reward)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



def train_single_param(args, n_seeds=5):
    """
    Main training loop for a single parameter setting. Returns results instead of plotting.
    """
    T = y.shape[0]
    accuracy_values_avg = np.zeros((T,))
    accuracy_values_recent_avg = np.zeros((T,))

    for sd in range(n_seeds): 
        torch.manual_seed(sd)
        random.seed(sd)
        np.random.seed(sd)

        print(f"running for seed = {sd}")

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        model = BanditModel(context_dim, n_actions, args)

        accuracy_values = np.zeros((T,))
        accuracy_values_recent = np.zeros((T,))
        t = 0
        batch_context = []
        batch_action = []
        batch_reward = []
        batch_action_prob = []
        
        
        for context, label in train_loader:
            switch_point = y.shape[0] // 2
            if t < switch_point: 
                reward_vector = 0.5 * F.one_hot(label, n_actions).float() 
            else:  
                reward_vector = 0.5 * F.one_hot(label, n_actions).float() + F.one_hot( (label+1)%n_actions, n_actions).float() 
                
            action_prob_all = model.act(context, t)
            
            action = torch.multinomial(action_prob_all, num_samples=1)
            reward = torch.gather(reward_vector, 1, action)
            action_prob = torch.gather(action_prob_all, 1, action)

            batch_context.append(context)
            batch_action.append(action)
            batch_reward.append(reward)
            batch_action_prob.append(action_prob)
                    
            if (t + 1) % model.N == 0:
                batch_context = torch.cat(batch_context, dim=0)
                batch_action = torch.cat(batch_action, dim=0)
                batch_reward = torch.cat(batch_reward, dim=0)
                batch_action_prob = torch.cat(batch_action_prob, dim=0)
                model.update_batch(
                    batch_context, batch_action, batch_reward, batch_action_prob
                )
                batch_context = []
                batch_action = []
                batch_reward = []
                batch_action_prob = []
                        

            accuracy_values[t] = reward
            recent = accuracy_values[max(0,t-500): t]
            accuracy_values_recent[t] = np.mean(recent)
            t += 1

        accuracy_values_avg += accuracy_values
        accuracy_values_recent_avg += accuracy_values_recent
    
    accuracy_values_avg /= n_seeds
    accuracy_values_recent_avg /= n_seeds

    return accuracy_values_avg, accuracy_values_recent_avg


def run_multiple_params(algorithm, param_values, param_name, n_seeds=5):
    """
    Run a single algorithm with multiple parameter values and plot results.
    """
    T = y.shape[0]
    all_results = []
    param_labels = []
    
    for param_val in param_values:
        print(f"\n=== Running {algorithm} with {param_name}={param_val} ===")
        
        # Create args object with current parameter value
        class Args:
            def __init__(self, algorithm, param_name, param_val):
                self.algorithm = algorithm
                if param_name == 'eps':
                    self.eps = param_val
                elif param_name == 'ld':
                    self.ld = param_val
                elif param_name == 'b':
                    self.b = param_val
                    self.kl_coeff = 0.1  # default
                elif param_name == 'kl_coeff':
                    self.b = 0.1       # default
                    self.kl_coeff = param_val
        
        args = Args(algorithm, param_name, param_val)
        accuracy_values_avg, accuracy_values_recent_avg = train_single_param(args, n_seeds)
        
        all_results.append(accuracy_values_recent_avg)
        param_labels.append(f"{param_name}={param_val}")
        
        # Print summary for this parameter
        switch_point = T // 2
        print(f"Average reward (1st phase): {np.mean(accuracy_values_avg[:switch_point]):.4f}")
        print(f"Average reward (2nd phase): {np.mean(accuracy_values_avg[switch_point:]):.4f}")
        print(f"Average reward (overall): {np.mean(accuracy_values_avg):.4f}")
    
    # Plot all results together
    plt.figure(figsize=(12, 8))
    for i, (result, label) in enumerate(zip(all_results, param_labels)):
        plt.plot(np.arange(T), result, label=label, linewidth=2)
    
    plt.title(f"Average Reward over Time ({algorithm})")
    plt.xlabel("Time Step")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()
    
    # Save results
    for i, (result, param_val) in enumerate(zip(all_results, param_values)):
        filename = f"results_{algorithm}_{param_name}_{param_val}.npy"
        np.save(filename, result)
        print(f"Saved results to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bandit algorithms with multiple parameters")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="EG",
        choices=["Rand", "EG", "IGW", "BE", "PPO", "Greedy", "PG"],
        help="Algorithm to run",
    )
    parser.add_argument(
        "--eps_values",
        type=float,
        nargs='+',
        default=[0.01],
        help="epsilon values for epsilon greedy (space-separated)",
    )
    parser.add_argument(
        "--ld_values",
        type=float,
        nargs='+',
        default=[10],
        help="lambda values for boltzmann exploration or inverse gap weighting (space-separated)",
    )
    parser.add_argument(
        "--b_values",
        type=float,
        nargs='+',
        default=[0.1],
        help="baseline values for PPO or PG (space-separated)",
    )
    parser.add_argument(
        "--kl_coeff_values",
        type=float,
        nargs='+',
        default=[0.1],
        help="KL penalty coefficient values for PPO (space-separated)",
    )
    args = parser.parse_args()
    
    # Determine which parameter to vary based on what's provided
    if args.algorithm == "EG":
        param_values = args.eps_values
        param_name = "eps"
    elif args.algorithm in ["IGW", "BE"]:
        param_values = args.ld_values
        param_name = "ld"
    elif args.algorithm == "PPO":
        if len(args.kl_coeff_values) > 1:  # non-default kl_coeff_values provided
            param_values = args.kl_coeff_values
            param_name = "kl_coeff"
        else:  
            param_values = args.b_values
            param_name = "b"
    elif args.algorithm == "PG":
        param_values = args.b_values
        param_name = "b"
    else:
        param_values = [None]
        param_name = "none"
    
    run_multiple_params(args.algorithm, param_values, param_name)
