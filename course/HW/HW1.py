import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import random


# may experiment over multiple different seeds to get more precise performance estimation
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# =============================================
#  Data loading and preprocessing 
#  - converting labels to one hot reward vectors
# =============================================

digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

def to_one_hot(y_tensor, num_classes=None):
    # Create a mapping from the unique labels to the range 0 to n-1
    unique_labels = torch.unique(y_tensor)
    label_to_index = {label.item(): index for index, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    mapped_labels = torch.tensor([label_to_index[label.item()] for label in y_tensor])
    return torch.eye(num_classes)[mapped_labels]

y_one_hot = to_one_hot(y_tensor)
train_dataset = TensorDataset(X_tensor, y_one_hot)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

context_dim = X_tensor.shape[1]
n_actions = y_one_hot.shape[1]

print("n_rounds=", X_tensor.shape[0])
print("context_dim=", context_dim)
print("n_actions=", n_actions)

# ===================================================
#  Define Contextual Bandit Models
#  - multiple TODO's
#  - feel free to modify the structure, add functions or variables 
#    even in places not marked with TODO's
# ===================================================

class LinearCB: 
    def __init__(self, dim, n_actions): 
        self.dim = dim
        self.n_actions = n_actions

    def act(self, x, algorithm='Random'):
        if algorithm == 'Random': 
            return random.randrange(self.n_actions)
        
        elif algorithm == 'Greedy': 
            # TODO
            pass

        elif algorithm == 'LinUCB': 
            # TODO
            pass 
        
        elif algorithm == 'ThompsonSampling': 
            # TODO
            pass
        
        elif algorithm == 'EpsilonGreedy': 
            # TODO
            pass
        
        elif algorithm == 'InverseGapWeighting': 
            # TODO
            pass
        
        elif algorithm == 'Boltzmann':
            # TODO
            pass

    def update(self, x, action, reward):
        pass

class ReplayBuffer:
    def __init__(self):
        self.buffer = []
    
    def push(self, data):
        self.buffer.append(data)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class GeneralCB(nn.Module): 
    def __init__(self, dim, n_actions): 
        self.dim = dim
        self.n_actions = n_actions
        super(GeneralCB, self).__init__()

        # define network structure 
        self.fc1 = nn.Linear(self.dim, 32)
        self.fc2 = nn.Linear(32, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.buffer = ReplayBuffer()
        self.batch_size = 128

    def forward(self, x): 
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

    def act(self, x, algorithm='Random'):
        if algorithm == 'Random': 
            return random.randrange(self.n_actions)
        
        elif algorithm == 'Greedy': 
            values = self(x)
            return torch.argmax(values).item()
        
        elif algorithm == 'EpsilonGreedy': 
            # TODO
            pass

        elif algorithm == 'InverseGapWeighting': 
            # TODO
            pass

        elif algorithm == 'Boltzmann':
            # TODO
            pass
            
       
    def update(self, x, action, reward):
        self.buffer.push((x, action, reward))

        # Updating the regression procedure using one batch of data from the replay buffer
        # no need to change
        for iter in range(1):
            batch = self.buffer.sample(self.batch_size)
            batch_x = torch.stack([item[0] for item in batch])  # shape = [batch_size, ]
            batch_actions = torch.tensor([item[1] for item in batch])
            batch_rewards = torch.tensor([item[2] for item in batch], dtype=torch.float)
        
            all_predicted_rewards = self(batch_x)  # shape = [batch_size, n_actions]
            batch_actions = batch_actions.unsqueeze(1)  # change the shape of batch_actions from [batch_size] to [batch_size, 1]
            predicted_rewards = all_predicted_rewards.gather(1, batch_actions).squeeze(1) # shape = [batch_size]
        
            loss = F.mse_loss(predicted_rewards, batch_rewards)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# ==========================================
#  Main training loop
# ==========================================

t_values = []
accuracy_values = []
accuracy_values_avg = []
t = 0

model = LinearCB(context_dim, n_actions)
bandit_feedback = True   # True to simulate contextual bandit environments
                         # False to simulate full-information online learning environments (benchmark)

for context, reward_vector in train_loader:
    context = context.squeeze(0)   
    reward_vector = reward_vector.squeeze(0)
    action = model.act(context)
    reward = reward_vector[action]
    
    if bandit_feedback: 
        model.update(context, action, reward)
    else: 
        for a in range(n_actions):
            model.update(context, torch.tensor(a), reward_vector[a])

    t_values.append(t)
    accuracy_values.append(reward)
    accuracy_values_avg.append(np.sum(accuracy_values)/(1+t))
    t += 1


# =======================================
#  Output some results 
# =======================================
print("average_reward=", accuracy_values_avg[-1])


# plt.figure()
# plt.plot(t_values, accuracy_values_avg)
# plt.title('Average Reward over Time')
# plt.xlabel('Time Step')
# plt.ylabel('Avrage Reward')
# plt.show()
