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
import torch.distributions as dist


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
#  - finish the update_batch function
# ===================================================

class GeneralPolicyBasedCB(nn.Module): 
    def __init__(self, dim, n_actions): 
        super(GeneralPolicyBasedCB, self).__init__() 
        self.dim = dim
        self.n_actions = n_actions
        self.M = 30    # the number of the inner iteration for updating \theta (denoted as M in HW2.pdf)

        # define a policy network
        self.fc1 = nn.Linear(self.dim, 32)
        self.fc2 = nn.Linear(32, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=0.002)

    def forward(self, x): 
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.Softmax(dim=1)(x)
        return x
    
    def update_batch(self, batch_x, batch_action, batch_reward, batch_action_prob):
        # TODO: calculate \hat{r}_t(x,a), p_{\theta_t}(x,a) for (x,a) in the batch
        # Need to perform the "detach" step for \hat{r}_t(x,a), p_{\theta_t}(x,a)  
        # so that the following for-loop won't calculate the gradient of them   

        for iter in range(self.M):
            # TODO
            pass
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
                

# ==========================================
#  Main training loop
# ==========================================

t_values = []
accuracy_values = []
accuracy_values_avg = []
t = 0
N = 128

batch_context = []
batch_action = []
batch_reward = []
batch_action_prob = []

model = GeneralPolicyBasedCB(context_dim, n_actions)

for context, reward_vector in train_loader:
    action_prob_all = model(context)
    action = torch.multinomial(action_prob_all, num_samples=1)
    reward = torch.gather(reward_vector, 1, action)
    action_prob = torch.gather(action_prob_all, 1, action)

    batch_context.append(context)
    batch_action.append(action)
    batch_reward.append(reward)
    batch_action_prob.append(action_prob)

    if (t+1) % N == 0: 
        batch_context = torch.cat(batch_context, dim=0)
        batch_action = torch.cat(batch_action, dim=0)
        batch_reward = torch.cat(batch_reward, dim=0)
        batch_action_prob = torch.cat(batch_action_prob, dim=0)
        model.update_batch(batch_context, batch_action, batch_reward, batch_action_prob)
        batch_context = []
        batch_action = []
        batch_reward = []
        batch_action_prob = []
    
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
