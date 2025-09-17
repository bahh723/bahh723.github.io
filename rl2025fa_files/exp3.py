import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

random.seed(0)

def bernoulli(p):
    return 1 if random.random() < p else 0

c = 1
n_arm = 2
r1 = [0.2, 0.8]
r2 = [0.8, 0.2]
n_steps = 1500
n_draws = [0,0]

probs = []
time_steps = range(n_steps)
rewards = np.zeros((n_arm, n_steps))    # arm * n_steps
for i in range(n_steps):
    for j in range(n_arm): 
        rewards[j,i] = r1[j] if i < 500 else r2[j]


p = np.array([0.5, 0.5])
eta = np.sqrt(n_arm)/np.sqrt(n_steps)
eps = 1/np.sqrt(n_steps)
eta = 1/np.sqrt(n_steps)

exploration = 'implicit'    # could be naive, explicit (first solution), implicit (second solution)


def update(i): 
    global p
    global exploration

    if exploration == 'naive': 
        p_act = p
        arm = np.random.choice(range(n_arm), p=p_act)
        reward = bernoulli(rewards[arm, i])
        reward_estimator = reward / p_act[arm]
        
    elif exploration == 'explicit': 
        p_act = (1 - 2 * eps) * p + 2 * eps * np.array([0.5, 0.5])
        arm = np.random.choice(range(n_arm), p=p_act) 
        reward = bernoulli(rewards[arm, i])
        reward_estimator = (reward) / (p_act[arm])

    elif exploration == 'implicit':
        p_act = p 
        arm = np.random.choice(range(n_arm), p=p_act)
        reward = bernoulli(rewards[arm, i])
        reward_estimator = (reward-1) / p_act[arm]
         

    # exponential weights
    p[arm] *= np.exp(eta * reward_estimator)
    p = p / np.sum(p)
    
    n_draws[arm] += 1
        
    probs.append(p_act[0])
    

for i in range(n_steps): 
    update(i)


plt.plot(time_steps, rewards[0,:], label="expected reward of arm 1", color='blue')
plt.plot(time_steps, rewards[1,:], label="expected reward of arm 2", color='green')
plt.plot(time_steps, probs, label="Probability of Choosing Arm 1", color='orange')
plt.legend()
plt.ylim(0, 1)
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.show()

