import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(0)

def bernoulli(p):
    # Simulate Bernoulli trial with normal approximation
    return np.random.normal(p, 1)

c = 1
n_arm = 2
r = np.array([0.6, 0.4])  # Rewards
n_steps = 1000
p0 = np.array([0.0001, 0.9999])  # Initial probabilities

eta_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 1000, 2000, 5000, 10000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000]
avg_rewards_method1 = []  # Store average rewards for method 1
avg_rewards_method2 = []  # Store average rewards for method 2

for eta in eta_list:
    avg_reward_method1 = 0
    avg_reward_method2 = 0
    for test_time in range(100):
        p_method1 = p0.copy()
        p_method2 = p0.copy()
        reward_method1 = 0
        reward_method2 = 0
        for t in range(n_steps):
            reward_vec = [bernoulli(r[i]) for i in range(n_arm)]
            reward_vec = np.array(reward_vec)
            adv1 = reward_vec - np.dot(p_method1, reward_vec)
            adv2 = reward_vec - np.dot(p_method2, reward_vec)

            # Method 1 Update
            logp_method1 = np.log(p_method1)
            logp_method1 += eta * adv1
            logp_method1 -= np.max(logp_method1)
            p_method1 = np.exp(logp_method1) / np.sum(np.exp(logp_method1))

            # Method 2 Update
            logp_method2 = np.log(p_method2)
            logp_method2 += eta * (p_method2 * adv2)
            logp_method2 -= np.max(logp_method2)
            p_method2 = np.exp(logp_method2) / np.sum(np.exp(logp_method2))

            reward_method1 += np.dot(p_method1, reward_vec)
            reward_method2 += np.dot(p_method2, reward_vec)

        avg_reward_method1 += reward_method1
        avg_reward_method2 += reward_method2

    avg_rewards_method1.append(avg_reward_method1 / 100)
    avg_rewards_method2.append(avg_reward_method2 / 100)
    print("Exponential weights", avg_reward_method1 / 100)
    print("Policy gradient", avg_reward_method2 / 100)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(eta_list, avg_rewards_method1, marker='o', label='Exponential weights')
plt.plot(eta_list, avg_rewards_method2, marker='x', label='Policy gradidnet')
plt.xscale('log')
plt.xlabel(r'$\eta$')
plt.ylabel('Average Reward')
# plt.title('')
plt.legend()
plt.grid(True)
plt.show()
