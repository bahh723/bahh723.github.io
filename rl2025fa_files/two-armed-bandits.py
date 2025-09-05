import numpy as np
import matplotlib.pyplot as plt

T = 10000
Delta_list = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.30]

# Flexible testing framework for methods and parameters
def BE(reward_sq, reward_mean, lamb_func):
    r_sum = np.zeros(2)
    n = np.ones(2)
    regret = 0
    for t in range(1, T + 1):
        lamb = lamb_func(t)
        mu = r_sum / n
        probs = np.exp(lamb * mu) / np.sum(np.exp(lamb * mu))
        at = np.random.choice(2, p=probs)
        rt = reward_sq[t - 1, at]
        r_sum[at] += rt
        n[at] += 1
        regret += max(reward_mean) - reward_mean[at]
    return regret

def IGW(reward_sq, reward_mean, lamb_func): 
    r_sum = np.zeros(2)
    n = np.ones(2)
    regret = 0
    for t in range(1, T + 1):
        lamb = lamb_func(t)
        mu = r_sum / n
        bt = np.argmax(mu)
        gap = mu[bt] - mu[1-bt]
        probs = np.zeros(2)
        probs[1-bt] = 1/(2 + lamb * gap)
        probs[bt] = 1 - probs[1-bt]
        at = np.random.choice(2, p=probs)
        rt = reward_sq[t - 1, at]
        r_sum[at] += rt
        n[at] += 1
        regret += max(reward_mean) - reward_mean[at]
    return regret

def EG(reward_sq, reward_mean, epsilon_func):
    r_sum = np.zeros(2)
    n = np.ones(2)
    regret = 0
    for t in range(1, T + 1):
        epsilon = epsilon_func(t)
        if np.random.rand() < epsilon:
            at = np.random.choice(2)
        else:
            mu = r_sum / n
            at = np.argmax(mu)
        rt = reward_sq[t - 1, at]
        r_sum[at] += rt
        n[at] += 1
        regret += max(reward_mean) - reward_mean[at]
    return regret

def generate_rewards(T, reward_mean):
    reward_sq = np.zeros((T, 2))
    for i in range(2):
        reward_sq[:, i] = np.random.binomial(1, reward_mean[i], size=T)  # Binary rewards (0 or 1)
    return reward_sq

def test_methods(methods, Delta_list, T, num_seeds=30):
    results = {method_name: {"means": [], "stds": []} for method_name in methods}

    for d in Delta_list:
        reward_mean = [0.5, 0.5 - d]
        delta_results = {method_name: [] for method_name in methods}

        for seed in range(num_seeds):
            print(f"Running Delta={d}, Seed={seed+1}")
            np.random.seed(seed)
            reward_sq = generate_rewards(T, reward_mean)

            for method_name, (method_func, param_func) in methods.items():
                delta_results[method_name].append(method_func(reward_sq, reward_mean, param_func))

        for method_name in methods:
            results[method_name]["means"].append(np.mean(delta_results[method_name]))
            results[method_name]["stds"].append(np.std(delta_results[method_name]))

    return results

# Define the five configurations to compare
methods = {
    r"BE ($\lambda = \sqrt{t}$)": (BE, lambda t: np.sqrt(t)),
    r"BE ($\lambda = 2\log(t)$)": (BE, lambda t: 2 * np.log(t)),
    r"EG ($\epsilon = t^{-1/3}$)": (EG, lambda t: t**(-1/3)),
    r"EG ($\epsilon = t^{-1/3} / 8$)": (EG, lambda t: (t**(-1/3)) / 8),
    r"IGW ($\lambda = 8\sqrt{t}$)": (IGW, lambda t: 8 * np.sqrt(t))
}

# Run the experiment
results = test_methods(methods, Delta_list, T)

# Determine y-range
all_means = [data["means"] for data in results.values()]
y_min = -150 #min(min(m) for m in all_means)
y_max = 300 #max(max(m) for m in all_means)

# Plot results
plt.figure()
for method_name, data in results.items():
    means = data["means"]
    stds = data["stds"]
    plt.plot(Delta_list, means, label=method_name, marker='o')
    plt.fill_between(Delta_list, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)

plt.ylim(y_min, y_max)
plt.xlabel("Delta (Difference in Mean Rewards)", fontsize=12)
plt.ylabel("Total Regret", fontsize=12)
plt.title("Comparison of BE, EG, and IGW Methods", fontsize=14)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
