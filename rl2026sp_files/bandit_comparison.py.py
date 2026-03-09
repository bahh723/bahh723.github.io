import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


T = 4000
WINDOW = 10
NUM_RUNS = 10

def get_reward(arm, t):
    """Get reward for arm (1-indexed) at time t (1-indexed)."""
    if t <= 1000:
        means = {1: 0.75, 2: 0.5}
    else:
        means = {1: 0.25, 2: 0.5}
    return np.random.binomial(1, means[arm])

def sliding_window_prob(actions, window=WINDOW):
    """Compute Pr[a_t=1] using a sliding window."""
    probs = np.full(T, np.nan)
    for t in range(T):
        start = max(0, t - window + 1)
        probs[t] = np.mean(np.array(actions[start:t+1]) == 1)
    return probs

# -------------------------
# 1. Thompson Sampling (c=1)
# -------------------------
def thompson_sampling():
    actions = []
    counts = {1: 0, 2: 0}
    sum_rewards = {1: 0.0, 2: 0.0}
    for t in range(1, T + 1):
        samples = {}
        for a in [1, 2]:
            emp_mean = sum_rewards[a] / counts[a] if counts[a] > 0 else 0.0
            noise = np.random.normal(0, 1)
            explore = 1.0 / np.sqrt(counts[a]) if counts[a] > 0 else 1.0
            samples[a] = emp_mean + 1.0 * explore * noise
        arm = 1 if samples[1] >= samples[2] else 2
        actions.append(arm)
        r = get_reward(arm, t)
        counts[arm] += 1
        sum_rewards[arm] += r
    return actions

# -------------------------
# 2. Epsilon-Greedy (eps=0.01, window=100)
# -------------------------
def epsilon_greedy():
    eps = 0.01
    actions = []
    recent = {1: [], 2: []}
    for t in range(1, T + 1):
        emp_mean = {}
        for a in [1, 2]:
            if len(recent[a]) > 0:
                emp_mean[a] = np.mean(recent[a])
            else:
                emp_mean[a] = 0.0
        if np.random.random() < eps:
            arm = np.random.choice([1, 2])
        else:
            arm = 1 if emp_mean[1] >= emp_mean[2] else 2
        actions.append(arm)
        r = get_reward(arm, t)
        recent[arm].append(r)
        if len(recent[arm]) > 100:
            recent[arm].pop(0)
    return actions

# -------------------------
# 3. EXP3 (eta=0.2)
# -------------------------
def exp3():
    eta = 0.2
    actions = []
    log_w = {1: 0.0, 2: 0.0}
    for t in range(1, T + 1):
        pi_arr = softmax([log_w[1], log_w[2]])
        pi = {1: pi_arr[0], 2: pi_arr[1]}
        arm = np.random.choice([1, 2], p=[pi[1], pi[2]])
        actions.append(arm)
        r = get_reward(arm, t)
        log_w[arm] += eta * r / pi[arm]
    return actions

# -------------------------
# 4. Exp Weight w/ Recent Mean (eta=0.05, window=100)
# -------------------------
def exp_weight():
    eta = 0.05
    actions = []
    recent = {1: [], 2: []}
    log_w = {1: 0.0, 2: 0.0}
    for t in range(1, T + 1):
        pi_arr = softmax([log_w[1], log_w[2]])
        pi = {1: pi_arr[0], 2: pi_arr[1]}
        arm = np.random.choice([1, 2], p=[pi[1], pi[2]])
        actions.append(arm)
        r = get_reward(arm, t)
        recent[arm].append(r)
        if len(recent[arm]) > 100:
            recent[arm].pop(0)
        for a in [1, 2]:
            emp_mean = np.mean(recent[a]) if len(recent[a]) > 0 else 0.0
            log_w[a] += eta * emp_mean
    return actions

# -------------------------
# Run simulations (NUM_RUNS each) and average
# -------------------------
algorithms = [
    ("Thompson Sampling (c=1)", thompson_sampling),
    (r"$\epsilon$-Greedy ($\epsilon=0.01$, window=100)", epsilon_greedy),
    (r"EXP3 ($\eta=0.2$)", exp3),
    (r"EXP w/ Recent Mean ($\eta=0.05$, window=100)", exp_weight),
]

avg_probs = {}
for name, algo_fn in algorithms:
    print(f"Running {name}...")
    all_probs = np.zeros((NUM_RUNS, T))
    for run in range(NUM_RUNS):
        np.random.seed(run)
        actions = algo_fn()
        all_probs[run] = sliding_window_prob(actions)
    avg_probs[name] = np.mean(all_probs, axis=0)

# -------------------------
# Plot
# -------------------------
fig, axes = plt.subplots(4, 1, figsize=(5, 13), sharex=True)
t_range = np.arange(1, T + 1)

for ax, (name, _) in zip(axes, algorithms):
    ax.plot(t_range, avg_probs[name], color='steelblue', linewidth=1)
    ax.axvline(1000, color='red', linestyle='--', alpha=0.7, label='Phase change')
    ax.set_ylabel(r"$\Pr[a_t = 1]$")
    ax.set_title(name)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([1000, 2000, 3000, 4000])
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.92))
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time $t$")
plt.tight_layout()
plt.savefig("bandit_comparison.png", bbox_inches='tight')
plt.show()
