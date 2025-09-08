import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


random.seed(1)

def bernoulli(p):
    return 1 if random.random() < p else 0

c = 1
n_arm = 4
r = [0.2, 0.4, 0.6, 0.7]
n_steps = 800
n_draws = [0,0,0,0]
avg_reward = [0,0,0,0]

r_est = [0, 0, 0, 0]

# plt.ion()

def update(i): 
    if i < n_arm: 
        arm = i
        reward = bernoulli(r[arm])
        n_draws[arm] += 1
        avg_reward[arm] = reward
    else: 
        opt_reward = [avg_reward[a] + c * np.sqrt(np.log(i)/n_draws[a]) for a in range(n_arm)] 
        arm = np.argmax(opt_reward)
        reward = bernoulli(r[arm])
        n_draws[arm] += 1
        avg_reward[arm] = (n_draws[arm] -1) / n_draws[arm] * avg_reward[arm] + 1/n_draws[arm] * reward
    
        err = [  c * np.sqrt(np.log(i)/n_draws[a])  for a in range(n_arm) ]

        plt.clf()
        plt.errorbar(range(n_arm), avg_reward, yerr=err, fmt='o', ecolor='black', capsize=5)
        plt.scatter(range(n_arm), r, marker='*', color='red')
        for i in range(n_arm):
            plt.text(i, avg_reward[i], f'{n_draws[i]}', fontsize=14, ha='right', va='bottom')

        plt.ylim(-1, 1.5)
        plt.xticks([0,1,2,3], ['1', '2', '3', '4'])
        plt.title("UCB")
        plt.xlabel("Arm Index")
        plt.ylabel("Confidence Interval")



        # plt.show()
        # plt.pause(0.001)

fig = plt.figure()

# Create the animation
ani = FuncAnimation(fig, update, frames=n_steps, repeat=False)

# Save the animation as a GIF
ani.save('ucb_animation.gif', writer='pillow', fps=60)

# To display the animation in a Jupyter notebook or a Python script
# plt.show()




