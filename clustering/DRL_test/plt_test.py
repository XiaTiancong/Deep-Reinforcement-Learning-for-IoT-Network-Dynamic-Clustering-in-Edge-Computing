import matplotlib.pyplot as plt
import numpy as np

total_point_num = 5000

def moving_average(a, n=20):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

average_reward = []
f = open("dqn_results_static_.txt", "r")    #dqn_results_full_synthetic_data   #dqn_results_full_synthetic_data_but_without_combined_probability #dqn_results_static #dqn_results_full_real_world_data
counter = 0
for x in f:
    average_reward.append(float(x.strip()))
    counter += 1
    if counter > total_point_num:
        break
f.close()

#average_reward = average_reward1 + average_reward
average_reward = moving_average(average_reward)
'''
counter = 0
average_reward2 = []
f2 = open("dqn_results_test_with_behavior.txt", "r")
for x in f2:
    average_reward2.append(float(x.strip()))
    counter += 1
    if counter > total_point_num:
        break
f1.close()
'''
x = np.arange(0, len(average_reward))
#x2 = np.arange(0, len(average_reward2))

parameter = np.polyfit(x, average_reward, 4)
#parameter2 = np.polyfit(x2, average_reward2, 4)

p = np.poly1d(parameter)
#p2 = np.poly1d(parameter2)
plt.plot(x, average_reward, "c-")
#plt.plot(x1, average_reward1, "bo")
plt.plot(x, p(x), 'b--', linewidth = 3)

#plt.legend(("Overfitting-reduced Dynamic Clustering with Wxtceighted Voronoi and D3QN (Moving Average)", "Overfitting-reduced Dynamic Clustering with Weighted Voronoi and D3QN (Polynomial Curve Fitting)"), loc="lower right")
#plt.title('DQN Model (Real-world Truck Trace) Training Convergence Diagram')
plt.xlabel('Test Epochs')
plt.ylabel('Reward')
#plt.plot(x, p1(x), color = 'b')
#plt.plot(x, p2(x), color = 'y')
plt.ylim(ymin=0, ymax=1)
plt.show()