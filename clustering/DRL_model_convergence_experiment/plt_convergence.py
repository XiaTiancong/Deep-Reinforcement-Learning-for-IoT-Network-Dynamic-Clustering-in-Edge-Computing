import matplotlib.pyplot as plt
import numpy as np



total_point_num = 5000

def moving_average(a, n=20):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

average_reward = []


f = open("DRL_results_25%_real_data_benchmark.txt", "r")    #DRL_results_real_data_new   #DRL_model_full_synthetic_data #DRL_results_real_data_benchmark #"DRL_results_real_data_new_25%
counter = 0
for x in f:
    average_reward.append(float(x.strip()))
    counter += 1
    if counter > total_point_num:
        break
f.close()

#average_reward = average_reward1 + average_reward
average_reward = moving_average(average_reward)

counter = 0
average_reward2 = []
f2 = open("DRL_model_full_synthetic_data.txt", "r")
for x in f2:
    average_reward2.append(float(x.strip()))
    counter += 1
    if counter > total_point_num:
        break
f2.close()


average_reward1 = moving_average(average_reward2)

x = np.arange(0, len(average_reward))
x2 = np.arange(0, len(average_reward2))

parameter = np.polyfit(x, average_reward, 4)
parameter2 = np.polyfit(x2, average_reward2, 4)

p = np.poly1d(parameter)
p2 = np.poly1d(parameter2)
plt.plot(x, average_reward, "c-")
#plt.plot(x1, average_reward1, "bo")
plt.plot(x, p(x), 'b--', linewidth = 3)

plt.legend(("Overfitting-reduced Dynamic Clustering with Weighted Voronoi and D3QN (Moving Average)", "Overfitting-reduced Dynamic Clustering with Weighted Voronoi and D3QN (Polynomial Curve Fitting)"), loc="lower right")
plt.title('DQN Model (Real-world Truck Trace) Training Convergence Diagram')
plt.xlabel('Training Epochs')
plt.ylabel('Reward')
#plt.plot(x, p1(x), color = 'b')
#plt.plot(x, p2(x), color = 'y')
plt.ylim(ymin=0, ymax=1)
plt.show()