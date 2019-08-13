import matplotlib.pyplot as plt
import numpy as np

total_point_num = 5000

def moving_average(a, n=20):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

average_reward = []
f = open("dqn_results_25%_real_world_data_benchmark_40_nodes.txt", "r")    #dqn_results_full_synthetic_data   #dqn_results_full_synthetic_data_but_without_combined_probability #dqn_results_static #dqn_results_full_real_world_data
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
average_reward1 = []
f1 = open("dqn_results_25%_real_world_data_40_nodes.txt", "r") #dqn_results_full_real_world_data_2_clusters
for x in f1:
    average_reward1.append(float(x.strip()))
    counter += 1
    if counter > total_point_num:
        break
f1.close()

average_reward1 = moving_average(average_reward1)

counter = 0
average_reward2 = []
f2 = open("dqn_results_25%_synthetic_data_40_nodes_new.txt", "r")
for x in f2:
    average_reward2.append(float(x.strip()))
    counter += 1
    if counter > total_point_num:
        break
f2.close()

average_reward2 = moving_average(average_reward2)

counter = 0
average_reward3 = []
f3 = open("dqn_results_static_40_nodes.txt", "r")
for x in f3:
    average_reward3.append(float(x.strip()))
    counter += 1
    if counter > total_point_num:
        break
f3.close()

average_reward3 = moving_average(average_reward3)

x = np.arange(0, len(average_reward))

x1 = np.arange(0, len(average_reward1))
x2 = np.arange(0, len(average_reward2))
x3 = np.arange(0, len(average_reward3))

parameter3 = np.polyfit(x3, average_reward3, 4)
parameter2 = np.polyfit(x2, average_reward2, 4)

parameter1 = np.polyfit(x1, average_reward1, 4)
parameter = np.polyfit(x, average_reward, 4)

p = np.poly1d(parameter)
p1 = np.poly1d(parameter1)
p2 = np.poly1d(parameter2)

p3 = np.poly1d(parameter3)
#plt.plot(x, average_reward, "c-")
#plt.plot(x1, average_reward1, "bo")
plt.plot(x, p(x), 'g-.', linewidth = 2)
plt.plot(x1, p1(x1), 'b--', linewidth = 2)

plt.plot(x2, p2(x2), 'r-', linewidth = 2)


plt.plot(x3, p3(x3), 'k:', linewidth = 2)

plt.legend(( "Dynamic Clustering", "Dynamic Clustering with Target Behavior Prediction","Dynamic Clustering with Synthetic Data and Target Behavior Prediction","Static Clustering"), loc="higher right")
#plt.title('DQN Model (Real-world Truck Trace) Training Convergence Diagram')
plt.xlabel('Test Epochs')
plt.ylabel('Reward')
#plt.plot(x, p1(x), color = 'b')
#plt.plot(x, p2(x), color = 'y')
plt.ylim(ymin=0, ymax=1)
plt.xlim(xmin=0, xmax=1000)
plt.show()