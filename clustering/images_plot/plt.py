import matplotlib.pyplot as plt
import numpy as np

total_point_num = 1000

def moving_average(a, n=20):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

average_reward = []
f = open("dqn_results_Final_without_sufficient_data.txt", "r")    #dqn_results_with_naive_model_and_insufficient_data
counter = 0
for x in f:
    average_reward.append(float(x.strip()))
    counter += 1
    if counter > total_point_num:
        break
f.close()
counter = 0
average_reward1 = []
f1 = open("dqn_results_Final_static.txt", "r")
for x in f1:
    average_reward1.append(float(x.strip()))
    counter += 1

    if counter > total_point_num:
        break
f1.close()
'''
counter = 0
average_reward1 = []
f1 = open("DRL_results_real_data_new.txt", "r")
for x in f1:
    average_reward1.append(float(x.strip()))
    counter += 1

    if counter > total_point_num:
        break
f1.close()
'''


#average_reward = average_reward1 + average_reward
average_reward = moving_average(average_reward)
average_reward1 = moving_average(average_reward1)
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
x1 = np.arange(0, len(average_reward1))
#x2 = np.arange(0, len(average_reward2))

parameter = np.polyfit(x, average_reward, 4)
parameter1 = np.polyfit(x1, average_reward1, 4)
#parameter2 = np.polyfit(x2, average_reward2, 4)

p = np.poly1d(parameter)
p1 = np.poly1d(parameter1)
#p2 = np.poly1d(parameter2)
plt.plot(x, average_reward, "b-")
#plt.plot(x1, average_reward1, "bo")
plt.plot(x, p(x), 'b--', linewidth = 3)
plt.plot(x, p1(x), 'g--', linewidth = 3)
#plt.plot(x, p2(x), color = 'y')
plt.legend(("Dynamic Clustering with Weighted Voronoi and D3QN (Moving Average)", "Dynamic Clustering with Weighted Voronoi and D3QN (Polynomial Curve Fitting)", "Static Clustering (Polynomial Curve Fitting)"), loc="lower right")
plt.title('DQN Model (Real-world Truck Trace) Test Performance Diagram')
plt.xlabel('Test Epochs')
plt.ylabel('Reward')

#plt.plot(x, p1(x), color = 'b')
#plt.plot(x, p2(x), color = 'y')
plt.ylim(ymin=0, ymax=1)
plt.show()