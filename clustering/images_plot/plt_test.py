import matplotlib.pyplot as plt
import numpy as np

total_point_num = 17000

def moving_average(a, n=30):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

average_reward = []
f = open("dqn_results_Final_without_combined_prob.txt", "r")
counter = 0
for x in f:
    average_reward.append(float(x.strip()))
    counter += 1
    if counter > total_point_num:
        break
f.close()

counter = 0
average_reward1 = []
f1 = open("dqn_results_Final_with_simi_combined_prob.txt", "r")
for x in f1:
    average_reward1.append(float(x.strip()))
    counter += 1

    if counter > total_point_num:
        break
f1.close()


counter = 0
average_reward2 = []
f2 = open("dqn_results_Final_with_combined_prob.txt", "r")
for x in f2:
    average_reward2.append(float(x.strip()))
    counter += 1
    if counter > total_point_num:
        break
f1.close()

average_reward1 = moving_average(average_reward1)
average_reward = moving_average(average_reward)
average_reward2 = moving_average(average_reward2)

x = np.arange(0, len(average_reward))
x1 = np.arange(0, len(average_reward1))
x2 = np.arange(0, len(average_reward2))

parameter = np.polyfit(x, average_reward, 3)
parameter1 = np.polyfit(x1, average_reward1, 3)
parameter2 = np.polyfit(x2, average_reward2, 3)

p = np.poly1d(parameter)
p1 = np.poly1d(parameter1)
p2 = np.poly1d(parameter2)

plt.plot(x, p(x), 'g:', linewidth = 3)
plt.plot(x1, p1(x1), 'b--', linewidth = 3)
plt.plot(x2, p2(x2), 'r-', linewidth =3)
plt.legend(("Argmax Action Selection (Polynomial Curve Fitting)", "Semi-combined Probability Action Selection (Polynomial Curve Fitting)", "Fully Combined Probability Action Selection (Polynomial Curve Fitting)"), loc = 'lower right')
'''
plt.plot(x, average_reward, "g-")
plt.plot(x1, average_reward1, "b-")
plt.plot(x2, average_reward2, "r-")

plt.plot(x, p(x), 'g:', linewidth = 5)
plt.plot(x1, p1(x1), 'b--', linewidth = 5)
plt.plot(x2, p2(x2), 'r-', linewidth =5)
'''

plt.title('Dynamic Clustering with LSTM Predictor at Decision Making Phase')
plt.xlabel('Test Epochs')
plt.ylabel('Reward')
plt.show()