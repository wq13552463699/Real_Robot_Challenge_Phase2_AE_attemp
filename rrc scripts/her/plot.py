import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# data1 = np.genfromtxt('/home/robert/summer_research/robochallenge/workspace/src/rrc_example_package/rrc_example_package/her/saved_models/difficulty1_1/ddpg.log',delimiter='')
# data1line, = plt.plot(data1[:,-1]*100, alpha=1, linewidth=1, label='run1', color='b')

data2 = np.genfromtxt('/home/robert/summer_research/robochallenge/workspace/src/rrc_example_package/rrc_example_package/her/saved_models/difficulty1_2/ddpg.log',delimiter='')
data2line, = plt.plot(-data2[:,-1], alpha=1, linewidth=1, label='actor_loss', color='r')


plt.ylabel('Mean Q-value of actor actions')
plt.xlabel('Epoch')
plt.title('HER: move cube difficulty 1')
# plt.ylim([0, 100])
# plt.xlim([0, 1e6])
plt.legend(handles=[data2line])

